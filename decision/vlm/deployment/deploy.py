"""Explicit wheel download and remote install/inspect/serve.

Installation and serving are print-only unless --execute is explicitly supplied.
Installation targets the invoking Python environment and requires an additional
--allow-env-change acknowledgement. Never reads credential configurations, creates
virtual environments, downloads weights, or launches the benchmark automatically.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def verify_templates(root=HERE):
    """Protect packaged code/defaults without ever reading server_config.json."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return  # Source checkout, not an exported portable bundle.
    manifest = read_json(manifest_path)
    for name in ("deploy.py", "installation_profile.json", "deployment_defaults.json",
                 "requirements.server.txt", "README.md", "compare_results.py"):
        expected = manifest.get("files", {}).get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing deployment asset hash; extract the complete ZIP")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or linked deployment asset")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Deployment asset changed; use flags, not edits to frozen templates")


def profile(root=HERE):
    result = read_json(Path(root) / "installation_profile.json")
    for name, version in result["packages"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not re.fullmatch(r"[0-9][a-z0-9.+]*", version):
            raise ValueError("Unexpected installation dependency")
    return result


def serving_defaults(root=HERE):
    root = Path(root)
    if (root / "deployment_defaults.json").is_file():
        return read_json(root / "deployment_defaults.json")
    project = read_json(root.parent / "configs/project.json")
    return {"deployment": project["deployment"], "served_name": project["model"]["served_name"]}


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def version_matches(actual, expected):
    # PEP 440: an == pin without a local build suffix also accepts +cu130.
    return isinstance(actual, str) and (actual == expected if "+" in expected
                                       else actual.split("+", 1)[0] == expected)


def output_directory(path, *, model_path=None):
    out = Path(path).expanduser().resolve()
    if out == HERE or out in HERE.parents or (
            out.is_relative_to(HERE) and out.relative_to(HERE).parts[0] != "results"):
        raise ValueError("Use results/ under the bundle or a separate new output directory")
    if model_path is not None and out.is_relative_to(Path(model_path).expanduser().resolve()):
        raise ValueError("Do not write reports into the model directory")
    out.mkdir(parents=True, exist_ok=False)
    return out


def installed_packages():
    """Names/versions only; never export URL-bearing pip freeze or private config."""
    return sorted([{"name": d.metadata["Name"], "version": d.version}
                   for d in metadata.distributions()], key=lambda d: d["name"].lower())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid wheel package name")
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_inventory(folder):
    """Read only wheel metadata/bytes; never import or extract third-party code."""
    rows = []
    names = set()
    for path in sorted(Path(folder).glob("*.whl")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Wheel files must be regular files, not links")
        with zipfile.ZipFile(path) as archive:
            # e.g. setuptools also contains _vendor/*/*.dist-info/METADATA.
            # Only the top-level dist-info describes the wheel being installed.
            metas = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
            if len(metas) != 1 or archive.getinfo(metas[0]).file_size > 2 * 1024**2:
                raise ValueError("Invalid wheel metadata")
            data = BytesParser().parsebytes(archive.read(metas[0]))
            name, version = canonical_name(data["Name"]), data["Version"]
            if not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.+!_-]*", version):
                raise ValueError("Invalid wheel version")
            if name in names:
                raise ValueError("Use exactly one wheel per package in the locked wheelhouse")
            if any("@" in dep for dep in data.get_all("Requires-Dist", [])):
                raise ValueError("Direct-URL wheel dependencies are not allowed in offline resources")
            names.add(name)
            wheel_meta = metas[0].rsplit("/", 1)[0] + "/WHEEL"
            if archive.getinfo(wheel_meta).file_size > 64 * 1024:
                raise ValueError("Invalid wheel tags")
            tags = BytesParser().parsebytes(archive.read(wheel_meta)).get_all("Tag", [])
            if not tags:
                raise ValueError("Wheel is missing compatibility tags")
        rows.append({"file": path.name, "name": name, "version": version,
                     "bytes": path.stat().st_size, "sha256": file_sha256(path), "tags": tags})
    if not rows:
        raise ValueError("No wheel files found")
    return rows


def locked_requirements(rows):
    return "# Complete resolved wheel set; install with --no-index --require-hashes.\n" + "".join(
        f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}\n"
        for row in sorted(rows, key=lambda r: r["name"]))


def offline_resolution_command(folder, python=None):
    folder = Path(folder).resolve()
    return [python or sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
            "install", "--dry-run", "--ignore-installed", "--no-user", "--no-index", "--find-links", str(folder),
            "--only-binary=:all:", "--require-hashes", "--no-cache-dir", "--no-input",
            "-r", str(folder / "requirements.lock")]


def seal_wheelhouse(folder, root=HERE):
    """Lock a completed download, prove dependency closure without installing."""
    folder = Path(folder).resolve()
    rows = wheel_inventory(folder)
    actual = {r["name"]: r["version"] for r in rows}
    for name, version in profile(root)["packages"].items():
        if not version_matches(actual.get(canonical_name(name)), version):
            raise ValueError("Downloaded wheels do not match the installation profile")
    lock = folder / "requirements.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(locked_requirements(rows))
    subprocess.run(offline_resolution_command(folder), check=True,
                   env={**os.environ, "PIP_CONFIG_FILE": os.devnull})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_minor": "3.12", "system": "Linux", "machine": "x86_64",
              "installation_profile_sha256": file_sha256(Path(root) / "installation_profile.json"),
              "requirements_lock_sha256": file_sha256(lock), "wheels": rows,
              "wheel_count": len(rows), "wheel_bytes": sum(r["bytes"] for r in rows),
              "offline_dependency_resolution_passed": True,
              "environment_modified": False, "gpu_or_model_inference_verified": False}
    save_json(folder / "wheelhouse_manifest.json", result)
    return result


def verify_wheelhouse(folder, root=HERE):
    folder = Path(folder).expanduser().resolve(strict=True)
    for name in ("wheelhouse_manifest.json", "requirements.lock"):
        if not (folder / name).is_file() or (folder / name).is_symlink():
            raise ValueError("Missing or linked wheelhouse manifest/lock; upload the complete resources")
    manifest = read_json(folder / "wheelhouse_manifest.json")
    if manifest.get("installation_profile_sha256") != file_sha256(Path(root) / "installation_profile.json"):
        raise ValueError("Wheelhouse belongs to a different installation profile")
    if (manifest.get("python_minor"), manifest.get("system"), manifest.get("machine")) != ("3.12", "Linux", "x86_64"):
        raise ValueError("Wheelhouse target does not match Linux x86_64/Python 3.12")
    if manifest.get("offline_dependency_resolution_passed") is not True:
        raise ValueError("Wheelhouse dependency closure was not validated")
    rows = wheel_inventory(folder)
    if rows != manifest.get("wheels"):
        raise ValueError("Wheelhouse files changed, missing, or unexpected; do not edit the manifest")
    lock = folder / "requirements.lock"
    if (file_sha256(lock) != manifest.get("requirements_lock_sha256") or
            lock.read_text(encoding="utf-8") != locked_requirements(rows)):
        raise ValueError("Wheelhouse requirements lock changed")
    actual = {r["name"]: r["version"] for r in rows}
    if any(not version_matches(actual.get(canonical_name(n)), v) for n, v in profile(root)["packages"].items()):
        raise ValueError("Wheelhouse roots do not match the installation profile")
    return manifest


def download_command(folder, root=HERE, python=None):
    p = profile(root)
    indexes = ["--index-url", p["index_url"]]
    for url in p["extra_index_urls"]:
        indexes.extend(["--extra-index-url", url])
    return [python or sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
            "download", "--dest", str(Path(folder).expanduser().resolve()), *indexes,
            "--only-binary=:all:", "--no-cache-dir", "--no-input", "--retries", "0",
            *(f"{n}=={v}" for n, v in p["packages"].items())]


def archive_wheelhouse(folder):
    folder = Path(folder).resolve()
    manifest = verify_wheelhouse(folder)
    target = folder.with_suffix(".zip")
    if target.exists():
        raise ValueError("Wheel archive already exists; preserve existing resources")
    # Wheels are already ZIP-compressed. Store them without another compression pass.
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in [*(r["file"] for r in manifest["wheels"]), "requirements.lock", "wheelhouse_manifest.json"]:
            archive.write(folder / name, folder.name + "/" + name)
    return {"archive": str(target), "archive_bytes": target.stat().st_size,
            "archive_sha256": file_sha256(target)}


def download(args):
    command = download_command(args.out)
    print(json.dumps({"execute": args.execute, "environment_will_be_modified": False,
                      "command": shlex.join(command)}, indent=2))
    if not args.execute:
        return 0
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Prepare wheels using Linux x86_64 and Python 3.12")
    out = output_directory(args.out)
    subprocess.run(command, check=True, env={**os.environ, "PIP_CONFIG_FILE": os.devnull})
    result = seal_wheelhouse(out)
    summary = {"wheel_count": result["wheel_count"], "wheel_bytes": result["wheel_bytes"],
               "offline_dependency_resolution_passed": True, "environment_modified": False}
    if args.archive:
        summary.update(archive_wheelhouse(out))
    print(json.dumps(summary, indent=2))
    return 0


def install_commands(root=HERE, python=None, wheelhouse=None):
    p = profile(root)
    requirements = [f"{name}=={version}" for name, version in p["packages"].items()]
    interpreter = python or sys.executable
    pip = [interpreter, "-m", "pip", "--isolated"]
    if wheelhouse is None:
        indexes = ["--index-url", p["index_url"]]
        for url in p["extra_index_urls"]:
            indexes.extend(["--extra-index-url", url])
    else:
        folder = Path(wheelhouse).expanduser().resolve()
        pip.append("--disable-pip-version-check")
        indexes = ["--no-index", "--find-links", str(folder), "--require-hashes"]
        requirements = ["-r", str(folder / "requirements.lock")]
    return [
        [*pip, "install", "--upgrade", "--upgrade-strategy", "only-if-needed", "--no-user",
         *indexes,
         "--only-binary=:all:", "--no-cache-dir", "--no-input", "--retries", "0", *requirements],
        [*pip, "check"],
    ]


def install(args):
    wheelhouse = getattr(args, "wheelhouse", None)
    commands = install_commands(wheelhouse=wheelhouse)
    print(json.dumps({"installation_mode": "current_environment",
                      "python_executable": sys.executable, "environment_prefix": sys.prefix,
                      "creates_virtual_environment": False,
                      "dependency_source": "offline_wheelhouse" if wheelhouse else "official_indexes",
                      "environment_will_be_modified_on_execute": True,
                      "warning": "Existing PyTorch and related packages will be replaced; no automatic rollback.",
                      "execute": args.execute, "commands": [shlex.join(c) for c in commands]}, indent=2))
    if not args.execute:
        return 0
    if not args.allow_env_change:
        raise ValueError("Installing into the current environment requires --allow-env-change; save the container image first")
    if args.out is None:
        raise ValueError("--execute requires a NEW --out directory for installation records")
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("This installation profile requires Linux x86_64 and Python 3.12")
    if any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME")):
        raise ValueError("Unset PYTHONPATH/PYTHONHOME in this terminal before changing the target environment")
    if shutil.disk_usage(sys.prefix).free < args.min_free_gib * 1024**3:
        raise ValueError("Insufficient disk space for CUDA/PyTorch replacement; expand storage first")
    wheel_manifest = verify_wheelhouse(wheelhouse) if wheelhouse else None
    out = output_directory(args.out)
    before = installed_packages()
    save_json(out / "installed_packages_before.json", before)
    details = {"installation_mode": "current_environment", "python_executable": sys.executable,
               "environment_prefix": sys.prefix, "creates_virtual_environment": False,
               "profile_sha256": hashlib.sha256((HERE / "installation_profile.json").read_bytes()).hexdigest(),
               "runtime_profile": profile(), "installation_completed": False,
               "gpu_or_model_inference_verified": False, "automatic_rollback": False}
    if wheel_manifest is not None:
        details["dependency_source"] = "offline_wheelhouse"
        details["wheelhouse_manifest_sha256"] = file_sha256(Path(wheelhouse).expanduser().resolve() / "wheelhouse_manifest.json")
        details["locked_packages"] = {r["name"]: r["version"] for r in wheel_manifest["wheels"]}
    save_json(out / "installation_started.json", details)
    install_env = dict(os.environ)
    install_env["PIP_CONFIG_FILE"] = os.devnull
    for name in ("PYTHONPATH", "PYTHONHOME"):
        install_env.pop(name, None)
    # pip output stays on the remote terminal. Do not persist proxy URLs/env or
    # pip's complete dependency report, which could carry authenticated URLs.
    try:
        for command in commands:
            subprocess.run(command, check=True, env=install_env)
        importlib.invalidate_caches()
        actual = package_versions(profile()["packages"])
        if any(not version_matches(actual[name], version) for name, version in profile()["packages"].items()):
            raise ValueError("Installed package metadata does not match the profile; inspect before continuing")
        if wheel_manifest is not None:
            locked = details["locked_packages"]
            installed = package_versions(locked)
            if installed != locked:
                raise ValueError("Installed package metadata does not match the complete offline lock")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        save_json(out / "installation_failed.json", {
            **details, "error_type": type(error).__name__, "environment_may_be_partially_modified": True})
        raise
    after = installed_packages()
    save_json(out / "installed_packages_after.json", after)
    save_json(out / "installation.json", {
        **details, "installation_completed": True, "target_environment_modified": before != after,
        "runtime_packages": actual})
    print("Installation completed. Keep using the same Python for inspect and serve; restart any old Python/Jupyter processes.")
    return 0


def inspect_model(model_path):
    root = Path(model_path).expanduser().resolve()
    config = read_json(root / "config.json")
    index = read_json(root / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(s, str) for s in mapping.values()):
        raise ValueError("Invalid safetensors shard index")
    shards = sorted(set(mapping.values()))
    total = 0
    for name in shards:
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or path.suffix != ".safetensors":
            raise ValueError("Unsafe shard path in model index")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("A referenced model shard is missing or empty")
        total += path.stat().st_size
    quantization = config.get("quantization_config", {})
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise ValueError("Expected an existing FP8 checkpoint, not online re-quantization")
    if not isinstance(config.get("vision_config"), dict):
        raise ValueError("Expected a multimodal model with vision_config")
    if not (root / "tokenizer_config.json").is_file() or not any(
            (root / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Missing tokenizer assets")
    if not any((root / name).is_file() for name in ("preprocessor_config.json", "processor_config.json")):
        raise ValueError("Missing vision processor configuration")
    files = ("config.json", "model.safetensors.index.json", "tokenizer_config.json",
             "preprocessor_config.json", "processor_config.json")
    return {"path": str(root), "architectures": config.get("architectures"),
            "model_type": config.get("model_type"), "quant_method": quantization["quant_method"],
            "weight_block_size": quantization.get("weight_block_size"),
            "shards": len(shards), "weight_bytes": total,
            "metadata_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in files if (root / name).is_file()},
            "all_weight_bytes_hashed": False, "checkpoint_identity_verified": False}


def inspect_environment(*, gpu=False, model_path=None):
    p = profile()
    packages = package_versions(p["packages"])
    issues = [f"package_mismatch:{name}" for name, expected in p["packages"].items()
              if not version_matches(packages[name], expected)]
    if sys.version_info[:2] != (3, 12):
        issues.append("python_minor_mismatch")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        issues.append("platform_mismatch")
    if any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME")):
        issues.append("python_path_overrides_set_unset_before_using_target_env")
    result = {"python": platform.python_version(), "python_executable": sys.executable,
              "platform": platform.system(), "machine": platform.machine(), "packages": packages,
              "gpu_checked": gpu, "model_loaded": False, "inference_requests_sent": 0,
              "issues": issues}
    if model_path is not None:
        result["model"] = inspect_model(model_path)
    if gpu:
        torch = importlib.import_module("torch")
        result["torch_cuda"] = torch.version.cuda
        if torch.version.cuda != p["runtime_torch_cuda"]:
            issues.append("torch_cuda_mismatch")
        if not torch.cuda.is_available():
            issues.append("cuda_unavailable")
        else:
            device = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            result["gpu"] = {"name": device.name, "compute_capability": [device.major, device.minor],
                             "total_mib": total // 1024**2, "free_mib": free // 1024**2}
            if total // 1024**2 < p["minimum_total_memory_mib"]:
                issues.append("insufficient_total_vram_use_48GB_or_larger")
            if (device.major, device.minor) < (8, 9):
                issues.append("gpu_outside_this_native_fp8_profile")
            if free // 1024**2 < 36000:
                issues.append("insufficient_free_vram_check_other_processes")
            # Metadata existence alone cannot prove CUDA is operational.
            tensor = torch.ones((2, 2), device="cuda:0")
            _ = tensor @ tensor
            torch.cuda.synchronize()
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                                  "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                 text=True, timeout=15)
        rows = list(csv.reader(process.stdout.splitlines()))
        result["driver_versions"] = sorted({row[1].strip() for row in rows if len(row) == 4})
        if not result["driver_versions"] or any(
                int(v.split(".")[0]) < p["minimum_driver_major"] for v in result["driver_versions"]):
            issues.append("driver_below_cuda13_minimum")
        result["native_imports"] = {}
        for name in ("sglang", "sglang_kernel", "flashinfer", "xgrammar"):
            try:
                importlib.import_module(name)
                result["native_imports"][name] = "ok"
            except Exception as error:
                result["native_imports"][name] = type(error).__name__
                issues.append(f"native_import_failed:{name}")
    result["environment_matches_profile"] = not issues
    result["launch_preflight_passed"] = bool(gpu and model_path is not None and not issues)
    result["real_multimodal_schema_inference_verified"] = False
    return result


def server_command(args):
    config = serving_defaults()
    defaults = config["deployment"]
    if not 1 <= args.port <= 65535 or not 0.5 <= args.mem_fraction_static <= 0.95:
        raise ValueError("Invalid port/memory fraction")
    if not 4096 <= args.context_length <= 12288:
        raise ValueError("Use a bounded context length between 4096 and 12288")
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path",
               str(Path(args.model_path).expanduser().resolve()), "--served-model-name", config["served_name"],
               "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16",
               "--context-length", str(args.context_length), "--max-running-requests", "1",
               "--mem-fraction-static", str(args.mem_fraction_static), "--enable-multimodal",
               "--grammar-backend", defaults["grammar_backend"]]
    if args.disable_radix_cache:
        command.append("--disable-radix-cache")
    if args.disable_cuda_graph:
        command.append("--disable-cuda-graph")
    return command


def stop_owned_process(process):
    """Only our still-running child's process group; never pkill another service."""
    if process.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=15)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def serve(args):
    command = server_command(args)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    if args.out is None:
        raise ValueError("--execute requires a NEW --out directory for preflight and server logs")
    model = Path(args.model_path).expanduser().resolve()
    out = output_directory(args.out, model_path=model)
    report = inspect_environment(gpu=True, model_path=model)
    save_json(out / "preflight.json", report)
    if not report["launch_preflight_passed"]:
        raise ValueError("Preflight failed; inspect preflight.json. No server started")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))  # Fail instead of terminating an existing service.
    save_json(out / "launch.json", {"command": command, "environment_modified_during_launch": False,
                                    "native_mtp_enabled": False, "inference_requests_sent": 0,
                                    "readiness_is_not_inference_validation": True})
    with (out / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Own server PID: {process.pid}. Log: {out / 'server.log'}. Ctrl+C stops only this service.", flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            stop_owned_process(process)
            code = 130
        finally:
            stop_owned_process(process)
    save_json(out / "exit.json", {"exit_code": code, "automatic_restarts": 0})
    return code if code >= 0 else 1


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    installing = sub.add_parser("install", help="Print a CURRENT Python environment install plan; explicit opt-in required")
    installing.add_argument("--allow-env-change", action="store_true", help="Acknowledge replacement of current PyTorch/dependencies")
    installing.add_argument("--out", type=Path, help="NEW installation record directory; required with --execute")
    installing.add_argument("--min-free-gib", type=int, default=20)
    installing.add_argument("--execute", action="store_true")
    installing.add_argument("--wheelhouse", type=Path, help="Verified local wheel directory; no network fallback")
    downloading = sub.add_parser("download", help="Print wheel download plan; --execute downloads without installing")
    downloading.add_argument("--out", type=Path, required=True, help="NEW wheelhouse output directory")
    downloading.add_argument("--execute", action="store_true")
    downloading.add_argument("--archive", action="store_true", help="Also create a ZIP beside the new wheelhouse")
    inspection = sub.add_parser("inspect", help="Package metadata only; --gpu opts into a small CUDA check")
    inspection.add_argument("--gpu", action="store_true")
    inspection.add_argument("--model-path", type=Path)
    inspection.add_argument("--out", type=Path, help="NEW output directory; optional")
    defaults = serving_defaults()["deployment"]
    serving = sub.add_parser("serve", help="Print command; --execute explicitly starts the GPU service")
    serving.add_argument("--model-path", type=Path, required=True)
    serving.add_argument("--port", type=int, default=defaults["port"])
    serving.add_argument("--context-length", type=int, default=defaults["context_length"])
    serving.add_argument("--mem-fraction-static", type=float, default=defaults["mem_fraction_static"])
    serving.add_argument("--disable-radix-cache", action="store_true")
    serving.add_argument("--disable-cuda-graph", action="store_true")
    serving.add_argument("--execute", action="store_true")
    serving.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.action == "install" and args.min_free_gib < 1:
        parser.error("--min-free-gib must be positive")
    return args


def main(argv=None):
    try:
        verify_templates()
        args = arguments(argv)
        if args.action == "install":
            return install(args)
        if args.action == "download":
            return download(args)
        if args.action == "serve":
            return serve(args)
        report = inspect_environment(gpu=args.gpu, model_path=args.model_path)
        if args.out is not None:
            out = output_directory(args.out, model_path=args.model_path)
            save_json(out / "preflight.json", report)
            # Names and versions only; never persist pip config, env or URL-bearing freeze entries.
            save_json(out / "installed_packages.json", installed_packages())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["environment_matches_profile"] else 1
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print("Deployment stopped:", type(error).__name__, str(error) if isinstance(error, ValueError) else "Check paths/dependencies and remote logs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
