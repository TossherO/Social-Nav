"""Offline portable SGLang test ZIP; copy only frozen non-secret input assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from decision.vlm import vlm_api_common as common
from decision.vlm import benchmark_defaults as defaults
from decision.vlm.project_config import load_config, project_path

VLM_ROOT = Path(__file__).resolve().parent
ROOT = VLM_ROOT.parents[1]
DEPLOYMENT_GUIDE = ROOT / "docs/decision/VLM部署与调用.md"
BUILD_GUIDE = ROOT / "docs/decision/SGLang服务部署方案.md"
DEPLOYMENT_FILES = (
    "start_server.sh", "runtime_env.sh", "compare_results.py",
    "build/build_sglang.sh", "build/source-cu129.patch", "config/constraints-cu129.txt",
)
SOURCE = project_path(load_config()["development"]["bundle"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(out, source=SOURCE, *, include_deployment=False):
    out, source = Path(out).resolve(), Path(source).resolve()
    archive = out.with_suffix(".zip")
    if out.exists() or archive.exists():
        raise ValueError("Use a NEW output directory and archive; preserve existing assets")
    if out.is_relative_to(source) or archive.is_relative_to(source):
        raise ValueError("Package output must be outside the source input bundle")
    exported_defaults = {"settings": defaults.from_project(common.read_json(VLM_ROOT / "configs/project.json"))}
    public_config = load_config()
    manifest, _, cases = common.load_bundle(source)
    if (not common.is_object_profile(manifest.get("prompt_profile")) or manifest.get("wire_protocol") != "object"
            or not cases or manifest.get("credentials_included") is not False):
        raise ValueError("Expected a verified, three-image object source bundle")
    # Derive an exact whitelist, never copy source directories or credentials.
    inputs = {"system_prompt.txt"}
    for case in cases:
        inputs.add(case["entry"]["input_file"])
        inputs.update(case["model"]["images"])
    if len(inputs) != 4 * len(cases) + 1 or any(not p.startswith("cases/") and p != "system_prompt.txt" for p in inputs):
        raise ValueError("Expected case metadata, three images per case and one prefix")
    if any(Path(p).suffix not in {".txt", ".json", ".png"} for p in inputs):
        raise ValueError("Unexpected input asset type")
    out.mkdir(parents=True)
    for name in sorted(inputs):
        original = common.asset_path(source, name)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    for original, name in ((VLM_ROOT / "benchmark_vlm.py", "benchmark.py"),
                          (VLM_ROOT / "vlm_api_common.py", "vlm_api_common.py"),
                          (VLM_ROOT / "benchmark_defaults.py", "benchmark_defaults.py"),
                          (VLM_ROOT / "configs/vlm_test_guide.md", "README.md")):
        shutil.copyfile(original, out / name)
    if include_deployment:
        deployment = VLM_ROOT / "deployment"
        for name in DEPLOYMENT_FILES:
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(deployment / name, out / name)
        shutil.copyfile(DEPLOYMENT_GUIDE, out / DEPLOYMENT_GUIDE.name)
        shutil.copyfile(BUILD_GUIDE, out / BUILD_GUIDE.name)
    (out / "requirements.txt").write_text("httpx>=0.28.1,<0.29\njsonschema>=4.23,<5\n", encoding="utf-8")
    (out / ".gitignore").write_text("/server_config.json\n/results/\n/.venv/\n/.client-env/\n/__pycache__/\n", encoding="utf-8")
    common.save_json(out / "server_config.json", {
        "base_url": f"http://127.0.0.1:{public_config['deployment']['port']}/v1" if include_deployment else "",
        "api_key": "", "model": public_config["model"]["served_name"] if include_deployment else "",
                                                "allow_insecure_http": False})
    (out / "server_config.json").chmod(0o600)
    common.save_json(out / "benchmark_defaults.json", exported_defaults)
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "server_config.json"}
    result = {"task": "JRDB_T1_REL_remote_SGLang_latency",
        "profile": manifest["profile"], "prompt_profile": "object", "wire_protocol": "object",
        "model": "discovered_from_user_SGLang_endpoint", "native_social_labels_included": False,
        "credentials_included": False, "source_manifest_sha256": sha(source / "manifest.json"),
        "source_frozen_input_files": {name: sha(source / name) for name in sorted(inputs)},
        "cases": manifest["cases"], "image_bytes_total": manifest["image_bytes_total"], "files": files,
        "defaults": exported_defaults,
        "limitations": ["development_NOT_blind_or_semantic_gold", "oracle_geometry_and_PID",
                        "server_not_deployed_or_called_during_packaging", "remote_grammar_and_speculation_not_verified",
                        "no_online_detection_rendering_navigation_timing"]}
    if include_deployment:
        result["deployment"] = {
            "installation_mode": "native_conda_source",
            "launch_entrypoint": "start_server.sh",
            "constraints_file": "config/constraints-cu129.txt",
            "environment_modified_during_packaging": False,
            "service_started_during_packaging": False,
            "model_weights_included": False,
            "third_party_sources_or_wheels_included": False}
    common.save_json(out / "manifest.json", result)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*sorted(files), "manifest.json", "server_config.json"]:
            z.write(out / name, out.name + "/" + name)
    return {"archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha(archive), "cases": len(cases), "images": 3 * len(cases),
            "credentials_included": False, "inference_requests_sent": 0,
            "deployment_guide_included": include_deployment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--with-deployment", action="store_true", help="Include native Conda/source-build service materials and current guides")
    args = parser.parse_args()
    print(json.dumps(package(args.out, args.source, include_deployment=args.with_deployment), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
