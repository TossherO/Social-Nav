"""Project defaults and print-only manual SGLang command; no environment changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

VLM_ROOT = Path(__file__).resolve().parent
ROOT = VLM_ROOT.parents[1]
CONFIG_DIR = VLM_ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "project.json"


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sections = {"dataset_roots", "model", "deployment", "inference", "benchmark", "development"}
    if not isinstance(config, dict) or any(not isinstance(config.get(name), dict) for name in sections):
        raise ValueError("Expected the complete project configuration")
    return config


def project_path(value):
    """Resolve a configured relative path against the repo, preserving aliases."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def dataset_path(name):
    return project_path(load_config()["dataset_roots"][name])


def server_command(*, model_path=None, port=None, python=None):
    config = load_config()
    model, deployment = config["model"], config["deployment"]
    return [python or deployment["python_executable"], "-m", "sglang.launch_server",
            "--model-path", model_path or model["path"],
            "--served-model-name", model["served_name"], "--host", deployment["host"],
            "--port", str(port or deployment["port"]), "--dtype", "bfloat16",
            "--context-length", str(deployment["context_length"]),
            "--max-running-requests", str(deployment["max_running_requests"]),
            "--mem-fraction-static", str(deployment["mem_fraction_static"]),
            "--enable-multimodal", "--grammar-backend", deployment["grammar_backend"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Actual FP8 checkpoint path on the remote GPU server")
    parser.add_argument("--python", help="Python in the remote, already prepared SGLang environment")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535")
    print(shlex.join(server_command(model_path=args.model_path, port=args.port, python=args.python)))


if __name__ == "__main__":
    main()
