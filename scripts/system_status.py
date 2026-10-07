"""Read-only system scaffold inspection; no component imports or execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/system.json"
STATUSES = {"source_available", "offline_ready", "planned", "not_implemented"}


def inspect_layout(root=ROOT):
    root = Path(root).resolve()
    config = json.loads((root / "configs/system.json").read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or not isinstance(config.get("modules"), dict)
            or set(config["modules"]) != {"perception", "decision", "navigation"}):
        raise ValueError("Unsupported system layout")
    components, absent, pending = [], [], []
    for domain, entries in config["modules"].items():
        for name, entry in entries.items():
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root) or entry["status"] not in STATUSES:
                raise ValueError("Invalid component path/status")
            present = path.is_dir() and (path / "__init__.py").is_file()
            key = f"{domain}.{name}"
            components.append({"component": key, "path": entry["path"], "status": entry["status"], "directory_present": present})
            if not present:
                absent.append(key)
            if entry["status"] in {"planned", "not_implemented"}:
                pending.append(key)
    contracts = (root / config["interfaces"]["path"]).resolve()
    if not contracts.is_relative_to(root) or not contracts.is_file():
        absent.append("interfaces")
    return {"system": config["name"], "layout_valid": not absent,
            "online_system_ready": False, "status": config["runtime_status"],
            "components": components, "pending_components": pending,
            "missing_layout_paths": absent, "interfaces_status": config["interfaces"]["status"],
            "processes_started": 0, "network_requests": 0,
            "notice": "Directories and draft contracts are not implementations. No integrated robot runtime exists."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-online", action="store_true", help="Fail until an actual integrated runtime is implemented")
    args = parser.parse_args(argv)
    report = inspect_layout()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["layout_valid"] and not args.require_online else 1


if __name__ == "__main__":
    raise SystemExit(main())
