"""Read-only comparison of two benchmark result directories or ZIP phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import zipfile


def read_results(path, phase):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            def read(name):
                matches = [n for n in archive.namelist() if n == f"results/{phase}/{name}"
                           or n.endswith(f"/results/{phase}/{name}")]
                if len(matches) != 1 or archive.getinfo(matches[0]).file_size > 20 * 1024**2:
                    raise ValueError("Missing, ambiguous or oversized result file in ZIP")
                return archive.read(matches[0]).decode("utf-8-sig")
            summary, rows = read("summary.json"), read("requests.jsonl")
    else:
        summary = (path / "summary.json").read_text(encoding="utf-8-sig")
        rows = (path / "requests.jsonl").read_text(encoding="utf-8-sig")
    return json.loads(summary), [json.loads(line) for line in rows.splitlines() if line.strip()]


def compare(reference, candidate):
    left_summary, left = reference
    right_summary, right = candidate
    for rows in (left, right):
        ids = [row["case_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate case IDs; do not combine runs")
    right = {row["case_id"]: row for row in right}
    aligned, deltas, ttft = [], [], []
    raw_identical, binding_methods = 0, set()
    for row in left:
        other = right.get(row["case_id"])
        if other is None or not row.get("valid") or not other.get("valid"):
            continue
        # A scene/case name alone does not prove the same input. Old user ZIPs
        # lack input_fingerprint but do retain the exact HTTP request hash.
        method = next((name for name in ("request_sha256", "input_fingerprint")
                       if row.get(name) and other.get(name)), None)
        if method is None or not re.fullmatch(r"[0-9a-f]{64}", str(row[method])) or row[method] != other[method]:
            raise ValueError("Inputs/request fingerprints do not match; not a paired comparison")
        if method == "input_fingerprint":
            controlled = ("model", "thinking", "temperature", "max_tokens", "constraint_requested")
            if any(left_summary.get("settings", {}).get(k) != right_summary.get("settings", {}).get(k)
                   for k in controlled):
                raise ValueError("Same observations but different inference parameters")
        binding_methods.add(method)
        aligned.append(row["case_id"])
        deltas.append(other["timing"]["end_to_end_s"] - row["timing"]["end_to_end_s"])
        if all("first_output_text_s" in r["timing"] for r in (row, other)):
            ttft.append(other["timing"]["first_output_text_s"] - row["timing"]["first_output_text_s"])
        raw_identical += isinstance(row.get("raw_content"), str) and row["raw_content"] == other.get("raw_content")
    if not aligned:
        raise ValueError("No paired valid results")
    return {"paired_valid_cases": len(aligned), "input_binding_methods": sorted(binding_methods),
            "identical_raw_responses": raw_identical,
            "candidate_minus_reference_mean_complete_s": statistics.mean(deltas),
            "candidate_minus_reference_mean_ttft_s": statistics.mean(ttft) if ttft else None,
            "reference_complete": left_summary["latency_valid_only_s"]["end_to_end_s"],
            "candidate_complete": right_summary["latency_valid_only_s"]["end_to_end_s"],
            "reference_submitted_valid": [left_summary["submitted"], left_summary["valid"]],
            "candidate_submitted_valid": [right_summary["submitted"], right_summary["valid"]],
            "pure_gpu_or_network_time_measured": False,
            "notice": "Difference includes client/transport/server/cache/run variation. Not pure RTT or a hardware-normalized comparison. Paired means only include shared valid cases; summary distributions may include others."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", default="serial", choices=("serial", "serial_prefix", "paced", "smoke"))
    args = parser.parse_args(argv)
    try:
        result = compare(read_results(args.reference, args.phase), read_results(args.candidate, args.phase))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print("Comparison stopped:", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
