#!/usr/bin/env python3
"""Visualize native JRDB reference labels or reviewed T1 REL ground truth.

Native reference is not automatically exported as T1 gold. Reviewed labels must
be explicitly tied to the exact model-input fingerprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageDraw

try:
    from decision.vlm import jrdb_t1_common as t1
except ModuleNotFoundError:
    import jrdb_t1_common as t1


def load_reviewed_label(path: Path, manifest: dict, allow_draft=False) -> dict:
    label = t1.jrdb.load_json(path)
    if label.get("input_fingerprint") != manifest["input_fingerprint"]:
        raise ValueError("Label was not created for these exact model inputs/PIDs")
    if label.get("review_status") != "reviewed" and not allow_draft:
        raise ValueError("Unreviewed draft is not ground truth; review it or use --allow-draft for a preview")
    if label.get("review_status") == "reviewed":
        for key in ("reviewer", "reviewed_at", "label_provenance"):
            if not isinstance(label.get(key), str) or not label[key].strip():
                raise ValueError(f"Reviewed label requires nonempty {key}")
    t1.validate_response(label["response"], manifest["model"]["VALID_PID"])
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Reviewed annotation_template-style T1 JSON")
    parser.add_argument("--allow-draft", action="store_true", help="Preview only; never export as gold")
    parser.add_argument("--out", type=Path, required=True, help="Separate label-only output directory")
    args = parser.parse_args()
    window, manifest = t1.load_bundle(args.input_dir)
    t1.check_output(window.root, args.out, input_dirs=(args.input_dir,))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Label output already exists; choose a new --out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.labels is None:
        if args.allow_draft:
            parser.error("--allow-draft requires --labels")
        styles, links, facts = t1.native_reference(window)
        title, filename = "JRDB REFERENCE - NOT T1 GOLD", "native_reference"
        facts["input_fingerprint"] = manifest["input_fingerprint"]
        t1.write_json(args.out / "native_reference.json", facts)
    else:
        label = load_reviewed_label(args.labels, manifest, args.allow_draft)
        styles, links = t1.relation_layout(label["response"])
        reviewed = label.get("review_status") == "reviewed"
        title, filename = ("T1 REVIEWED GT", "gt") if reviewed else ("T1 DRAFT - NOT GOLD", "draft")
        t1.write_json(args.out / f"{filename}.json", label)
        if reviewed:
            t1.write_json(args.out / "assistant_response.json", label["response"])
    # Target-time truth only: do not imply that the same relations held in older frames.
    image = t1.render_frame(window, 2, styles, links)
    draw = ImageDraw.Draw(image)
    t1.badge(draw, (8, window.height - 26), title, (245, 210, 104), 15)
    image.save(args.out / f"{filename}.png")
    print(f"Rendered {args.out.resolve() / (filename + '.png')}: {title}")


if __name__ == "__main__":
    main()
