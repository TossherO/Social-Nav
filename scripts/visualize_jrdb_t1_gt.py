"""Compatibility CLI; implementation lives in decision.vlm.visualize_jrdb_t1_gt."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.vlm.visualize_jrdb_t1_gt import main

if __name__ == "__main__":
    main()
