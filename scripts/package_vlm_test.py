"""Compatibility CLI; implementation lives in decision.vlm.package_vlm_test."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.vlm.package_vlm_test import main

if __name__ == "__main__":
    main()
