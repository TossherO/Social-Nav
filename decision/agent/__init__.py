"""Long-horizon task development: hosted VLM, bounded tools and simulation.

No chassis output. Robot and online runtime feedback adapters remain separate
integration work; simulation completion never denotes physical task success.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent/'configs'
