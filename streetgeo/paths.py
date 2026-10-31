"""Stable workspace paths, independent of package and launch locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_ASSETS = ROOT / "assets" / "maps"
