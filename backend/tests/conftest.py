"""Pytest bootstrap: put backend/ on sys.path for imports."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
