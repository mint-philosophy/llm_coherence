#!/usr/bin/env python3
"""Prepare, run, or analyze the bounded Ministral temperature pilot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llm_coherence.experiments.temperature_sensitivity import main

if __name__ == "__main__":
    raise SystemExit(main())
