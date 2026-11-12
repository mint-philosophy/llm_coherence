#!/usr/bin/env python3
"""Subject-model rankings on the fixed 100 ladders; preview unless --run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llm_coherence.experiments.full_ladder_ranking import main


if __name__ == "__main__":
    raise SystemExit(main())
