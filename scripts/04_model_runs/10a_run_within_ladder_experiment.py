#!/usr/bin/env python3
"""Step 10a: within-ladder tier-pair experiment (Instance~1).

Writes artifacts under ``outputs/<model_key>/within_ladder/`` (input.jsonl,
output.jsonl, cost logs, summary.json).
"""

from llm_coherence.experiments.within_ladder.run_within_ladder_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
