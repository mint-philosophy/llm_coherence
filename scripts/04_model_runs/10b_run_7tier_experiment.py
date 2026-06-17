#!/usr/bin/env python3
"""Step 10b: run forced-choice model experiments (Instance~2).

Writes artifacts under ``outputs/<model_key>/ladder_vs_comparison_statements/``
(per-ladder ``results.json``, cost logs, checkpoints under ``outputs/checkpoints/``).
"""

from llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
