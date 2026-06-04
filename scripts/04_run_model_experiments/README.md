# 04 Run Model Experiments

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment --model gpt-54-nano --trials 10 --resume
```

Writes model runs to:

```text
outputs/04_model_runs/<model>/
```

