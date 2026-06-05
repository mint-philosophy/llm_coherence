# Rerun Guide

Run commands from the repository root. Either install the package in editable
mode or set `PYTHONPATH=src` as shown below.

## 01. Validate Existing Artifacts

```bash
PYTHONPATH=src python scripts/validate_artifacts.py
```

Refresh the browsable indexes after changing comparison files or model outputs:

```bash
PYTHONPATH=src python scripts/validate_artifacts.py --write-indexes
```

## 02. Build Comparison Inputs

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

This reads:

- `data/01_ladders/phase6b_variations_pruned_final.json`

and writes:

- `data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/03_comparisons/phase6b_variations_pruned/*_comparisons.json`

## 03. Run Model Experiments

Smoke run:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano \
  --trials 1 \
  --max-variation-sets 2 \
  --smoke \
  --resume
```

Full non-thinking run:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano \
  --trials 10 \
  --resume
```

Thinking-on run:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano-thinking \
  --trials 10 \
  --with-reasoning \
  --reasoning-mode thinking \
  --resume
```

To shard a run:

```bash
for s in 0 10 20 30 40 50 60 70 80 90; do
  PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
    --model gpt-54-nano \
    --trials 10 \
    --start-from "$s" \
    --max-variation-sets 10 \
    --resume
done
```

## 04. Analyze Coherence

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

## 05. Make Figures And Tables

```bash
PYTHONPATH=src python -m llm_coherence.reporting.make_paper_figures
```

## Artifact Tracking

Raw model outputs are generated artifacts. Keep them locally under `outputs/`
while running experiments. Commit new raw outputs only when intentionally
updating a publication snapshot or external artifact bundle. The public GitHub
repo keeps lightweight indexes and documentation rather than thousands of raw
result payloads.
