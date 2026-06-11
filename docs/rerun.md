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

## 02. Rebuild Source Filtering

```bash
PYTHONPATH=src python -m llm_coherence.generation.create_filtered_dataset
```

This reads `data/01_source_outcomes/options_hierarchical.json` and writes the
category-filtered source under `data/02_category_filtering/`.

## 03. Rerun Outcome Screening

```bash
PYTHONPATH=src python -m llm_coherence.generation.filter_statements
```

This Opus screening stage requires API access. It reads
`data/02_category_filtering/options_hierarchical_filtered_phase1.json` and
writes `data/03_outcome_screening/phase2_filtering_results.json`.

## 04. Regenerate Seven-Tier Ladders

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_variations
```

This stage also requires API access. It writes
`data/04_ladder_generation/phase6b_variations.json`.

Most reruns should start from the tracked canonical post-validation input below
unless the goal is to audit dataset construction itself.

## 05. Build Comparison Inputs

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

This reads:

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`

and writes:

- `data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/*_comparisons.json`

## 06. Run Model Experiments

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

## 07. Analyze Coherence

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

## 08. Make Figures And Tables

```bash
PYTHONPATH=src python -m llm_coherence.reporting.make_paper_figures
```

## Artifact Tracking

Raw model outputs are generated artifacts. Keep them locally under `outputs/`
while running experiments. Commit new raw outputs only when intentionally
updating a publication snapshot or external artifact bundle. The public GitHub
repo keeps lightweight indexes and documentation rather than thousands of raw
result payloads.

For the difference between rerunning the experiment and reproducing archived
paper outputs without rerunning APIs, see `docs/replication.md`.
