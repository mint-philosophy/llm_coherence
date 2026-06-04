# LLM Preference Coherence

This repo tests whether LLM forced-choice preferences are coherent over
parametric 7-tier outcome ladders. The repository is organized in the same order
as the experiment pipeline.

## Pipeline Order

```text
data/
  01_ladders/        canonical ladder set
  02_validation/     ladder quality checks and pruning inputs
  03_comparisons/    forced-choice comparison files and manifest

outputs/
  01_ladders/        generated ladder artifacts
  02_validation/     validation run outputs
  03_comparisons/    generated comparison artifacts
  04_model_runs/     raw per-model experiment runs
  05_analysis/       coherence metrics and derived summaries
  06_figures_tables/ paper figures and tables

src/llm_coherence/
  generation/        build comparison inputs
  validation/        validate and prune ladders
  experiments/       run model forced-choice experiments
  runtime/           local API helpers formerly supplied by compute_utilities
  analysis/          compute coherence and predictive-utility metrics
  reporting/         build paper figures and tables
```

## Current Canonical Inputs

- Ladders: `data/01_ladders/phase6b_variations_pruned_final.json`
- Comparison sample: `data/03_comparisons/comparison_sample.json`
- Comparison manifest: `data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`

## Typical Commands

Run commands from the repo root with the package on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment --model gpt-54-nano --trials 10 --resume
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.reporting.make_paper_figures
```

Live model runs use `src/llm_coherence/runtime/` for LiteLLM agents, prompt
templates, cost estimates, and OpenRouter budget checks. API keys are read from
environment variables or `api_keys/api_key_<provider>.txt` under the repo root.

Thinking on/off runs are separate model keys and separate output folders, for
example `outputs/04_model_runs/gpt-54/` and
`outputs/04_model_runs/gpt-54-thinking/`.

For the ordered experiment map, see `docs/pipeline.md`.
For exact rerun commands, see `docs/rerun.md`.
For current model-run inventory, see `outputs/04_model_runs/model_run_index.json`.

## Artifact Policy

Keep small canonical inputs and manifests in git. Treat raw model runs,
reasoning traces, checkpoints, regenerated figures, and large derived outputs
as artifacts. Those should live under `outputs/` locally and be mirrored to an
external dataset/archive when needed for publication.
