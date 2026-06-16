# LLM Preference Coherence

This repository contains the code and all necessary input materials for a paper on whether
LLM forced-choice preferences remain coherent under controlled seven-tier
outcome variations.

The central test is simple: when one value-relevant property is increased across
an ordered ladder, does the model's preference probability move in the intended
direction, or does it reverse, flatten, or become erratic?

The repository is organized as a reproducible research artifact. It includes
the validated ladder inputs, forced-choice comparison inputs, model-run
wrappers, analysis code, and lightweight public summaries. Raw model responses
and full paper-output payloads are not tracked in Git and should be archived
externally for exact reproduction.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `data/` | Canonical inputs and intermediate instrument data. Numbered subfolders follow the experiment order. |
| `results/` | Generated model, analysis, figure, and table outputs. Git tracks only small public summaries here. |
| `outputs/` | Local scratch space for checkpoints, smoke-test artifacts, and legacy outputs. Ignored by Git. |
| `scripts/` | User-facing numbered command wrappers for rerunning the pipeline. |
| `src/llm_coherence/` | Importable Python package containing the implementation used by the wrappers. |

Use `scripts/` when running the experiment. Use `src/llm_coherence/` when
editing or auditing the implementation. The wrapper files in `scripts/` are
intentionally small and mostly delegate to `main()` functions in `src/`.

## Tracked Inputs

The canonical validated ladder set is:

```text
data/05_ladder_validation/phase6b_variations_pruned_final.json
```

The canonical forced-choice inputs are:

```text
data/06_forced_choice_inputs/phase6b_variations_pruned/
```

The main count progression is:

| Stage | Count |
| --- | ---: |
| Source outcomes | 510 |
| Screened candidate outcomes | 181 |
| Generated ladder candidates | 146 |
| Final validated ladders | 100 |

The per-category ladder split is included for inspection under:

```text
data/05_ladder_validation/phase6b_variations_pruned_final_by_category/
```

## Installation

Create an isolated environment and install the package:

```bash
bash scripts/00_repository/00_create_environment.sh
source .venv/bin/activate
```

The environment script installs the dependencies declared in `pyproject.toml`,
including the NumPy pin used by the analysis stack. If you already have a clean
Python 3.11 or 3.12 environment, the manual equivalent is:

```bash
python -m pip install -e .
```

Validate the tracked inputs and lightweight indexes:

```bash
PYTHONPATH=src python scripts/00_repository/validate_artifacts.py
```

## Quick Smoke Test

For ordinary replication, run a bounded smoke test before launching full model
runs. The example below starts from the tracked validated ladders, creates a
small forced-choice slice, runs one inexpensive model, and runs both analysis
stages.

The model-run step requires provider access. API keys may be supplied through
environment variables or local `api_keys/api_key_<provider>.txt` files. API
keys are not included in this repository.

```bash
PYTHONPATH=src python scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py \
  --variations data/05_ladder_validation/phase6b_variations_pruned_final.json \
  --comparison-sample data/06_forced_choice_inputs/comparison_sample.json \
  --max-variations 2 \
  --max-comparison-samples 10 \
  --output-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10
```

```bash
PYTHONPATH=src python scripts/04_model_runs/10_run_7tier_experiment.py \
  --model ministral-3b-2512-openrouter \
  --trials 1 \
  --data-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10 \
  --results-dir results/smoke_pipeline/07_model_runs_tiny10 \
  --checkpoints-dir outputs/smoke_pipeline/checkpoints_tiny10 \
  --max-variation-sets 2 \
  --max-concurrent 1 \
  --infrastructure openrouter \
  --resume
```

```bash
PYTHONPATH=src python scripts/05_analysis/11_analyze_7tier_coherence.py \
  --model ministral-3b-2512-openrouter \
  --data-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10 \
  --results-dir results/smoke_pipeline/07_model_runs_tiny10 \
  --output results/smoke_pipeline/08_analysis/ministral-3b-2512-openrouter_tiny10_coherence.json
```

```bash
PYTHONPATH=src python scripts/05_analysis/12_predictive_utility.py \
  --model ministral-3b-2512-openrouter \
  --results-dir results/smoke_pipeline/07_model_runs_tiny10 \
  --out-dir results/smoke_pipeline/08_analysis/ministral-3b-2512-openrouter_tiny10_pred_utility \
  --n-perm 20
```

For a full rerun, remove or increase the smoke bounds
`--max-variation-sets`, `--max-variations`, and `--max-comparison-samples`, and
set the desired trial count.

## Pipeline

Run scripts from the repository root with `PYTHONPATH=src python <script>`.

| Step | Command wrapper | Implementation |
| ---: | --- | --- |
| 1 | `scripts/01_instrument_design/01_create_filtered_dataset.py` | `src/llm_coherence/generation/create_filtered_dataset.py` |
| 2 | `scripts/01_instrument_design/02_screen_outcomes.py` | `src/llm_coherence/generation/filter_statements.py` |
| 3 | `scripts/01_instrument_design/03_generate_7tier_ladders.py` | `src/llm_coherence/generation/generate_7tier_variations.py` |
| 4 | `scripts/02_ladder_validation/04_within_ladder_pruning.py` | `src/llm_coherence/validation/within_ladder_validation.py` |
| 5 | `scripts/02_ladder_validation/05_property_validation.py` | `src/llm_coherence/validation/property_ladder_pruning.py` |
| 6 | `scripts/02_ladder_validation/06_build_pairtest_pruned_ladders.py` | `src/llm_coherence/validation/generate_pairtest_pruned.py` |
| 7 | `scripts/02_ladder_validation/07_ranking_ladder_pruning.py` | `src/llm_coherence/validation/full_ladder_ranking_pruning.py` |
| 8 | `scripts/02_ladder_validation/08_build_final_pruned_ladders.py` | `src/llm_coherence/validation/build_final_pruned_variations.py` |
| 9 | `scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py` | `src/llm_coherence/generation/generate_7tier_comparisons.py` |
| 10 | `scripts/04_model_runs/10_run_7tier_experiment.py` | `src/llm_coherence/experiments/ladder_statement_pair/run_7tier_experiment.py` |
| 11 | `scripts/05_analysis/11_analyze_7tier_coherence.py` | `src/llm_coherence/analysis/analyze_7tier_coherence.py` |
| 12 | `scripts/05_analysis/12_predictive_utility.py` | `src/llm_coherence/analysis/predictive_utility.py` |
| 13 | `scripts/06_reporting/13_make_paper_figures.py` | `src/llm_coherence/reporting/make_paper_figures.py` |

The early instrument-design and ladder-audit stages require API access and are
not necessary for most replication workflows. Most users should start from the
tracked validated ladders and forced-choice inputs.

## Outputs and External Artifacts

Tracked GitHub contents are sufficient to inspect the instrument and rerun the
pipeline. Exact reproduction of archived paper outputs without rerunning APIs
requires an external artifact bundle containing raw model responses, reasoning
traces, and derived analysis payloads.

Expected local output layout:

| Path | Contents |
| --- | --- |
| `results/07_model_runs/` | Raw model choices, reasoning traces, cost logs, and run summaries. |
| `results/08_analysis/` | Coherence and predictive-utility analysis outputs. |
| `results/09_figures_tables/` | Generated figures and tables. |
| `outputs/` | Checkpoints and scratch files that are not part of the public artifact. |

To prepare an external artifact bundle:

```bash
PYTHONPATH=src python scripts/00_repository/prepare_hf_artifact_bundle.py \
  --bundle-dir /path/to/artifact_bundle
```

## Public Summaries

Two small summary files are tracked for inspection:

```text
results/phase6b_coherence_summary.json
results/model_run_index.json
```

These are not substitutes for the raw model-response artifact bundle.

## Notes on Scope

This repository implements the coherence and predictive-utility analyses used
for the paper. Bradley-Terry and Thurstonian utility estimation are not part of
the canonical analysis pipeline here unless separately regenerated and
documented. Refusal-like behavior is tracked through parseability and
unparseable-response statistics in the forced-choice outputs.

