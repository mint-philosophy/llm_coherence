# LLM Preference Coherence

This repository contains the code and input materials for research on whether LLM forced-choice preferences remain coherent under controlled seven-tier outcome variations.

The central test is: when one value-relevant property is increased across an ordered ladder, does the model's preference probability move in the intended direction, or does it reverse, flatten, or become erratic?

The repository is organized as a reproducible research artifact from the [MINT Lab](https://mintresearch.org/). It includes validated ladder inputs, forced-choice comparison inputs, model-run wrappers, analysis code, and lightweight public summaries. Raw model responses and full paper-output payloads are not tracked in Git; the complete experiment datasets are hosted on Hugging Face (see below).

## Experiment Data

All datasets created during the experiment—including canonical inputs under `data/` and model-run payloads under `outputs/`—are available on Hugging Face:

**[MINTLABJHUANU/LLMCoherence_Var_100](https://huggingface.co/datasets/MINTLABJHUANU/LLMCoherence_Var_100/tree/main)**

Clone or download that dataset repo to populate `data/` and `outputs/` locally without rerunning API calls.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `data/` | Canonical inputs and intermediate instrument data. Numbered subfolders (`01_`–`06_`) follow the experiment order. |
| `outputs/` | Model-run payloads, per-model analysis, and checkpoints. Ignored by Git. |
| `results/` | Paper figures, tables, and small tracked summaries. |
| `api_keys/` | Local provider API keys (`api_key_<provider>.txt`). Ignored by Git. |
| `scripts/` | Numbered command wrappers for rerunning the pipeline. |
| `src/llm_coherence/` | Importable Python package used by the wrappers. |

Use `scripts/` to run the pipeline. Use `src/llm_coherence/` to edit or audit the implementation. Wrapper files are intentionally small and delegate to `main()` functions in `src/`.

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

## Installation

Create an isolated environment and install the package:

```bash
bash scripts/00_repository/00_create_environment.sh
source .venv/bin/activate
```

The environment script installs the dependencies declared in `pyproject.toml`, including the NumPy pin used by the analysis stack. If you already have a clean Python 3.11 or 3.12 environment, the manual equivalent is:

```bash
python -m pip install -e .
```

Validate tracked inputs and lightweight indexes:

```bash
PYTHONPATH=src python scripts/00_repository/validate_artifacts.py
```

Refresh browsable indexes after adding local model-run payloads under `outputs/`:

```bash
PYTHONPATH=src python scripts/00_repository/validate_artifacts.py --write-indexes
```

## API Keys

Model-run steps require provider access. Set environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) or create local files under `api_keys/`:

```text
api_keys/api_key_openai.txt
api_keys/api_key_anthropic.txt
api_keys/api_key_openrouter.txt
```

Keys are loaded through `src/llm_coherence/runtime/api_keys.py` and are not included in this repository.

## Quick Smoke Test

For ordinary replication, run a bounded smoke test before launching full model runs. The example below starts from the tracked validated ladders, creates a small forced-choice slice, runs both model experiments (step 10a: within-ladder tier-pair preferences; step 10b: ladder-vs-comparison forced choice), and runs both analysis stages.

```bash
PYTHONPATH=src python scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py \
  --variations data/05_ladder_validation/phase6b_variations_pruned_final.json \
  --comparison-sample data/06_forced_choice_inputs/comparison_sample.json \
  --max-variations 2 \
  --max-comparison-samples 10 \
  --output-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10
```

```bash
PYTHONPATH=src python scripts/04_model_runs/10a_run_within_ladder_experiment.py \
  --model ministral-3b-2512-openrouter \
  --smoke
```

```bash
PYTHONPATH=src python scripts/04_model_runs/10b_run_7tier_experiment.py \
  --model ministral-3b-2512-openrouter \
  --trials 1 \
  --data-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10 \
  --max-variation-sets 2 \
  --max-concurrent 1 \
  --infrastructure openrouter \
  --smoke \
  --resume
```

```bash
PYTHONPATH=src python scripts/05_analysis/11_analyze_7tier_coherence.py \
  --model ministral-3b-2512-openrouter \
  --data-dir data/06_forced_choice_inputs/phase6b_variations_pruned_smoke_tiny10 \
  --results-dir outputs/ministral-3b-2512-openrouter/smoke_ministral-3b-2512-openrouter/ladder_vs_comparison_statements
```

```bash
# Optional on a tiny smoke slice (may produce no rows if too few comparison pairs).
PYTHONPATH=src python scripts/05_analysis/12_predictive_utility.py \
  --model ministral-3b-2512-openrouter \
  --results-dir outputs/ministral-3b-2512-openrouter/smoke_ministral-3b-2512-openrouter/ladder_vs_comparison_statements \
  --out-dir outputs/ministral-3b-2512-openrouter/smoke_ministral-3b-2512-openrouter/ladder_vs_comparison_statements/pred_utility_test \
  --n-perm 20
```

```bash
PYTHONPATH=src python scripts/06_reporting/13_make_fig_table.py \
  --model ministral-3b-2512-openrouter \
  --results-dir outputs
```

For a full rerun, remove or increase the smoke bounds (`--max-variation-sets`, `--max-variations`, and `--max-comparison-samples`), omit `--smoke`, and set the desired trial count.

## Pipeline

Run scripts from the repository root with `PYTHONPATH=src python <script>`.

| Step | Command wrapper | Implementation |
| ---: | --- | --- |
| 1 | `scripts/01_instrument_design/01_create_filtered_dataset.py` | `src/llm_coherence/generation/create_filtered_dataset.py` |
| 2 | `scripts/01_instrument_design/02_screen_outcomes.py` | `src/llm_coherence/generation/filter_statements.py` |
| 3 | `scripts/01_instrument_design/03_generate_7tier_ladders.py` | `src/llm_coherence/generation/generate_7tier_variations.py` |
| 4 | `scripts/02_ladder_validation/04_within_ladder_pruning.py` | `src/llm_coherence/validation/within_ladder_pruning.py` |
| 5 | `scripts/02_ladder_validation/05_property_ladder_pruning.py` | `src/llm_coherence/validation/property_ladder_pruning.py` |
| 7 | `scripts/02_ladder_validation/07_ranking_ladder_pruning.py` | `src/llm_coherence/validation/ranking_ladder_pruning.py` |
| 8 | `scripts/02_ladder_validation/08_build_final_pruned_variations.py` | `src/llm_coherence/validation/build_final_pruned_variations.py` |
| 9 | `scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py` | `src/llm_coherence/generation/generate_7tier_comparisons.py` |
| 10a | `scripts/04_model_runs/10a_run_within_ladder_experiment.py` | `src/llm_coherence/experiments/within_ladder/run_within_ladder_experiment.py` |
| 10b | `scripts/04_model_runs/10b_run_7tier_experiment.py` | `src/llm_coherence/experiments/ladder_statement_pair/run_7tier_experiment.py` |
| 11 | `scripts/05_analysis/11_analyze_7tier_coherence.py` | `src/llm_coherence/analysis/analyze_7tier_coherence.py` |
| 12 | `scripts/05_analysis/12_predictive_utility.py` | `src/llm_coherence/analysis/predictive_utility.py` |
| 13 | `scripts/06_reporting/13_make_fig_table.py` | `src/llm_coherence/reporting/make_fig_table.py` |

The early instrument-design and ladder-audit stages require API access and are not necessary for most replication workflows. Most users should start from the tracked validated ladders and forced-choice inputs.

## Outputs and External Artifacts

Tracked GitHub contents are sufficient to inspect the instrument and rerun the pipeline. For exact reproduction without rerunning APIs, download the full artifact tree from the [Hugging Face dataset](https://huggingface.co/datasets/MINTLABJHUANU/LLMCoherence_Var_100/tree/main) (`data/` and `outputs/`).

Expected local output layout:

| Path | Contents |
| --- | --- |
| `data/05_ladder_validation/` | Ladder validation: pruned ladder JSONs, audit reports, and judge run folders (`within_ladder_validation_tier/`, `property/`, `ranking/`). |
| `outputs/<model_key>/within_ladder/` | Instance 1 (step 10a): tier-pair preferences, cost logs, `summary.json`. |
| `outputs/<model_key>/ladder_vs_comparison_statements/` | Instance 2 (step 10b): per-ladder `results.json`, reasoning traces, cost logs. |
| `outputs/<model_key>/ladder_vs_comparison_statements/coherence_test/` | Step 11: `phase6b_coherence_*.json`, justification analysis, per-category summaries. |
| `outputs/<model_key>/ladder_vs_comparison_statements/pred_utility_test/` | Step 12: predictive-utility CSVs and summaries. |
| `outputs/checkpoints/<model_key>/` | Resumable checkpoints for step 10b. |
| `results/figures/` | Generated figures (step 13). |
| `results/tables/` | Generated tables (step 13). |

Smoke runs for step 10b write under `outputs/<model_key>/smoke_<model_key>/ladder_vs_comparison_statements/` instead of the full-run path above.

The tracked `results/model_run_index.json` snapshot inventories local payloads under `outputs/<model_key>/`. Refresh it with `validate_artifacts.py --write-indexes` after copying or generating model-run artifacts.


```bash
# Update the dataset README on Hugging Face
python scripts/00_repository/hf_upload/hf_dataset.py upload readme

# Upload outputs/ (resume incomplete models)
python scripts/00_repository/hf_upload/hf_dataset.py upload outputs --skip-existing

# Stage a local bundle with manifest (optional)
python scripts/00_repository/hf_upload/hf_dataset.py prepare /path/to/artifact_bundle
```

## Public Summaries

Two small summary files are tracked for inspection:

```text
results/phase6b_coherence_summary.json
results/model_run_index.json
```

These are not substitutes for the raw model-response artifact bundle on [Hugging Face](https://huggingface.co/datasets/MINTLABJHUANU/LLMCoherence_Var_100/tree/main).

## License

Released under the MIT License. See [LICENSE](LICENSE).
