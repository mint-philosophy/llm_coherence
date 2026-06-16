# LLM Preference Coherence

Public research artifact for a paper on whether LLM forced-choice preferences
remain coherent when one value-relevant property is varied across a seven-tier
ladder.

The experiment asks a simple question: if a model prefers an outcome at a
baseline tier, does it keep preferring stronger versions of that same outcome
when the relevant property is increased? The repository is organized around the
paper's experiment order: instrument design, ladder audit, forced-choice model
runs, coherence analysis, predictive-utility analysis, and reporting.

## Research Question

Do LLM forced-choice preferences remain coherent when outcomes are varied
monotonically along a controlled seven-tier ladder? In this repository,
coherence means that a model's choices should track the intended direction of a
single value-relevant property rather than reversing, flattening, or becoming
unpredictable as the property changes.

This repository accompanies a paper manuscript. The citation and arXiv link
will be added when publicly available.

## Start Here

If you are reading the repo for the first time:

1. Read the workflow overview below.
2. Inspect the final validated ladders in
   `data/05_ladder_validation/phase6b_variations_pruned_final.json`, or browse
   the per-ladder category split in
   `data/05_ladder_validation/phase6b_variations_pruned_final_by_category/`.
3. Inspect the forced-choice inputs in `data/06_forced_choice_inputs/`.
4. Check the summary files in `results/`.
5. Use the numbered directories under `scripts/` only if you want to rerun
   code.

Raw model responses are not tracked in Git. Fresh reruns write their canonical
generated outputs under `results/`; checkpoints and disposable scratch files
can go under `outputs/`. Full paper-result payloads should also be mirrored in
an external artifact archive for reproduction.

## What This Repository Supports

From GitHub alone, you can:

- inspect the canonical ladder and comparison inputs;
- rerun the model experiments, subject to provider access and model
  availability;
- regenerate analysis outputs from rerun model responses;
- inspect lightweight summary snapshots and model-run inventories.

Exact reproduction of archived paper outputs without rerunning APIs requires a
separate artifact bundle containing raw model responses, reasoning traces, and
derived analysis payloads. The GitHub repo tracks the code, canonical inputs,
and lightweight summaries; raw outputs should be archived externally, for
example in a Hugging Face Dataset repository.

## Workflow Overview

The experiment proceeds in this order: source outcome filtering, outcome
screening, seven-tier ladder generation, ladder validation and pruning,
forced-choice input construction, model runs, coherence analysis,
predictive-utility analysis, and reporting. The detailed script-level order is
listed in the reproduction pipeline below.

## Folder Guide

| Folder | Purpose |
| --- | --- |
| `data/` | Canonical inputs and intermediate data products that define the instrument. Numbered subfolders follow the experiment order. |
| `scripts/00_repository/` | Repository maintenance scripts, including artifact validation and artifact-bundle preparation. |
| `scripts/01_instrument_design/` | Public wrappers for source filtering, outcome screening, and ladder generation. |
| `scripts/02_ladder_validation/` | Public wrappers for ladder audit, pruning, and final validation. |
| `scripts/03_forced_choice_inputs/` | Public wrapper for forced-choice input construction. |
| `scripts/04_model_runs/` | Public wrapper for model forced-choice runs. |
| `scripts/05_analysis/` | Public wrappers for monotonicity and predictive-utility analyses. |
| `scripts/06_reporting/` | Public wrapper for paper figure and table generation. |
| `src/llm_coherence/` | Importable Python package containing the implementation used by the wrappers. See note below. |
| `results/` | Generated model, analysis, figure, and table outputs. Git tracks only the small public summaries in this folder; full rerun payloads are ignored and should be archived externally. |
| `outputs/` | Local-only scratch space for checkpoints, temporary smoke artifacts, and legacy local outputs. This folder is ignored by Git. |

Use the numbered `scripts/` directories when reading the repository as a paper
artifact. GitHub sorts folders alphabetically, so the numeric prefixes make the
methodology order visible in the file browser. Use `src/llm_coherence/` when
editing the implementation. The `src` layout is the standard Python package
layout: it keeps importable code separate from data and generated artifacts.
The package name is `llm_coherence`, which is why direct module commands use
`python -m llm_coherence...`.

The count progression should be easy to audit: 510 source outcomes, 181
screened candidate outcomes, 146 generated ladder candidates, and 100 final
validated ladders used in the main experiments.

## Key Inputs

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`
- `data/05_ladder_validation/phase6b_variations_pruned_final_by_category/category_index.json`
- `data/06_forced_choice_inputs/comparison_sample.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/category_index.json`

The combined final-ladder JSON is the canonical input used by scripts. The
`phase6b_variations_pruned_final_by_category/` directory is a browsable
per-ladder split of the same 100 records. The pruned comparison files are also
grouped into category folders; their manifest stores category-relative file
paths, while each JSON file keeps a flat `test_name` so existing model-run
outputs remain comparable.

## Final Ladder Categories

| Category | Folder | Ladders |
| --- | --- | ---: |
| AI and human romantic relationships | `AI_and_human_romantic_relationships/` | 3 |
| AI moral patienthood | `AI_moral_patienthood/` | 5 |
| Global economy | `Global_economy/` | 6 |
| Global politics and geopolitics | `Global_politics_and_geopolitics/` | 7 |
| Life and species | `Life_and_species/` | 4 |
| Personal accomplishments | `Personal_accomplishments/` | 7 |
| Personal finances | `Personal_finances/` | 15 |
| Personal freedom and autonomy | `Personal_freedom_and_autonomy/` | 3 |
| Religion and spirituality | `Religion_and_spirituality/` | 1 |
| United States economy | `United_States_economy/` | 9 |
| United States politics and policies | `United_States_politics_and_policies/` | 29 |
| Wellbeing of humans | `Wellbeing_of_humans/` | 11 |

## Result Summaries

- `results/phase6b_coherence_summary.json`: compact public summary of local
  phase 6b coherence metrics.
- `results/model_run_index.json`: inventory of the local publication run
  snapshot, including model keys, thinking on/off status, and result counts.

These files are for inspection and sanity checks. They do not replace the raw
model-response artifact bundle needed to audit every individual choice.
Full per-category analysis payloads are generated locally under `results/`
when analyses are rerun and should be mirrored in the external artifact bundle,
not committed directly to Git.

## Paper Model Slate

The main paper reports 15 model configurations: 10 reasoning-off or
non-reasoning configurations and 5 reasoning-on configurations.

| Paper label | Repo model key | Reasoning mode |
| --- | --- | --- |
| GPT-5.4 Nano | `gpt-54-nano` | off |
| GPT-5.4 Nano Thinking | `gpt-54-nano-thinking` | on |
| GPT-5.4 Mini | `gpt-54-mini` | off |
| GPT-5.4 Mini Thinking | `gpt-54-mini-thinking` | on |
| GPT-5.4 | `gpt-54` | off |
| GPT-5.4 Thinking | `gpt-54-thinking` | on |
| Opus 4.6 | `opus-46` | off |
| Nemotron 3 Super 120B | `nemotron-3-super` | off |
| Nemotron 3 Super 120B Thinking | `nemotron-3-super-thinking` | on |
| GLM-4.5 Base | `glm-45-base-logprobs` | logprob-scored base model |
| GLM-4.5 Hybrid | `glm-45-hybrid` | off |
| GLM-4.5 Hybrid Thinking | `glm-45-hybrid-thinking` | on |
| Llama 3.1 8B Instruct | `llama-31-8b-instruct-openrouter` | off |
| Ministral 3B 2512 | `ministral-3b-2512-openrouter` | off |
| Mistral Small 3.1 Thinking | `mistral-small-2603-openrouter-thinking` | on |

The ladder-audit stage is a separate judge-model use case. The pruning judge is
`gpt-55-openai`, and the paper also reports tier-pair sanity checks for GPT-5.5,
Opus 4.6, GPT-5.4, GLM-4.5 Hybrid, GPT-5.4 Mini, Nemotron 3 Super 120B, and
Llama 3.1 8B. These audit checks should not be counted as additional main
experiment configurations.

Local output inventories may include support or exploratory model keys that are
not part of the 15-configuration paper table. In particular,
`opus-46-thinking` is configured for local/exploratory runs but is not included
in the main paper's reported model slate unless the paper is updated.

## Quick Start

Create an isolated environment and install the repo:

```bash
bash scripts/00_repository/00_create_environment.sh
source .venv/bin/activate
```

The environment script installs from `pyproject.toml`, including the NumPy
pin used by the analysis stack. If you already have a clean environment, the
manual equivalent is:

```bash
python -m pip install -e .
```

Validate tracked inputs and lightweight indexes:

```bash
PYTHONPATH=src python scripts/00_repository/validate_artifacts.py
```

Regenerate early instrument-design stages when needed. The first command is
local; the screening and ladder-generation stages require provider API access:

```bash
PYTHONPATH=src python scripts/01_instrument_design/01_create_filtered_dataset.py
PYTHONPATH=src python scripts/01_instrument_design/02_screen_outcomes.py
PYTHONPATH=src python scripts/01_instrument_design/03_generate_7tier_ladders.py
```

For ordinary replication, most users should start from the tracked canonical
post-validation inputs rather than rerunning the API-heavy instrument-design
and ladder-audit stages.

The ladder-quality audit/validation code lives under:

```text
src/llm_coherence/validation/
```

The canonical validated ladder file is:

```text
data/05_ladder_validation/phase6b_variations_pruned_final.json
```

Regenerate the full forced-choice comparison inputs:

```bash
PYTHONPATH=src python scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py
```

For ordinary replication, run this recommended smoke test first. It starts from
the tracked validated ladders, builds a bounded forced-choice slice, runs one
cheap model, then runs both analysis stages. The example below uses 2 ladders,
10 comparison statements, 7 tiers, flipped prompts, and 1 trial, for 280 model
calls plus the pre-launch health check. API keys are read from environment
variables or `api_keys/api_key_<provider>.txt` under the repo root; API keys are
not included in the repository.

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

For a full rerun, increase `--max-variation-sets` or remove it, raise
`--trials`, and shard with `--start-from` across non-overlapping ranges.

## Reproduction Pipeline

Use the numbered wrapper directories under `scripts/` when tracing the
experiment from the paper methodology. Run them from the repository root with
`PYTHONPATH=src python <script>`.

| Step | Purpose and entry point |
| ---: | --- |
| 1 | Source filtering: `scripts/01_instrument_design/01_create_filtered_dataset.py` -> `src/llm_coherence/generation/create_filtered_dataset.py` |
| 2 | Outcome screening: `scripts/01_instrument_design/02_screen_outcomes.py` -> `src/llm_coherence/generation/filter_statements.py` |
| 3 | Ladder generation: `scripts/01_instrument_design/03_generate_7tier_ladders.py` -> `src/llm_coherence/generation/generate_7tier_variations.py` |
| 4 | Within-ladder pruning: `scripts/02_ladder_validation/04_within_ladder_pruning.py` -> `src/llm_coherence/validation/within_ladder_validation.py` |
| 5 | Property validation: `scripts/02_ladder_validation/05_property_validation.py` -> `src/llm_coherence/validation/property_ladder_pruning.py` |
| 6 | Pair-test pruning: `scripts/02_ladder_validation/06_build_pairtest_pruned_ladders.py` -> `src/llm_coherence/validation/generate_pairtest_pruned.py` |
| 7 | Ranking ladder pruning: `scripts/02_ladder_validation/07_ranking_ladder_pruning.py` -> `src/llm_coherence/validation/full_ladder_ranking_pruning.py` |
| 8 | Final ladder intersection: `scripts/02_ladder_validation/08_build_final_pruned_ladders.py` -> `src/llm_coherence/validation/build_final_pruned_variations.py` |
| 9 | Forced-choice inputs: `scripts/03_forced_choice_inputs/09_generate_forced_choice_inputs.py` -> `src/llm_coherence/generation/generate_7tier_comparisons.py` |
| 10 | Model runs: `scripts/04_model_runs/10_run_7tier_experiment.py` -> `src/llm_coherence/experiments/ladder_statement_pair/run_7tier_experiment.py` |
| 11 | Coherence analysis: `scripts/05_analysis/11_analyze_7tier_coherence.py` -> `src/llm_coherence/analysis/analyze_7tier_coherence.py` |
| 12 | Predictive utility: `scripts/05_analysis/12_predictive_utility.py` -> `src/llm_coherence/analysis/predictive_utility.py` |
| 13 | Figures and tables: `scripts/06_reporting/13_make_paper_figures.py` -> `src/llm_coherence/reporting/make_paper_figures.py` |

Wrapper files are intentionally small; the implementation path after `->`
shows where the substantive code lives.

## Methodology Guardrails

- Predictive utility in this repository is the cross-validated
  logistic-regression test implemented in
  `src/llm_coherence/analysis/predictive_utility.py`.
- Bradley-Terry and Thurstonian utility estimation are not part of the
  canonical analysis pipeline tracked here. Upstream computed-utility files
  should not be cited as outputs of this repo unless they are regenerated and
  documented explicitly.
- The current analysis does not include the earlier value-chart/worldview
  section, contested-ladder split, or contradiction-alternator audit as active
  paper results.
- Refusal-like behavior is tracked through parseability/unparseable-response
  statistics in forced-choice outputs, not through a separate refusal
  classifier.

## Artifact Policy

This public repository tracks source code, canonical inputs, documentation, and
small summaries. It does not track raw model responses, reasoning traces,
checkpoints, regenerated figures, or large derived outputs.

Generated run, analysis, figure, and table artifacts should use the `results/`
layout locally. Only the small summary files are tracked by Git; full payloads
under `results/07_model_runs/`, `results/08_analysis/`, and
`results/09_figures_tables/` are ignored. For paper-result reproduction without
rerunning APIs, mirror the full output bundle to a stable external archive,
preferably a Hugging Face Dataset repository, and record the DOI or URL in this
README before public release.
