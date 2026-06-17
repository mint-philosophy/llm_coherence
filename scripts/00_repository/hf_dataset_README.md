---
task_categories:
  - text-classification
language:
  - en
tags:
  - llm-evaluation
  - preference-coherence
  - utility
  - forced-choice
  - preference-elicitation
  - mint-lab
size_categories:
  - <1K
---

# LLM Preference Coherence — 100 validated parametric ladders

Dataset accompanying the [MINT Lab](https://mintresearch.org/) study of **LLM preference coherence** over parametric outcome ladders. Each ladder is a 7-tier scale (T1→T7) varying one choice-relevant property within a value category. After judge-model quality audits, **100 ladders** across **12 categories** are evaluated on **16 subject models** (GPT-5.4 nano/mini/std, Opus 4.6, Nemotron-3 Super, GLM-4.5 Hybrid, GLM-4.5 Base, Llama 3.1 8B, Ministral 3B, Mistral Small 2603; reasoning on/off where supported).

**Code and pipeline:** [mint-philosophy/llm_coherence](https://github.com/mint-philosophy/llm_coherence)

This repository is an archive for reproduction and audit, not a standalone training dataset.

## Contents

| Top-level path | Files | Description |
| --- | ---: | --- |
| `data/` | 150 | Canonical experiment inputs (`01_`–`06_` pipeline stages) |
| `outputs/` | 2,758 | Full model-run payloads and derived analysis for all 16 paper models (~3.3 GB) |

Clone or download this dataset to populate `data/` and `outputs/` locally without rerunning API calls:

```bash
huggingface-cli download MINTLABJHUANU/LLMCoherence_Var_100 --repo-type dataset --local-dir .
```

## Experiment design

| Instance | Task | Queries per ladder | What it tests |
| --- | --- | ---: | --- |
| **1 — Within-ladder** | All tier-pair A/B choices (both orientations) | 42 | Local ladder ordering |
| **2 — Cross-ladder** | Each of the 7 tiers vs. 30 fixed comparison statements × 20 trials | 4,200 | Win-rate curves → strict monotonicity, isotonic R², JT significance |

Forced-choice prompts, **temperature = 0**. Positive-valence ladders: higher tier = more of a good property. Negative-valence: higher tier = less harm. In both cases T1 is least choice-worthy and T7 is most.

### Instrument counts

| Stage | Count |
| --- | ---: |
| Source outcomes | 510 |
| Screened candidate outcomes | 181 |
| Generated ladder candidates | 146 |
| Final validated ladders | 100 |

## Repository layout

### `data/` — inputs (stimuli & audit)

Numbered subfolders follow the experiment pipeline order:

| Path | Description |
| --- | --- |
| `data/01_source_outcomes/` | Source outcome pool |
| `data/02_category_filtering/` | Category filter outputs |
| `data/03_outcome_screening/` | Screened candidate outcomes |
| `data/04_ladder_generation/` | Generated ladder candidates |
| `data/05_ladder_validation/phase6b_variations_pruned_final.json` | Canonical **100 ladder** definitions (tiers, category, valence, property) |
| `data/05_ladder_validation/within_ladder_validation_tier/` | Tier-pair audit (judge model) |
| `data/05_ladder_validation/within_ladder_validation_property/` | Adjacent-pair property audit |
| `data/05_ladder_validation/within_ladder_validation_ranking/` | Full ranking audit |
| `data/06_forced_choice_inputs/phase6b_variations_pruned/` | Per-ladder comparison-statement files |

### `outputs/<model_key>/` — model runs & analysis

Each of the 16 `model_key` directories contains:

**`within_ladder/`** (Instance 1, step 10a)

| File | Description |
| --- | --- |
| `summary.json` | `overall_accuracy`, per-ladder accuracy, parse-error counts |
| `input.jsonl` | Batch/API request payloads (one line per query) |
| `output.jsonl` | Model responses |
| `cost_log.json`, `phase6b_cost_log.json` | API cost logs |
| `batch_id.txt` | Provider batch job id (batch runs only) |

**`ladder_vs_comparison_statements/`** (Instance 2, step 10b)

| File / pattern | Description |
| --- | --- |
| `phase6b_variations_prune_*/results.json` | Raw trial outcomes per ladder |
| `phase6b_variations_prune_*/reasoning_traces.jsonl` | Reasoning-channel content (reasoning-on models) |
| `phase6b_variations_prune_*/cost_log.json` | Per-ladder cost logs |

**`ladder_vs_comparison_statements/coherence_test/`** (step 11)

| File / pattern | Description |
| --- | --- |
| `phase6b_coherence_<model>.json` | Aggregated coherence metrics (monotonicity, isotonic R², JT, …) |
| `phase6b_by_category_<model>/*.json` | Per-category coherence summaries |
| `phase6b_justification_analysis_<model>.json` | Justification analysis |

**`ladder_vs_comparison_statements/pred_utility_test/`** (step 12)

| File / pattern | Description |
| --- | --- |
| `*.csv` | Predictive-utility test outputs |
| Summary JSONs | AUC, permutation-null statistics |

Paper figures and tables (step 13) are generated into `results/figures/` and `results/tables/` by the code repo; they are not included in this dataset.

## Model keys (paper slate)

```
glm-45-base-logprobs          glm-45-hybrid                 glm-45-hybrid-thinking
gpt-54                        gpt-54-thinking               gpt-54-mini
gpt-54-mini-thinking          gpt-54-nano                   gpt-54-nano-thinking
llama-31-8b-instruct-openrouter
ministral-3b-2512-openrouter
mistral-small-2603-openrouter-thinking
nemotron-3-super              nemotron-3-super-thinking
opus-46                       opus-46-thinking
```

## Usage with the code repository

After downloading this dataset into a local clone of [llm_coherence](https://github.com/mint-philosophy/llm_coherence):

```bash
# Validate tracked inputs and refresh indexes
PYTHONPATH=src python scripts/00_repository/validate_artifacts.py --write-indexes

# Regenerate paper figures/tables from downloaded outputs
PYTHONPATH=src python scripts/06_reporting/13_make_fig_table.py --results-dir outputs
```

## Citation

MINT Research Lab, Johns Hopkins University / Australian National University — LLM coherence parametric variations experiment.

## License

Released under the MIT License. See the [code repository LICENSE](https://github.com/mint-philosophy/llm_coherence/blob/main/LICENSE).
