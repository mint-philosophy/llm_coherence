# LLM Preference Coherence

Public research artifact for an AIES 2026 project on whether LLM
forced-choice preferences remain coherent over parametric seven-tier outcome
ladders.

This README is the main runbook. The repository is organized in the order the
experiment was conducted in the paper methodology: instrument design,
ladder-quality audit/validation, forced-choice preference elicitation,
monotonicity analysis, predictive-utility analysis, and reporting.

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

## Experiment Order

1. **Instrument design.** Start from the Mosaic-style source outcome hierarchy,
   exclude unsuitable categories, screen outcomes for a monotonic property, and
   generate seven-tier candidate ladders.
2. **Ladder-quality audit and validation.** Audit ladder ordering, property
   consistency, and valence direction with a strong model judge; fix
   valence-reversal issues and prune to the final validated ladder set.
3. **Forced-choice preference elicitation.** For each model variant, compare
   every tier in each validated ladder against fixed comparison statements and
   save per-pair choice counts and probabilities.
4. **Monotonicity analysis.** Convert forced-choice outputs into strict
   monotonicity, Kendall tau, Spearman rho, isotonic R-squared, bootstrap
   monotonicity, and parseability summaries.
5. **Predictive-utility analysis.** Run the cross-validated logistic-regression
   predictive-utility test on each model's forced-choice outputs.
6. **Reporting.** Build figures, tables, compact summaries, and external
   artifact bundles.

## Repository Map

```text
data/
  01_source_outcomes/       510 source outcomes across 30 categories
  02_category_filtering/    category-exclusion output and report
  03_outcome_screening/     Opus screening of 181 candidate outcomes
  04_ladder_generation/     146 generated seven-tier ladder candidates
  05_ladder_validation/     ladder audit/pruning outputs; final 100 ladders
  06_forced_choice_inputs/  30 comparison statements and per-ladder pairs

src/llm_coherence/
  generation/        filtering, screening, ladder generation, comparisons
  validation/        tier-pair, property, ranking, and final pruning scripts
  experiments/       forced-choice model experiment runners
  runtime/           API clients, model configuration, budget helpers
  analysis/          monotonicity, trend, and predictive-utility metrics
  reporting/         paper figures and tables

outputs/
  04_ladder_generation/     generated run artifacts and checkpoints
  05_ladder_validation/     validation run outputs
  06_forced_choice_inputs/  regenerated comparison artifacts
  07_model_runs/            raw per-model forced-choice outputs
  08_analysis/              derived coherence and predictive-utility outputs
  09_figures_tables/        final paper figures and tables

results/
  phase6b_coherence_summary.json
```

The count progression should be easy to audit: 510 source outcomes, 181
screened candidate outcomes, 146 generated ladder candidates, and 100 final
validated ladders used in the main experiments.

## Key Inputs

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`
- `data/06_forced_choice_inputs/comparison_sample.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/category_index.json`

The pruned comparison files remain flat on disk because the manifest and
loaders resolve comparison filenames relative to one data directory. Use the
category index for browsing by topic without changing executable paths.

## Result Summaries

- `results/phase6b_coherence_summary.json`: compact public summary of local
  phase 6b coherence metrics.
- `outputs/07_model_runs/model_run_index.json`: inventory of the local
  publication run snapshot, including model keys, thinking on/off status, and
  result counts.

These files are for inspection and sanity checks. They do not replace the raw
model-response artifact bundle needed to audit every individual choice.

## Quick Start

Install locally:

```bash
python -m pip install -e .
```

Validate tracked inputs and lightweight indexes:

```bash
PYTHONPATH=src python scripts/validate_artifacts.py
```

Regenerate early instrument-design stages when needed. The first command is
local; the screening and ladder-generation stages require provider API access:

```bash
PYTHONPATH=src python -m llm_coherence.generation.create_filtered_dataset
PYTHONPATH=src python -m llm_coherence.generation.filter_statements
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_variations
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

Regenerate forced-choice comparison inputs:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

Run a tiny end-to-end smoke test with the same small slice under thinking off
and thinking on. Each command uses one variation set, one trial, and
single-request concurrency so the test is cheap and easy to inspect.

Thinking off:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano \
  --trials 1 \
  --max-variation-sets 1 \
  --max-concurrent 1 \
  --smoke \
  --resume
```

Thinking on:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano-thinking \
  --trials 1 \
  --max-variation-sets 1 \
  --max-concurrent 1 \
  --smoke \
  --with-reasoning \
  --reasoning-mode thinking \
  --resume
```

Smoke outputs are written under:

```text
outputs/07_model_runs/gpt-54-nano/smoke_gpt54nano/
outputs/07_model_runs/gpt-54-nano-thinking/smoke_gpt54nanothinking/
```

After the smoke run, analyze the corresponding model output:

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

For a full rerun, increase `--max-variation-sets` or remove it, raise
`--trials`, and shard with `--start-from` across non-overlapping ranges.

## Canonical Script Sequence

Use this order when tracing or rerunning the experiment:

1. Instrument design / source filtering:
   `llm_coherence.generation.create_filtered_dataset`
2. Outcome screening:
   `llm_coherence.generation.filter_statements`
3. Seven-tier ladder generation:
   `llm_coherence.generation.generate_7tier_variations`
4. Ladder audit / pruning:
   `llm_coherence.validation.*`
5. Forced-choice comparison construction:
   `llm_coherence.generation.generate_7tier_comparisons`
6. Forced-choice model runs:
   `llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment`
7. Core monotonicity and trend analysis:
   `llm_coherence.analysis.analyze_7tier_coherence`
8. Predictive-utility analysis:
   `llm_coherence.analysis.predictive_utility`
9. Figure and table generation:
   `llm_coherence.reporting.make_paper_figures`

The model-run wrapper calls the forced-choice elicitation machinery in
`src/llm_coherence/experiments/ladder_statement_pair/experiment_runner_tradeoff.py`.
That runner writes per-pair `count_prefer_a`, `count_prefer_b`,
`prob_prefer_a`, `prob_prefer_b`, raw response fields, usage metadata, and
unparseable-response statistics.

## Model Runs

Thinking on/off variants are represented as separate model keys and separate
output folders. For example:

- `gpt-54`
- `gpt-54-thinking`
- `gpt-54-nano`
- `gpt-54-nano-thinking`

Runtime helpers live in `src/llm_coherence/runtime/`. API keys are read from
environment variables or from `api_keys/api_key_<provider>.txt` under the repo
root. API keys are not included in the repository.

For the paired tiny smoke test, use:

- thinking off: `gpt-54-nano`
- thinking on: `gpt-54-nano-thinking`

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

Generated artifacts should stay under `outputs/` locally. For paper-result
reproduction without rerunning APIs, mirror the full output bundle to a stable
external archive, preferably a Hugging Face Dataset repository, and record the
DOI or URL in this README before public release.
