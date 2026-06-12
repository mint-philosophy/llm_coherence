# LLM Preference Coherence

Public research artifact for an AIES 2026 project on whether LLM
forced-choice preferences remain coherent when one value-relevant property is
varied across a seven-tier ladder.

The experiment asks a simple question: if a model prefers an outcome at a
baseline tier, does it keep preferring stronger versions of that same outcome
when the relevant property is increased? The repository is organized around the
paper's experiment order: instrument design, ladder audit, forced-choice model
runs, coherence analysis, predictive-utility analysis, and reporting.

## Start Here

If you are reading the repo for the first time:

1. Read the experiment order below.
2. Inspect the final validated ladders in
   `data/05_ladder_validation/phase6b_variations_pruned_final.json`.
3. Inspect the forced-choice inputs in `data/06_forced_choice_inputs/`.
4. Check the summary files in `results/`.
5. Use the quick-start commands only if you want to rerun code.

Raw model responses are not tracked in Git. They should live locally under
`outputs/` during reruns and in an external artifact archive for paper
reproduction.

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

1. **Instrument design**
   Start from the Mazeika source outcomes, exclude unsuitable categories,
   screen outcomes for a monotonic property, and generate seven-tier candidate
   ladders.
   Look at: `data/01_source_outcomes/` to `data/04_ladder_generation/`.

2. **Ladder audit**
   Check ordering, property consistency, and valence direction with a strong
   judge model; prune to the final 100 ladders.
   Look at: `data/05_ladder_validation/` and `src/llm_coherence/validation/`.

3. **Forced-choice inputs**
   Pair every validated ladder tier with fixed comparison statements.
   Look at: `data/06_forced_choice_inputs/`.

4. **Model runs**
   Run each model configuration and save per-pair choice counts, probabilities,
   raw responses, and parseability metadata.
   Writes to: local-only `outputs/07_model_runs/`.

5. **Coherence analysis**
   Compute strict monotonicity, Kendall tau, Spearman rho, isotonic R-squared,
   bootstrap monotonicity, and parseability summaries.
   Look at: `src/llm_coherence/analysis/`.

6. **Predictive utility**
   Run the cross-validated logistic-regression predictive-utility test.
   Look at: `src/llm_coherence/analysis/predictive_utility.py`.

7. **Reporting**
   Build figures, tables, compact summaries, and external artifact bundles.
   Look at: `src/llm_coherence/reporting/` and `results/`.

## Folder Guide

| Folder | Purpose |
| --- | --- |
| `data/` | Canonical inputs and intermediate data products that define the instrument. Numbered subfolders follow the experiment order. |
| `src/llm_coherence/` | Importable Python package containing the experiment code. See note below. |
| `scripts/` | Repository maintenance scripts, including artifact validation. |
| `results/` | Lightweight public summaries and inventories that are small enough to track in Git. |
| `outputs/` | Local-only generated artifacts from reruns: raw model responses, reasoning traces, derived analyses, figures, tables, and checkpoints. This folder is ignored by Git. |

Why `src/llm_coherence/`? This is the standard Python `src` layout. It keeps
importable code separate from data and generated artifacts. The package name is
`llm_coherence`, which is why commands use `python -m llm_coherence...`.

The count progression should be easy to audit: 510 source outcomes, 181
screened candidate outcomes, 146 generated ladder candidates, and 100 final
validated ladders used in the main experiments.

## Key Inputs

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`
- `data/06_forced_choice_inputs/comparison_sample.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/category_index.json`

The pruned comparison files are grouped into category folders for browsing.
The manifest stores category-relative file paths, while each JSON file keeps a
flat `test_name` so existing model-run outputs remain comparable.

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
Full per-category analysis payloads are generated locally under `outputs/`
when analyses are rerun and should be mirrored in the external artifact bundle,
not expanded into Git-tracked `results/`.

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
