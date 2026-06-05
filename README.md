# LLM Preference Coherence

Public research artifact for an AIES 2026 project on whether LLM
forced-choice preferences remain coherent over parametric 7-tier outcome
ladders.

The repository is organized around the experiment pipeline: build and validate
ladders, generate forced-choice comparisons, run model preference experiments,
analyze coherence, and produce paper-ready summaries.

## What This Repository Supports

From GitHub alone, you can:

- inspect the canonical ladder and comparison inputs;
- rerun the model experiments, subject to provider access and model
  availability;
- regenerate analysis outputs from rerun model responses;
- inspect lightweight summary snapshots and model-run inventories.

Exact reproduction of archived paper outputs without rerunning APIs requires a
separate artifact bundle containing raw model responses, reasoning traces, and
derived analysis payloads. See `docs/replication.md`.

## Repository Map

```text
data/
  01_ladders/        canonical pruned ladder set
  02_validation/     validation-stage notes and regenerated intermediates
  03_comparisons/    canonical forced-choice comparison files and manifest

src/llm_coherence/
  generation/        build comparison inputs
  validation/        validate and prune ladders
  experiments/       run model forced-choice experiments
  runtime/           local API/runtime helpers
  analysis/          compute coherence and predictive-utility metrics
  reporting/         build paper figures and tables

results/
  phase6b_coherence_summary.json

outputs/
  README files and lightweight indexes only in Git;
  generated payloads stay local or in an external artifact archive
```

## Key Inputs

- `data/01_ladders/phase6b_variations_pruned_final.json`
- `data/03_comparisons/comparison_sample.json`
- `data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/03_comparisons/phase6b_variations_pruned/category_index.json`

The pruned comparison files remain flat on disk because the manifest and
loaders resolve comparison filenames relative to one data directory. Use the
category index for browsing by topic without changing executable paths.

## Result Summaries

- `results/phase6b_coherence_summary.json`: compact public summary of local
  phase 6b coherence metrics.
- `outputs/04_model_runs/model_run_index.json`: inventory of the local
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

Regenerate comparison inputs:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

Run a small smoke experiment:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
  --model gpt-54-nano \
  --trials 1 \
  --max-variation-sets 2 \
  --smoke \
  --resume
```

Run analysis for one model:

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

For full rerun commands, sharding examples, and artifact handling, see
`docs/rerun.md`.

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

## Artifact Policy

This public repository tracks source code, canonical inputs, documentation, and
small summaries. It does not track raw model responses, reasoning traces,
checkpoints, regenerated figures, or large derived outputs.

Generated artifacts should stay under `outputs/` locally. For paper-result
reproduction without rerunning APIs, mirror the full output bundle to a stable
external archive and record the DOI or URL in `docs/replication.md`.

## Documentation

- `docs/pipeline.md`: ordered experiment map.
- `docs/rerun.md`: commands for rerunning the pipeline.
- `docs/replication.md`: what can be replicated from GitHub and what requires
  an external artifact bundle.
- `docs/artifact_policy.md`: Git and artifact-tracking policy.
