# Replication And Reproduction

This repository supports two different use cases.

## Replicate From Scratch

Supported from GitHub alone, subject to API access and model availability.

Tracked inputs:

- `data/01_ladders/phase6b_variations_pruned_final.json`
- `data/03_comparisons/comparison_sample.json`
- `data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/03_comparisons/phase6b_variations_pruned/*_comparisons.json`
- source code under `src/llm_coherence/`
- model/runtime configuration under `src/llm_coherence/config.py` and
  `src/llm_coherence/runtime/`

Run commands:

- `docs/rerun.md`

Limitations:

- Live API replication requires the same provider access and compatible model
  availability.
- Even with deterministic settings, hosted model APIs and provider routing can
  change over time.
- API keys are intentionally not included.

## Reproduce Archived Results Without Rerunning APIs

Not complete from GitHub alone yet. The public GitHub repo intentionally omits
raw model responses, reasoning traces, checkpoints, and large derived analysis
payloads so the repository remains readable.

The external artifact bundle should include:

- `outputs/04_model_runs/`
- `outputs/05_analysis/`
- `outputs/06_figures_tables/` if final generated figures/tables are part of
  the camera-ready artifact
- checksums for the archived files
- the Git commit SHA used to generate the archive

Once the archive exists, add its DOI or stable URL to this document and to the
root README.

## Inspect Current Summary Results

The repo tracks a small summary snapshot:

- `results/phase6b_coherence_summary.json`

This file is useful for public browsing and sanity checks. It is not enough to
audit individual forced-choice responses or regenerate every table exactly.
