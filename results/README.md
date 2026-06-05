# Public Result Summaries

This directory contains small summary snapshots that make the public repository
inspectable without tracking raw model responses.

- `phase6b_coherence_summary.json`: aggregate phase 6b coherence metrics for
  the model runs that have local coherence-summary outputs.

These summaries are not a replacement for the full artifact bundle. Exact
paper-result reproduction without rerunning model APIs requires the raw
`outputs/04_model_runs/` payloads and derived `outputs/05_analysis/` artifacts
from an external archive.
