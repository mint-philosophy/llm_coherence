# Artifact Policy

Use git for source code, docs, canonical input data, and small manifests.

Avoid committing newly generated raw outputs unless they are intentionally part
of a publication snapshot. In particular, treat these as generated artifacts:

- `outputs/04_model_runs/`
- reasoning traces
- retry/checkpoint files
- regenerated figures and tables
- temporary validation outputs

If a large artifact must be preserved for review or publication, mirror it to a
dataset/archive system and record the external location in the README or paper
supplement.

