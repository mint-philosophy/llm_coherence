# Artifact Policy

Use git for source code, docs, canonical input data, and small manifests.

Avoid committing newly generated raw outputs unless they are intentionally part
of a publication snapshot. In particular, treat these as generated artifacts:

- `outputs/04_model_runs/`
- reasoning traces
- retry/checkpoint files
- regenerated figures and tables
- temporary validation outputs

## Current Decision

The existing tracked outputs are treated as a retained publication snapshot for
now. New raw outputs should stay local unless the snapshot is intentionally
updated. Lightweight index files such as
`outputs/04_model_runs/model_run_index.json` are kept in git so GitHub remains
browsable without moving the underlying artifacts.

If a large artifact must be preserved for review or publication, mirror it to a
dataset/archive system and record the external location in the README or paper
supplement.

Some generated results may still appear on GitHub because they were committed
before the current ignore rules. `.gitignore` prevents new untracked files from
being added; it does not remove files that Git already tracks.

To stop tracking generated result artifacts while keeping local copies on disk,
use `git rm --cached` on the chosen output folders and commit that index-only
removal.
