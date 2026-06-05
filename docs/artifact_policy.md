# Artifact Policy

Use git for source code, docs, canonical input data, and small manifests.

Avoid committing newly generated raw outputs unless they are intentionally part
of a publication snapshot. In particular, treat these as generated artifacts:

- `outputs/04_model_runs/`
- reasoning traces
- retry/checkpoint files
- regenerated figures and tables
- temporary validation outputs

## Public Repository Decision

For the public AIES 2026 repository, generated output payloads are not tracked
in Git. Keep README files, lightweight indexes, canonical inputs, code, and
reproducibility scripts in the repository. Keep raw model runs, reasoning
traces, and large derived outputs local or mirror them to an external artifact
archive.

`outputs/04_model_runs/model_run_index.json` is a retained snapshot inventory:
it records what was present in the local publication outputs without requiring
the public GitHub tree to show thousands of generated files.

If a large artifact must be preserved for review or publication, mirror it to a
dataset/archive system and record the external location in the README or paper
supplement.

`.gitignore` prevents new untracked files from being added; it does not remove
files that Git already tracks. Use `git rm --cached` for explicit index-only
cleanup while keeping local copies on disk.
