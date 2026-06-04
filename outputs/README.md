# Output Stages

`outputs/` is ordered by pipeline stage.

- `01_ladders/`: generated ladder artifacts.
- `02_validation/`: validation run outputs.
- `03_comparisons/`: generated comparison artifacts.
- `04_model_runs/`: raw per-model forced-choice runs.
- `05_analysis/`: derived metrics and summaries.
- `06_figures_tables/`: paper-ready figures and tables.

Most files here are generated artifacts. Keep only intentional publication
snapshots tracked in git.

If a generated result is already tracked, `.gitignore` will not remove it from
GitHub by itself. Use `git rm --cached` for an explicit index-only cleanup when
the publication snapshot no longer needs to live in the repository.
