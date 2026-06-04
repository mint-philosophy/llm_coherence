# Phase 6b Pruned Comparison Sets

This directory is intentionally flat. The experiment runner and analysis code
read `phase6b_variations_pruned_final_manifest.json`, then resolve each
`variation_files` entry as:

```text
<data-dir>/<comparison-filename>
```

Moving the comparison JSON files into category subdirectories would break that
lookup unless the manifest and every loader are updated at the same time.

Use `category_index.json` to browse the same files by category without changing
the paths consumed by code. Filenames encode both category and ladder id:

```text
phase6b_variations_pruned_final_<category>_<ladder_id>_comparisons.json
```

## Category Counts

| Category | Sets |
| --- | ---: |
| AI and human romantic relationships | 3 |
| AI moral patienthood | 5 |
| Global economy | 6 |
| Global politics and geopolitics | 7 |
| Life and species | 4 |
| Personal accomplishments | 7 |
| Personal finances | 15 |
| Personal freedom and autonomy | 3 |
| Religion and spirituality | 1 |
| United States economy | 9 |
| United States politics and policies | 29 |
| Wellbeing of humans | 11 |
