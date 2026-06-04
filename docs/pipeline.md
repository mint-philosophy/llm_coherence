# Pipeline

The project is arranged by experiment order. Numbered folders are for human
navigation; Python packages keep semantic names so imports stay readable.

## 01. Generate Ladders

Canonical input:

- `data/01_ladders/phase6b_variations_pruned_final.json`

Related code:

- `src/llm_coherence/generation/`

## 02. Validate Ladders

Validation checks:

- within-ladder pair tests
- property ladder pruning
- full-ladder ranking pruning
- final intersection of pruned ladder sets

Related data:

- `data/02_validation/ladder_validation/`

Related code:

- `src/llm_coherence/validation/`

## 03. Build Comparisons

Comparison inputs:

- `data/03_comparisons/comparison_sample.json`
- `data/03_comparisons/phase6b_variations_pruned/`

The pruned phase 6b comparison files stay in a flat directory because the
manifest stores filenames and the runner resolves them relative to the data
directory. Use
`data/03_comparisons/phase6b_variations_pruned/category_index.json` to browse
the same comparison files by category.

Command:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

## 04. Run Model Experiments

Raw model runs live under:

- `outputs/04_model_runs/<model>/`

Thinking on/off variants are segmented as separate model keys. For example,
`gpt-54` and `gpt-54-thinking` write to different folders.

Runtime helpers for live API calls live under:

- `src/llm_coherence/runtime/`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment --model gpt-54-nano --trials 10 --resume
```

## 05. Analyze Coherence

Derived analysis outputs live under:

- `outputs/05_analysis/`

Commands:

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

## 06. Make Figures And Tables

Paper assets live under:

- `outputs/06_figures_tables/figures/`
- `outputs/06_figures_tables/tables/`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.reporting.make_paper_figures
```
