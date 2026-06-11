# Pipeline

The project is arranged in the order used by the paper methodology. Numbered
folders are for human navigation; Python packages keep semantic names so imports
stay readable.

## 01. Source Outcomes

Input:

- `data/01_source_outcomes/options_hierarchical.json`

This is the original 510-outcome hierarchy across 30 value categories.

## 02. Category Filtering

Input:

- `data/01_source_outcomes/options_hierarchical.json`

Outputs:

- `data/02_category_filtering/options_hierarchical_filtered_phase1.json`
- `data/02_category_filtering/phase1_filtering_report.json`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.generation.create_filtered_dataset
```

## 03. Outcome Screening

Input:

- `data/02_category_filtering/options_hierarchical_filtered_phase1.json`

Output:

- `data/03_outcome_screening/phase2_filtering_results.json`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.generation.filter_statements
```

This stage screens the 181 candidate outcomes for a choice-relevant property
that can be varied by quality or magnitude.

## 04. Ladder Generation

Inputs:

- `data/03_outcome_screening/phase2_filtering_results.json`
- `data/04_ladder_generation/phase3_variations.json`

Output:

- `data/04_ladder_generation/phase6b_variations.json`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_variations
```

This stage produces the 146 seven-tier ladder candidates.

## 05. Ladder Validation And Pruning

Validation checks:

- tier-pair validation
- property validation
- ranking validation
- final intersection of ladders passing all validation tests

Canonical output:

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`

Related code:

- `src/llm_coherence/validation/`

The final file contains the 100 validated ladders used in the main model
experiments.

## 06. Forced-Choice Inputs

Inputs:

- `data/05_ladder_validation/phase6b_variations_pruned_final.json`
- `data/06_forced_choice_inputs/comparison_sample.json`

Outputs:

- `data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json`
- `data/06_forced_choice_inputs/phase6b_variations_pruned/*_comparisons.json`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
```

The pruned comparison files stay in a flat directory because the manifest stores
filenames and the runner resolves them relative to the data directory. Use
`data/06_forced_choice_inputs/phase6b_variations_pruned/category_index.json` to
browse the same comparison files by category.

## 07. Model Experiments

Raw model runs live under:

- `outputs/07_model_runs/<model>/`

Thinking on/off variants are segmented as separate model keys. For example,
`gpt-54` and `gpt-54-thinking` write to different folders.

Runtime helpers for live API calls live under:

- `src/llm_coherence/runtime/`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment --model gpt-54-nano --trials 10 --resume
```

## 08. Coherence Analysis

Derived analysis outputs live under:

- `outputs/08_analysis/`

Commands:

```bash
PYTHONPATH=src python -m llm_coherence.analysis.analyze_7tier_coherence --model gpt-54-nano
PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model gpt-54-nano
```

The analysis stage computes strict monotonicity, trend metrics, and predictive
utility from forced-choice model outputs.

## 09. Figures And Tables

Paper assets live under:

- `outputs/09_figures_tables/figures/`
- `outputs/09_figures_tables/tables/`

Command:

```bash
PYTHONPATH=src python -m llm_coherence.reporting.make_paper_figures
```
