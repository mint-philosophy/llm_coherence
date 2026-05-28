# predictive_utility/

Predictive-utility results, organized by value category (Paper Appendix E).

Each category folder contains a `per_set_pred_util.csv` with the rows (one per
model x ladder) whose `variation_id` belongs to that category. Columns include
observed/held-out AUC, log-loss, permutation-null statistics, raw p-values, and
Benjamini-Hochberg adjusted significance decisions.

```
predictive_utility/{category}/per_set_pred_util.csv   # per-ladder rows for this category, all models
predictive_utility/per_model_pred_util.csv            # model-level summary (no category dimension)
```

`per_model_pred_util.csv` lives at the top level because it is aggregated per
model across all ladders and has no category breakdown.

Provenance: split by category from `aies/analysis_outputs/pred_util/`
(mint-re-engineering-fig, elena_dev snapshot). Category is derived from the
`variation_id` prefix; rows were only partitioned, never recomputed.

## Categories

- AI_and_human_romantic_relationships
- AI_moral_patienthood
- Global_economy
- Global_politics_and_geopolitics
- Life_and_species
- Personal_accomplishments
- Personal_finances
- Personal_freedom_and_autonomy
- Religion_and_spirituality
- United_States_economy
- United_States_politics_and_policies
- Wellbeing_of_humans
