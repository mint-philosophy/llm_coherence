# coherence_metrics/

Monotonicity / coherence analysis, organized by value category
(Paper Section 3.4.1 / Appendix F).

Each category folder contains one JSON per model. A file is that model's
coherence breakdown for the ladders in that category: an `aggregate` block plus
`per_variation_set` entries with monotonicity rate, Kendall's tau, Spearman's
rho, isotonic R^2, logistic slope, bootstrap monotonicity probability, and
related metrics.

```
coherence_metrics/{category}/{model}.json
  -> { "model", "category", "aggregate", "per_variation_set": [...] }
```

Provenance: copied from the in-repo per-model coherence outputs
(`outputs/{model}/phase6b_by_category_{model}/{Category}.json`), produced by
`src/analyze_results/analyze_7tier_coherence.py`. This is the complete
in-repo model set; thinking variants appear as their own `{model}` key
(e.g. `gpt-54-nano-thinking`).

Models included (14): the gpt-54 family (6), opus-46, nemotron-3-super(+thinking),
glm-45-hybrid(+thinking), glm-45-base-logprobs, llama-31-8b-instruct-openrouter,
ministral-3b-2512-openrouter. (opus-46-thinking and mistral-small-2603-openrouter-thinking
have raw results only; their coherence analysis has not been generated yet.)

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
