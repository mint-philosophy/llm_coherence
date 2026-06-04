# forced_choice/

Aggregated forced-choice elicitation results, organized by value category
(Paper Section 3.3 / Appendix D).

Each category folder contains one JSON per model. A file holds that model's
per-ladder forced-choice preferences for ladders in that category: for every
comparison (7 tiers x 30 cross-ladder reference statements) it records the
A/B preference counts and probabilities, aggregated over trials and flipped
prompt orders.

```
forced_choice/{category}/{model}.json
  -> { "model", "category", "n_ladders", "set_list", "ladders": { "{Category}_{id}": { "preferences": [...] } } }
```

Provenance: sliced by category from the `aies/forced_choice/` per-model
aggregates (mint-re-engineering-fig, elena_dev snapshot). No values were
recomputed; ladders were only partitioned by their `category` field.

Models included (9): gpt-54, gpt-54-thinking, gpt-54-mini, gpt-54-mini-thinking,
gpt-54-nano, gpt-54-nano-thinking, opus-46, opus-46-thinking, glm-45-base-logprobs.

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
