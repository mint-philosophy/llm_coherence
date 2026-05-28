# reasoning_summaries/

Extended-thinking reasoning-trace summaries, organized by value category
(Paper Section 4.2).

Each category folder contains one JSON per thinking-variant model. A file holds
that model's chain-of-thought reasoning traces for the ladders in that category:
per-comparison entries with the prompt direction, parsed choice, and the raw
reasoning text the provider exposed.

```
reasoning_summaries/{category}/{model}.json
  -> { "model", "category", "n_ladders_with_traces", "ladders": { "{Category}_{id}": [ {trace}, ... ] } }
```

Coverage is limited to ladders where the provider returned reasoning text, so
some category/model combinations are absent or sparse (e.g. opus-46-thinking has
traces for only a subset of its ladders).

Provenance: sliced by category from the `aies/reasoning_summaries/` per-model
aggregates (mint-re-engineering-fig, elena_dev snapshot). Ladders were only
partitioned by the category prefix of their key; traces are unmodified.

Models included (4, thinking variants only): gpt-54-thinking, gpt-54-mini-thinking,
gpt-54-nano-thinking, opus-46-thinking.

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
