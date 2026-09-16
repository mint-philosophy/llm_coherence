# Missing-response and visible-rationale analysis

This analysis separates two questions:

1. **Behavioral robustness:** Do missing or unparseable A/B responses change
   the reported forced-choice result?
2. **Visible-rationale mechanism:** When a response abstains, does its retained
   rationale favor an option, reject the comparison, or fail to reach a
   conclusion?

Behavioral results are the primary evidence. Provider-exposed reasoning and
prompted justifications are observable response artifacts, not privileged
access to a model's internal chain of thought.

## Run the behavioral audit

Download or clone the public output artifacts, then run:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model kimi-k2-openrouter \
  --results-dir outputs \
  --affected-cells-csv results/kimi-k2-affected-cells.csv
```

Repeat with `--model kimi-k2-openrouter-thinking` for the Thinking condition.
The JSON report contains:

- integrity checks for stored counts, conditional probabilities, and duplicate
  comparison cells;
- expected, parseable, and missing trial counts;
- missingness by variation category, comparison category, tier, and recorded
  failure reason;
- every affected comparison cell with `(A, B, missing)` counts;
- conditional `P(A)` among parseable responses;
- `P(A)` when all missing responses are assigned to B or to A;
- whether the pairwise winner is invariant to either assignment;
- monotonicity under the complete-case and two extreme-allocation scenarios.

The extreme-allocation scenarios are sensitivity checks. They are not formal
upper and lower bounds on every coherence statistic.

## Trace-provenance gate

The command also inventories `reasoning_traces.jsonl`. Quantitative
trial-level analysis is allowed only when:

- missing-response records contain stable `custom_id` values;
- trace rows contain the same IDs;
- every missing trial has a matching trace; and
- no trace rows are malformed or lack an ID.

Legacy sidecars containing only `message_idx` cannot be joined safely after
asynchronous execution, retries, or resumed runs. The report marks these files
as qualitative-only evidence and blocks annotation-template generation.

When linkage passes, generate the coding template:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model MODEL_KEY \
  --results-dir outputs \
  --annotation-template results/refusal-annotations.jsonl
```

## Code the visible rationales

Each trial should be coded independently by two researchers. Complete these
fields using the allowed values below.

| Field | Allowed values |
| --- | --- |
| `reasoning_conclusion` | `favors_a`, `favors_b`, `equal`, `incomparable`, `no_conclusion`, `unclear` |
| `refusal_reason` | `none`, `no_personal_preference`, `political_or_religious_neutrality`, `ethical_sensitivity`, `false_dichotomy_or_incomparability`, `insufficient_information`, `instruction_or_format_failure`, `other` |
| `final_response` | `A`, `B`, `explicit_refusal`, `both_without_choice`, `malformed_or_incomplete` |
| `relationship` | `reasoning_and_answer_agree`, `reasoning_favors_a_or_b_but_final_refuses`, `reasoning_and_final_both_abstain`, `reasoning_and_final_disagree`, `unclear` |

Keep one JSON object per line and identify the coder in `coder`. To double-code
a trial, duplicate its template row and use a different coder name.

Analyze completed annotations with:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model MODEL_KEY \
  --results-dir outputs \
  --annotations results/refusal-annotations-coded.jsonl
```

The report adds category counts, a `reasoning_conclusion -> final_response`
transition table, percent agreement, and Cohen's kappa for double-coded trials.

## Interpretation

Use the transition table to distinguish:

- **deliberative abstention:** the rationale itself rejects or cannot resolve
  the comparison;
- **answer suppression:** the rationale favors A or B but the final response
  refuses;
- **reasoning-answer disagreement:** the rationale favors one option and the
  final answer selects the other; and
- **format failure:** the substantive choice is recoverable from the text but
  the required A/B answer is absent or malformed.

Report the model-wide missing rate separately from rates within affected cells.
Affected cells were selected because they contain missing responses, so their
rate is descriptive of the case study and is not an estimate of the model's
general refusal propensity.

## Compare model conditions statistically

After running the behavioral audit for each condition, compare matched model
conditions with the same 100 ladders:

```bash
PYTHONPATH=src python scripts/05_analysis/11c_compare_model_results.py \
  --left-model kimi-k2-openrouter \
  --right-model kimi-k2-openrouter-thinking \
  --results-dir outputs \
  --output results/kimi-k2-instruct-vs-thinking.json
```

The comparison uses the ladder as the resampling unit, preserving the pairing
between models. It reports cluster-bootstrap confidence intervals, paired
sign-flip randomization tests, paired effect sizes, and Holm-adjusted p-values
across the overall coherence metrics. A second Holm adjustment covers the three
designated primary endpoints: monotonicity rate, mean isotonic R-squared, and
within-ladder accuracy. Within-ladder accuracy is analyzed overall and by
valence.

Category-level monotonicity comparisons are exploratory. Categories with fewer
than five paired ladders receive descriptive estimates only. Because the stored
Kendall and Spearman headline aggregates use Fisher transformations, the paired
comparison reports both the original headline aggregate and the equal-ladder
macro mean used for inference.
