# Missing-response and visible-rationale analysis

This analysis separates two questions:

1. **Behavioral robustness:** Do missing or unparseable A/B responses change
   the reported forced-choice result?
2. **Visible-rationale description:** When a response abstains, does its retained
   rationale favor an option, reject the comparison, or fail to reach a
   conclusion?

Behavioral results are the primary evidence. Provider-exposed reasoning and
prompted justifications are observable response artifacts, not privileged
access to a model's internal chain of thought.

The numerical audit and paired model comparison do not require a human-labeling
study. The rationale-coding paths below are optional extensions. Step 11e can
instead produce [exploratory automated diagnostics](trace_annotation_validation.md)
without human labels; such flags remain provisional and never change original
votes or establish verified refusal rates.

## Run the behavioral audit

Download or clone the public output artifacts, then run:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model kimi-k2-openrouter \
  --results-dir outputs \
  --expected-result-files 100 \
  --expected-comparison-groups 3000 \
  --affected-cells-csv results/kimi-k2-affected-cells.csv
```

Repeat with `--model kimi-k2-openrouter-thinking` for the Thinking condition.
The JSON report contains:

- integrity checks for model identity, expected file coverage, unique ladders,
  ordered tiers, stored counts, probabilities, and duplicate comparison cells;
- expected, parseable, and missing trial counts;
- missingness by variation category, comparison category, tier, and recorded
  failure reason;
- every affected comparison cell with `(A, B, missing)` counts;
- conditional `P(A)` among parseable responses;
- `P(A)` when all missing responses are assigned to B or to A;
- whether the pairwise winner is invariant to either assignment;
- monotonicity under the complete-case and two extreme-allocation scenarios.
- the exact minimum and maximum number of monotonic groups over every integer
  allocation of missing responses to A or B.

The extreme-allocation scenarios are sensitivity checks. They are not formal
upper and lower bounds on every coherence statistic.

## Trace-provenance gate

The command also inventories provider `reasoning_traces.jsonl` and OpenAI Batch
`reasoning_summaries.jsonl` sidecars. Quantitative trial-level analysis is
allowed only when:

- missing-response records contain stable `custom_id` values;
- trace rows contain the same IDs;
- every missing trial has a matching trace; and
- both the trace and missing-response record retain response text, with matching
  content and trial metadata;
- every linked trial has a visible, substantive rationale; and
- no trace rows are malformed or lack an ID.

Legacy sidecars containing only `message_idx` cannot be joined safely after
asynchronous execution, retries, or resumed runs. The report marks these files
as qualitative-only evidence and blocks annotation-template generation.

The Hugging Face inventory checked on 2026-09-17 at revision
`f978c15eb328a99f86cbe2057f918a38af25282f` contains 200 Kimi
`reasoning_traces.jsonl` sidecars, 100 per model. The first 20 records sampled
from each of Instruct's Religion and spirituality 1357 and Thinking's US politics
5556 and 1458 files contain `message_idx`, `attempt`, `content`, and `reasoning`,
but no stable `custom_id`. The previously audited local download contained only
the result files; its zero-sidecar inventory did not describe the full HF upload.

The downloaded result artifacts contain aggregate missing/unparseable counts
without trial-level `missing_responses`. They support behavioral sensitivity
analysis. The available traces support qualitative examination, but quantitative
coding of unique trials requires verified request/pair/direction and retry
mapping from the original run records. The current command blocks that coding
when linkage cannot be established; it does not infer trial IDs from line order
or treat repeated log entries as independent trials.

When linkage passes, generate the coding template:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model MODEL_KEY \
  --results-dir outputs \
  --annotation-template results/refusal-annotations.jsonl
```

## Code the visible rationales

For semantic annotation across models with a frozen random validation sample,
see [Step 11e: trace annotation and sampled validation](trace_annotation_validation.md).
That workflow reduces human coding to a reference sample and selected difficult
cases. It remains separate from the full trial-linked coding route below and
cannot convert unlinked legacy entries into unique-trial counts.

Step 11e's [offline parser comparison](trace_annotation_validation.md#2a-compare-parser-output-with-the-meaning-of-the-response)
also checks whether an unparseable final response nevertheless expresses a choice,
or a parseable response contains an objection. It preserves parser output,
expressed choice, comparison acceptance/rejection, and stated reason separately.
It does not turn a parsing failure into a refusal label or revise recorded votes.

### Automated first-pass screening, including legacy traces

Step 11d runs offline lexical rules on trace files without calling another model:

```bash
PYTHONPATH=src python scripts/05_analysis/11d_screen_trace_responses.py \
  --traces /path/to/reasoning_traces.jsonl /path/to/another/reasoning_traces.jsonl \
  --output-dir results/kimi-trace-screen
```

Use a new output directory each time. The command writes `screened_entries.jsonl`,
`review.csv`, and a completion `summary.json`. Each entry includes a source path,
file hash, line number, provisional label, and exact character offsets for its
matched evidence. Original results are never edited. Malformed records fail the
command; output without `summary.json` is incomplete and must not be analyzed.

Suggested labels distinguish explicit choice cues, conditional preferences
inside refusal text, abstention cues with unknown preference, conflicting text,
and cases needing review. A refusal followed by a letter is flagged as conflicting,
including a formatted arbitrary answer. The letters refer to the text as written;
no canonical option or AB/BA mapping is inferred. Refusal without an expressed
option never receives an imputed choice. Final response and provider-exposed
reasoning are screened separately; reasoning cues never replace the final answer.

These are **provisional lexical suggestions**, not validated semantic labels.
In particular, `conflicting_response` is a review flag for mixed textual cues,
not a finding of logical contradiction or preference incoherence. Rejecting a
comparison while stating a conditional choice can be internally consistent.
Rules can miss paraphrases, quotations, negation, and longer arguments. Every entry
requires human review. The reasoning screen may identify tentative or abandoned
statements, not a settled conclusion. Inspect the original trace before coding.
The CSV's `human_label` and `human_notes` fields are blank for that review.

This screening can be used on legacy traces because its unit is a **log entry**.
Counts retain repeated attempts and are not unique-trial counts or refusal rates.
The screening output is deliberately separate from Step 11b's trial-linked,
double-coded annotations and cannot bypass its provenance requirements. Establish
trial linkage and validate the coding before using annotations in research claims.

### Optional human coding of linked trials

Each trial should be coded independently by two researchers. Complete these
fields using the allowed values below.

The template retains the canonical outcomes, prompt-order A/B text, AB/BA
direction, tier and comparison metadata, raw response, and visible rationale so
coders can interpret each choice in its original context.

| Field | Allowed values |
| --- | --- |
| `reasoning_conclusion` | `favors_a`, `favors_b`, `equal`, `incomparable`, `no_conclusion`, `unclear` |
| `refusal_reason` | `none`, `no_personal_preference`, `political_or_religious_neutrality`, `ethical_sensitivity`, `false_dichotomy_or_incomparability`, `insufficient_information`, `instruction_or_format_failure`, `other` |
| `final_response` | `A`, `B`, `explicit_refusal`, `both_without_choice`, `malformed_or_incomplete` |
| `relationship` | `reasoning_and_answer_agree`, `reasoning_favors_a_or_b_but_final_refuses`, `reasoning_and_final_both_abstain`, `reasoning_and_final_disagree`, `unclear` |

Keep one JSON object per line and identify the coder in `coder`. Use the same
two independent coder identities for every trial. The summary counts each trial
once when the two coders agree. Disagreements are listed under
`trials_requiring_adjudication` and excluded from substantive counts until a
third consensus row is added with a distinct coder name and
`"adjudicated": true`.

Analyze completed annotations with:

```bash
PYTHONPATH=src python scripts/05_analysis/11b_analyze_refusal_robustness.py \
  --model MODEL_KEY \
  --results-dir outputs \
  --annotations results/refusal-annotations-coded.jsonl
```

The report adds category counts, a `reasoning_conclusion -> final_response`
transition table, percent agreement, and Cohen's kappa for double-coded trials.
The annotation trial keys must exactly match the linked missing-response set;
unlinked, partial, or unknown trial sets are rejected.

## Interpretation

Use the transition table to distinguish:

- **deliberative abstention:** the rationale itself rejects or cannot resolve
  the comparison;
- **expressed-choice/final-refusal pattern:** the rationale expresses support
  for A or B but the final response refuses; this describes the text and does
  not establish a hidden preference or causal suppression mechanism;
- **reasoning-answer disagreement:** the rationale favors one option and the
  final answer selects the other; and
- **format failure:** the substantive choice is recoverable from the text but
  the required A/B answer is absent or malformed.

Report the model-wide missing rate separately from rates within affected cells.
Affected cells were selected because they contain missing responses, so their
rate is descriptive of the case study and is not an estimate of the model's
general refusal propensity.

## Research basis and follow-up annotation

Refusal alone does not establish preference incoherence. Forced-choice and
open-ended elicitation can produce different expressed opinions
([Röttger et al., ACL 2024](https://aclanthology.org/2024.acl-long.816/)).
Visible reasoning is evidence about stated reasons, not guaranteed access to
the factors that caused an answer
([Chen et al., 2025](https://arxiv.org/abs/2505.05410)).

A follow-up workflow should code expressed choice, comparison acceptance or
refusal, and stated reasons separately, retaining exact evidence for final
responses and reasoning. Mixed responses require more than keyword detection
([XSTest, NAACL 2024](https://aclanthology.org/2024.naacl-long.301/)).
Claims of measured automated-label accuracy require task-specific human validation
([Pangakis and Wolken, ICWSM 2025](https://arxiv.org/abs/2409.09467)).
Uncertain cases can receive extra human attention
([CoAnnotating, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.92/)),
but validation must also sample ordinary cases to detect confident errors.
Pilot examples used to revise the rubric are not held-out validation data.
The full double-coding route above remains available as an optional follow-up.
Automated-only diagnostics require neither full-corpus human coding nor completion
of the reserved validation sample, but cannot establish annotation accuracy.

If estimating category prevalence, prediction-powered inference is a possible
later extension using a probability-sampled human reference set
([Angelopoulos et al., 2023](https://arxiv.org/abs/2301.09633)). It is not
implemented here and cannot resolve missing trial/retry provenance. Software
tests validate implementation behavior, not annotation accuracy.

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
Kendall and Spearman headline aggregates use Fisher transformations, and other
means can exclude non-finite observations whose denominators are not retained,
the comparison omits source headline values it cannot verify. It reports the
equal-ladder macro means used for inference; only count-reconciled monotonicity
and erratic-flip headline rates are retained. Valence subgroup comparisons are also
exploratory and their p-values are not multiplicity-adjusted.
The comparison also requires complete, internally reconciled summary coverage
and records every input path and SHA-256 hash in its output.
