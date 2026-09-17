# Trace annotation with sampled human validation

Step 11e adds a model-independent workflow for describing **expressed choice,
comparison acceptance/refusal, and stated reasons** in existing trace files.
It supports provider `reasoning_traces.jsonl` and batch
`reasoning_summaries.jsonl`, across any number of model directories. It does
not rerun the original preference experiment.

Behavioral audits and paired comparisons remain Steps 11b/11c. Step 11d is
lexical screening. Step 11e supplies semantic judge prompts, evidence checks,
blinded human reference templates, a frozen random sample, disagreement
triage, and annotation-error metrics. None of these stages imputes votes.

## Research and interpretation

The design follows these findings, with our implementation choices distinguished
from the papers' results:

| Research | Finding | Application here |
| --- | --- | --- |
| [Röttger et al., ACL 2024](https://aclanthology.org/2024.acl-long.816/) | Forced-choice elicitation can change expressed opinions. | Refusal is an observable response, not automatically preference incoherence. |
| [XSTest, NAACL 2024](https://aclanthology.org/2024.naacl-long.301/) | Full/partial refusal and compliance differ; keyword and model classifiers can misclassify mixed responses. | Separate choice from acceptance of the comparison and require task-specific validation. |
| [Chen et al., 2025](https://arxiv.org/abs/2505.05410) | Visible reasoning can omit influences on the answer. | Code stated reasons and expressed choices; do not infer hidden preferences or causal suppression. |
| [Pangakis and Wolken, ICWSM 2025](https://arxiv.org/abs/2409.09467) | Automated annotation performance varies across tasks. | Evaluate each dimension against independent human reference labels on this corpus. |
| [CoAnnotating, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.92/) | Uncertainty can guide allocation of human annotation effort. | Offer extra triage for difficult cases while retaining a random validation sample that includes ordinary cases. |
| [Prediction-Powered Inference, 2023](https://arxiv.org/abs/2301.09633) | Automated predictions plus a labeled reference sample can support error-corrected inference. | Possible later extension; prediction-powered prevalence estimates are **not implemented** in this version. |

The workflow measures observable text. A conditional A accompanied by rejection
of the framing can be coherent. A bare A is an expressed selection, not proof
of personal preference. A rationale/final-answer mismatch is a textual pattern,
not evidence of a causal mechanism. Exact quotations establish where evidence
comes from; the human sample tests whether labels correctly interpret it.

## 1. Prepare all available traces offline

```bash
PYTHONPATH=src python scripts/05_analysis/11e_annotate_trace_responses.py prepare \
  --inputs /path/to/outputs \
  --output-dir results/trace-annotation \
  --pilot-size 20 --validation-size 100 --triage-size 20 --seed 20260917
```

Inputs may be directories, individual JSONL files, or a mixture. Directories are
searched recursively for the two supported sidecar names; repeated input paths
are loaded once. Counts retain repeated records and retries. No model-specific
filter is imposed. Files outside the supplied inputs are outside the study scope.

The new output directory contains:

- `entries.jsonl`: immutable text snapshots, source SHA-256 hashes, line numbers,
  entry hashes, and original metadata. Letters remain in their original order;
  missing prompts, canonical options, and stable trial IDs are not reconstructed.
- `rubric.json`: versioned instructions and allowed labels.
- `pilot_texts.jsonl` and `pilot_human_template.jsonl`: rubric-development examples.
- `validation_texts.jsonl` and `validation_human_template.jsonl`: a simple random
  sample without replacement from **all supplied log entries**, including ordinary
  answers. These views contain no model identity, lexical flags, or judge predictions.
- `triage_texts.jsonl` and `triage_human_template.jsonl`: additional lexical-screen
  candidates for qualitative exploration. This purposive subset is not used for accuracy.
- `manifest.json`: completion marker, frozen split IDs, seed, inclusion probability,
  source inventory, and corpus hash. A missing manifest means preparation is incomplete.

The validation sample is selected first. Pilot/triage selection excludes exact
copies of its final-response/reasoning pair. If too few distinct texts remain,
reduce `--pilot-size`, possibly to zero. This avoids exact-text tuning leakage;
it cannot establish trial independence or remove near-duplicate/retry leakage.
Do not inspect validation predictions while revising the rubric. Any rubric
change needs a new version/bundle and a new untouched validation set; do not keep
tuning to the same validation errors. Tests and pilot labels are not validation.

The example sample sizes are starting points, **not guarantees of precision**.
A rare class may have no human examples; its recall then remains unestimated.
Choose sample size before looking at validation outcomes based on the intended
claims and acceptable uncertainty. Cases from extra triage can illustrate rare
patterns, but must not be mixed into the random-sample accuracy denominator.

## 2. Run one or more semantic annotators

First plan a bounded run using an API-backed model key from project configuration:

```bash
PYTHONPATH=src python scripts/05_analysis/11e_annotate_trace_responses.py annotate \
  --bundle results/trace-annotation \
  --model YOUR_JUDGE_MODEL_KEY --split pilot \
  --max-entries 20 --max-tokens 4096 \
  --output-dir results/judge-pilot
```

The default is offline: it reports request count and input characters, creates no
output directory, and accesses no API credentials. Add `--execute` to send the
requests through the existing runtime. `--max-entries` is a hard limit: exceeding
it fails before sending anything, rather than silently annotating a truncated sample.
No input text is truncated. Providers may reject inputs exceeding their context.
The run preserves invalid outputs and token-cap failures as missing annotations,
not model refusals, and retries only transport failures (at most three attempts).
There is no automatic retry to force a different semantic label.

After fixing the rubric on the pilot, run `--split validation` with its corresponding
entry limit, or `--split all` to obtain provisional labels for the full corpus.
Use a new directory for each run. A second independently configured judge can
produce another predictions file for disagreement review; judge agreement alone
does not establish accuracy. The runner is sequential and intended to be bounded;
it does not implement checkpoint resume or automatically merge runs.

Each channel receives these dimensions, each with an exact evidence quotation:

| Dimension | Labels |
| --- | --- |
| `expressed_choice` | `A`, `B`, `conditional_A`, `conditional_B`, `both`, `equal`, `incomparable`, `none`, `unclear`, `unobservable` |
| `comparison_response` | `accepts`, `rejects`, `mixed`, `unclear`, `unobservable` |
| `stated_reason` | `no_personal_preference`, `neutrality`, `ethical_objection`, `incomparability`, `insufficient_information`, `format_constraint`, `substantive_comparison`, `multiple`, `other`, `none`, `unclear`, `unobservable` |

The final response and visible reasoning are separate channels. An empty channel
must be `unobservable`; present text cannot be `unobservable`. `none` means no
expressed choice or stated reason in present text. `multiple` records several
reasons with no primary one. Ambiguous/mixed cases require `needs_review: true`.
The validator checks schema and quotation provenance, not semantic entailment.

Rubric 1.2 presents each channel as numbered literal source passages, each at
most 400 characters, preserving every character and its original offsets. The
automatic judge selects channel-scoped `evidence_ids`; the program resolves them
to exact quotations and records the selected offsets. Unknown, duplicate, or
cross-channel IDs fail validation. Human coding still uses direct quotations.
Passage selection establishes source provenance, not whether the evidence
semantically supports the selected label; human validation remains necessary.

Live development runs of 1.0 and 1.1 produced cross-channel, overescaped, or
paraphrased quotations, which the validator rejected. These failed outputs are
method-development evidence, not subject-model refusals or evidence of accuracy.
Old bundles remain tied to their rubric version and cannot be silently reused
with 1.2. A previously frozen validation sample may be retained when it has
never been inspected or used for tuning; re-export templates with the new rubric
binding and document that its IDs and source text are unchanged.

`run.json` records model, rubric, corpus binding, settings and selected IDs.
`attempts.jsonl` retains raw judge outputs, runtime outcomes, and validation failures.
`predictions.jsonl` contains only schema/evidence-valid annotations.
`summary.json` marks execution completion and separately reports whether all
entries received valid annotations. Capped and empty provider responses remain
missing annotations and processing continues. Infrastructure/configuration
failures stop the run, leaving attempts available but no completion summary.
Always inspect coverage.
Three consecutive invalid annotations also stop the run and write `incomplete.json`,
preserving usage and failures for debugging instead of spending through the corpus.

## 3. Human-check the frozen sample

Two researchers independently copy `validation_human_template.jsonl`, enter their
own `annotator.id`, and fill each label/evidence field using
`validation_texts.jsonl` and `rubric.json`. Leave the entry IDs and hashes intact.
Evidence is an exact substring of its channel. `none`/`unclear` can have no
quotation when based on absence; all substantive labels require quotations.
Do not consult the judge output while creating reference labels.

This is sample-based validation, not a requirement to manually annotate every
trace. The original Step 11b full trial-linked double-coding route remains an
alternative when unique missing trials can actually be linked.

## 4. Evaluate and resolve disagreements

```bash
PYTHONPATH=src python scripts/05_analysis/11e_annotate_trace_responses.py evaluate \
  --bundle results/trace-annotation \
  --predictions results/judge-all/predictions.jsonl \
  --human results/coder-one.jsonl results/coder-two.jsonl \
  --output-dir results/annotation-validation
```

Omit `--human` for descriptive provisional counts only. Multiple `--predictions`
paths compare distinct judge identities and populate a review queue.

The report contains channel/dimension counts, prediction coverage, per-class
precision/recall/F1, confusion matrices, accuracy with approximate Wilson 95%
intervals, and human agreement/kappa before adjudication. Unobserved denominators
produce null metrics, not perfect scores. Human files must each cover exactly
the frozen sample with two distinct fixed coder identities. Unknown/duplicate
IDs, mismatched hashes, invalid evidence, and inconsistent schemas are rejected.

Human label disagreements generate `adjudication_template.jsonl`. A third
identified researcher resolves those entries; rerun evaluation in a new directory
with `--adjudications completed-adjudications.jsonl`. Evidence wording and
`needs_review` differences alone are not substantive-label disagreements.
Missing predictions or unresolved human disagreements block that judge's entire
validation-metric table, avoiding selectively discarding difficult cases.

Review queues include uncertainty and disagreement cases; they are not a
probability sample. Their inspection does not replace validation. There is no
automatic accuracy threshold or automatic promotion to scientifically validated
labels. Review class-specific errors and uncertainty before choosing claims.

## Scope of conclusions

All counts describe **log entries in the supplied files**, including retries.
The validation sample is representative of that fixed entry population by design;
the approximate intervals do not establish independent-trial or model-wide
generalization. Source files selected because of failures remain a case study.
The pipeline does not estimate unique-trial refusal prevalence, run
prediction-powered inference, or bypass Step 11b's provenance gate.
Repeated refusals are not independent evidence simply because they occupy
different log lines. A stable `custom_id` field alone does not verify retry mapping.

Successful software tests and mocked judge runs verify implementation, not a live
judge's semantic accuracy. Publishing category frequencies as validated findings
requires the human-reference step and an appropriately defined population.
