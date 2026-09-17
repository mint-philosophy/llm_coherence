"""Scientific failure modes and offline end-to-end annotation/validation checks."""

import asyncio
import copy
import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_coherence.analysis.annotate_trace_responses import annotate, judge_messages
from llm_coherence.analysis.trace_annotation import (
    FIELDS,
    classification_metrics,
    decode_judge_annotation,
    evidence_passages,
    evaluate,
    load_bundle,
    prepare,
    read_jsonl,
    template,
    validate_annotation,
    write_jsonl,
)


def annotation(entry, choice="A"):
    result = {"needs_review": False}
    for channel, text in entry["text"].items():
        if not text:
            result[channel] = {
                f: {"label": "unobservable", "evidence": []} for f in FIELDS
            }
        else:
            result[channel] = {
                "expressed_choice": {"label": choice, "evidence": [text]},
                "comparison_response": {"label": "accepts", "evidence": [text]},
                "stated_reason": {"label": "none", "evidence": []},
            }
    return result


def wire_annotation(entry, choice="A"):
    result = annotation(entry, choice)
    for channel in entry["text"]:
        for item in result[channel].values():
            quotes = item.pop("evidence")
            item["evidence_ids"] = [f"{channel}:0000"] if quotes else []
    return result


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.traces = self.root / "inputs" / "model-one" / "reasoning_traces.jsonl"
        self.traces.parent.mkdir(parents=True)
        write_jsonl(
            self.traces,
            (
                {
                    "content": f"I choose A. Example {i}.",
                    "reasoning": None,
                    "message_idx": i,
                    "attempt": 0,
                }
                for i in range(12)
            ),
        )
        self.bundle = self.root / "bundle"
        self.manifest = prepare(
            [self.root / "inputs"],
            self.bundle,
            pilot_size=2,
            validation_size=4,
            triage_size=2,
            seed=42,
        )
        _, self.entries = load_bundle(self.bundle)
        self.sample = sorted(self.manifest["splits"]["validation"])

    def records(self, kind, name, ids=None):
        records = []
        for i in ids if ids is not None else self.sample:
            row = template(self.entries[i], name)
            row["annotator"]["kind"] = kind
            row["annotation"] = annotation(self.entries[i])
            records.append(row)
        return records

    def save(self, name, rows):
        path = self.root / name
        write_jsonl(path, rows)
        return path

    def files(self):
        pred = self.save("judge.jsonl", self.records("model", "judge"))
        h1 = self.save("human1.jsonl", self.records("human", "coder-1"))
        h2 = self.save("human2.jsonl", self.records("human", "coder-2"))
        return pred, h1, h2

    def test_recursive_discovery_across_models_and_summary_formats(self):
        second = self.root / "inputs/model-two/reasoning_summaries.jsonl"
        second.parent.mkdir()
        write_jsonl(second, [{"content": "B", "summary": "I select B."}])
        manifest = prepare(
            [self.root / "inputs", self.traces],
            self.root / "all",
            pilot_size=0,
            validation_size=13,
            triage_size=0,
            seed=2,
        )
        self.assertEqual(manifest["entry_count"], 13)
        self.assertEqual(len(manifest["sources"]), 2)
        _, entries = load_bundle(self.root / "all")
        self.assertTrue(
            any(
                e["text"]["visible_reasoning"] == "I select B."
                for e in entries.values()
            )
        )

    def test_judge_input_keeps_literal_channels_separate(self):
        entry = {
            "text": {
                "final_response": "A",
                "visible_reasoning": 'I prefer "B".\nBut this is tentative.\u2028',
            }
        }
        messages = judge_messages(entry)
        content = messages[1]["content"]
        final_block, reasoning_block = content.split(
            "\n\nChannel: visible_reasoning", 1
        )
        self.assertIn("Channel: final_response; characters: 1", final_block)
        self.assertIn("\nA\nEND_final_response_", final_block)
        self.assertNotIn("tentative", final_block)
        self.assertIn(entry["text"]["visible_reasoning"], reasoning_block)
        self.assertNotIn('\\"B\\"', reasoning_block)
        self.assertIn("Its stated_reason is none", messages[0]["content"])

    def test_evidence_passages_preserve_all_characters_and_offsets(self):
        text = ('Quoted "A".\nLiteral \\n. Unicode\u2028separator. ' * 40) + " final"
        passages = evidence_passages(text, "visible_reasoning")
        self.assertEqual("".join(p["text"] for p in passages), text)
        self.assertEqual(len({p["id"] for p in passages}), len(passages))
        for p in passages:
            self.assertEqual(text[p["start"] : p["end"]], p["text"])
            self.assertLessEqual(len(p["text"]), 400)

    def test_judge_ids_resolve_exact_evidence_without_rewriting_quotes(self):
        entry = {
            "text": {"final_response": "A", "visible_reasoning": 'I choose "A".\n'}
        }
        labels, spans = decode_judge_annotation(wire_annotation(entry), entry)
        self.assertEqual(
            labels["visible_reasoning"]["expressed_choice"]["evidence"],
            ['I choose "A".\n'],
        )
        self.assertEqual(
            spans["final_response"]["expressed_choice"],
            [{"id": "final_response:0000", "start": 0, "end": 1}],
        )
        for bad in ("visible_reasoning:0000", "final_response:9999"):
            raw = wire_annotation(entry)
            raw["final_response"]["expressed_choice"]["evidence_ids"] = [bad]
            with self.assertRaisesRegex(ValueError, "Unknown or cross-channel"):
                decode_judge_annotation(raw, entry)
        raw = wire_annotation(entry)
        raw["final_response"]["expressed_choice"]["evidence_ids"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            decode_judge_annotation(raw, entry)

    def test_sampling_is_reproducible_blinded_and_text_disjoint(self):
        other = prepare(
            [self.traces],
            self.root / "other",
            pilot_size=2,
            validation_size=4,
            triage_size=2,
            seed=42,
        )
        self.assertEqual(other["splits"], self.manifest["splits"])
        self.assertEqual(self.manifest["validation_inclusion_probability"], 1 / 3)
        view = read_jsonl(self.bundle / "validation_texts.jsonl")
        self.assertEqual(set(view[0]), {"entry_id", "text"})
        self.assertNotIn("model-one", str(judge_messages(self.entries[self.sample[0]])))
        val_text = {self.entries[i]["text_sha256"] for i in self.sample}
        for split in ("pilot", "triage"):
            self.assertFalse(
                val_text.intersection(
                    self.entries[i]["text_sha256"] for i in other["splits"][split]
                )
            )

    def test_retries_are_retained_without_invented_unique_trials(self):
        source = self.save("repeat.jsonl", [{"content": "A"}] * 5)
        manifest = prepare(
            [source],
            self.root / "repeats",
            pilot_size=0,
            validation_size=2,
            triage_size=3,
            seed=1,
        )
        _, entries = load_bundle(self.root / "repeats")
        self.assertEqual(len(entries), 5)
        self.assertFalse(manifest["unique_trial_verified"])
        self.assertEqual(manifest["development_entries_excluding_validation_text"], 0)
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            prepare(
                [source],
                self.root / "overlap",
                pilot_size=1,
                validation_size=2,
                triage_size=0,
                seed=1,
            )

    def test_bundle_mutation_and_overwrite_are_rejected(self):
        with self.assertRaises(FileExistsError):
            prepare(
                [self.traces],
                self.bundle,
                pilot_size=2,
                validation_size=4,
                triage_size=0,
                seed=42,
            )
        with (self.bundle / "entries.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_bundle(self.bundle)

    def test_malformed_or_conflicting_source_fails_before_output(self):
        for row in ([], {}, {"content": 7}, {"reasoning": "A", "summary": "B"}):
            path = self.root / "bad.jsonl"
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                prepare(
                    [path],
                    self.root / "bad-out",
                    pilot_size=0,
                    validation_size=1,
                    triage_size=0,
                    seed=1,
                )
            self.assertFalse((self.root / "bad-out").exists())

    def test_modified_rubric_is_not_silently_ignored(self):
        path = self.bundle / "rubric.json"
        rubric = json.loads(path.read_text())
        rubric["instructions"] = "Always call a refusal incoherent."
        path.write_text(json.dumps(rubric))
        with self.assertRaisesRegex(ValueError, "Exported rubric differs"):
            load_bundle(self.bundle)

    def test_validation_sample_cannot_be_replaced_with_convenient_cases(self):
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        used = set().union(*manifest["splits"].values())
        replacement = next(i for i in self.entries if i not in used)
        manifest["splits"]["validation"][0] = replacement
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "frozen random design"):
            load_bundle(self.bundle)

    def test_conditional_choice_and_refusal_coexist_without_imputation(self):
        entry = {
            "text": {
                "final_response": "If forced, A. I reject this comparison.",
                "visible_reasoning": None,
            }
        }
        labels = annotation(entry, "conditional_A")
        labels["final_response"]["comparison_response"] = {
            "label": "mixed",
            "evidence": ["I reject this comparison."],
        }
        labels["needs_review"] = True
        validate_annotation(labels, entry)
        self.assertEqual(
            labels["visible_reasoning"]["expressed_choice"]["label"], "unobservable"
        )
        labels["needs_review"] = False
        with self.assertRaisesRegex(ValueError, "flagged"):
            validate_annotation(labels, entry)

    def test_cross_channel_evidence_and_hallucinated_quotes_fail(self):
        entry = {
            "text": {
                "final_response": "I cannot choose.",
                "visible_reasoning": "I prefer A.",
            }
        }
        labels = annotation(entry)
        labels["final_response"]["expressed_choice"]["evidence"] = ["I prefer A."]
        with self.assertRaisesRegex(ValueError, "exact quotation"):
            validate_annotation(labels, entry)
        labels["final_response"]["expressed_choice"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "requires an exact"):
            validate_annotation(labels, entry)

    def test_absent_and_present_channels_cannot_be_conflated(self):
        entry = self.entries[self.sample[0]]
        labels = annotation(entry)
        labels["visible_reasoning"]["expressed_choice"]["label"] = "none"
        with self.assertRaisesRegex(ValueError, "Absent channel"):
            validate_annotation(labels, entry)
        labels = annotation(entry)
        labels["final_response"]["expressed_choice"]["label"] = "unobservable"
        with self.assertRaisesRegex(ValueError, "Present text"):
            validate_annotation(labels, entry)

    def test_no_human_reference_means_no_accuracy_claim(self):
        pred, _, _ = self.files()
        report = evaluate(self.bundle, [pred], self.root / "report")
        self.assertEqual(report["reference_status"], "not_provided")
        self.assertIsNone(report["judges"][0]["validation_metrics"])

    def test_complete_reference_produces_per_class_metrics(self):
        pred, h1, h2 = self.files()
        rows = read_jsonl(pred)
        # A syntactically valid but semantically wrong model label must be
        # counted as an error, not certified by matching evidence strings.
        rows[0]["annotation"]["final_response"]["expressed_choice"]["label"] = "B"
        write_jsonl(pred, rows)
        report = evaluate(self.bundle, [pred], self.root / "report", [h1, h2])
        metric = report["judges"][0]["validation_metrics"][
            "final_response.expressed_choice"
        ]
        self.assertEqual(metric["accuracy"], 0.75)
        self.assertEqual(metric["confusion"]["A"], {"A": 3, "B": 1})
        self.assertEqual(metric["per_class"]["B"]["precision"], 0)
        self.assertIsNone(metric["per_class"]["B"]["recall"])

    def test_unresolved_disagreements_block_all_accuracy_metrics(self):
        pred, h1, h2 = self.files()
        rows = read_jsonl(h2)
        rows[0]["annotation"]["final_response"]["expressed_choice"]["label"] = "B"
        write_jsonl(h2, rows)
        report = evaluate(self.bundle, [pred], self.root / "pending", [h1, h2])
        self.assertEqual(report["pending_adjudication"], [self.sample[0]])
        self.assertIsNone(report["judges"][0]["validation_metrics"])
        adjudication = self.records("human", "adjudicator", [self.sample[0]])
        adj = self.save("adj.jsonl", adjudication)
        report = evaluate(self.bundle, [pred], self.root / "resolved", [h1, h2], adj)
        self.assertEqual(report["reference_status"], "complete")
        self.assertEqual(
            report["judges"][0]["validation_metrics"][
                "final_response.expressed_choice"
            ]["accuracy"],
            1,
        )

    def test_missing_predictions_are_not_silently_dropped(self):
        pred, h1, h2 = self.files()
        write_jsonl(pred, read_jsonl(pred)[:-1])
        report = evaluate(self.bundle, [pred], self.root / "report", [h1, h2])
        self.assertEqual(
            report["judges"][0]["missing_validation_predictions"], [self.sample[-1]]
        )
        self.assertIsNone(report["judges"][0]["validation_metrics"])

    def test_wrong_partial_duplicate_or_same_coder_reference_rejected(self):
        pred, h1, h2 = self.files()
        original = read_jsonl(h2)
        for rows in (original[:-1], original + original[:1], read_jsonl(h1)):
            write_jsonl(h2, rows)
            with self.assertRaises(ValueError):
                evaluate(self.bundle, [pred], self.root / "bad", [h1, h2])
        rows = copy.deepcopy(original)
        rows[0]["entry_sha256"] = "wrong"
        write_jsonl(h2, rows)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            evaluate(self.bundle, [pred], self.root / "bad", [h1, h2])

    def test_extra_adjudication_is_rejected(self):
        pred, h1, h2 = self.files()
        adj = self.save("adj.jsonl", self.records("human", "third", [self.sample[0]]))
        with self.assertRaisesRegex(ValueError, "only to disputed"):
            evaluate(self.bundle, [pred], self.root / "bad", [h1, h2], adj)

    def test_judge_agreement_is_not_validation_and_disagreement_is_queued(self):
        pred, _, _ = self.files()
        rows = self.records("model", "second")
        rows[0]["annotation"]["final_response"]["expressed_choice"]["label"] = "B"
        second = self.save("second.jsonl", rows)
        report = evaluate(self.bundle, [pred, second], self.root / "report")
        self.assertEqual(report["review_queue_entries"], 1)
        self.assertTrue(all(j["validation_metrics"] is None for j in report["judges"]))

    def test_known_metrics_and_degenerate_kappa(self):
        result = classification_metrics(["A", "A", "B"], ["A", "B", "B"], ("A", "B"))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["cohen_kappa"], 0.4)
        self.assertEqual(result["per_class"]["A"]["recall"], 0.5)
        self.assertIsNone(
            classification_metrics(["A"], ["A"], ("A", "B"))["cohen_kappa"]
        )

    def test_dry_run_and_entry_limit_never_instantiate_agent(self):
        def factory(**kwargs):
            self.fail("Offline preparation must not instantiate a model")

        result = asyncio.run(
            annotate(
                self.bundle,
                self.root / "dry",
                model="test",
                split="validation",
                max_entries=4,
                max_tokens=100,
                agent_factory=factory,
            )
        )
        self.assertEqual(result["selected_entries"], 4)
        self.assertFalse((self.root / "dry").exists())
        with self.assertRaisesRegex(ValueError, "exceeds"):
            asyncio.run(
                annotate(
                    self.bundle,
                    self.root / "dry",
                    model="test",
                    split="all",
                    max_entries=4,
                    max_tokens=100,
                    execute=True,
                    agent_factory=factory,
                )
            )

    def test_mocked_judge_end_to_end_preserves_invalid_outputs(self):
        sample_entries = [
            self.entries[i] for i in self.manifest["splits"]["validation"]
        ]
        responses = [json.dumps(wire_annotation(e)) for e in sample_entries]
        responses[1] = "I cannot annotate that."

        class Agent:
            async def async_completions(self, messages, verbose):
                return [responses.pop(0)]

        result = asyncio.run(
            annotate(
                self.bundle,
                self.root / "judge",
                model="test",
                split="validation",
                max_entries=4,
                max_tokens=500,
                execute=True,
                agent_factory=lambda **kw: Agent(),
            )
        )
        self.assertEqual(result["valid_annotations"], 3)
        self.assertEqual(result["invalid_annotations"], 1)
        self.assertFalse(result["semantic_accuracy_validated"])
        attempts = read_jsonl(self.root / "judge/attempts.jsonl")
        self.assertEqual(attempts[1]["raw_response"], "I cannot annotate that.")
        self.assertEqual(attempts[1]["status"], "invalid_annotation")

    def test_runtime_failure_leaves_no_completion_marker(self):
        class Agent:
            async def async_completions(self, messages, verbose):
                return [None]

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            asyncio.run(
                annotate(
                    self.bundle,
                    self.root / "failed",
                    model="test",
                    split="validation",
                    max_entries=4,
                    max_tokens=500,
                    execute=True,
                    agent_factory=lambda **kw: Agent(),
                )
            )
        self.assertFalse((self.root / "failed/summary.json").exists())
        self.assertEqual(len(read_jsonl(self.root / "failed/attempts.jsonl")), 1)

    def test_unicode_separators_survive_corpus_and_evidence_roundtrip(self):
        text = "I\u2028choose\u2029A.\u0085"
        source = self.save("unicode.jsonl", [{"content": text}])
        bundle = self.root / "unicode"
        prepare(
            [source], bundle, pilot_size=0, validation_size=1, triage_size=0, seed=42
        )
        _, entries = load_bundle(bundle)
        entry = next(iter(entries.values()))
        self.assertEqual(entry["text"]["final_response"], text)
        paths = []
        for kind, name in (("model", "judge"), ("human", "one"), ("human", "two")):
            row = template(entry, name)
            row["annotator"]["kind"] = kind
            row["annotation"] = annotation(entry)
            paths.append(self.save(f"unicode-{name}.jsonl", [row]))
        report = evaluate(bundle, paths[:1], self.root / "unicode-report", paths[1:])
        self.assertEqual(
            report["judges"][0]["validation_metrics"][
                "final_response.expressed_choice"
            ]["accuracy"],
            1,
        )

    def test_real_runtime_capped_and_empty_responses_are_preserved_and_continue(self):
        # Exercise the actual runtime adapter, mocking only the provider call.
        from llm_coherence.runtime.agents import LiteLLMAgent

        ids = self.manifest["splits"]["validation"]
        replies = [('{"final_response":', "length"), ("", "stop")]
        replies += [
            (json.dumps(wire_annotation(self.entries[i])), "stop") for i in ids[2:]
        ]

        async def completion(**kwargs):
            content, finish = replies.pop(0)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content), finish_reason=finish
                    )
                ]
            )

        fake_sdk = SimpleNamespace(
            acompletion=completion,
            BadRequestError=type("BadRequestError", (Exception,), {}),
        )

        def factory(**kwargs):
            agent = LiteLLMAgent(
                model="mock/provider", max_retries=1, retry_transport_only=True
            )
            agent._log_usage = lambda response: None
            return agent

        with patch.dict(sys.modules, {"litellm": fake_sdk}):
            report = asyncio.run(
                annotate(
                    self.bundle,
                    self.root / "runtime",
                    model="mock",
                    split="validation",
                    max_entries=4,
                    max_tokens=100,
                    execute=True,
                    agent_factory=factory,
                )
            )
        self.assertEqual(report["valid_annotations"], 2)
        self.assertEqual(report["invalid_annotations"], 2)
        self.assertFalse(replies)
        attempts = read_jsonl(self.root / "runtime/attempts.jsonl")
        self.assertEqual(attempts[0]["raw_response"], '{"final_response":')
        self.assertEqual(attempts[0]["provider_outcome"]["status"], "token_capped")
        self.assertIn("token cap", attempts[0]["validation_error"])
        self.assertEqual(attempts[1]["raw_response"], "")
        self.assertEqual(attempts[1]["provider_outcome"]["status"], "empty_response")

    def test_repeated_invalid_annotations_stop_before_spending_entire_split(self):
        calls = []

        class Agent:
            async def async_completions(self, messages, verbose):
                calls.append(1)
                return ["not-json"]

        with self.assertRaisesRegex(RuntimeError, "Three consecutive invalid"):
            asyncio.run(
                annotate(
                    self.bundle,
                    self.root / "invalid-run",
                    model="test",
                    split="validation",
                    max_entries=4,
                    max_tokens=500,
                    execute=True,
                    agent_factory=lambda **kwargs: Agent(),
                )
            )
        self.assertEqual(len(calls), 3)
        self.assertTrue((self.root / "invalid-run/incomplete.json").exists())
        self.assertFalse((self.root / "invalid-run/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
