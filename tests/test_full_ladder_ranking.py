"""Offline checks for experiment coverage, scoring, and duplicate-call prevention."""

import asyncio
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_coherence.experiments import full_ladder_ranking as experiment


class RankingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.requests, self.manifest = experiment.prepare(
            experiment.DEFAULT_INPUT, list(experiment.PAPER_MODELS)
        )

    def test_all_100_for_all_23_without_judge_metadata(self):
        self.assertEqual(self.manifest["n_planned_requests"], 2300)
        self.assertEqual(len({t.ladder_id for t in self.tasks}), 100)
        self.assertNotIn("gpt-55-openai", self.manifest["models"])
        self.assertNotIn("glm-45-base-logprobs", self.manifest["models"])
        for request in self.requests:
            text = json.dumps(request["messages"])
            self.assertNotIn("reasonable person", text)
            self.assertNotIn("PROPERTY BEING VARIED", text)
            for label in "ABCDEFG":
                self.assertEqual(text.count(f"[{label}]"), 1)

    def test_reject_short_input_and_duplicate_ids(self):
        original = json.loads(experiment.DEFAULT_INPUT.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ladders.json"
            for bad in [original[:99], original[:99] + [original[0]]]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    experiment.prepare(path, ["gpt-54-nano"])

    def test_preview_cannot_create_agent_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            with patch.object(experiment, "create_agent", side_effect=AssertionError("paid call")):
                with redirect_stdout(io.StringIO()):
                    result = experiment.main(["--paper-slate", "--output-dir", str(root)])
            self.assertEqual(result, 0)
            self.assertFalse(root.exists())

    def test_execution_scores_and_resume_never_replays_calls(self):
        tasks = self.tasks[:3]
        responses = {}
        for i, task in enumerate(tasks):
            order = [task.labels[task.shuffle_indices.index(t)] for t in range(7)]
            if i == 1:
                order[0], order[1] = order[1], order[0]
            if i == 2:
                order[-1] = order[0]  # Invalid duplicate label.
            responses[experiment.digest(experiment.messages_for(task))] = json.dumps({"ranking": order})
        calls = []

        class FakeAgent:
            def __init__(self):
                self.usage_log = []
                self.last_completion_outcomes = []

            async def async_completions(self, messages, **kwargs):
                key = experiment.digest(messages[0])
                calls.append(key)
                raw = responses[key]
                self.usage_log.append({"total_tokens": 5})
                self.last_completion_outcomes = [{"status": "completed", "raw_response": raw}]
                return [raw]

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            run = lambda: asyncio.run(experiment.run_model(
                "gpt-54-nano", tasks, {}, Path(temp), 2, factory=lambda *_: FakeAgent()
            ))
            summary = run()
            self.assertEqual(len(calls), 3)
            self.assertEqual(summary["n_valid_rankings"], 2)
            self.assertEqual(summary["exact_match_count"], 1)
            self.assertAlmostEqual(summary["mean_kendall_tau"], (1 + 19 / 21) / 2)
            self.assertEqual(summary["n_invalid_or_failed"], 1)
            self.assertEqual(run(), summary)
            self.assertEqual(len(calls), 3)

    def test_unresolved_paid_request_is_not_replayed(self):
        task = self.tasks[0]
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "responses.jsonl").write_text(json.dumps({
                "event": "started", "ladder_id": task.ladder_id,
            }) + "\n")
            summary = asyncio.run(experiment.run_model(
                "gpt-54-nano", [task], {}, root, 1,
                factory=lambda *_: self.fail("An unresolved request was replayed"),
            ))
            self.assertEqual(summary["n_unresolved_requests"], 1)
            self.assertEqual(summary["n_valid_rankings"], 0)

    def test_failed_pilot_stops_remaining_requests(self):
        class FakeAgent:
            usage_log = []
            last_completion_outcomes = [{"status": "token_capped", "raw_response": "{}"}]

            async def async_completions(self, *args, **kwargs):
                return [None]

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            summary = asyncio.run(experiment.run_model(
                "gpt-54-nano", self.tasks, {}, Path(temp), 3, factory=lambda *_: FakeAgent()
            ))
            self.assertEqual(summary["n_started"], 1)
            self.assertEqual(summary["n_not_requested"], 99)
            self.assertEqual(summary["n_valid_rankings"], 0)

    def test_full_command_covers_all_models_preserves_instrument_and_checks_resume(self):
        calls = []

        class FakeAgent:
            def __init__(self):
                self.usage_log = []
                self.last_completion_outcomes = []

            async def async_completions(self, messages, **kwargs):
                calls.append(messages)
                raw = json.dumps({"ranking": list("ABCDEFG")})
                self.usage_log.append({"total_tokens": 10})
                self.last_completion_outcomes = [{"status": "completed", "raw_response": raw}]
                return [raw]

        protected = list(experiment.DEFAULT_INPUT.parent.glob("*.json"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp) / "experiment"
            argv = ["--paper-slate", "--output-dir", str(root), "--run", "--max-cost-usd", "40"]
            with patch.object(experiment, "agent_for", side_effect=lambda *_: FakeAgent()), \
                 patch.object(experiment, "ensure_api_key_env"), \
                 patch.object(experiment, "close_api_async_clients", new_callable=AsyncMock):
                self.assertEqual(experiment.main(argv), 0)
                self.assertEqual(len(calls), 2300)
                summaries = json.loads((root / "summary.json").read_text())
                self.assertEqual(len(summaries), 23)
                self.assertTrue(all(s["n_valid_rankings"] == 100 for s in summaries))
                self.assertEqual(experiment.main(argv), 0)
                self.assertEqual(len(calls), 2300)
                with self.assertRaisesRegex(ValueError, "Saved protocol differs"):
                    experiment.main(argv + ["--seed", "99"])
                self.assertEqual(len(calls), 2300)
            self.assertFalse(list(root.rglob("*pruned*")))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})

    def test_missing_or_low_budget_never_creates_output_or_calls_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            with patch.object(experiment, "create_agent", side_effect=AssertionError("paid call")):
                for budget in ([], ["--max-cost-usd", "3"], ["--max-cost-usd", "nan"]):
                    with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                        experiment.main(["--paper-slate", "--run", "--output-dir", str(root)] + budget)
                    self.assertFalse(root.exists())

    def test_provider_prices_and_sdk_retry_controls(self):
        for model, profile in self.manifest["models"].items():
            if profile["endpoint"]["model_type"] == "openrouter":
                self.assertTrue(profile["extra_body"]["provider"]["require_parameters"])
                self.assertEqual(profile["extra_body"]["provider"]["max_price"]["request"], 0)
        complete = AsyncMock(return_value=SimpleNamespace(id="response", model="gpt-5.6-sol", usage={
            "prompt_tokens": 200, "completion_tokens": 30,
        }))
        fake = SimpleNamespace(_acompletion=complete)
        with patch.object(experiment, "create_agent", return_value=fake):
            agent = experiment.agent_for("gpt-56-sol", self.manifest["models"]["gpt-56-sol"])
        asyncio.run(agent._acompletion(model="openai/gpt-5.6-sol"))
        self.assertEqual(complete.await_args.kwargs["max_retries"], 0)
        self.assertEqual(complete.await_args.kwargs["num_retries"], 0)
        self.assertEqual(complete.await_args.kwargs["service_tier"], "default")
        self.assertEqual(agent.ranking_receipts[0]["usage"]["completion_tokens"], 30)

    def test_allowance_violation_stops_after_first_response(self):
        task = self.tasks[0]
        profile = self.manifest["models"]["gpt-54-nano"]
        class FakeAgent:
            usage_log = []
            ranking_receipts = []
            last_completion_outcomes = []
            async def async_completions(self, *args, **kwargs):
                raw = json.dumps({"ranking": list("ABCDEFG")})
                self.ranking_receipts.append({"usage": {"prompt_tokens": 1000000, "completion_tokens": 30}})
                self.last_completion_outcomes = [{"status": "completed", "raw_response": raw}]
                return [raw]
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            summary = asyncio.run(experiment.run_model(
                "gpt-54-nano", self.tasks, profile, Path(temp), 3, factory=lambda *_: FakeAgent(),
            ))
            self.assertEqual(summary["n_started"], 1)
            self.assertEqual(summary["n_not_requested"], 99)


if __name__ == "__main__":
    unittest.main()
