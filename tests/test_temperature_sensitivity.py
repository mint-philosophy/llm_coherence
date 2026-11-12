import asyncio
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import httpx

from llm_coherence.experiments import temperature_sensitivity as experiment


class TemperaturePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ladders = json.loads(experiment.LADDERS.read_text())
        cls.references = json.loads(experiment.COMPARISONS.read_text())["comparison_sample"]

    def test_balanced_paired_design_and_fixed_parameters(self):
        manifest, prompts, tasks = experiment.build_plan(self.ladders, self.references)
        self.assertEqual(manifest["n_requests"], 20160)
        self.assertEqual(manifest["n_ladders"], 12)
        self.assertEqual(manifest["n_curves_per_temperature"], 36)
        self.assertLess(manifest["reserved_cost_usd"], 5)
        self.assertEqual(Counter(t["temperature"] for t in tasks), dict.fromkeys([0.0, .3, .7, 1.0], 5040))
        groups = defaultdict(list)
        for task in tasks:
            key = (task["ladder_id"], task["comparison_id"], task["tier"], task["order"], task["repeat"])
            groups[key].append(task)
        for group in groups.values():
            bodies = [experiment.request_body(task, prompts) for task in group]
            self.assertEqual(len(bodies), 4)
            for body in bodies:
                body.pop("temperature")
            self.assertTrue(all(body == bodies[0] for body in bodies))
        self.assertEqual(experiment.build_plan(self.ladders, self.references), (manifest, prompts, tasks))

    def test_parser_excludes_caps_and_ambiguous_answers(self):
        payload = lambda text, finish="stop": {"choices": [{"finish_reason": finish, "message": {"content": text}}]}
        self.assertEqual(experiment.parse_reply(payload("B")), "B")
        self.assertIsNone(experiment.parse_reply(payload("B", "length")))
        self.assertIsNone(experiment.parse_reply(payload("A or B")))

    def test_paired_analysis_detects_known_reversal_and_excludes_missing_curve(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            experiment.prepare(root)
            manifest, _, tasks = experiment.load_plan(root)
            keep_ladder = manifest["ladder_ids"][0]
            keep_reference = manifest["reference_ids"][0]
            chosen = [t for t in tasks if t["ladder_id"] == keep_ladder and t["comparison_id"] == keep_reference]
            counts = defaultdict(int)
            with (root / "responses.jsonl").open("w") as out:
                for task in chosen:
                    cell = (task["temperature"], task["tier"])
                    k = counts[cell]
                    counts[cell] += 1
                    wins = [2, 4, 6, 8, 10, 12, 14]
                    if task["temperature"] == 1.0:
                        wins[4] = 6  # .40 -> .30 is an intentional reversal.
                    tier_wins = k < wins[task["tier"] - 1]
                    answer = ("A" if task["order"] == "AB" else "B")
                    if not tier_wins:
                        answer = "B" if answer == "A" else "A"
                    out.write(json.dumps({"event": "started", "request_id": task["request_id"]}) + "\n")
                    out.write(json.dumps({"event": "result", "request_id": task["request_id"], "answer": answer}) + "\n")
            summaries = experiment.analyze(root)
            self.assertTrue(all(s["common_full_response_curves"] == 1 for s in summaries))
            self.assertEqual([s["monotonicity_percent"] for s in summaries], [100, 100, 100, 0])
            self.assertAlmostEqual(summaries[-1]["mean_largest_decrease"], .1)
            self.assertEqual(summaries[-1]["change_from_temperature_0_pp"], -100)
            lines = (root / "responses.jsonl").read_text().splitlines()
            (root / "responses.jsonl").write_text("\n".join(lines[:-1]) + "\n")
            summaries = experiment.analyze(root)
            self.assertTrue(all(s["common_full_response_curves"] == 0 for s in summaries))
            self.assertTrue(all(s["monotonicity_percent"] is None for s in summaries))

    def test_modified_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            experiment.prepare(root)
            tasks = json.loads((root / "tasks.json").read_text())
            tasks[0]["temperature"] = .99
            (root / "tasks.json").write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, "request files have changed"):
                experiment.load_plan(root)

    def test_live_path_budget_pin_temperature_and_no_duplicate_requests_offline(self):
        manifest, prompts, tasks = experiment.build_plan(self.ladders, self.references)
        tasks = tasks[:4]  # One paired trial at each temperature.
        manifest["tasks_sha256"] = experiment.sha(experiment.encoded(tasks))
        manifest["n_requests"] = len(tasks)
        manifest["reserved_cost_usd"] = sum(
            prompts[t["prompt_id"]]["token_allowance"] * experiment.PRICE_PER_TOKEN for t in tasks
        )
        posted = []

        def respond(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": {"endpoints": [{
                    "tag": experiment.ENDPOINT, "supported_parameters": ["temperature"],
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000001"},
                }]}})
            body = json.loads(request.content)
            posted.append(body)
            self.assertEqual(body["provider"]["only"], [experiment.ENDPOINT])
            self.assertFalse(body["provider"]["allow_fallbacks"])
            return httpx.Response(200, json={
                "provider": "Mistral", "usage": {"prompt_tokens": 20, "completion_tokens": 1, "cost": .0000021},
                "choices": [{"finish_reason": "stop", "message": {"content": "A"}}],
            })

        client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            for name, value in [("manifest", manifest), ("prompts", prompts), ("tasks", tasks)]:
                (root / f"{name}.json").write_bytes(experiment.encoded(value))
            with patch.object(experiment, "require_api_key", return_value="test-only-key"), \
                 patch.object(httpx, "AsyncClient", side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs)):
                with self.assertRaises(ValueError):
                    asyncio.run(experiment.run(root, max_cost_usd=0.0000001, rate=10000))
                self.assertEqual(len(posted), 0)
                asyncio.run(experiment.run(root, max_cost_usd=5, rate=10000))
                self.assertEqual(len(posted), 4)
                self.assertEqual({body["temperature"] for body in posted}, {0.0, .3, .7, 1.0})
                asyncio.run(experiment.run(root, max_cost_usd=5, rate=10000))
                self.assertEqual(len(posted), 4)

    def test_unrun_analysis_reports_no_score(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            experiment.prepare(root)
            rows = experiment.analyze(root)
            self.assertTrue(all(row["parsed"] == 0 for row in rows))
            self.assertTrue(all(row["monotonicity_percent"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
