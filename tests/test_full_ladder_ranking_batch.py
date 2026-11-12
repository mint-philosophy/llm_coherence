import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import httpx

from llm_coherence.experiments import full_ladder_ranking as original
from llm_coherence.experiments import full_ladder_ranking_batch as batch
from llm_coherence.experiments import full_ladder_ranking_recovery as recovery

MODELS = ["gpt-56-luna-thinking", "opus-46-thinking"]
KEYS = {original.API_KEY_ENV_BY_TYPE["openai"]: "test-openai",
        original.API_KEY_ENV_BY_TYPE["anthropic"]: "test-anthropic"}


def no_calls(request):
    raise AssertionError(f"unexpected HTTP call: {request.method} {request.url}")


class FakeProviders:
    """Just enough of the OpenAI and Anthropic batch endpoints for offline tests."""

    def __init__(self, openai_lines=None, anthropic_lines=None, ended=True, fail_create=False, openai_batch=None):
        self.calls, self.uploads, self.anthropic_payloads = [], [], []
        self.openai_lines, self.anthropic_lines, self.ended = openai_lines, anthropic_lines, ended
        self.fail_create, self.openai_batch = fail_create, openai_batch

    def __call__(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        if request.method == "POST" and path == "/v1/files":
            self.uploads.append(request.content)
            return httpx.Response(200, json={"id": f"file-{len(self.uploads)}"})
        if request.method == "POST" and path == "/v1/batches":
            if self.fail_create:  # e.g. the provider accepted it but the reply was a gateway error
                return httpx.Response(502, text="<html>bad gateway</html>")
            return httpx.Response(200, json={"id": "batch_o1", "status": "validating"})
        if request.method == "POST" and path == "/v1/messages/batches":
            self.anthropic_payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "msgbatch_1", "processing_status": "in_progress"})
        if request.method == "GET" and path == "/v1/batches/batch_o1":
            if self.openai_batch is not None:
                return httpx.Response(200, json=self.openai_batch)
            return httpx.Response(200, json={"id": "batch_o1", "status": "completed" if self.ended else "in_progress",
                                             "output_file_id": "file-out", "error_file_id": None})
        if request.method == "GET" and path == "/v1/files/file-out/content":
            return httpx.Response(200, text="".join(json.dumps(x) + "\n" for x in self.openai_lines))
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_1":
            return httpx.Response(200, json={
                "id": "msgbatch_1", "processing_status": "ended" if self.ended else "in_progress",
                "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"})
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_1/results":
            return httpx.Response(200, text="".join(json.dumps(x) + "\n" for x in self.anthropic_lines))
        raise AssertionError(f"unhandled {request.method} {path}")


class BatchRankingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"
        self.env = patch.dict(os.environ, KEYS)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def prepared(self):
        manifest = batch.prepare(self.out, MODELS, limit=2)
        requests = json.loads((self.out / "requests.json").read_text())
        return manifest, requests

    def test_rejects_openrouter_models(self):
        with self.assertRaisesRegex(ValueError, "no batch API"):
            batch.prepare(self.out, ["llama-31-8b-instruct-openrouter"], limit=2)
        self.assertFalse(self.out.exists())

    def test_requests_match_live_prompt_and_native_bodies(self):
        manifest, requests = self.prepared()
        tasks, _, live = original.prepare(original.DEFAULT_INPUT, MODELS, limit=2)
        self.assertEqual(manifest["source_manifest"], live)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len({r["custom_id"] for r in requests}), 4)
        by_ladder = {t.ladder_id: t for t in tasks}
        for r in requests:
            # Anthropic custom_id limit; OpenAI accepts any string
            self.assertRegex(r["custom_id"], re.compile(r"^[A-Za-z0-9_-]{1,64}$"))
            messages = original.messages_for(by_ladder[r["ladder_id"]])
            _, body = recovery.native_request(live["models"][r["model_key"]], messages)
            body.pop("service_tier", None)
            self.assertEqual(r["body"], body)
        lines = [json.loads(x) for x in (self.out / "gpt-56-luna-thinking" / "batch_input.jsonl").read_text().splitlines()]
        self.assertEqual({x["url"] for x in lines}, {"/v1/responses"})
        self.assertEqual({x["method"] for x in lines}, {"POST"})
        self.assertNotIn("service_tier", lines[0]["body"])

    def test_prepare_needs_empty_directory(self):
        self.out.mkdir(parents=True)
        (self.out / "stray.txt").write_text("x")
        with self.assertRaisesRegex(ValueError, "empty"):
            batch.prepare(self.out, MODELS, limit=2)

    def test_submit_refuses_low_budget_before_any_call(self):
        manifest, _ = self.prepared()
        with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
            with self.assertRaisesRegex(ValueError, "reservation"):
                batch.submit(self.out, manifest["reserved_cost_usd"] / 2, client)
        self.assertFalse((self.out / "batches.json").exists())

    def test_submit_one_batch_per_model_and_never_twice(self):
        manifest, requests = self.prepared()
        fake = FakeProviders()
        with httpx.Client(transport=httpx.MockTransport(fake)) as client:
            batch.submit(self.out, manifest["reserved_cost_usd"], client)
        saved = json.loads((self.out / "batches.json").read_text())
        self.assertEqual(saved["gpt-56-luna-thinking"]["batch_id"], "batch_o1")
        self.assertEqual(saved["opus-46-thinking"]["batch_id"], "msgbatch_1")
        uploaded = (self.out / "gpt-56-luna-thinking" / "batch_input.jsonl").read_bytes()
        self.assertIn(uploaded, fake.uploads[0])
        sent = fake.anthropic_payloads[0]["requests"]
        self.assertEqual([x["custom_id"] for x in sent],
                         [r["custom_id"] for r in requests if r["model_key"] == "opus-46-thinking"])
        with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
            batch.submit(self.out, manifest["reserved_cost_usd"], client)  # already submitted: no calls

    def test_unconfirmed_create_blocks_a_blind_retry(self):
        manifest, _ = self.prepared()
        with httpx.Client(transport=httpx.MockTransport(FakeProviders(fail_create=True))) as client:
            with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
                batch.submit(self.out, manifest["reserved_cost_usd"], client)
        saved = json.loads((self.out / "batches.json").read_text())
        self.assertIn("create_attempted_at", saved["gpt-56-luna-thinking"])
        self.assertNotIn("batch_id", saved["gpt-56-luna-thinking"])
        with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
            with self.assertRaisesRegex(ValueError, "may have reached the provider"):
                batch.submit(self.out, manifest["reserved_cost_usd"], client)

    def test_submit_refuses_a_second_runner(self):
        import fcntl
        manifest, _ = self.prepared()
        with (self.out / ".run.lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
                with self.assertRaisesRegex(RuntimeError, "active runner"):
                    batch.submit(self.out, manifest["reserved_cost_usd"], client)

    def submitted_with_results(self, ended=True):
        manifest, requests = self.prepared()
        tasks = {t.ladder_id: t for t in original.prepare(original.DEFAULT_INPUT, MODELS, limit=2)[0]}
        luna = [r for r in requests if r["model_key"] == "gpt-56-luna-thinking"]
        opus = [r for r in requests if r["model_key"] == "opus-46-thinking"]
        # intended order: labels sorted by the tier of the statement they show (T1 first)
        good = lambda r: json.dumps({"ranking": [label for _, label in sorted(
            zip(tasks[r["ladder_id"]].shuffle_indices, tasks[r["ladder_id"]].labels))]})
        openai_lines = [
            {"custom_id": luna[0]["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                "id": "resp_1", "model": "gpt-5.6-luna", "status": "completed",
                "usage": {"input_tokens": 300, "output_tokens": 80},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": good(luna[0])}]}]}}},
            {"custom_id": luna[1]["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                "id": "resp_2", "model": "gpt-5.6-luna", "status": "completed",
                "usage": {"input_tokens": 300, "output_tokens": 5},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "I would rather not."}]}]}}},
        ]
        anthropic_lines = [
            {"custom_id": opus[0]["custom_id"], "result": {"type": "succeeded", "message": {
                "id": "msg_1", "model": "claude-opus-4-6", "stop_reason": "end_turn",
                "usage": {"input_tokens": 300, "output_tokens": 600},
                "content": [{"type": "thinking", "thinking": "a different order"},
                            {"type": "text", "text": good(opus[0])}]}}},
            {"custom_id": opus[1]["custom_id"], "result": {"type": "expired"}},
        ]
        fake = FakeProviders(openai_lines, anthropic_lines, ended)
        with httpx.Client(transport=httpx.MockTransport(fake)) as client:
            batch.submit(self.out, manifest["reserved_cost_usd"], client)
            summaries = batch.fetch(self.out, client)
        return summaries, tasks, fake

    def test_fetch_writes_live_format_events(self):
        summaries, tasks, _ = self.submitted_with_results()
        by_model = {s["model_key"]: s for s in summaries}
        self.assertEqual(by_model["gpt-56-luna-thinking"]["n_valid_rankings"], 1)
        self.assertEqual(by_model["gpt-56-luna-thinking"]["n_invalid_or_failed"], 1)
        self.assertEqual(by_model["gpt-56-luna-thinking"]["exact_match_count"], 1)
        self.assertEqual(by_model["opus-46-thinking"]["n_valid_rankings"], 1)
        self.assertEqual(by_model["opus-46-thinking"]["n_invalid_or_failed"], 1)
        # the live runner's reader and summarizer accept the converted log unchanged
        started, results = original.read_events(self.out / "opus-46-thinking" / "responses.jsonl", set(tasks))
        self.assertEqual(original.summarize("opus-46-thinking", list(tasks.values()), started, results),
                         by_model["opus-46-thinking"])
        statuses = sorted(r["outcome"]["status"] for r in results.values())
        self.assertEqual(statuses, ["completed", "expired"])
        self.assertTrue((self.out / "opus-46-thinking" / "batch_results.raw.jsonl").exists())

    def test_fetch_waits_for_unfinished_batches(self):
        summaries, _, _ = self.submitted_with_results(ended=False)
        self.assertEqual(summaries, [])
        self.assertFalse((self.out / "gpt-56-luna-thinking" / "responses.jsonl").exists())

    def test_bad_result_line_leaves_no_partial_log(self):
        real = batch.result_row
        seen = []

        def flaky(*args):
            seen.append(1)
            if len(seen) == 2:
                raise KeyError("result")
            return real(*args)

        with patch.object(batch, "result_row", flaky):
            with self.assertRaises(KeyError):
                self.submitted_with_results()
        self.assertFalse((self.out / "gpt-56-luna-thinking" / "responses.jsonl").exists())

    def test_fetch_refuses_changed_protocol(self):
        self.submitted_with_results(ended=False)
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["source_manifest"]["input_sha256"] = "0" * 64
        original.write_json(self.out / "manifest.json", manifest)
        with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                batch.fetch(self.out, client)

    def test_failed_batch_keeps_status_and_writes_no_log(self):
        manifest, _ = self.prepared()
        failed = {"id": "batch_o1", "status": "failed", "output_file_id": None, "error_file_id": None,
                  "errors": {"data": [{"code": "invalid_request", "message": "bad line"}]}}
        fake = FakeProviders(anthropic_lines=[], ended=False, openai_batch=failed)
        with httpx.Client(transport=httpx.MockTransport(fake)) as client:
            batch.submit(self.out, manifest["reserved_cost_usd"], client)
            self.assertEqual(batch.fetch(self.out, client), [])
        status = json.loads((self.out / "gpt-56-luna-thinking" / "batch_status.json").read_text())
        self.assertEqual(status["errors"]["data"][0]["code"], "invalid_request")
        self.assertFalse((self.out / "gpt-56-luna-thinking" / "responses.jsonl").exists())

    def test_fetch_never_overwrites_saved_results(self):
        self.submitted_with_results()
        before = (self.out / "gpt-56-luna-thinking" / "responses.jsonl").read_bytes()
        with httpx.Client(transport=httpx.MockTransport(no_calls)) as client:
            batch.fetch(self.out, client)
        self.assertEqual((self.out / "gpt-56-luna-thinking" / "responses.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
