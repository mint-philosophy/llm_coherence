import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from llm_coherence.experiments import full_ladder_ranking as original
from llm_coherence.experiments import full_ladder_ranking_recovery as recovery


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.requests, cls.manifest = original.prepare(original.DEFAULT_INPUT, list(original.PAPER_MODELS))

    def test_native_openai_preserves_reasoning_and_omits_incompatible_fields(self):
        for key in ["gpt-56-luna-thinking", "gpt-56-terra-thinking", "gpt-56-sol-thinking"]:
            profile = self.manifest["models"][key]
            url, body = recovery.native_request(profile, self.requests[0]["messages"])
            self.assertEqual(url, "https://api.openai.com/v1/responses")
            self.assertEqual(body["input"], self.requests[0]["messages"])
            self.assertEqual(body["reasoning"], {"effort": "high", "summary": "auto"})
            self.assertEqual(body["max_output_tokens"], 3000)
            for name in ["extra_body", "reasoning_effort", "temperature", "max_tokens", "messages"]:
                self.assertNotIn(name, body)

    def test_anthropic_thinking_is_top_level(self):
        profile = self.manifest["models"]["opus-46-thinking"]
        url, body = recovery.native_request(profile, self.requests[0]["messages"])
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(body["max_tokens"], 2048)
        self.assertEqual(body["temperature"], 1)
        self.assertNotIn("extra_body", body)
        self.assertEqual(body["messages"], self.requests[0]["messages"][1:])

    def test_replay_only_explicit_api_rejections(self):
        def row(status, error="", raw="", receipts=None):
            return dict(outcome=dict(status=status, error=error), raw_response=raw, receipts=receipts or [])
        self.assertTrue(recovery.eligible_retry(row("provider_error", "Unknown parameter: 'reasoning'.")))
        self.assertTrue(recovery.eligible_retry(row("provider_error", "extra_body: Extra inputs are not permitted")))
        self.assertTrue(recovery.eligible_retry(row("transport_failure", "RateLimitError 429")))
        for item in [None, row("completed", raw="I cannot rank these."), row("completed", raw='[A,B,B]'),
                     row("token_capped"), row("transport_failure", "TimeoutError"), row("provider_error", "401"),
                     row("transport_failure", "RateLimitError 429", receipts=[{"usage": {}}])]:
            self.assertFalse(recovery.eligible_retry(item))

    def test_parse_only_visible_text_and_preserve_token_caps(self):
        text = '{"ranking":["A","B","C","D","E","F","G"]}'
        body = {"status": "completed", "usage": {"input_tokens": 100, "output_tokens": 400}, "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "a different order"}]},
            {"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        raw, status, _, tokens = recovery.response_fields("openai", body)
        self.assertEqual(raw, text); self.assertEqual(status, "completed"); self.assertEqual(tokens, (100, 400))
        body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        self.assertEqual(recovery.response_fields("openai", body)[1], "token_capped")

    def test_unresolved_request_remains_started_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "responses.jsonl"
            event = json.dumps({"event": "started", "request_id": "one"}) + "\n"
            path.write_text(event)
            started, results = recovery.read_log(path, {"one"})
            self.assertEqual(started, {"one"}); self.assertFalse(results)
            path.write_text(event + event)
            with self.assertRaises(ValueError):
                recovery.read_log(path, {"one"})

    def test_native_workers_issue_once_and_resume_does_not_replay(self):
        model = "gpt-56-luna-thinking"
        profile = self.manifest["models"][model]
        planned = []
        for task, request in zip(self.tasks[:6], self.requests[:6]):
            url, body = recovery.native_request(profile, request["messages"])
            planned.append(dict(request_id=task.ladder_id, model_key=model, ladder_id=task.ladder_id,
                                url=url, body=body, reserved_cost_usd=1, input_token_allowance=1000))
        calls = []
        class Response:
            status_code = 200
            def json(self):
                return {"status": "completed", "usage": {"input_tokens": 200, "output_tokens": 30},
                        "output": [{"type": "message", "content": [{"type": "output_text", "text":
                                   '{"ranking":["A","B","C","D","E","F","G"]}'}]}]}
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, headers, json):
                calls.append(json)
                return Response()
        loaded = ({"combined_reservation_usd": 1, "requests_per_model": {model: 6}},
                  planned, {"models": {model: profile}}, self.tasks[:6])
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(recovery, "load", return_value=loaded), \
             patch.object(recovery, "analyze", return_value=[]), \
             patch.object(original, "load_key_directory"), \
             patch.object(recovery.httpx, "AsyncClient", return_value=Client()), \
             patch.object(recovery.asyncio, "sleep", new_callable=AsyncMock), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
            root = Path(temp)
            asyncio.run(recovery.run(root, root, 40))
            self.assertEqual(len(calls), 6)
            self.assertEqual(len({original.digest(c) for c in calls}), 6)
            asyncio.run(recovery.run(root, root, 40))
            self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main()
