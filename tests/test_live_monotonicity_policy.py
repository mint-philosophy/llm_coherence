from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

import httpx

from llm_coherence.experiments.ladder_statement_pair.experiment_runner_tradeoff import (
    run_single_comparison,
)
from llm_coherence.runtime.agents import LiteLLMAgent


def _completion(content: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None, _hidden_params={})


class LiveMonotonicityRetryPolicyTests(unittest.TestCase):
    def _agent(self) -> LiteLLMAgent:
        return LiteLLMAgent(
            model="openrouter/qwen/qwen3.7-max-20260520",
            max_tokens=16,
            max_retries=3,
            base_delay=0,
            use_jitter=False,
            retry_transport_only=True,
        )

    def test_transport_failure_is_retried(self) -> None:
        agent = self._agent()
        calls = 0

        async def fake_completion(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("connection failed")
            return _completion("A")

        agent._acompletion = fake_completion
        result = asyncio.run(agent.async_completions([[{"role": "user", "content": "x"}]], verbose=False))

        self.assertEqual(result, ["A"])
        self.assertEqual(calls, 2)
        self.assertEqual(agent.retry_counts["transport_retries"], 1)
        self.assertEqual(agent.last_completion_outcomes[0]["status"], "completed")

    def test_transport_failure_honors_ten_attempt_limit(self) -> None:
        agent = LiteLLMAgent(
            model="openrouter/qwen/qwen3.7-max-20260520",
            max_tokens=16,
            max_retries=10,
            base_delay=0,
            max_delay=0,
            use_jitter=False,
            retry_transport_only=True,
        )
        calls = 0

        async def fake_completion(**_kwargs):
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("connection failed")

        agent._acompletion = fake_completion
        result = asyncio.run(
            agent.async_completions(
                [[{"role": "user", "content": "x"}]],
                verbose=False,
            )
        )

        self.assertEqual(result, [None])
        self.assertEqual(calls, 10)
        self.assertEqual(agent.retry_counts["transport_retries"], 9)
        self.assertEqual(agent.retry_counts["transport_failures"], 1)

    def test_non_transport_error_is_not_retried(self) -> None:
        agent = self._agent()
        calls = 0

        async def fake_completion(**_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("provider rejected request")

        agent._acompletion = fake_completion
        result = asyncio.run(agent.async_completions([[{"role": "user", "content": "x"}]], verbose=False))

        self.assertEqual(result, [None])
        self.assertEqual(calls, 1)
        self.assertEqual(agent.last_completion_outcomes[0]["status"], "provider_error")

    def test_token_cap_is_retained_as_missing_without_retry(self) -> None:
        agent = self._agent()
        calls = 0

        async def fake_completion(**_kwargs):
            nonlocal calls
            calls += 1
            return _completion("Answer: A", finish_reason="length")

        agent._acompletion = fake_completion
        result = asyncio.run(agent.async_completions([[{"role": "user", "content": "x"}]], verbose=False))

        self.assertEqual(result, [None])
        self.assertEqual(calls, 1)
        outcome = agent.last_completion_outcomes[0]
        self.assertEqual(outcome["status"], "token_capped")
        self.assertEqual(outcome["raw_response"], "Answer: A")

    def test_comparison_discloses_capped_trial_and_uses_parseable_denominator(self) -> None:
        class StubAgent:
            accepts_system_message = True
            uses_logits = False
            base_timeout = 120.0

            def __init__(self):
                self.calls = 0
                self.last_completion_outcomes = []
                self.timeouts = []

            async def async_completions(self, _messages, **kwargs):
                self.calls += 1
                self.timeouts.append(kwargs.get("timeout"))
                if self.calls == 1:
                    self.last_completion_outcomes = [{
                        "status": "token_capped",
                        "finish_reason": "length",
                        "raw_response": "Answer: A",
                        "error": None,
                        "attempts": 1,
                        "transport_retries": 0,
                    }]
                    return [None]
                self.last_completion_outcomes = [{
                    "status": "completed",
                    "finish_reason": "stop",
                    "raw_response": "B",
                    "error": None,
                    "attempts": 1,
                    "transport_retries": 0,
                }]
                return ["B"]

        comparison = {
            "outcome_a": {"text": "better", "tier": 1},
            "outcome_b": {"text": "worse", "tier": 2},
        }
        agent = StubAgent()
        result = asyncio.run(run_single_comparison(
            agent,
            comparison,
            num_trials=1,
            include_flipped=True,
            system_message="system",
            with_reasoning=False,
            verbose=False,
        ))

        self.assertEqual(result["expected_trials"], 2)
        self.assertEqual(result["parseable_trials"], 1)
        self.assertEqual(result["missing_trials"], 1)
        self.assertEqual(result["missing_by_reason"], {"token_capped": 1})
        self.assertEqual(agent.timeouts, [120.0, 120.0])
        missing = result["missing_responses"][0]
        self.assertEqual(missing["raw_response"], "Answer: A")
        self.assertEqual(missing["custom_id"], "c0000-dab-t000")
        self.assertEqual(missing["response_status"], "token_capped")
        self.assertEqual(missing["schema_version"], "1.0")
        self.assertEqual(
            set(missing),
            {
                "schema_version",
                "direction",
                "trial_index",
                "custom_id",
                "reason",
                "finish_reason",
                "response_status",
                "raw_response",
                "error",
                "attempts",
                "transport_retries",
            },
        )
        self.assertEqual(result["prob_prefer_a"], 1.0)
        self.assertEqual(
            result["prob_prefer_a_bounds"],
            {"lower_missing_prefer_b": 0.5, "upper_missing_prefer_a": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
