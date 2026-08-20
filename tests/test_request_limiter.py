"""Offline tests for run-wide live-request controls."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment import (
    run_phase6b,
)
from llm_coherence.runtime.agents import AsyncRequestLimiter, LiteLLMAgent


def _completion_response(letter: str = "A") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=letter,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_details=None,
                ),
            )
        ],
        usage=None,
        _hidden_params={},
    )


class AsyncRequestLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_limiter_caps_multiple_agents_together(self) -> None:
        limiter = AsyncRequestLimiter(max_concurrency=2)
        agents = [
            LiteLLMAgent(
                model="openrouter/qwen/qwen3.7-flash",
                max_tokens=16,
                concurrency_limit=10,
                request_limiter=limiter,
            )
            for _ in range(2)
        ]

        async def fake_completion(**_kwargs):
            await asyncio.sleep(0.01)
            return _completion_response()

        for agent in agents:
            agent._acompletion = fake_completion

        messages = [[{"role": "user", "content": "Choose A or B."}]] * 5
        first, second = await asyncio.gather(
            agents[0].async_completions(messages, verbose=False),
            agents[1].async_completions(messages, verbose=False),
        )

        self.assertEqual(first, ["A"] * 5)
        self.assertEqual(second, ["A"] * 5)
        stats = limiter.snapshot()
        self.assertEqual(stats["attempts_started"], 10)
        self.assertEqual(stats["attempts_completed"], 10)
        self.assertEqual(stats["peak_in_flight"], 2)
        self.assertEqual(stats["in_flight"], 0)

    async def test_request_starts_are_evenly_spaced(self) -> None:
        limiter = AsyncRequestLimiter(
            max_concurrency=4,
            requests_per_second=25,
        )
        starts: list[float] = []

        async def request() -> None:
            async with limiter.slot():
                starts.append(asyncio.get_running_loop().time())

        await asyncio.gather(*(request() for _ in range(4)))

        gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
        self.assertEqual(len(gaps), 3)
        self.assertTrue(all(gap >= 0.03 for gap in gaps), gaps)
        stats = limiter.snapshot()
        self.assertEqual(stats["attempts_started"], 4)
        self.assertGreater(stats["pacing_wait_seconds"], 0)
        self.assertIsNotNone(stats["observed_start_rate"])

    async def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            AsyncRequestLimiter(0)
        with self.assertRaisesRegex(ValueError, "requests_per_second"):
            AsyncRequestLimiter(1, requests_per_second=0)

    async def test_phase_runner_shares_one_limiter_across_ladders(self) -> None:
        module = (
            "llm_coherence.experiments.ladder_statement_pair."
            "run_7tier_experiment"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            names = [
                f"phase6b_variations_pruned_final_Test_{index}_comparisons.json"
                for index in (1, 2)
            ]
            for name in names:
                (data_dir / name).write_text(
                    json.dumps({"comparisons": []}),
                    encoding="utf-8",
                )
            manifest_path = data_dir / "phase6b_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "variation_files": names,
                        "n_tiers": 7,
                        "n_comparison_samples": 30,
                    }
                ),
                encoding="utf-8",
            )

            run_single_mock = AsyncMock(return_value={})
            budget = SimpleNamespace(
                should_stop=False,
                last_usage=None,
                force_check=AsyncMock(),
                on_task_completed=AsyncMock(),
                summary=lambda: "test budget",
            )
            with (
                patch(f"{module}.run_single", run_single_mock),
                patch(f"{module}.BudgetMonitor", return_value=budget),
                patch(f"{module}._write_cost_logs", return_value=None),
                patch(f"{module}.print_phase6b_live_cost_estimate"),
                redirect_stdout(io.StringIO()),
            ):
                await run_phase6b(
                    model_key="qwen-37-flash-openrouter",
                    num_trials=10,
                    with_reasoning=False,
                    max_tokens=16,
                    data_dir=data_dir,
                    manifest_path=manifest_path,
                    results_dir=root / "results",
                    checkpoints_dir=root / "checkpoints",
                    variation_ids=None,
                    max_concurrent=2,
                    resume=False,
                    verbose=False,
                    infrastructure="openrouter",
                    skip_smoke_test=True,
                    request_concurrency=4,
                    requests_per_second=5,
                )

        self.assertEqual(run_single_mock.await_count, 2)
        first_limiter = run_single_mock.await_args_list[0].kwargs["request_limiter"]
        second_limiter = run_single_mock.await_args_list[1].kwargs["request_limiter"]
        self.assertIs(first_limiter, second_limiter)
        self.assertEqual(first_limiter.max_concurrency, 4)
        self.assertEqual(first_limiter.requests_per_second, 5)


if __name__ == "__main__":
    unittest.main()
