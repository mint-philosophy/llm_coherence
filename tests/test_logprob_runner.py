"""Offline checks of self-hosted engine ownership, evidence and accounting."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from llm_coherence.experiments.ladder_statement_pair import experiment_runner_tradeoff as runner
from llm_coherence.experiments.ladder_statement_pair import run_7tier_experiment as phase6b
from llm_coherence.runtime.agents import VLLMLogprobAgent, model_name_for_key


MODEL_KEY = "glm-45-base-logprobs"
COMPARISON = {
    "outcome_a": {"text": "A child’s wellbeing improves."},
    "outcome_b": {"text": "The child receives no additional support."},
}


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = []

    def generate(self, prompts, sampling_params, **kwargs):
        outputs = []
        for prompt in prompts:
            # Original P(A)=0.8; flipped P(B)=0.7. Their average must be 0.75.
            p_a = 0.8 if len(self.prompts) % 2 == 0 else 0.3
            self.prompts.append(prompt)
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                101: SimpleNamespace(logprob=math.log(p_a)),
                102: SimpleNamespace(logprob=math.log(1 - p_a)),
            }])]))
        return outputs


class LogprobEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.factory = Mock(side_effect=FakeEngine)
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [{" A": 101, " B": 102}[text]])
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 8)),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))),
            "vllm": SimpleNamespace(LLM=self.factory, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        self.owner = VLLMLogprobAgent(model_name_for_key(MODEL_KEY), temperature=0.0)
        # Keep the test entirely offline and avoid collecting host-specific metadata.
        self.enterContext(patch.object(runner, "_package_versions", return_value={}))
        self.enterContext(patch.object(runner, "_host_info", return_value={}))
        self.enterContext(patch.object(runner, "_actual_cost", return_value=None))
        self.enterContext(patch.object(runner, "_estimate_cost", return_value=None))

    async def compare(self, agent, include_flipped=True):
        return await runner.run_single_comparison(
            agent, COMPARISON, num_trials=10, include_flipped=include_flipped,
            system_message="unused by base model", with_reasoning=False, verbose=False,
        )

    async def experiment(self, name="test", **kwargs):
        path = self.root / f"{name}_comparisons.json"
        path.write_text(json.dumps({"comparisons": [COMPARISON]}), encoding="utf-8")
        return await runner.run_experiment(
            name, MODEL_KEY, num_trials=10, include_flipped=True, verbose=False,
            data_dir=self.root, comparison_path=path, results_dir=self.root / "results",
            checkpoints_dir=self.root / "checkpoints", shared_logprob_agent=self.owner,
            **kwargs,
        )

    async def test_one_lazy_engine_and_independent_ladder_telemetry(self):
        first, second = self.owner.new_session(), self.owner.new_session()
        self.factory.assert_not_called()
        runner._restore_agent_telemetry(first, {
            "usage_log": [{"prompt_tokens": 7}], "retry_counts": {"errors": 3},
        })
        first.reasoning_log.append({"reasoning": "first ladder only"})
        result1, result2 = await self.compare(first), await self.compare(second)
        self.factory.assert_called_once()
        self.assertIs(first._llm, second._llm)
        self.assertEqual(first._llm.kwargs["tensor_parallel_size"], 8)
        self.assertEqual(first._sampling_params.allowed_token_ids, [101, 102])
        self.assertEqual(first._sampling_params.temperature, 0.0)
        self.assertEqual(len(first._llm.prompts), 4)
        self.assertEqual(result1["prob_prefer_a"], 0.75)
        self.assertEqual(result1, result2)
        self.assertEqual(first.retry_counts["errors"], 3)
        for untouched in (second, self.owner):
            self.assertEqual(untouched.retry_counts["errors"], 0)
            self.assertEqual(untouched.usage_log, [])
            self.assertEqual(untouched.reasoning_log, [])

    async def test_both_order_scores_are_retained_without_changing_estimator(self):
        result = await self.compare(self.owner.new_session())
        self.assertAlmostEqual(result["order_probabilities"]["original"]["A"], 0.8)
        self.assertAlmostEqual(result["order_probabilities"]["flipped"]["B"], 0.7)
        self.assertEqual(result["prob_prefer_a"], 0.75)
        self.assertEqual(result["count_prefer_a"], round(0.75 * 10))
        self.assertEqual(result["count_kind"], "probability_derived_pseudocounts")
        self.assertEqual(result["scoring_calls"], 2)
        evidence = result["logprob_evidence"]
        self.assertEqual(evidence["original"][0]["token_ids"], {"A": 101, "B": 102})
        self.assertAlmostEqual(evidence["original"][0]["logprobs"]["A"], math.log(0.8))
        self.assertAlmostEqual(evidence["flipped"][0]["logprobs"]["B"], math.log(0.7))

    async def test_unflipped_score_has_one_call_and_no_flipped_evidence(self):
        result = await self.compare(self.owner.new_session(), include_flipped=False)
        self.assertEqual(result["prob_prefer_a"], 0.8)
        self.assertEqual(result["scoring_calls"], 1)
        self.assertIsNone(result["order_probabilities"]["flipped"])
        self.assertEqual(result["logprob_evidence"]["flipped"], [])

    async def test_result_reports_scoring_calls_instead_of_sampled_trials(self):
        result = await self.experiment()
        self.assertEqual(result["config"]["count_kind"], "probability_derived_pseudocounts")
        self.assertEqual(result["config"]["scoring_method"], "logprobs")
        metadata = result["metadata"]
        self.assertEqual(metadata["total_scoring_calls"], 2)
        self.assertEqual(metadata["total_api_calls"], 2)
        self.assertEqual(metadata["sampled_trials"], 0)
        self.assertEqual(metadata["missing_response_count"], 0)
        self.assertEqual(metadata["unparseable_rate"], 0.0)
        self.assertEqual(metadata["run_status"], "complete")

    async def test_shared_engine_rejects_mismatched_model_or_temperature(self):
        for field, value in (("model", "wrong/model"), ("temperature", 1.0)):
            with self.subTest(field=field), patch.object(self.owner, field, value):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    await self.experiment()
        self.factory.assert_not_called()

    async def test_resume_preserves_evidence_without_generating_again(self):
        # Simulate failure after the final comparison checkpoint but before save.
        with patch.object(runner, "save_results", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                await self.experiment()
        checkpoint_path = next((self.root / "checkpoints").glob("ckpt_*.json"))
        checkpoint = json.loads(checkpoint_path.read_text())
        self.assertEqual(checkpoint["run_config"]["logprob_result_protocol"], "per_order_scores_v1")
        result = await self.experiment(resume=True)
        self.assertEqual(result["preferences"], checkpoint["preferences"])
        self.assertEqual(len(self.owner._llm.prompts), 2)

    async def test_legacy_logprob_checkpoint_is_not_mixed_with_new_evidence(self):
        with patch.object(runner, "save_results", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                await self.experiment()
        checkpoint_path = next((self.root / "checkpoints").glob("ckpt_*.json"))
        checkpoint = json.loads(checkpoint_path.read_text())
        del checkpoint["run_config"]["logprob_result_protocol"]
        checkpoint["run_fingerprint"] = runner._run_fingerprint(checkpoint["run_config"])
        checkpoint_path.write_text(json.dumps(checkpoint))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            await self.experiment(resume=True)
        self.assertEqual(len(self.owner._llm.prompts), 2)

    async def test_incomplete_or_invalid_scores_cannot_become_half_probabilities(self):
        bad_scores = [
            {"A": 0.5}, {"B": 0.5}, {"A": None, "B": 0.5},
            {"A": float("nan"), "B": 0.5}, {"A": 1.1, "B": -0.1},
            {"A": 0.4, "B": 0.4},
        ]
        for score in bad_scores:
            with self.subTest(score=score):
                fake = SimpleNamespace(uses_logits=True)
                with patch.object(runner, "generate_responses", new=AsyncMock(return_value={0: [score]})):
                    with self.assertRaises(ValueError):
                        await self.compare(fake)


class Phase6bOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "n_tiers": 7, "n_comparison_samples": 30,
            "variation_files": [f"ladder_{i}_comparisons.json" for i in range(3)],
        }))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.budget_check = self.enterContext(patch.object(phase6b.BudgetMonitor, "force_check", new_callable=AsyncMock))
        self.enterContext(patch.object(phase6b.BudgetMonitor, "on_task_completed", new_callable=AsyncMock))
        self.enterContext(patch.object(phase6b, "print_phase6b_live_cost_estimate"))
        self.smoke_call = self.enterContext(patch.object(phase6b, "smoke_call", new_callable=AsyncMock, return_value=True))
        self.enterContext(patch.object(phase6b, "_write_cost_logs", return_value=None))
        self.enterContext(patch.object(phase6b, "is_complete", return_value=False))

    async def run_phase(self, model_key=MODEL_KEY, **kwargs):
        options = dict(
            model_key=model_key, num_trials=10, with_reasoning=False, max_tokens=10,
            data_dir=self.root, manifest_path=self.manifest, results_dir=self.root / "results",
            checkpoints_dir=self.root / "checkpoints", variation_ids=None,
            max_concurrent=3, resume=False, verbose=False,
        )
        options.update(kwargs)
        await phase6b.run_phase6b(**options)

    async def measure_concurrency(self, model_key):
        active, peak = 0, 0
        owners = []

        async def run_single(*args, **kwargs):
            nonlocal active, peak
            owners.append(kwargs["shared_logprob_agent"])
            active += 1
            peak = max(active, peak)
            await asyncio.sleep(0)
            active -= 1
            return {"metadata": {}}

        with patch.object(phase6b, "run_single", side_effect=run_single):
            await self.run_phase(model_key)
        return peak, owners

    async def test_glm_uses_one_owner_serially_and_never_api_smoke_or_budget(self):
        peak, owners = await self.measure_concurrency(MODEL_KEY)
        self.assertEqual(peak, 1)
        self.assertEqual(len(owners), 3)
        self.assertTrue(all(owner is owners[0] for owner in owners))
        self.assertIsInstance(owners[0], VLLMLogprobAgent)
        self.assertIsNone(owners[0]._llm)  # Creating a run owner is lazy.
        self.smoke_call.assert_not_awaited()
        self.budget_check.assert_not_awaited()

    async def test_api_ladders_keep_requested_parallelism_and_smoke(self):
        peak, owners = await self.measure_concurrency("glm-45-hybrid")
        self.assertEqual(peak, 3)
        self.assertEqual(owners, [None, None, None])
        self.smoke_call.assert_awaited_once()
        self.budget_check.assert_awaited_once()

    async def test_self_hosted_failure_stops_later_ladders(self):
        with patch.object(phase6b, "run_single", new=AsyncMock(return_value=None)) as run:
            with self.assertRaisesRegex(RuntimeError, "failed for 1"):
                await self.run_phase()
        self.assertEqual(run.await_count, 1)

    async def test_completed_resume_does_not_construct_engine(self):
        with patch.object(phase6b, "is_complete", return_value=True), \
                patch.object(phase6b, "create_experiment_agent") as create:
            await self.run_phase(resume=True)
        create.assert_not_called()

    async def test_zero_concurrency_fails_before_work(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            await self.run_phase(max_concurrent=0)
        self.smoke_call.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
