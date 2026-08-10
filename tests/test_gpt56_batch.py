"""Offline coverage for GPT-5.6 Sol/Terra/Luna batch configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.experiments.ladder_statement_pair.openai_batch_runner import (
    BATCH_JOBS_NAME,
    BATCH_MANIFEST_NAME,
    build_openai_chat_batch_body,
    generate_batch_run,
    process_batch_run,
)
from llm_coherence.experiments.within_ladder.run_within_ladder_experiment import (
    _openai_batch_request_body,
)
from llm_coherence.runtime.agents import MODEL_SPECS
from llm_coherence.runtime.usage_cost import (
    OPENAI_BATCH_PRICE_PER_MTOK,
    OPENAI_STANDARD_PRICE_PER_MTOK,
)


GPT56_CASES = {
    "gpt-56": ("gpt-5.6-sol", "none", 10),
    "gpt-56-thinking": ("gpt-5.6-sol", "high", 200),
    "gpt-56-terra": ("gpt-5.6-terra", "none", 10),
    "gpt-56-terra-thinking": ("gpt-5.6-terra", "high", 150),
    "gpt-56-luna": ("gpt-5.6-luna", "none", 10),
    "gpt-56-luna-thinking": ("gpt-5.6-luna", "high", 150),
}


class GPT56ConfigurationTests(unittest.TestCase):
    def test_all_six_conditions_have_matching_model_and_effort(self) -> None:
        for model_key, (model_id, effort, max_tokens) in GPT56_CASES.items():
            with self.subTest(model_key=model_key):
                config = MODEL_CONFIGS[model_key]
                spec = MODEL_SPECS[model_key]
                self.assertEqual(spec.model_type, "openai")
                self.assertEqual(spec.model_name, f"openai/{model_id}")
                self.assertEqual(config.extra_body, {"reasoning_effort": effort})
                self.assertEqual(config.max_tokens, max_tokens)
                self.assertEqual(
                    config.reasoning_artifact_type,
                    "prose_justification" if effort == "high" else "none",
                )

    def test_all_three_variants_have_standard_and_half_price_batch_rates(self) -> None:
        expected = {
            "gpt-5.6-sol": {"input": 5.0, "output": 30.0},
            "gpt-5.6-terra": {"input": 2.5, "output": 15.0},
            "gpt-5.6-luna": {"input": 1.0, "output": 6.0},
        }
        for model_id, standard in expected.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(OPENAI_STANDARD_PRICE_PER_MTOK[model_id], standard)
                self.assertEqual(
                    OPENAI_BATCH_PRICE_PER_MTOK[model_id],
                    {name: price * 0.5 for name, price in standard.items()},
                )

    def test_step10b_body_keeps_explicit_off_and_omits_temperature_for_high(self) -> None:
        off = build_openai_chat_batch_body(
            model_id="gpt-5.6-sol",
            prompt="Choose A or B",
            max_tokens=10,
            temperature=0.0,
            extra_body={"reasoning_effort": "none"},
            system_message=None,
        )
        self.assertEqual(off["reasoning_effort"], "none")
        self.assertEqual(off["temperature"], 0.0)
        self.assertEqual(off["messages"], [{"role": "user", "content": "Choose A or B"}])

        high = build_openai_chat_batch_body(
            model_id="gpt-5.6-terra",
            prompt="Choose A or B",
            max_tokens=150,
            temperature=0.0,
            extra_body={"reasoning_effort": "high"},
            system_message=None,
        )
        self.assertEqual(high["reasoning_effort"], "high")
        self.assertNotIn("temperature", high)

    def test_step10a_body_uses_the_same_reasoning_rules(self) -> None:
        off = _openai_batch_request_body(
            "gpt-5.6-luna",
            "Choose A or B",
            10,
            0.0,
            {"reasoning_effort": "none"},
        )
        high = _openai_batch_request_body(
            "gpt-5.6-luna",
            "Choose A or B",
            150,
            0.0,
            {"reasoning_effort": "high"},
        )
        self.assertEqual(off["reasoning_effort"], "none")
        self.assertEqual(off["temperature"], 0.0)
        self.assertEqual(high["reasoning_effort"], "high")
        self.assertNotIn("temperature", high)


class GPT56BatchRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "data"
        self.results_dir = self.root / "results"
        self.data_dir.mkdir()
        self.source_manifest = self.data_dir / "source_manifest.json"
        self.source_manifest.write_text('{"variation_files": []}\n', encoding="utf-8")
        self.test_name = "phase6b_variations_pruned_final_Test_category_1234"
        self.comparison_path = self.data_dir / f"{self.test_name}_comparisons.json"
        self.comparison_path.write_text(
            json.dumps(
                {
                    "comparisons": [
                        {
                            "outcome_a": {"text": "Outcome A", "tier": 1},
                            "outcome_b": {"text": "Outcome B", "tier": None},
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _generate(self, model_key: str = "gpt-56-thinking") -> Path:
        return generate_batch_run(
            run_items=[
                {
                    "test_name": self.test_name,
                    "comparison_path": self.comparison_path,
                }
            ],
            data_dir=self.data_dir,
            source_manifest_path=self.source_manifest,
            results_dir=self.results_dir,
            model_key=model_key,
            num_trials=2,
            include_flipped=True,
            with_reasoning=model_key.endswith("-thinking"),
            system_message=None,
            max_requests_per_batch=2,
        )

    def test_generation_shards_four_requests_without_submitting(self) -> None:
        run_dir = self._generate()
        manifest = json.loads((run_dir / BATCH_MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_id"], "gpt-5.6-sol")
        self.assertEqual(manifest["reasoning_effort"], "high")
        self.assertEqual(manifest["max_tokens"], 200)
        self.assertTrue(manifest["with_reasoning"])
        self.assertEqual(
            manifest["prompt_template_used"],
            "comparison_prompt_template_reasoning_default",
        )
        self.assertIsNone(manifest["system_message"])
        self.assertEqual(manifest["total_requests"], 4)
        self.assertEqual([s["request_count"] for s in manifest["shards"]], [2, 2])
        self.assertFalse((run_dir / BATCH_JOBS_NAME).exists())

        rows = []
        for shard in manifest["shards"]:
            rows.extend(
                json.loads(line)
                for line in (run_dir / shard["input_file"]).read_text(encoding="utf-8").splitlines()
            )
        self.assertEqual(len({row["custom_id"] for row in rows}), 4)
        for row in rows:
            self.assertEqual(row["method"], "POST")
            self.assertEqual(row["url"], "/v1/chat/completions")
            self.assertEqual(row["body"]["reasoning_effort"], "high")
            self.assertEqual(row["body"]["max_completion_tokens"], 200)
            self.assertNotIn("temperature", row["body"])
            self.assertEqual(row["body"]["messages"][0]["role"], "user")

    def test_completed_outputs_reconstruct_normal_result_schema(self) -> None:
        run_dir = self._generate()
        output_name = "batch_output_000.jsonl"
        response_choices = {
            "s0000-c0000-dab-t000": "Reasoning one. Answer: A",
            "s0000-c0000-dab-t001": "Reasoning two. Answer: A",
            "s0000-c0000-dba-t000": "Reasoning three. Answer: B",
            "s0000-c0000-dba-t001": "Reasoning four. Answer: A",
        }
        rows = []
        for custom_id, answer in response_choices.items():
            rows.append(
                {
                    "id": f"request-{custom_id}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": f"req-{custom_id}",
                        "body": {
                            "model": "gpt-5.6-sol",
                            "choices": [{"message": {"content": answer}}],
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 5,
                                "total_tokens": 25,
                                "completion_tokens_details": {"reasoning_tokens": 4},
                            },
                        },
                    },
                    "error": None,
                }
            )
        (run_dir / output_name).write_text(
            "\n".join(json.dumps(row) for row in reversed(rows)) + "\n",
            encoding="utf-8",
        )
        (run_dir / BATCH_JOBS_NAME).write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "jobs": [
                        {
                            "batch_id": "batch-test",
                            "status": "completed",
                            "output_file": output_name,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        summary = process_batch_run(run_dir)
        self.assertEqual(summary["successful_responses"], 4)
        self.assertEqual(summary["missing_responses"], 0)
        self.assertEqual(summary["result_files_written"], 1)

        result = json.loads(Path(summary["result_files"][0]).read_text(encoding="utf-8"))
        self.assertEqual(result["config"]["reasoning_mode"], "high")
        self.assertTrue(result["config"]["with_reasoning"])
        self.assertEqual(result["config"]["infrastructure"], "openai_batch_api")
        self.assertEqual(result["metadata"]["usage_stats"]["reasoning_tokens"]["total"], 16)
        preference = result["preferences"][0]
        self.assertEqual(preference["count_prefer_a"], 3)
        self.assertEqual(preference["count_prefer_b"], 1)
        self.assertEqual(len(preference["raw_responses_original"]), 2)
        self.assertEqual(len(preference["raw_responses_flipped"]), 2)

    def test_processing_rejects_nonterminal_jobs(self) -> None:
        run_dir = self._generate()
        (run_dir / BATCH_JOBS_NAME).write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "jobs": [{"batch_id": "batch-test", "status": "in_progress"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "before every submitted job is terminal"):
            process_batch_run(run_dir)


if __name__ == "__main__":
    unittest.main()
