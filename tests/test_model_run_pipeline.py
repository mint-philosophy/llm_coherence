"""Offline coverage for model configuration and model-run pipelines."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_coherence.config import (
    MODEL_CONFIGS,
    canonical_model_key,
    get_model_config,
    results_dir_name,
)
from llm_coherence.experiments.ladder_statement_pair.openai_batch_runner import (
    BATCH_JOBS_NAME,
    BATCH_MANIFEST_NAME,
    MAX_BATCH_FILE_BYTES,
    OPENAI_BATCH_ENDPOINT,
    REASONING_SUMMARIES_NAME,
    _write_sharded_requests,
    batch_queue_limit_for_model,
    build_openai_responses_batch_body,
    create_retry_shards,
    estimate_batch_run_pre_submit_cost,
    generate_batch_run,
    print_batch_run_pre_submit_cost_estimate,
    process_batch_run,
    submit_pending_batch_shards,
    validate_batch_run_binding,
)
from llm_coherence.experiments.within_ladder.run_within_ladder_experiment import (
    MODELS,
    _guard_duplicate_batch_submission,
    _is_complete_finish_reason,
    _is_retryable_transport_exception,
    _is_transport_failure_row,
    _openai_responses_batch_request_body,
    _prune_cost_log,
    _resolve_smoke_max_variation_sets,
    _within_ladder_cost_pricing,
    analyze,
    estimate_pre_submit_batch_cost,
    estimate_pre_submit_live_cost,
    extract_clean_row,
    generate_pairs,
    print_pre_submit_batch_cost_estimate,
    print_pre_submit_live_cost_estimate,
    run_live,
    write_clean_and_cost_log,
)
from llm_coherence.runtime.agents import LiteLLMAgent, MODEL_SPECS, model_name_for_key
from llm_coherence.runtime.model_run_index import model_reasoning_mode
from llm_coherence.runtime.preflight_check import (
    MODEL_COST_ESTIMATES,
    is_thinking_run,
)
from llm_coherence.runtime.usage_cost import (
    OPENAI_BATCH_PRICE_PER_MTOK,
    OPENAI_STANDARD_PRICE_PER_MTOK,
    actual_cost_usd_from_usage,
    usage_cost_breakdown,
)


GPT56_CASES = {
    "gpt-56-sol": ("gpt-5.6-sol", "none", 16),
    "gpt-56-sol-thinking": ("gpt-5.6-sol", "high", 3000),
    "gpt-56-terra": ("gpt-5.6-terra", "none", 16),
    "gpt-56-terra-thinking": ("gpt-5.6-terra", "high", 3000),
    "gpt-56-luna": ("gpt-5.6-luna", "none", 16),
    "gpt-56-luna-thinking": ("gpt-5.6-luna", "high", 3000),
}


OPENROUTER_CASES = {
    "qwen-37-max-openrouter": {
        "model": "qwen/qwen3.7-max-20260520",
        "reasoning": {"enabled": False},
        "max_tokens": 16,
        "thinking": False,
        "prices": {"input": 1.475, "output": 4.425},
    },
    "qwen-37-max-openrouter-thinking": {
        "model": "qwen/qwen3.7-max-20260520",
        "reasoning": {"enabled": True},
        "max_tokens": 3000,
        "thinking": True,
        "prices": {"input": 1.475, "output": 4.425},
    },
}


def _one_ladder() -> list[dict]:
    return [
        {
            "original_statement_id": "offline_smoke",
            "variations": [{"text": f"Tier {index}"} for index in range(1, 8)],
        }
    ]


def _responses_body(
    text: str | None,
    *,
    model: str = "gpt-5.6-sol",
    status: str = "completed",
    incomplete_reason: str | None = None,
    reasoning_summary: str | None = "A concise native reasoning summary.",
) -> dict:
    content = [] if text is None else [{"type": "output_text", "text": text}]
    output = []
    if reasoning_summary is not None:
        output.append(
            {
                "id": "rs-test",
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": reasoning_summary}
                ],
            }
        )
    output.append({"type": "message", "role": "assistant", "content": content})
    body = {
        "id": "resp-test",
        "model": model,
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }
    if incomplete_reason:
        body["incomplete_details"] = {"reason": incomplete_reason}
    return body


class _FakeFiles:
    def __init__(self) -> None:
        self.created: list[str] = []

    def create(self, *, file, purpose):
        self.created.append(Path(file.name).name)
        return SimpleNamespace(id=f"file-{len(self.created)}")


class _FakeBatches:
    def __init__(self, remote: list | None = None) -> None:
        self.remote = list(remote or [])
        self.created: list[dict] = []

    def list(self, *, limit):
        return SimpleNamespace(data=list(self.remote))

    def create(self, **kwargs):
        self.created.append(kwargs)
        batch = SimpleNamespace(
            id=f"batch-{len(self.created)}",
            status="validating",
            output_file_id=None,
            error_file_id=None,
            metadata=kwargs["metadata"],
        )
        self.remote.insert(0, batch)
        return batch


class _FakeClient:
    def __init__(self, remote: list | None = None) -> None:
        self.files = _FakeFiles()
        self.batches = _FakeBatches(remote)


class WithinLadderSmokeScopeTests(unittest.TestCase):
    def test_smoke_defaults_to_one_ladder(self) -> None:
        self.assertEqual(
            _resolve_smoke_max_variation_sets(
                smoke=True,
                max_variation_sets=None,
            ),
            1,
        )

    def test_smoke_preserves_explicit_multi_ladder_slice(self) -> None:
        self.assertEqual(
            _resolve_smoke_max_variation_sets(
                smoke=True,
                max_variation_sets=5,
            ),
            5,
        )

    def test_non_smoke_run_remains_unbounded_by_default(self) -> None:
        self.assertIsNone(
            _resolve_smoke_max_variation_sets(
                smoke=False,
                max_variation_sets=None,
            )
        )


class GPT56ConfigurationTests(unittest.TestCase):
    def test_opus_thinking_artifact_is_classified_as_summary(self) -> None:
        self.assertEqual(
            MODEL_CONFIGS["opus-46-thinking"].reasoning_artifact_type,
            "summary",
        )

    def test_all_six_conditions_have_matching_model_and_effort(self) -> None:
        for model_key, (model_id, effort, max_tokens) in GPT56_CASES.items():
            with self.subTest(model_key=model_key):
                config = MODEL_CONFIGS[model_key]
                spec = MODEL_SPECS[model_key]
                self.assertEqual(spec.model_type, "openai")
                self.assertEqual(spec.model_name, f"openai/{model_id}")
                expected_extra_body = {"reasoning_effort": effort}
                if effort == "high":
                    expected_extra_body["reasoning"] = {"summary": "auto"}
                self.assertEqual(config.extra_body, expected_extra_body)
                self.assertEqual(config.max_tokens, max_tokens)
                self.assertEqual(
                    config.reasoning_artifact_type,
                    "summary" if effort == "high" else "none",
                )

    def test_legacy_sol_keys_resolve_to_explicit_canonical_names(self) -> None:
        aliases = {
            "gpt-56": "gpt-56-sol",
            "gpt-56-thinking": "gpt-56-sol-thinking",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(canonical_model_key(alias), canonical)
                self.assertIs(get_model_config(alias), MODEL_CONFIGS[canonical])
                self.assertEqual(results_dir_name(alias), canonical)
                self.assertEqual(
                    model_name_for_key(alias), MODEL_SPECS[canonical].model_name
                )

    def test_current_standard_and_half_price_batch_rates(self) -> None:
        expected = {
            "gpt-5.6-sol": {"input": 5.0, "output": 30.0},
            "gpt-5.6-terra": {"input": 2.0, "output": 12.0},
            "gpt-5.6-luna": {"input": 0.2, "output": 1.2},
        }
        for model_id, standard in expected.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(OPENAI_STANDARD_PRICE_PER_MTOK[model_id], standard)
                self.assertEqual(
                    OPENAI_BATCH_PRICE_PER_MTOK[model_id],
                    {name: price * 0.5 for name, price in standard.items()},
                )

    def test_within_ladder_rollup_uses_batch_rates_for_all_gpt56_conditions(self) -> None:
        for model_key, (model_id, _, _) in GPT56_CASES.items():
            with self.subTest(model_key=model_key):
                self.assertEqual(
                    _within_ladder_cost_pricing(model_key),
                    OPENAI_BATCH_PRICE_PER_MTOK[model_id],
                )

    def test_reasoning_off_pre_submit_estimate_uses_projected_and_cap_costs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            rows = [
                {
                    "custom_id": f"ladder__T1_vs_T2__{direction}",
                    "body": {
                        "model": "gpt-5.6-terra",
                        "input": "a" * 5_000,
                        "instructions": "é" * 10,
                        "max_output_tokens": 16,
                        "reasoning": {"effort": "none"},
                    },
                }
                for direction in ("AB", "BA")
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            estimate = estimate_pre_submit_batch_cost(
                "gpt-56-terra",
                input_path=input_path,
            )

            self.assertIsNotNone(estimate)
            self.assertEqual(estimate["request_count"], 2)
            self.assertEqual(estimate["reasoning_on_requests"], 0)
            self.assertEqual(estimate["estimated_input_tokens"], 2_024)
            self.assertEqual(estimate["projected_output_tokens"], 10)
            self.assertEqual(estimate["output_token_cap"], 32)
            self.assertEqual(estimate["estimated_cost_usd"], 0.002084)
            self.assertEqual(estimate["maximum_output_cost_usd"], 0.002216)

            stream = io.StringIO()
            with redirect_stdout(stream):
                displayed = print_pre_submit_batch_cost_estimate(
                    "gpt-56-terra",
                    input_path=input_path,
                )
            self.assertEqual(displayed, estimate)
            output = stream.getvalue()
            self.assertIn("Pre-submit OpenAI Batch cost estimate", output)
            self.assertIn("Reasoning-off assumption: 5 output tokens/request", output)
            self.assertIn("after --fetch", output)

    def test_reasoning_on_pre_submit_estimate_uses_configured_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            rows = [
                {
                    "custom_id": f"ladder__T1_vs_T2__{direction}",
                    "body": {
                        "model": "gpt-5.6-luna",
                        "input": "a" * 5_000,
                        "max_output_tokens": 200,
                        "reasoning": {"effort": "high"},
                    },
                }
                for direction in ("AB", "BA")
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            estimate = estimate_pre_submit_batch_cost(
                "gpt-56-luna-thinking",
                input_path=input_path,
            )

            self.assertIsNotNone(estimate)
            self.assertEqual(estimate["reasoning_on_requests"], 2)
            self.assertEqual(estimate["estimated_input_tokens"], 2_016)
            self.assertEqual(estimate["projected_output_tokens"], 400)
            self.assertEqual(estimate["output_token_cap"], 400)
            self.assertEqual(
                estimate["estimated_cost_usd"],
                estimate["maximum_output_cost_usd"],
            )

            stream = io.StringIO()
            with redirect_stdout(stream):
                print_pre_submit_batch_cost_estimate(
                    "gpt-56-luna-thinking",
                    input_path=input_path,
                )
            output = stream.getvalue()
            self.assertIn("conservative planning estimate", output)
            self.assertIn("configured output-token caps", output)

    def test_responses_body_keeps_explicit_off_and_omits_temperature_for_high(self) -> None:
        off = build_openai_responses_batch_body(
            model_id="gpt-5.6-sol",
            prompt="Choose A or B",
            max_tokens=16,
            temperature=0.0,
            extra_body={"reasoning_effort": "none"},
            system_message=None,
        )
        self.assertEqual(off["reasoning"], {"effort": "none"})
        self.assertEqual(off["temperature"], 0.0)
        self.assertEqual(off["input"], "Choose A or B")
        self.assertEqual(off["max_output_tokens"], 16)
        self.assertNotIn("messages", off)

        high = build_openai_responses_batch_body(
            model_id="gpt-5.6-terra",
            prompt="Choose A or B",
            max_tokens=150,
            temperature=0.0,
            extra_body={
                "reasoning_effort": "high",
                "reasoning": {"summary": "auto"},
            },
            system_message="Follow the protocol.",
        )
        self.assertEqual(
            high["reasoning"], {"effort": "high", "summary": "auto"}
        )
        self.assertEqual(high["instructions"], "Follow the protocol.")
        self.assertNotIn("temperature", high)

    def test_step10a_uses_the_same_responses_reasoning_rules(self) -> None:
        off = _openai_responses_batch_request_body(
            "gpt-5.6-luna", "Choose A or B", 16, 0.0, {"reasoning_effort": "none"}
        )
        high = _openai_responses_batch_request_body(
            "gpt-5.6-luna",
            "Choose A or B",
            150,
            0.0,
            {
                "reasoning_effort": "high",
                "reasoning": {"summary": "auto"},
            },
        )
        self.assertEqual(off["reasoning"], {"effort": "none"})
        self.assertEqual(off["temperature"], 0.0)
        self.assertEqual(
            high["reasoning"], {"effort": "high", "summary": "auto"}
        )
        self.assertNotIn("temperature", high)

    def test_both_responses_builders_reject_gpt56_caps_below_16(self) -> None:
        with self.assertRaisesRegex(ValueError, r"max_output_tokens >= 16"):
            build_openai_responses_batch_body(
                model_id="gpt-5.6-sol",
                prompt="Choose A or B",
                max_tokens=15,
                temperature=0.0,
                extra_body={"reasoning_effort": "none"},
                system_message=None,
            )
        with self.assertRaisesRegex(ValueError, r"max_output_tokens >= 16"):
            _openai_responses_batch_request_body(
                "gpt-5.6-luna",
                "Choose A or B",
                15,
                0.0,
                {"reasoning_effort": "none"},
            )

    def test_responses_usage_reasoning_tokens_and_actual_cost_semantics(self) -> None:
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 12},
        }
        fields = usage_cost_breakdown(
            usage, provider="openai", model_id="gpt-5.6-terra", batch=True
        )
        self.assertEqual(fields["prompt_tokens"], 100)
        self.assertEqual(fields["completion_tokens"], 20)
        self.assertEqual(fields["reasoning_tokens"], 12)
        self.assertEqual(fields["cost_source"], "computed_from_usage")
        self.assertIsNone(
            actual_cost_usd_from_usage(
                usage, provider="openai", model_id="gpt-5.6-terra", batch=True
            )
        )

    def test_queue_limit_requires_verified_tier_or_override(self) -> None:
        self.assertEqual(
            batch_queue_limit_for_model("gpt-5.6-luna", usage_tier=3),
            40_000_000,
        )
        self.assertEqual(
            batch_queue_limit_for_model(
                "gpt-5.6-luna", usage_tier=None, explicit_limit=1234
            ),
            1234,
        )
        with self.assertRaisesRegex(ValueError, "requires --batch-usage-tier"):
            batch_queue_limit_for_model("gpt-5.6-sol", usage_tier=None)


class OpenRouterModelConfigurationTests(unittest.TestCase):
    def test_registry_uses_pinned_models_and_current_reasoning_shapes(self) -> None:
        for model_key, expected in OPENROUTER_CASES.items():
            with self.subTest(model_key=model_key):
                config = MODEL_CONFIGS[model_key]
                spec = MODEL_SPECS[model_key]
                self.assertEqual(spec.model_type, "openrouter")
                self.assertEqual(spec.model_name, f"openrouter/{expected['model']}")
                self.assertEqual(config.max_tokens, expected["max_tokens"])
                self.assertEqual(
                    config.extra_body,
                    {"reasoning": expected["reasoning"]},
                )
                self.assertEqual(
                    config.reasoning_artifact_type,
                    "raw_cot" if expected["thinking"] else "none",
                )
                self.assertEqual(MODEL_COST_ESTIMATES[model_key], expected["prices"])
                self.assertEqual(is_thinking_run(model_key), expected["thinking"])
                self.assertEqual(
                    model_reasoning_mode(model_key),
                    "thinking_on" if expected["thinking"] else "thinking_off",
                )

    def test_within_ladder_requests_keep_nested_openrouter_reasoning(self) -> None:
        for model_key, expected in OPENROUTER_CASES.items():
            with self.subTest(model_key=model_key):
                requests = generate_pairs(
                    _one_ladder(),
                    model_key,
                    with_reasoning=expected["thinking"],
                )
                self.assertEqual(len(requests), 42)
                first = requests[0]
                self.assertEqual(first["url"], "/v1/chat/completions")
                self.assertEqual(first["body"]["model"], expected["model"])
                self.assertEqual(first["body"]["max_tokens"], expected["max_tokens"])
                self.assertEqual(first["body"]["reasoning"], expected["reasoning"])
                self.assertNotIn("reasoning_effort", first["body"])
                self.assertEqual(MODELS[model_key][1], "openrouter")

    def test_live_preflight_reports_projected_and_all_cap_costs(self) -> None:
        requests = [
            {
                "custom_id": f"ladder__T1_vs_T2__{direction}",
                "body": {
                    "model": "qwen/qwen3.7-max-20260520",
                    "messages": [{"role": "user", "content": "a" * 5_000}],
                    "max_tokens": 16,
                    "reasoning": {"enabled": False},
                },
            }
            for direction in ("AB", "BA")
        ]

        estimate = estimate_pre_submit_live_cost(
            "qwen-37-max-openrouter",
            requests=requests,
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["request_count"], 2)
        self.assertEqual(estimate["reasoning_on_requests"], 0)
        self.assertEqual(estimate["estimated_input_tokens"], 2_018)
        self.assertEqual(estimate["projected_output_tokens"], 10)
        self.assertEqual(estimate["output_token_cap"], 32)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.003021)
        self.assertAlmostEqual(estimate["maximum_output_cost_usd"], 0.003118)

        stream = io.StringIO()
        with redirect_stdout(stream):
            displayed = print_pre_submit_live_cost_estimate(
                "qwen-37-max-openrouter",
                requests=requests,
            )
        self.assertEqual(displayed, estimate)
        output = stream.getvalue()
        self.assertIn("Preflight OpenRouter live cost estimate", output)
        self.assertIn("all-cap estimate", output)
        self.assertIn("no API request has been sent", output)

    def test_reasoning_live_preflight_projects_below_strict_cap(self) -> None:
        requests = [
            {
                "custom_id": f"ladder__T1_vs_T2__{direction}",
                "body": {
                    "model": "qwen/qwen3.7-max-20260520",
                    "messages": [{"role": "user", "content": "a" * 5_000}],
                    "max_tokens": 3_000,
                    "reasoning": {"enabled": True},
                },
            }
            for direction in ("AB", "BA")
        ]

        estimate = estimate_pre_submit_live_cost(
            "qwen-37-max-openrouter-thinking",
            requests=requests,
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["reasoning_on_requests"], 2)
        self.assertEqual(estimate["projected_output_tokens"], 500)
        self.assertEqual(estimate["output_token_cap"], 6_000)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.005189)
        self.assertAlmostEqual(estimate["maximum_output_cost_usd"], 0.029527)
        self.assertLess(
            estimate["estimated_cost_usd"],
            estimate["maximum_output_cost_usd"],
        )

    def test_main_runner_keeps_reasoning_and_requests_cost_telemetry(self) -> None:
        model_key = "qwen-37-max-openrouter-thinking"
        config = MODEL_CONFIGS[model_key]
        agent = LiteLLMAgent(
            model=MODEL_SPECS[model_key].model_name,
            max_tokens=config.max_tokens,
            extra_body=config.extra_body,
        )
        kwargs = agent._completion_kwargs(
            [{"role": "user", "content": "Choose A or B."}],
            timeout=30.0,
        )
        self.assertEqual(kwargs["max_tokens"], 3000)
        self.assertEqual(
            kwargs["extra_body"]["reasoning"],
            {"enabled": True},
        )
        self.assertEqual(kwargs["extra_body"]["usage"], {"include": True})


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

    def _generate(
        self,
        model_key: str = "gpt-56-sol-thinking",
        *,
        max_requests_per_batch: int = 10,
    ) -> Path:
        return generate_batch_run(
            run_items=[
                {"test_name": self.test_name, "comparison_path": self.comparison_path}
            ],
            data_dir=self.data_dir,
            source_manifest_path=self.source_manifest,
            results_dir=self.results_dir,
            model_key=model_key,
            num_trials=2,
            include_flipped=True,
            with_reasoning=model_key.endswith("-thinking"),
            system_message=None,
            max_requests_per_batch=max_requests_per_batch,
        )

    def _manifest(self, run_dir: Path) -> dict:
        return json.loads((run_dir / BATCH_MANIFEST_NAME).read_text(encoding="utf-8"))

    def _write_output_and_jobs(
        self,
        run_dir: Path,
        rows: list[dict],
        *,
        status: str = "completed",
    ) -> None:
        manifest = self._manifest(run_dir)
        if len(manifest["shards"]) != 1:
            raise AssertionError("test helper expects a one-shard run")
        output_name = "batch_output_000.jsonl"
        (run_dir / output_name).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        (run_dir / BATCH_JOBS_NAME).write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "jobs": [
                        {
                            "input_file": manifest["shards"][0]["input_file"],
                            "batch_id": "batch-test",
                            "status": status,
                            "output_file": output_name,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _success_rows(self, *, empty: bool = False, one_truncated: bool = False) -> list[dict]:
        answers = {
            "s0000-c0000-dab-t000": "Reasoning one. Answer: A",
            "s0000-c0000-dab-t001": "Reasoning two. Answer: A",
            "s0000-c0000-dba-t000": "Reasoning three. Answer: B",
            "s0000-c0000-dba-t001": "Reasoning four. Answer: A",
        }
        rows = []
        for index, (custom_id, answer) in enumerate(answers.items()):
            body = _responses_body(None if empty else answer)
            if one_truncated and index == 0:
                body["status"] = "incomplete"
                body["incomplete_details"] = {"reason": "max_output_tokens"}
            rows.append(
                {
                    "id": f"request-{custom_id}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": f"req-{custom_id}",
                        "body": body,
                    },
                    "error": None,
                }
            )
        return rows

    def test_generation_uses_responses_shards_and_anonymous_relative_paths(self) -> None:
        run_dir = self._generate(max_requests_per_batch=2)
        manifest = self._manifest(run_dir)
        self.assertEqual(manifest["schema_version"], "2.0")
        self.assertEqual(manifest["api_endpoint"], OPENAI_BATCH_ENDPOINT)
        self.assertEqual(manifest["reasoning_artifact_type"], "summary")
        self.assertIsNone(manifest["temperature"])
        self.assertEqual(manifest["configured_temperature"], 0.0)
        self.assertEqual([s["request_count"] for s in manifest["shards"]], [2, 2])
        self.assertFalse(Path(manifest["results_dir"]).is_absolute())
        self.assertFalse(Path(manifest["source_manifest_path"]).is_absolute())
        self.assertNotIn(str(Path.home()), json.dumps(manifest))

        rows = []
        for shard in manifest["shards"]:
            self.assertLessEqual(shard["byte_size"], MAX_BATCH_FILE_BYTES)
            self.assertGreater(shard["input_token_upper_bound"], 0)
            rows.extend(
                json.loads(line)
                for line in (run_dir / shard["input_file"]).read_text().splitlines()
            )
        self.assertEqual(len({row["custom_id"] for row in rows}), 4)
        for row in rows:
            self.assertEqual(row["url"], "/v1/responses")
            self.assertEqual(
                row["body"]["reasoning"],
                {"effort": "high", "summary": "auto"},
            )
            self.assertEqual(row["body"]["max_output_tokens"], 3000)
            self.assertNotIn("temperature", row["body"])

    def test_step10b_pre_submit_estimate_uses_generated_shards(self) -> None:
        run_dir = self._generate(
            model_key="gpt-56-luna",
            max_requests_per_batch=2,
        )

        estimate = estimate_batch_run_pre_submit_cost(run_dir)

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["model_key"], "gpt-56-luna")
        self.assertEqual(estimate["model_id"], "gpt-5.6-luna")
        self.assertEqual(estimate["request_count"], 4)
        self.assertEqual(estimate["reasoning_on_requests"], 0)
        self.assertEqual(estimate["projected_output_tokens"], 20)
        self.assertEqual(estimate["output_token_cap"], 64)
        self.assertEqual(estimate["input_rate_per_mtok"], 0.1)
        self.assertEqual(estimate["output_rate_per_mtok"], 0.6)
        self.assertLess(
            estimate["estimated_cost_usd"], estimate["maximum_output_cost_usd"]
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            displayed = print_batch_run_pre_submit_cost_estimate(run_dir)
        self.assertEqual(displayed, estimate)
        output = stream.getvalue()
        self.assertIn("Pre-submit OpenAI Batch cost estimate", output)
        self.assertIn("Reasoning-off assumption: 5 output tokens/request", output)
        self.assertIn("after --batch-process", output)

    def test_step10b_reasoning_preflight_is_conservative(self) -> None:
        run_dir = self._generate(model_key="gpt-56-sol-thinking")

        estimate = estimate_batch_run_pre_submit_cost(run_dir)

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["request_count"], 4)
        self.assertEqual(estimate["reasoning_on_requests"], 4)
        self.assertEqual(estimate["projected_output_tokens"], 12_000)
        self.assertEqual(estimate["output_token_cap"], 12_000)
        self.assertEqual(
            estimate["estimated_cost_usd"], estimate["maximum_output_cost_usd"]
        )

    def test_byte_limit_creates_additional_shards(self) -> None:
        requests = [
            {
                "custom_id": f"s0000-c0000-dab-t{i:03d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"input": "x" * 100},
            }
            for i in range(4)
        ]
        shards, total = _write_sharded_requests(
            requests,
            run_dir=self.root,
            prefix="bytes",
            kind="initial",
            attempt=0,
            max_requests_per_batch=50_000,
            max_bytes_per_batch=350,
        )
        self.assertEqual(total, 4)
        self.assertGreater(len(shards), 1)
        self.assertTrue(all(shard["byte_size"] <= 350 for shard in shards))

    def test_completed_outputs_reconstruct_schema_and_telemetry(self) -> None:
        run_dir = self._generate()
        self._write_output_and_jobs(
            run_dir, list(reversed(self._success_rows(one_truncated=True)))
        )
        summary = process_batch_run(run_dir)
        self.assertEqual(summary["successful_responses"], 4)
        self.assertEqual(summary["missing_responses"], 0)
        self.assertTrue(summary["reasoning_summary_requested"])
        self.assertEqual(summary["responses_with_reasoning_summary"], 4)
        self.assertEqual(summary["reasoning_summary_files_written"], 1)
        result_path = self.results_dir / summary["result_files"][0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["config"]["reasoning_mode"], "high")
        self.assertIsNone(result["config"]["temperature"])
        self.assertEqual(result["metadata"]["run_status"], "incomplete")
        self.assertEqual(result["metadata"]["usage_stats"]["reasoning_tokens"]["total"], 16)
        self.assertEqual(
            result["metadata"]["response_diagnostics"]["finish_reason_counts"][
                "max_output_tokens"
            ],
            1,
        )
        self.assertIsNone(result["metadata"]["actual_cost_usd"])
        self.assertNotIn("hostname", result["metadata"])
        self.assertEqual(
            result["metadata"]["reasoning_artifacts"],
            {
                "artifact_type": "summary",
                "summary_requested": True,
                "summary_mode_requested": "auto",
                "responses_with_summary": 4,
                "responses_without_summary": 0,
                "summary_coverage_rate": 1.0,
                "missing_summary_examples": [],
                "summary_block_count": 4,
                "sidecar": REASONING_SUMMARIES_NAME,
                "raw_batch_output_retained": True,
            },
        )
        traces_path = self.results_dir / summary["reasoning_summary_files"][0]
        traces = [
            json.loads(line)
            for line in traces_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(traces), 4)
        self.assertEqual(
            {trace["custom_id"] for trace in traces},
            {
                "s0000-c0000-dab-t000",
                "s0000-c0000-dab-t001",
                "s0000-c0000-dba-t000",
                "s0000-c0000-dba-t001",
            },
        )
        self.assertTrue(
            all(
                trace["summary"] == "A concise native reasoning summary."
                for trace in traces
            )
        )
        self.assertTrue(
            all(
                set(trace)
                == {"custom_id", "pair_idx", "direction", "trial", "content", "summary"}
                for trace in traces
            )
        )
        preference = result["preferences"][0]
        self.assertEqual(preference["count_prefer_a"], 2)
        self.assertEqual(preference["count_prefer_b"], 1)
        self.assertEqual(preference["parseable_trials"], 3)
        self.assertEqual(preference["expected_trials"], 4)

    def test_processing_rejects_nonterminal_and_partial_terminal_jobs(self) -> None:
        run_dir = self._generate()
        manifest = self._manifest(run_dir)
        (run_dir / BATCH_JOBS_NAME).write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "input_file": manifest["shards"][0]["input_file"],
                            "batch_id": "batch-test",
                            "status": "in_progress",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "before every submitted job is terminal"):
            process_batch_run(run_dir)

        self._write_output_and_jobs(run_dir, self._success_rows()[:3], status="failed")
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            process_batch_run(run_dir)
        self.assertFalse((self.results_dir / "phase6b_ladder_Test_category_1234" / "results.json").exists())

    def test_processing_rejects_input_hash_drift_before_writing(self) -> None:
        run_dir = self._generate()
        self.comparison_path.write_text('{"comparisons": []}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "comparison file hash changed"):
            process_batch_run(run_dir)

    def test_processing_marks_zero_parseable_pair_incomplete(self) -> None:
        run_dir = self._generate()
        self._write_output_and_jobs(run_dir, self._success_rows(empty=True))
        summary = process_batch_run(run_dir)

        self.assertEqual(summary["processing_status"], "incomplete")
        self.assertEqual(summary["zero_parseable_comparison_count"], 1)
        result_path = self.results_dir / summary["result_files"][0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["metadata"]["run_status"], "incomplete")
        preference = result["preferences"][0]
        self.assertEqual(preference["parseable_trials"], 0)
        self.assertIsNone(preference["prob_prefer_a"])
        self.assertIsNone(preference["prob_prefer_b"])

    def test_processing_records_missing_requested_reasoning_summaries(self) -> None:
        run_dir = self._generate()
        rows = self._success_rows()
        for row in rows:
            body = row["response"]["body"]
            body["output"] = [
                item for item in body["output"] if item.get("type") != "reasoning"
            ]
        self._write_output_and_jobs(run_dir, rows)

        summary = process_batch_run(run_dir)
        artifact_dir = self.results_dir / "phase6b_ladder_Test_category_1234"
        result = json.loads(
            (artifact_dir / "results.json").read_text(encoding="utf-8")
        )
        reasoning_artifacts = result["metadata"]["reasoning_artifacts"]
        self.assertEqual(reasoning_artifacts["responses_with_summary"], 0)
        self.assertEqual(reasoning_artifacts["responses_without_summary"], 4)
        self.assertEqual(reasoning_artifacts["summary_coverage_rate"], 0.0)
        self.assertEqual(len(reasoning_artifacts["missing_summary_examples"]), 4)
        self.assertTrue((artifact_dir / REASONING_SUMMARIES_NAME).exists())
        self.assertEqual(
            (artifact_dir / REASONING_SUMMARIES_NAME).read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(summary["responses_with_reasoning_summary"], 0)
        self.assertEqual(summary["responses_without_reasoning_summary"], 4)
        self.assertEqual(summary["reasoning_summary_coverage_rate"], 0.0)

    def test_retry_shards_exclude_deterministic_4xx_failures(self) -> None:
        run_dir = self._generate()
        success = self._success_rows()[0]
        deterministic = {
            "custom_id": "s0000-c0000-dab-t001",
            "response": {
                "status_code": 400,
                "body": {"error": {"code": "unsupported_parameter"}},
            },
        }
        transient = {
            "custom_id": "s0000-c0000-dba-t000",
            "response": {
                "status_code": 500,
                "body": {"error": {"code": "server_error"}},
            },
        }
        self._write_output_and_jobs(run_dir, [success, deterministic, transient])
        shards, retry_count, classification = create_retry_shards(run_dir)
        self.assertEqual(retry_count, 2)  # one 500 plus one response missing entirely
        self.assertEqual(classification["non_retryable"], 1)
        retry_ids = {
            json.loads(line)["custom_id"]
            for shard in shards
            for line in (run_dir / shard["input_file"]).read_text().splitlines()
        }
        self.assertNotIn("s0000-c0000-dab-t001", retry_ids)

    def test_retry_shards_preserve_fixed_cap_and_exclude_token_capped_responses(self) -> None:
        run_dir = self._generate()
        rows = self._success_rows(one_truncated=True)
        self._write_output_and_jobs(run_dir, rows)

        shards, retry_count, classification = create_retry_shards(run_dir)

        self.assertEqual(shards, [])
        self.assertEqual(retry_count, 0)
        self.assertEqual(classification["missing_total"], 0)
        self.assertEqual(classification["incomplete_total"], 1)
        self.assertEqual(classification["token_capped_incomplete"], 1)
        self.assertEqual(classification["retryable_incomplete"], 0)
        self.assertEqual(classification["retryable"], 0)
        self.assertEqual(classification["incomplete_retry_token_caps"], [])

    def test_token_capped_response_is_processed_as_incomplete_without_retry(self) -> None:
        run_dir = self._generate()
        initial_rows = self._success_rows(one_truncated=True)
        self._write_output_and_jobs(run_dir, initial_rows)

        shards, retry_count, classification = create_retry_shards(run_dir)
        summary = process_batch_run(run_dir)

        self.assertEqual(shards, [])
        self.assertEqual(retry_count, 0)
        self.assertEqual(classification["token_capped_incomplete"], 1)
        self.assertEqual(summary["processing_status"], "incomplete")
        self.assertEqual(summary["unparseable_or_missing_responses"], 1)

    def test_queue_limit_submits_in_waves(self) -> None:
        run_dir = self._generate(max_requests_per_batch=2)
        manifest = self._manifest(run_dir)
        limit = max(shard["input_token_upper_bound"] for shard in manifest["shards"])
        client = _FakeClient()
        first = submit_pending_batch_shards(
            run_dir, max_queued_input_tokens=limit, client=client
        )
        self.assertEqual(first["submitted_this_call"], 1)
        self.assertEqual(first["pending_shards"], 1)

        jobs_path = run_dir / BATCH_JOBS_NAME
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        jobs["jobs"][0]["status"] = "completed"
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        second = submit_pending_batch_shards(
            run_dir, max_queued_input_tokens=limit, client=client
        )
        self.assertEqual(second["submitted_this_call"], 1)
        self.assertEqual(second["pending_shards"], 0)
        self.assertEqual(len(client.batches.created), 2)

    def test_interrupted_submission_recovers_remote_batch_without_duplicate(self) -> None:
        run_dir = self._generate()
        shard = self._manifest(run_dir)["shards"][0]
        submission_key = hashlib.sha256(
            f"{run_dir.name}:{shard['input_file']}".encode("utf-8")
        ).hexdigest()[:32]
        (run_dir / BATCH_JOBS_NAME).write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "jobs": [
                        {
                            "input_file": shard["input_file"],
                            "request_count": shard["request_count"],
                            "input_token_upper_bound": shard["input_token_upper_bound"],
                            "submission_key": submission_key,
                            "status": "pending_upload",
                            "batch_id": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        remote = SimpleNamespace(
            id="batch-recovered",
            status="validating",
            output_file_id=None,
            error_file_id=None,
            metadata={"submission_key": submission_key},
        )
        client = _FakeClient([remote])
        jobs = submit_pending_batch_shards(
            run_dir,
            max_queued_input_tokens=shard["input_token_upper_bound"],
            client=client,
        )
        self.assertEqual(jobs["jobs"][0]["batch_id"], "batch-recovered")
        self.assertEqual(client.files.created, [])
        self.assertEqual(client.batches.created, [])

    def test_explicit_run_binding_rejects_wrong_model_or_scope(self) -> None:
        run_dir = self._generate()
        with self.assertRaisesRegex(ValueError, "belongs to"):
            validate_batch_run_binding(
                run_dir, model_key="gpt-56-luna-thinking", results_dir=self.results_dir
            )
        with self.assertRaisesRegex(ValueError, "Check --smoke"):
            validate_batch_run_binding(
                run_dir,
                model_key="gpt-56-sol-thinking",
                results_dir=self.root / "other-results",
            )


class WithinLadderSafetyTests(unittest.TestCase):
    def test_live_retry_classification_is_transport_only(self) -> None:
        import httpx

        self.assertTrue(
            _is_retryable_transport_exception(
                httpx.ConnectError("connection failed")
            )
        )
        self.assertFalse(_is_retryable_transport_exception(RuntimeError("API error")))
        self.assertTrue(
            _is_transport_failure_row(
                {
                    "error": "All connection attempts failed",
                    "finish_reason": None,
                }
            )
        )
        self.assertTrue(
            _is_transport_failure_row(
                {"error": "socket failed", "error_type": "transport"}
            )
        )
        self.assertFalse(
            _is_transport_failure_row(
                {"error": "rate limited", "error_type": "non_transport"}
            )
        )
        self.assertFalse(
            _is_transport_failure_row(
                {"finish_reason": "length", "answer": None}
            )
        )

    def test_live_resume_retains_capped_and_unparseable_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            custom_ids = [
                "ladder__T1_vs_T2__AB",
                "ladder__T1_vs_T2__BA",
            ]
            requests = [
                {"custom_id": cid, "body": {"max_tokens": 16}}
                for cid in custom_ids
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in requests),
                encoding="utf-8",
            )
            retained_rows = [
                {
                    "custom_id": custom_ids[0],
                    "answer": None,
                    "finish_reason": "length",
                    "content": "A partial response",
                },
                {
                    "custom_id": custom_ids[1],
                    "answer": None,
                    "finish_reason": "stop",
                    "content": "I decline the forced choice.",
                },
            ]
            output_path.write_text(
                "".join(json.dumps(row) + "\n" for row in retained_rows),
                encoding="utf-8",
            )

            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )

            def artifact_path(_model_key: str, artifact: str) -> str:
                return str(root / artifact)

            stdout = io.StringIO()
            with (
                patch(f"{module}.model_output_path", side_effect=artifact_path),
                patch(f"{module}.sync_artifacts_to_input"),
                patch(
                    f"{module}.load_input_request_maps",
                    return_value=(set(custom_ids), {}),
                ),
                patch(f"{module}.require_api_key") as require_key,
                redirect_stdout(stdout),
            ):
                run_live("qwen-37-max-openrouter", concurrency=1)

            require_key.assert_not_called()
            self.assertIn("2/2 outcomes retained", stdout.getvalue())
            self.assertIn("transport retries pending=0", stdout.getvalue())
            self.assertEqual(
                [json.loads(line) for line in output_path.read_text().splitlines()],
                retained_rows,
            )

    def test_duplicate_submission_requires_explicit_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_id_path = root / "batch_id.txt"
            batch_id_path.write_text("batch-existing", encoding="utf-8")

            with patch(
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment.model_output_path",
                return_value=str(batch_id_path),
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    r"Refusing duplicate batch submission.*batch-existing",
                ):
                    _guard_duplicate_batch_submission("gpt-56-luna")

                self.assertEqual(
                    _guard_duplicate_batch_submission(
                        "gpt-56-luna",
                        force_resubmit=True,
                    ),
                    batch_id_path,
                )

    def test_responses_row_extracts_text_finish_reason_and_reasoning_usage(self) -> None:
        raw = {
            "custom_id": "ladder__T1_vs_T2__AB",
            "response": {"status_code": 200, "body": _responses_body("Answer: B")},
        }
        clean, cost = extract_clean_row(
            raw, "openai", model_id="gpt-5.6-sol", batch=True
        )
        self.assertEqual(clean["answer"], "B")
        self.assertEqual(clean["finish_reason"], "stop")
        self.assertEqual(cost["reasoning_tokens"], 4)

    def test_truncated_response_prose_is_preserved_but_not_scored(self) -> None:
        raw = {
            "custom_id": "ladder__T1_vs_T2__AB",
            "response": {
                "status_code": 200,
                "body": _responses_body(
                    "I would prefer Outcome A because it is more effective",
                    status="incomplete",
                    incomplete_reason="max_output_tokens",
                ),
            },
        }
        clean, _ = extract_clean_row(
            raw, "openai", model_id="gpt-5.6-luna", batch=True
        )
        self.assertIsNone(clean["answer"])
        self.assertEqual(clean["finish_reason"], "max_output_tokens")
        self.assertEqual(
            clean["content"],
            "I would prefer Outcome A because it is more effective",
        )

    def test_only_normal_terminal_reasons_are_scoreable(self) -> None:
        self.assertTrue(_is_complete_finish_reason("stop"))
        self.assertTrue(_is_complete_finish_reason("end_turn"))
        for reason in (None, "unknown", "max_output_tokens", "length", "max_tokens"):
            with self.subTest(reason=reason):
                self.assertFalse(_is_complete_finish_reason(reason))

    def test_cost_pruning_preserves_multiple_billable_attempts_per_id(self) -> None:
        custom_id = "ladder__T1_vs_T2__AB"
        entries = [
            {"custom_id": custom_id, "prompt_tokens": 10, "completion_tokens": 10},
            {"custom_id": custom_id, "prompt_tokens": 10, "completion_tokens": 2},
            {"custom_id": "stale", "prompt_tokens": 5, "completion_tokens": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cost_path = Path(tmp) / "cost_log.json"
            cost_path.write_text(
                json.dumps({"per_request": entries}),
                encoding="utf-8",
            )
            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )
            with (
                patch(f"{module}.model_output_path", return_value=str(cost_path)),
                patch(f"{module}.write_per_request_cost_log_file") as write_cost,
                patch(f"{module}.write_within_ladder_cost_artifacts"),
            ):
                dropped = _prune_cost_log(
                    "qwen-37-max-openrouter", {custom_id}, {}
                )

        self.assertEqual(dropped, 1)
        kept = write_cost.call_args.args[2]
        self.assertEqual(len(kept), 2)
        self.assertEqual([row["completion_tokens"] for row in kept], [10, 2])

    def test_analysis_rejects_truncated_row_even_with_stored_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            custom_id = "ladder__T1_vs_T2__AB"
            input_path.write_text(
                json.dumps({"custom_id": custom_id}) + "\n",
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "answer": "A",
                        "finish_reason": "max_output_tokens",
                        "content": "I would prefer Outcome A because...",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )
            with (
                patch(f"{module}.resolve_model_input_path", return_value=input_path),
                patch(f"{module}.sync_artifacts_to_input"),
                patch(
                    f"{module}.load_input_request_maps",
                    return_value=({custom_id}, {}),
                ),
                patch(f"{module}.backfill_output_answers", return_value=0),
                patch(f"{module}.resolve_model_output_path", return_value=output_path),
                patch(
                    f"{module}.load_ladders",
                    return_value=[
                        {"original_statement_id": "ladder", "valence": "positive"}
                    ],
                ),
            ):
                with self.assertRaisesRegex(ValueError, r"incomplete=1"):
                    analyze("gpt-56-luna-thinking")

    def test_analysis_can_retain_and_disclose_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"
            custom_ids = [
                "ladder__T1_vs_T2__AB",
                "ladder__T1_vs_T2__BA",
                "ladder__T1_vs_T3__AB",
            ]
            input_path.write_text(
                "".join(json.dumps({"custom_id": cid}) + "\n" for cid in custom_ids),
                encoding="utf-8",
            )
            rows = [
                {
                    "custom_id": custom_ids[0],
                    "answer": "B",
                    "finish_reason": "stop",
                    "content": "B",
                },
                {
                    "custom_id": custom_ids[1],
                    "answer": "A",
                    "finish_reason": "max_output_tokens",
                    "content": "I would choose A because...",
                },
                {
                    "custom_id": custom_ids[2],
                    "answer": None,
                    "finish_reason": "stop",
                    "content": "I decline the forced choice.",
                },
            ]
            output_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )
            with (
                patch(f"{module}.resolve_model_input_path", return_value=input_path),
                patch(f"{module}.sync_artifacts_to_input"),
                patch(
                    f"{module}.load_input_request_maps",
                    return_value=(set(custom_ids), {}),
                ),
                patch(f"{module}.backfill_output_answers", return_value=0),
                patch(f"{module}.resolve_model_output_path", return_value=output_path),
                patch(
                    f"{module}.load_ladders",
                    return_value=[
                        {"original_statement_id": "ladder", "valence": "positive"}
                    ],
                ),
                patch(f"{module}.model_output_path", return_value=summary_path),
                patch(f"{module}.refresh_cost_artifacts", return_value=None),
            ):
                analyze("gpt-56-luna-thinking", retain_missing=True)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["analysis_policy"], "retain_missing")
            self.assertFalse(summary["complete_parseable_coverage"])
            self.assertEqual(summary["n_requests_expected"], 3)
            self.assertEqual(summary["n_requests_received"], 3)
            self.assertEqual(summary["n_requests_parseable"], 1)
            self.assertEqual(summary["n_requests_missing_from_scoring"], 2)
            self.assertEqual(summary["incomplete_response_count"], 1)
            self.assertEqual(summary["unparseable_response_count"], 1)
            self.assertEqual(summary["overall_correct"], 1)
            self.assertEqual(summary["overall_accuracy"], 1.0)
            self.assertEqual(
                summary["overall_accuracy_bounds"],
                {
                    "lower_missing_incorrect": 1 / 3,
                    "upper_missing_correct": 1.0,
                },
            )
            self.assertEqual(summary["per_ladder"][0]["n_expected"], 3)
            self.assertEqual(summary["per_ladder"][0]["n_missing"], 2)
            self.assertEqual(summary["by_distance"]["1"]["missing"], 1)
            self.assertEqual(summary["by_distance"]["2"]["missing"], 1)

    def test_partial_rows_are_rejected_before_output_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.jsonl"
            raw = {
                "custom_id": "ladder__T1_vs_T2__AB",
                "response": {"status_code": 200, "body": _responses_body("B")},
            }
            with (
                patch(
                    "llm_coherence.experiments.within_ladder.run_within_ladder_experiment.load_input_request_maps",
                    return_value=(
                        {"ladder__T1_vs_T2__AB", "ladder__T1_vs_T2__BA"},
                        {},
                    ),
                ),
                patch(
                    "llm_coherence.experiments.within_ladder.run_within_ladder_experiment.model_output_path",
                    return_value=str(output),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "missing=1"):
                    write_clean_and_cost_log([raw], "openai", "gpt-56-sol")
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
