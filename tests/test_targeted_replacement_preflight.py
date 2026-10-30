"""Replacement scopes must fail before any output if source bindings drift."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from llm_coherence.analysis.prepare_targeted_replacements import prepare_bundle, verified_run_argv
from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.experiments.ladder_statement_pair import run_7tier_experiment as runner
from llm_coherence.runtime.agents import model_name_for_key


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplacementPreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = self.root / "bundle"
        self.plan_path = self.root / "plan.json"
        self.tasks = [self.task("glm-45-base-logprobs"), self.task("llama-31-8b-instruct-openrouter")]

    def task(self, model_key):
        local = model_key == "glm-45-base-logprobs"
        name = "phase6b_variations_pruned_final_Wellbeing_1"
        path = self.root / f"{name}_comparisons.json"
        inputs = {
            "test_name": name, "variation_id": "Wellbeing_1", "n_tiers": 2,
            "n_comparison_statements": 2, "num_comparisons": 4,
            "comparisons": [
                {"outcome_a": {"text": "A child’s wellbeing improves—at home.",
                               "tier": tier, "variation_id": "Wellbeing_1"},
                 "outcome_b": {"text": "Another outcome.", "comparison_id": comparison}}
                for comparison in (0, 1) for tier in (1, 2)
            ],
        }
        path.write_text(json.dumps(inputs, ensure_ascii=False), encoding="utf-8")
        config = {
            "test_name": name, "model_key": model_key, "model_variant": "base" if local else "instruct",
            "reasoning_mode": "none", "num_trials": 10, "include_flipped": True,
            "with_reasoning": False, "max_tokens": 10, "temperature": 1.0 if local else 0.0,
            "k_samples": 1, "infrastructure": "hf_jobs" if local else "openai_api",
            "gpu_type": "H200-8x" if local else None, "gpu_count": 8 if local else None,
            "quantization": None,
        }
        # Exercise compatibility with original metadata lacking the routing prefix.
        provider = model_name_for_key(model_key).removeprefix("openrouter/")
        result_path = self.root / f"{model_key}.json"
        result_path.write_text(json.dumps({
            "config": config, "metadata": {
                "model_name_full": provider, "extra_body": MODEL_CONFIGS[model_key].extra_body,
                "system_message": "You are a helpful assistant.",
            },
        }))
        return {
            "model_key": model_key, "ladder_id": "Wellbeing_1",
            "canonical_input_path": str(path), "canonical_input_sha256": digest(path),
            "original_result_path": str(result_path), "original_result_sha256": digest(result_path),
            "original_config": config, "original_provider_model": provider,
            "original_extra_body": MODEL_CONFIGS[model_key].extra_body,
            "expected_scoring_calls": 8 if local else 80,
        }

    def prepare(self):
        self.plan_path.write_text(json.dumps({"reruns": self.tasks}))
        return prepare_bundle(self.plan_path, self.output)

    def test_exact_input_bytes_scopes_and_separate_output_paths(self):
        report = self.prepare()
        self.assertEqual(report["replacement_ladders"], 2)
        self.assertEqual(report["base_scoring_calls"], 8)
        self.assertEqual(report["sampled_requests_before_retries"], 80)
        for task in self.tasks:
            root = self.output / task["model_key"]
            manifest = runner.load_manifest(root / "manifest.json")
            items = runner.scoped_manifest_items(manifest, root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["comparison_path"].read_bytes(), Path(task["canonical_input_path"]).read_bytes())
            self.assertEqual(manifest["replacement_sources"], [task])
            argv = verified_run_argv(self.output, task["model_key"])
            self.assertEqual(Path(argv[argv.index("--results-dir") + 1]), (root / "results").resolve())
            self.assertNotIn("--submit-hf-job", argv)
            self.assertFalse((root / "results").exists())

    def test_generated_arguments_parse_and_preserve_original_settings(self):
        report = self.prepare()
        for model_key, entry in report["models"].items():
            argv = verified_run_argv(self.output, model_key)
            with self.subTest(model=model_key), redirect_stdout(io.StringIO()), \
                    patch.object(sys, "argv", [argv[2]] + argv[3:]), \
                    patch.object(runner, "run_phase6b_with_client_cleanup", new_callable=AsyncMock) as launch:
                runner.main()
            kwargs = launch.call_args.kwargs
            self.assertEqual(kwargs["model_key"], model_key)
            for field in ("num_trials", "max_tokens", "temperature", "with_reasoning", "model_variant"):
                self.assertEqual(kwargs[field], entry["settings"][field])
            self.assertEqual(kwargs["max_concurrent"], 1 if model_key == "glm-45-base-logprobs" else 3)
            self.assertTrue(kwargs["resume"])

    def test_bundle_relocates_without_the_original_files(self):
        report = self.prepare()
        relocated = self.root / "new_machine" / "bundle"
        shutil.copytree(self.output, relocated)
        shutil.rmtree(self.output)
        for path in {task[key] for task in self.tasks for key in ("canonical_input_path", "original_result_path")}:
            Path(path).unlink()
        for model_key, entry in report["models"].items():
            argv = verified_run_argv(relocated, model_key)
            self.assertEqual(argv[argv.index("--data-dir") + 1], str((relocated / model_key).resolve()))
            self.assertIn("--resume", argv)
            self.assertNotIn(str(self.output), " ".join(argv))
            self.assertEqual(entry["argv"][0], "python3")
            self.assertEqual(entry["argv"][2], "llm_coherence.analysis.prepare_targeted_replacements")

    def test_changed_bundle_input_is_rejected_before_launch(self):
        self.prepare()
        key = self.tasks[0]["model_key"]
        path = next((self.output / key / "inputs").glob("*.json"))
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Bundled input hash mismatch"):
            verified_run_argv(self.output, key)

    def test_changed_bundle_settings_are_rejected_before_launch(self):
        self.prepare()
        key = self.tasks[0]["model_key"]
        path = self.output / key / "run_arguments.json"
        entry = json.loads(path.read_text())
        entry["settings"]["temperature"] = 0.5
        path.write_text(json.dumps(entry))
        with self.assertRaisesRegex(ValueError, "run settings changed"):
            verified_run_argv(self.output, key)

    def test_changed_provider_settings_are_rechecked_at_launch(self):
        self.prepare()
        key = self.tasks[0]["model_key"]
        replacement = type("Config", (), {"extra_body": {"new_setting": True}, "system_message": None})()
        with patch.dict(MODEL_CONFIGS, {key: replacement}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                verified_run_argv(self.output, key)

    def test_modified_source_or_input_fails_before_creating_bundle(self):
        for prefix in ("canonical_input", "original_result"):
            path = Path(self.tasks[0][f"{prefix}_path"])
            original = path.read_bytes()
            with self.subTest(prefix=prefix):
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    self.prepare()
                self.assertFalse(self.output.exists())
            path.write_bytes(original)

    def test_duplicate_tasks_fail_before_creating_bundle(self):
        self.tasks.append(dict(self.tasks[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate replacement"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_changed_provider_config_is_rejected(self):
        key = self.tasks[0]["model_key"]
        replacement = type("Config", (), {"extra_body": {"new_setting": True}})()
        with patch.dict(MODEL_CONFIGS, {key: replacement}):
            with self.assertRaisesRegex(ValueError, "Current provider settings differ"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_tier_pair_is_rejected_even_if_input_hash_matches(self):
        path = Path(self.tasks[0]["canonical_input_path"])
        data = json.loads(path.read_text())
        data["comparisons"][1] = data["comparisons"][0]
        path.write_text(json.dumps(data))
        for task in self.tasks:
            task["canonical_input_sha256"] = digest(path)
        with self.assertRaisesRegex(ValueError, "tier/comparator pairs"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_inconsistent_plan_calls_are_rejected(self):
        self.tasks[0]["expected_scoring_calls"] = 80
        with self.assertRaisesRegex(ValueError, "Planned call count differs"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_shuffled_complete_grid_is_rejected_before_writing(self):
        path = Path(self.tasks[0]["canonical_input_path"])
        data = json.loads(path.read_text())
        data["comparisons"][0], data["comparisons"][1] = data["comparisons"][1], data["comparisons"][0]
        path.write_text(json.dumps(data))
        for task in self.tasks:
            task["canonical_input_sha256"] = digest(path)
        with self.assertRaisesRegex(ValueError, "group ascending tiers by comparator"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_destination_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "existing_result.json"
        sentinel.write_text("original results")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "original results")


if __name__ == "__main__":
    unittest.main()
