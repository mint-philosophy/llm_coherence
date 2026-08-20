"""Crash/restart coverage for live within-ladder checkpointing."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llm_coherence.experiments.within_ladder.run_within_ladder_experiment import (
    LIVE_CHECKPOINT_NAME,
    LIVE_LOCK_NAME,
    _append_live_checkpoint_row,
    _atomic_write_jsonl,
    _durable_unlink,
    _exclusive_live_run_lock,
    _load_live_checkpoint_rows,
    run_live,
)


class _SimulatedProcessCrash(BaseException):
    """Bypass request-level ``except Exception`` like a hard interruption."""


class _FakeResponse:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "model": "qwen/qwen3.7-max",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": self._answer},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "cost": 0.0001,
            },
        }


class WithinLadderCheckpointTests(unittest.TestCase):
    def test_truncated_final_checkpoint_record_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / LIVE_CHECKPOINT_NAME
            first = {
                "custom_id": "ladder__T1_vs_T2__AB",
                "response": {"body": {"choices": []}},
            }
            checkpoint.write_bytes(
                (json.dumps(first) + "\n").encode("utf-8")
                + b'{"custom_id":"truncated'
            )

            rows = _load_live_checkpoint_rows(checkpoint)

            self.assertEqual(rows, [first])
            self.assertEqual(
                checkpoint.read_text(encoding="utf-8"),
                json.dumps(first) + "\n",
            )

            second = {
                "custom_id": "ladder__T1_vs_T2__BA",
                "response": {"body": {"choices": []}},
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                _append_live_checkpoint_row(handle, second)
            self.assertEqual(_load_live_checkpoint_rows(checkpoint), [first, second])

    def test_malformed_newline_terminated_final_record_is_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / LIVE_CHECKPOINT_NAME
            first = {
                "custom_id": "ladder__T1_vs_T2__AB",
                "response": {"body": {"choices": []}},
            }
            original = (
                (json.dumps(first) + "\n").encode("utf-8")
                + b'{"custom_id":"malformed"\n'
            )
            checkpoint.write_bytes(original)

            with self.assertRaisesRegex(
                ValueError, r"Corrupt checkpoint record at line 2"
            ):
                _load_live_checkpoint_rows(checkpoint)

            self.assertEqual(checkpoint.read_bytes(), original)

    def test_live_run_refuses_lock_contention_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            lock_path = root / LIVE_LOCK_NAME
            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )

            def artifact_path(_model_key: str, artifact: str) -> str:
                return str(root / artifact)

            with (
                _exclusive_live_run_lock(lock_path, "qwen-37-max-openrouter"),
                patch(f"{module}.model_output_path", side_effect=artifact_path),
                patch(f"{module}._run_live_locked") as run_locked,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"Another --run-live process holds the model lock"
                ):
                    run_live("qwen-37-max-openrouter", concurrency=1)

            run_locked.assert_not_called()

    def test_live_run_rejects_checkpoint_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            checkpoint_path = root / LIVE_CHECKPOINT_NAME
            custom_id = "ladder__T1_vs_T2__AB"
            request = {
                "custom_id": custom_id,
                "body": {
                    "model": "qwen/qwen3.7-max",
                    "messages": [{"role": "user", "content": "choose"}],
                    "max_tokens": 16,
                    "temperature": 0,
                    "reasoning": {"enabled": False},
                },
            }
            input_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            stale_row = {
                "custom_id": custom_id,
                "_run_fingerprint": "0" * 64,
                "response": {"body": {"choices": []}},
            }
            checkpoint_path.write_text(
                json.dumps(stale_row) + "\n", encoding="utf-8"
            )
            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )

            def artifact_path(_model_key: str, artifact: str) -> str:
                return str(root / artifact)

            with (
                patch(f"{module}.model_output_path", side_effect=artifact_path),
                patch(f"{module}.sync_artifacts_to_input") as sync,
                patch(f"{module}.require_api_key") as require_key,
            ):
                with self.assertRaisesRegex(
                    ValueError, r"Checkpoint fingerprint mismatch"
                ):
                    run_live("qwen-37-max-openrouter", concurrency=1)

            sync.assert_not_called()
            require_key.assert_not_called()

    def test_atomic_replace_and_unlink_fsync_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "checkpoint.jsonl"
            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )
            rows = [{"custom_id": "ladder__T1_vs_T2__AB"}]

            with patch(f"{module}._fsync_parent_directory") as fsync_parent:
                _atomic_write_jsonl(destination, rows)
                fsync_parent.assert_called_once_with(destination)

                fsync_parent.reset_mock()
                self.assertTrue(_durable_unlink(destination))
                fsync_parent.assert_called_once_with(destination)

    def test_live_run_resumes_only_missing_request_after_process_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            checkpoint_path = root / LIVE_CHECKPOINT_NAME
            custom_ids = [
                "ladder__T1_vs_T2__AB",
                "ladder__T1_vs_T2__BA",
            ]
            requests = [
                {
                    "custom_id": custom_id,
                    "body": {
                        "messages": [{"role": "user", "content": "choose"}],
                        "max_tokens": 16,
                    },
                }
                for custom_id in custom_ids
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in requests),
                encoding="utf-8",
            )

            module = (
                "llm_coherence.experiments.within_ladder."
                "run_within_ladder_experiment"
            )

            def artifact_path(_model_key: str, artifact: str) -> str:
                return str(root / artifact)

            first_run_calls = 0

            class CrashingClient:
                def __init__(self, **_kwargs) -> None:
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args) -> bool:
                    return False

                async def post(self, *_args, **_kwargs):
                    nonlocal first_run_calls
                    first_run_calls += 1
                    if first_run_calls == 1:
                        return _FakeResponse("B")
                    raise _SimulatedProcessCrash("simulated interruption")

            def apply_common_patches(stack: ExitStack) -> None:
                stack.enter_context(
                    patch(f"{module}.model_output_path", side_effect=artifact_path)
                )
                stack.enter_context(patch(f"{module}.sync_artifacts_to_input"))
                stack.enter_context(
                    patch(
                        f"{module}.load_input_request_maps",
                        return_value=(set(custom_ids), {}),
                    )
                )
                stack.enter_context(
                    patch(f"{module}.print_pre_submit_live_cost_estimate")
                )
                stack.enter_context(
                    patch(f"{module}.require_api_key", return_value="test-token")
                )
                stack.enter_context(
                    patch(f"{module}.persist_per_request_cost_log")
                )

            with ExitStack() as stack:
                apply_common_patches(stack)
                stack.enter_context(patch("httpx.AsyncClient", CrashingClient))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaises(_SimulatedProcessCrash):
                    run_live("qwen-37-max-openrouter", concurrency=1)

            self.assertFalse(output_path.exists())
            checkpoint_rows = _load_live_checkpoint_rows(checkpoint_path)
            self.assertEqual(
                [row["custom_id"] for row in checkpoint_rows],
                [custom_ids[0]],
            )

            second_run_calls = 0

            class RecoveryClient:
                def __init__(self, **_kwargs) -> None:
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args) -> bool:
                    return False

                async def post(self, *_args, **_kwargs):
                    nonlocal second_run_calls
                    second_run_calls += 1
                    return _FakeResponse("A")

            stdout = io.StringIO()
            with ExitStack() as stack:
                apply_common_patches(stack)
                stack.enter_context(patch("httpx.AsyncClient", RecoveryClient))
                stack.enter_context(redirect_stdout(stdout))
                run_live("qwen-37-max-openrouter", concurrency=1)

            self.assertEqual(second_run_calls, 1)
            self.assertFalse(checkpoint_path.exists())
            output_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["custom_id"] for row in output_rows],
                custom_ids,
            )
            self.assertEqual([row["answer"] for row in output_rows], ["B", "A"])
            self.assertIn("Recovered 1 checkpoint record", stdout.getvalue())
            self.assertIn("checkpoint cleared", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
