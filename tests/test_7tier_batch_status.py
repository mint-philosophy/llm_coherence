"""Tests for generic seven-tier OpenAI Batch status reporting."""

from __future__ import annotations

import unittest

from llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment import (
    aggregate_batch_progress,
    batch_job_progress,
    format_batch_job_status,
)


class BatchStatusFormattingTests(unittest.TestCase):
    def test_job_status_reports_total_remaining_and_percentage(self) -> None:
        entry = {
            "input_file": "batch_input_000.jsonl",
            "batch_id": "batch-test",
            "status": "in_progress",
            "request_count": 2_100,
            "request_counts": {
                "total": 2_100,
                "completed": 149,
                "failed": 0,
            },
        }

        self.assertEqual(
            batch_job_progress(entry),
            {
                "total": 2_100,
                "completed": 149,
                "failed": 0,
                "processed": 149,
                "remaining": 1_951,
                "percent": 149 / 2_100 * 100.0,
            },
        )
        self.assertEqual(
            format_batch_job_status(entry),
            "batch_input_000.jsonl -> batch-test: in_progress | "
            "processed=149/2,100 (7.1%), completed=149, failed=0, "
            "remaining=1,951",
        )

    def test_job_status_falls_back_to_recorded_shard_size(self) -> None:
        progress = batch_job_progress(
            {
                "request_count": 42_000,
                "request_counts": {"total": 0, "completed": 182, "failed": 0},
            }
        )

        self.assertEqual(progress["total"], 42_000)
        self.assertEqual(progress["remaining"], 41_818)

    def test_aggregate_includes_completed_and_failed_requests(self) -> None:
        overall = aggregate_batch_progress(
            [
                {
                    "request_count": 100,
                    "request_counts": {"total": 100, "completed": 90, "failed": 5},
                },
                {
                    "request_count": 50,
                    "request_counts": {"total": 50, "completed": 20, "failed": 0},
                },
            ]
        )

        self.assertEqual(overall["total"], 150)
        self.assertEqual(overall["completed"], 110)
        self.assertEqual(overall["failed"], 5)
        self.assertEqual(overall["processed"], 115)
        self.assertEqual(overall["remaining"], 35)
        self.assertAlmostEqual(overall["percent"], 76.6666666667)


if __name__ == "__main__":
    unittest.main()
