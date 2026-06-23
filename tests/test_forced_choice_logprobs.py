"""Tests for exact A/B next-token logprob scoring."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from llm_coherence.runtime.forced_choice_logprobs import (
    ForcedChoiceScoringError,
    normalized_choice_probabilities,
    resolve_choice_token_ids,
    vllm_load_kwargs_from_env,
)


class FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]):
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]


@dataclass
class FakeLogprob:
    logprob: float


class ResolveChoiceTokenIdsTests(unittest.TestCase):
    def test_resolves_distinct_single_tokens(self) -> None:
        tokenizer = FakeTokenizer({" A": [10], " B": [11]})
        self.assertEqual(resolve_choice_token_ids(tokenizer), (10, 11))

    def test_rejects_multi_token_label(self) -> None:
        tokenizer = FakeTokenizer({" A": [10, 12], " B": [11]})
        with self.assertRaisesRegex(ForcedChoiceScoringError, "exactly one token"):
            resolve_choice_token_ids(tokenizer)

    def test_rejects_identical_token_ids(self) -> None:
        tokenizer = FakeTokenizer({" A": [10], " B": [10]})
        with self.assertRaisesRegex(ForcedChoiceScoringError, "same token ID"):
            resolve_choice_token_ids(tokenizer)


class NormalizeChoiceProbabilitiesTests(unittest.TestCase):
    def test_normalizes_both_exact_logprobs(self) -> None:
        probabilities = normalized_choice_probabilities(
            {10: FakeLogprob(-2.0), 11: FakeLogprob(-1.0)}, 10, 11
        )
        self.assertAlmostEqual(probabilities["A"] + probabilities["B"], 1.0)
        self.assertAlmostEqual(probabilities["A"], 1.0 / (1.0 + math.e))
        self.assertGreater(probabilities["B"], probabilities["A"])

    def test_rejects_a_missing_choice(self) -> None:
        with self.assertRaisesRegex(ForcedChoiceScoringError, "B"):
            normalized_choice_probabilities({10: FakeLogprob(-2.0)}, 10, 11)

    def test_rejects_non_finite_logprob(self) -> None:
        with self.assertRaisesRegex(ForcedChoiceScoringError, "Non-finite"):
            normalized_choice_probabilities(
                {10: FakeLogprob(float("-inf")), 11: FakeLogprob(-1.0)}, 10, 11
            )


class VllmLoadKwargsTests(unittest.TestCase):
    def test_reads_hf_job_loading_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_COHERENCE_SAFETENSORS_LOAD_STRATEGY": "prefetch",
                "LLM_COHERENCE_SAFETENSORS_PREFETCH_THREADS": "16",
                "LLM_COHERENCE_MAX_MODEL_LEN": "4096",
            },
            clear=True,
        ):
            self.assertEqual(
                vllm_load_kwargs_from_env(),
                {
                    "safetensors_load_strategy": "prefetch",
                    "safetensors_prefetch_num_threads": 16,
                    "max_model_len": 4096,
                },
            )

    def test_rejects_unknown_loading_strategy(self) -> None:
        with patch.dict(
            "os.environ",
            {"LLM_COHERENCE_SAFETENSORS_LOAD_STRATEGY": "surprise"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                vllm_load_kwargs_from_env()


if __name__ == "__main__":
    unittest.main()
