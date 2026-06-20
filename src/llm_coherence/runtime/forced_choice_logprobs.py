"""Strict helpers for binary forced-choice scoring from next-token logprobs."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping


class ForcedChoiceScoringError(RuntimeError):
    """Raised when exact A/B probabilities cannot be recovered safely."""


def vllm_load_kwargs_from_env() -> dict[str, Any]:
    """Return optional vLLM loading overrides used by HF Jobs.

    Mounted Hub repositories use a FUSE filesystem. For very large models,
    vLLM's default lazy mmap strategy can spend hours on random reads, so the
    HF submitters select sequential multithreaded prefetching via environment
    variables while local runs retain vLLM defaults.
    """

    kwargs: dict[str, Any] = {}
    strategy = os.environ.get("LLM_COHERENCE_SAFETENSORS_LOAD_STRATEGY")
    if strategy:
        if strategy not in {"lazy", "eager", "prefetch"}:
            raise ValueError(f"Unsupported safetensors load strategy: {strategy!r}")
        kwargs["safetensors_load_strategy"] = strategy

    threads = os.environ.get("LLM_COHERENCE_SAFETENSORS_PREFETCH_THREADS")
    if threads:
        kwargs["safetensors_prefetch_num_threads"] = int(threads)

    max_model_len = os.environ.get("LLM_COHERENCE_MAX_MODEL_LEN")
    if max_model_len:
        kwargs["max_model_len"] = int(max_model_len)
    return kwargs


def resolve_choice_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve ``" A"`` and ``" B"`` to distinct single-token IDs.

    The experiment scores the token immediately following ``"Answer:"``. A
    multi-token label would make a one-step logprob comparison invalid, so this
    helper fails closed rather than silently using only the first token.
    """

    token_ids: dict[str, int] = {}
    for label in ("A", "B"):
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(encoded) != 1:
            raise ForcedChoiceScoringError(
                f"Forced-choice label {label!r} must encode as exactly one token; "
                f"got token IDs {encoded!r}."
            )
        token_ids[label] = int(encoded[0])

    if token_ids["A"] == token_ids["B"]:
        raise ForcedChoiceScoringError(
            "Forced-choice labels 'A' and 'B' resolved to the same token ID."
        )
    return token_ids["A"], token_ids["B"]


def _logprob_value(value: Any, *, label: str) -> float:
    raw = getattr(value, "logprob", value)
    try:
        score = float(raw)
    except (TypeError, ValueError) as exc:
        raise ForcedChoiceScoringError(
            f"Invalid logprob value for label {label}: {raw!r}."
        ) from exc
    if not math.isfinite(score):
        raise ForcedChoiceScoringError(
            f"Non-finite logprob value for label {label}: {score!r}."
        )
    return score


def normalized_choice_probabilities(
    top_logprobs: Mapping[int, Any],
    token_id_a: int,
    token_id_b: int,
) -> dict[str, float]:
    """Return normalized A/B probabilities, requiring both exact logprobs."""

    missing = [
        label
        for label, token_id in (("A", token_id_a), ("B", token_id_b))
        if token_id not in top_logprobs
    ]
    if missing:
        raise ForcedChoiceScoringError(
            "Missing constrained logprob value(s) for " + ", ".join(missing) + "."
        )

    score_a = _logprob_value(top_logprobs[token_id_a], label="A")
    score_b = _logprob_value(top_logprobs[token_id_b], label="B")
    maximum = max(score_a, score_b)
    exp_a = math.exp(score_a - maximum)
    exp_b = math.exp(score_b - maximum)
    total = exp_a + exp_b
    if not math.isfinite(total) or total <= 0:
        raise ForcedChoiceScoringError(
            f"Could not normalize A/B logprobs: A={score_a}, B={score_b}."
        )
    return {"A": exp_a / total, "B": exp_b / total}
