"""Cost estimates and lightweight preflight helpers for experiment runs."""

from __future__ import annotations

from pathlib import Path


# Approximate cost per 1M tokens in USD.
MODEL_COST_ESTIMATES = {
    "claude-opus-openrouter": {"input": 5.0, "output": 25.0},
    "claude-opus": {"input": 5.0, "output": 25.0},
    "gpt-4o-mini-openrouter": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "gpt-55": {"input": 5.0, "output": 30.0},
    "gpt-55-openai": {"input": 5.0, "output": 30.0},
    "gpt-54-nano": {"input": 0.20, "output": 1.25},
    "gpt-54-nano-thinking": {"input": 0.20, "output": 1.25},
    "gpt-54-mini": {"input": 0.75, "output": 4.50},
    "gpt-54-mini-thinking": {"input": 0.75, "output": 4.50},
    "gpt-54": {"input": 2.50, "output": 15.0},
    "gpt-54-thinking": {"input": 2.50, "output": 15.0},
    "opus-46": {"input": 5.0, "output": 25.0},
    "opus-46-thinking": {"input": 5.0, "output": 25.0},
    "opus-47": {"input": 5.0, "output": 25.0},
    "opus-47-thinking": {"input": 5.0, "output": 25.0},
    "gemini-31-pro": {"input": 1.25, "output": 10.0},
    "gemini-31-pro-thinking": {"input": 1.25, "output": 10.0},
    "deepseek-v31-hybrid": {"input": 0.55, "output": 2.19},
    "deepseek-v31-hybrid-thinking": {"input": 0.55, "output": 2.19},
    "gemini-25-pro": {"input": 1.25, "output": 10.0},
    "gemini-25-pro-thinking": {"input": 1.25, "output": 10.0},
    "nemotron-3-super": {"input": 0.15, "output": 0.40},
    "nemotron-3-super-thinking": {"input": 0.15, "output": 0.40},
    "glm-45-hybrid": {"input": 0.60, "output": 2.20},
    "glm-45-hybrid-thinking": {"input": 0.60, "output": 2.20},
}

EXPENSIVE_MODELS = {
    "claude-opus-openrouter",
    "claude-opus",
    "opus-46",
    "opus-46-thinking",
    "gpt-54",
    "gpt-54-thinking",
    "gpt-55",
    "gpt-55-openai",
    "gemini-25-pro-thinking",
}

GPU_COST_PER_HOUR = {
    "T4-small": 0.40,
    "T4": 0.60,
    "A100": 2.50,
    "H200": 5.00,
    "H200-8x": 40.00,
}

SELF_HOSTED_MODELS = {
    "qwen25-05b-base",
    "qwen25-7b-base",
    "glm-45-base",
    "glm-45-base-logprobs",
}

AVG_INPUT_TOKENS = 200
AVG_OUTPUT_TOKENS_NO_COT = 3
AVG_OUTPUT_TOKENS_COT = 250
AVG_OUTPUT_TOKENS_THINKING = 1500

CALIBRATED_OUTPUT_TOKENS: dict[str, int] = {
    "gpt-54-nano-thinking": 45,
    "gpt-54-mini-thinking": 82,
    "gpt-54-thinking": 121,
    "nemotron-3-super-thinking": 158,
    "glm-45-hybrid-thinking": 386,
    "opus-47-thinking": 300,
    "gemini-31-pro-thinking": 200,
    "deepseek-v31-hybrid-thinking": 250,
}


def is_thinking_run(model_key: str) -> bool:
    """Whether the model has native reasoning enabled."""
    return model_key.endswith("-thinking")


def estimate_cost(
    model_key: str,
    total_api_calls: int,
    with_reasoning: bool,
) -> float | None:
    """Estimate total cost in USD. Returns None if the model is not priced."""
    costs = MODEL_COST_ESTIMATES.get(model_key)
    if not costs:
        return None
    if is_thinking_run(model_key):
        avg_output = CALIBRATED_OUTPUT_TOKENS.get(model_key, AVG_OUTPUT_TOKENS_THINKING)
    elif with_reasoning:
        avg_output = AVG_OUTPUT_TOKENS_COT
    else:
        avg_output = AVG_OUTPUT_TOKENS_NO_COT
    input_cost = (total_api_calls * AVG_INPUT_TOKENS / 1_000_000) * costs["input"]
    output_cost = (total_api_calls * avg_output / 1_000_000) * costs["output"]
    return input_cost + output_cost


def estimate_gpu_cost(
    gpu_type: str,
    gpu_count: int,
    estimated_gpu_hours: float,
) -> float | None:
    """Estimate GPU cost in USD for self-hosted runs."""
    rate = GPU_COST_PER_HOUR.get(gpu_type)
    if rate is None:
        return None
    return rate * gpu_count * estimated_gpu_hours


def run_preflight_checks(
    model_key: str,
    num_trials: int,
    max_tokens: int,
    with_reasoning: bool,
    total_api_calls: int,
    resume: bool = True,
    max_concurrent: int | None = None,
    completed_sets: int = 0,
    total_sets: int = 0,
    results_dir: Path | None = None,
    max_expected_cost: float | None = None,
    gpu_type: str | None = None,
    gpu_count: int = 1,
    estimated_gpu_hours: float | None = None,
    auto_confirm: bool = False,
    full_run_api_calls: int | None = None,
) -> None:
    """Print warnings for common costly run configurations."""
    del resume, max_concurrent, completed_sets, total_sets, results_dir
    del auto_confirm, full_run_api_calls
    warnings: list[str] = []

    if with_reasoning and max_tokens > 50:
        warnings.append(
            f"CoT reasoning is enabled with max_tokens={max_tokens}; cost will be higher."
        )
    if not with_reasoning and max_tokens > 50 and not is_thinking_run(model_key):
        warnings.append(
            f"with_reasoning is off but max_tokens={max_tokens}; binary choices usually need 10."
        )
    if is_thinking_run(model_key) and max_tokens < 500:
        warnings.append(
            f"Thinking-on model {model_key!r} may need max_tokens >= 1500."
        )
    if num_trials > 15:
        warnings.append(f"num_trials={num_trials} is high; cost scales linearly.")

    est = estimate_cost(model_key, total_api_calls, with_reasoning)
    if max_expected_cost is not None and est is not None and est > max_expected_cost:
        warnings.append(
            f"Estimated cost ${est:.2f} exceeds max_expected_cost ${max_expected_cost:.2f}."
        )
    if gpu_type and estimated_gpu_hours is not None:
        gpu_est = estimate_gpu_cost(gpu_type, gpu_count, estimated_gpu_hours)
        if gpu_est is not None:
            warnings.append(f"Estimated GPU cost: ${gpu_est:.2f}.")

    for warning in warnings:
        print(f"[preflight] {warning}")

