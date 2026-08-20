"""
config.py

Configuration for parametric variation experiments.
Centralized settings for models, API parameters, and experiment parameters.

Trimmed to the publication workflow for llm_coherence. The public README
separates the 15 model configurations reported in the paper from audit judges
and local support or exploratory model keys.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Model configuration
GPT56_RESPONSES_MIN_OUTPUT_TOKENS = 16


def validate_openai_responses_max_output_tokens(
    model_id: str, max_output_tokens: int
) -> None:
    """Reject GPT-5.6 Responses requests below the provider's enforced floor."""
    if (
        model_id.startswith("gpt-5.6")
        and max_output_tokens < GPT56_RESPONSES_MIN_OUTPUT_TOKENS
    ):
        raise ValueError(
            f"{model_id} requires max_output_tokens >= "
            f"{GPT56_RESPONSES_MIN_OUTPUT_TOKENS}; got {max_output_tokens}. "
            "This is enforced by the OpenAI Responses API."
        )



@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_key: str  # Key in models.yaml
    temperature: float = 0.0
    max_tokens: int = 10
    concurrency_limit: int = 50
    base_timeout: float = 5.0
    extra_body: Optional[dict] = None  # Provider-specific params (reasoning_effort, thinking, etc.)
    enable_cache: bool = False  # Enable prompt caching (Anthropic via OpenRouter)
    # Native provider model name for APIs that need it. For Anthropic models this
    # is "claude-opus-4-6" / "claude-opus-4-7", not the model_key.
    model_name_full: Optional[str] = None
    # System message for forced-choice elicitation. Matches what nemotron / GLM /
    # Llama OFF runs used so cross-family comparisons stay apples-to-apples.
    system_message: Optional[str] = None
    # Declarative provenance of the reasoning artifact this model produces.
    # Values: none | prose_justification | summary | raw_cot | unverified
    # Used by reporting/inventory code to describe reasoning artifacts.
    reasoning_artifact_type: str = "none"

    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"Invalid max_tokens: {self.max_tokens}")


# Predefined model configurations (paper slate)

MODEL_CONFIGS = {
    # GPT 5.4 (OpenAI direct API)
    # reasoning_effort: "none" = no native reasoning, "high" = full reasoning
    'gpt-54-nano': ModelConfig(
        model_key='gpt-54-nano',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=100,
        base_timeout=5.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-54-nano-thinking': ModelConfig(
        model_key='gpt-54-nano-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=100,
        base_timeout=10.0,
        extra_body={"reasoning_effort": "high"},
        reasoning_artifact_type="summary",
    ),
    'gpt-54-mini': ModelConfig(
        model_key='gpt-54-mini',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=100,
        base_timeout=5.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-54-mini-thinking': ModelConfig(
        model_key='gpt-54-mini-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=100,
        base_timeout=10.0,
        extra_body={"reasoning_effort": "high"},
        reasoning_artifact_type="summary",
    ),
    'gpt-54': ModelConfig(
        model_key='gpt-54',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=50,
        base_timeout=10.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-54-thinking': ModelConfig(
        model_key='gpt-54-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=50,
        base_timeout=15.0,
        extra_body={"reasoning_effort": "high"},
        reasoning_artifact_type="summary",
    ),

    # GPT 5.5 (OpenAI direct API) — also serves as the within-ladder pruning judge.
    'gpt-55-openai': ModelConfig(
        model_key='gpt-55-openai',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=50,
        base_timeout=10.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),

    # GPT-5.6 Sol (OpenAI direct API). GPT-5.6 defaults to medium reasoning,
    # so the off condition must send reasoning_effort="none" explicitly.
    # OpenAI exposes reasoning-token usage and opt-in summaries, not the raw
    # private chain of thought, through the Responses Batch API path used here.
    'gpt-56-sol': ModelConfig(
        model_key='gpt-56-sol',
        temperature=0.0,
        # GPT-5.4 used 10; GPT-5.6 Responses enforces a minimum of 16.
        max_tokens=GPT56_RESPONSES_MIN_OUTPUT_TOKENS,
        concurrency_limit=50,
        base_timeout=15.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-56-sol-thinking': ModelConfig(
        model_key='gpt-56-sol-thinking',
        temperature=0.0,
        # Preserve the 3,000-token ceiling used by the completed GPT-5.6
        # reasoning-on runs so regenerated requests match their Batch manifests.
        max_tokens=3000,
        concurrency_limit=50,
        base_timeout=30.0,
        extra_body={
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
        },
        # OpenAI does not expose private reasoning tokens. Request the most
        # detailed supported reasoning summary as a separate response item;
        # the manuscript prompt still elicits its visible prose justification.
        reasoning_artifact_type="summary",
    ),
    'gpt-56-terra': ModelConfig(
        model_key='gpt-56-terra',
        temperature=0.0,
        # GPT-5.4 Mini used 10; GPT-5.6 Responses enforces a minimum of 16.
        max_tokens=GPT56_RESPONSES_MIN_OUTPUT_TOKENS,
        concurrency_limit=75,
        base_timeout=12.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-56-terra-thinking': ModelConfig(
        model_key='gpt-56-terra-thinking',
        temperature=0.0,
        # Match the completed GPT-5.6 reasoning-on Batch protocol.
        max_tokens=3000,
        concurrency_limit=75,
        base_timeout=30.0,
        extra_body={
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
        },
        reasoning_artifact_type="summary",
    ),
    'gpt-56-luna': ModelConfig(
        model_key='gpt-56-luna',
        temperature=0.0,
        # GPT-5.4 Nano used 10; GPT-5.6 Responses enforces a minimum of 16.
        max_tokens=GPT56_RESPONSES_MIN_OUTPUT_TOKENS,
        concurrency_limit=100,
        base_timeout=10.0,
        extra_body={"reasoning_effort": "none"},
        reasoning_artifact_type="none",
    ),
    'gpt-56-luna-thinking': ModelConfig(
        model_key='gpt-56-luna-thinking',
        temperature=0.0,
        # Match the completed GPT-5.6 reasoning-on Batch protocol.
        max_tokens=3000,
        concurrency_limit=100,
        base_timeout=30.0,
        extra_body={
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
        },
        reasoning_artifact_type="summary",
    ),

    # Opus 4.6 (via OpenRouter)
    # Extended thinking enabled by default on OpenRouter for Anthropic models.
    # Non-thinking: no extra_body needed (prompt template controls CoT).
    # Thinking: budget_tokens sets the thinking token budget (min 1024, calibrate).
    'opus-46': ModelConfig(
        model_key='opus-46',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=20,
        base_timeout=30.0,
        enable_cache=True,
        model_name_full="claude-opus-4-6",
        system_message="You are a helpful assistant.",
        reasoning_artifact_type="none",
    ),
    'opus-46-thinking': ModelConfig(
        model_key='opus-46-thinking',
        temperature=1.0,        # Anthropic requires temp=1 with extended thinking
        max_tokens=2048,        # must exceed budget_tokens (1024) + answer headroom
        concurrency_limit=20,
        base_timeout=120.0,     # extended thinking is slower than non-thinking
        enable_cache=True,
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        model_name_full="claude-opus-4-6",
        system_message="You are a helpful assistant.",
        # Claude 4 thinking blocks expose Anthropic-generated summaries of the
        # full private thinking process, not raw chain-of-thought.
        reasoning_artifact_type="summary",
    ),

    # OpenRouter (Nemotron)
    # Nemotron: reasoning via <think> tags, enabled by default
    'nemotron-3-super': ModelConfig(
        model_key='nemotron-3-super',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=50,
        base_timeout=10.0,
        extra_body={"reasoning": {"enabled": False}},  # Nemotron is a hybrid; without this it tries to reason and truncates at max_tokens=10 returning partial reasoning text instead of "A"/"B"
    ),
    'nemotron-3-super-thinking': ModelConfig(
        model_key='nemotron-3-super-thinking',
        temperature=0.0,
        max_tokens=3000,   # bumped from 150 to accommodate reasoning tokens
        concurrency_limit=20,
        base_timeout=30.0,  # bumped from 15s for reasoning latency
        extra_body={"reasoning": {"enabled": True}, "provider": {"order": ["nvidia"]}},
    ),

    # GLM 4.5 hybrid via OpenRouter (z-ai/glm-4.5). Free-form sampling +
    # regex parser, same code path as Nemotron. Reasoning toggled
    # via extra_body.reasoning.enabled.
    'glm-45-hybrid': ModelConfig(
        model_key='glm-45-hybrid',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=50,
        base_timeout=15.0,
        extra_body={"reasoning": {"enabled": False}},
    ),
    'glm-45-hybrid-thinking': ModelConfig(
        model_key='glm-45-hybrid-thinking',
        temperature=0.0,
        max_tokens=3000,  # calibrate on smoke
        concurrency_limit=50,
        base_timeout=30.0,
        # Provider pin to Z.AI's official endpoint to guarantee reasoning is honored.
        extra_body={
            "reasoning": {"enabled": True},
            "provider": {"order": ["Z.AI"]},
        },
    ),

    # GLM 4.5 base (logprob-scored). Greedy decoding via temp=1 + single-stream
    # (concurrency 1) since logprob extraction runs one request at a time.
    'glm-45-base-logprobs': ModelConfig(
        model_key='glm-45-base-logprobs',
        temperature=1.0,
        max_tokens=10,
        concurrency_limit=1,
        base_timeout=60.0,
    ),

    # Local / OpenRouter small models
    'llama-31-8b-instruct-openrouter': ModelConfig(
        model_key='llama-31-8b-instruct-openrouter',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=100,  # small fast model — high concurrency safe
        base_timeout=10.0,
    ),

    # Mistral (via OpenRouter)
    'ministral-3b-2512-openrouter': ModelConfig(
        model_key='ministral-3b-2512-openrouter',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=100,
        base_timeout=10.0,
        reasoning_artifact_type="none",
    ),
    'mistral-small-2603-openrouter-thinking': ModelConfig(
        model_key='mistral-small-2603-openrouter-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=50,
        base_timeout=30.0,
        extra_body={"reasoning": {"enabled": True}},
        reasoning_artifact_type="raw_cot",
    ),
    'kimi-k2-openrouter': ModelConfig(
        model_key='kimi-k2-openrouter',
        temperature=0.0,
        max_tokens=10,
        concurrency_limit=20,
        base_timeout=30.0,
        reasoning_artifact_type="none",
    ),
    'kimi-k2-openrouter-thinking': ModelConfig(
        model_key='kimi-k2-openrouter-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=20,
        base_timeout=120.0,
        reasoning_artifact_type="raw_cot",
    ),
    # Kimi K3 via OpenRouter (moonshotai/kimi-k3). Always reasons; effort is
    # low vs high (no true OFF). Keys include -thinking- to reflect that.
    'kimi-k3-openrouter-thinking-low': ModelConfig(
        model_key='kimi-k3-openrouter-thinking-low',
        temperature=0.0,
        max_tokens=1500,
        concurrency_limit=20,
        base_timeout=90.0,
        extra_body={"reasoning_effort": "low"},
        reasoning_artifact_type="raw_cot",
    ),
    'kimi-k3-openrouter-thinking-medium': ModelConfig(
        model_key='kimi-k3-openrouter-thinking-medium',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=20,
        base_timeout=120.0,
        extra_body={"reasoning_effort": "medium"},
        reasoning_artifact_type="raw_cot",
    ),
    'kimi-k3-openrouter-thinking-high': ModelConfig(
        model_key='kimi-k3-openrouter-thinking-high',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=20,
        base_timeout=120.0,
        extra_body={"reasoning_effort": "high"},
        reasoning_artifact_type="raw_cot",
    ),

    # Qwen 3.7 Flash via OpenRouter. This is a hybrid thinking model, so the
    # same endpoint supports a genuine reasoning-off condition and a bounded
    # reasoning-on condition. The latter retains the raw reasoning text that
    # OpenRouter returns while reserving 200 tokens for the final answer.
    'qwen-37-flash-openrouter': ModelConfig(
        model_key='qwen-37-flash-openrouter',
        temperature=0.0,
        max_tokens=16,
        concurrency_limit=20,
        base_timeout=60.0,
        extra_body={"reasoning": {"enabled": False}},
        reasoning_artifact_type="none",
    ),
    'qwen-37-flash-openrouter-thinking': ModelConfig(
        model_key='qwen-37-flash-openrouter-thinking',
        temperature=0.0,
        max_tokens=2400,
        concurrency_limit=20,
        base_timeout=120.0,
        extra_body={
            "reasoning": {
                "enabled": True,
                "max_tokens": 2200,
                "exclude": False,
            }
        },
        reasoning_artifact_type="raw_cot",
    ),

    # Qwen 3.7 Max via OpenRouter. This is a hybrid thinking model: reasoning
    # can be switched fully off or on, but the current model metadata does not
    # advertise selectable effort tiers. In the ON condition Qwen determines
    # the trace length within the configured output-token ceiling. Pin the
    # May 20 snapshot used by OpenRouter's canonical Qwen3.7 Max endpoint.
    'qwen-37-max-openrouter': ModelConfig(
        model_key='qwen-37-max-openrouter',
        temperature=0.0,
        max_tokens=16,
        concurrency_limit=20,
        base_timeout=60.0,
        extra_body={"reasoning": {"enabled": False}},
        reasoning_artifact_type="none",
    ),
    'qwen-37-max-openrouter-thinking': ModelConfig(
        model_key='qwen-37-max-openrouter-thinking',
        temperature=0.0,
        max_tokens=3000,
        concurrency_limit=20,
        base_timeout=120.0,
        extra_body={"reasoning": {"enabled": True}},
        # Alibaba returns the generated trace as reasoning_content; OpenRouter
        # normalizes it as visible reasoning text rather than a summary.
        reasoning_artifact_type="raw_cot",
    ),

}

# Default model for experiments
DEFAULT_MODEL = 'gpt-54-nano'

# Legacy CLI keys → canonical MODEL_CONFIGS entry.
MODEL_KEY_ALIASES: dict[str, str] = {
    "gpt-56": "gpt-56-sol",
    "gpt-56-thinking": "gpt-56-sol-thinking",
    "opus-46-openrouter-thinking": "opus-46-thinking",
}


def canonical_model_key(model_key: str) -> str:
    return MODEL_KEY_ALIASES.get(model_key, model_key)


def results_dir_name(model_key: str) -> str:
    """Folder name under ``results/07_model_runs/`` (without ``-openrouter``)."""
    return canonical_model_key(model_key).replace("-openrouter", "")


def _candidate_results_dir_names(model_key: str) -> list[str]:
    """Search order: preferred stripped name, then legacy names with -openrouter."""
    key = canonical_model_key(model_key)
    stripped = results_dir_name(key)
    names: list[str] = []
    legacy_aliases = [
        alias for alias, canonical in MODEL_KEY_ALIASES.items() if canonical == key
    ]
    for name in (stripped, key, model_key, *legacy_aliases):
        if name and name not in names:
            names.append(name)
    return names


def resolve_model_results_dir(model_key: str, results_root: Path) -> Path:
    """Existing model results dir, or preferred path for new writes."""
    for name in _candidate_results_dir_names(model_key):
        path = results_root / name
        if path.is_dir():
            return path
    return results_root / results_dir_name(model_key)


def model_key_from_results_folder(folder_name: str, known_keys: set[str]) -> str:
    """Map a ``results/07_model_runs/`` subdirectory name to a config model key."""
    candidates: list[str] = [folder_name]
    stripped = folder_name.replace("-openrouter", "")
    if stripped not in candidates:
        candidates.append(stripped)
    if "-openrouter" not in folder_name:
        if folder_name.endswith("-thinking"):
            base = folder_name[: -len("-thinking")]
            candidates.append(f"{base}-openrouter-thinking")
        candidates.append(f"{folder_name}-openrouter")
    for candidate in candidates:
        resolved = canonical_model_key(candidate)
        if resolved in known_keys:
            return resolved
    return canonical_model_key(stripped)


def get_model_config(model_key: str) -> ModelConfig:
    """Return ``ModelConfig`` for a paper model key."""
    resolved = MODEL_KEY_ALIASES.get(model_key, model_key)
    if resolved not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[resolved]
