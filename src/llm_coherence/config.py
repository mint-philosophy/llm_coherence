"""
config.py

Configuration for parametric variation experiments.
Centralized settings for models, API parameters, and experiment parameters.

Trimmed to the publication workflow for llm_coherence. The public README
separates the 15 model configurations reported in the paper from audit judges
and local support or exploratory model keys.
"""

from dataclasses import dataclass
from typing import Optional


# Model configuration

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
        reasoning_artifact_type="raw_cot",
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
        reasoning_artifact_type="raw_cot",
    ),
}

# Default model for experiments
DEFAULT_MODEL = 'gpt-54-nano'
