"""Lightweight API agent layer for llm_coherence experiment runs."""

from __future__ import annotations

import asyncio
import math
import os
import random
from dataclasses import dataclass
from typing import Any

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.runtime.api_keys import API_KEY_ENV_BY_TYPE, ensure_api_key_env


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_type: str
    accepts_system_message: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt-54-nano": ModelSpec("openai/gpt-5.4-nano-2026-03-17", "openai"),
    "gpt-54-nano-thinking": ModelSpec("openai/gpt-5.4-nano-2026-03-17", "openai"),
    "gpt-54-mini": ModelSpec("openai/gpt-5.4-mini-2026-03-17", "openai"),
    "gpt-54-mini-thinking": ModelSpec("openai/gpt-5.4-mini-2026-03-17", "openai"),
    "gpt-54": ModelSpec("openai/gpt-5.4-2026-03-05", "openai"),
    "gpt-54-thinking": ModelSpec("openai/gpt-5.4-2026-03-05", "openai"),
    "gpt-55-openai": ModelSpec("openai/gpt-5.5", "openai"),
    "opus-46": ModelSpec("claude-opus-4-6", "anthropic"),
    "opus-46-thinking": ModelSpec("claude-opus-4-6", "anthropic"),
    "nemotron-3-super": ModelSpec("openrouter/nvidia/nemotron-3-super-120b-a12b", "openrouter"),
    "nemotron-3-super-thinking": ModelSpec("openrouter/nvidia/nemotron-3-super-120b-a12b", "openrouter"),
    "glm-45-hybrid": ModelSpec("openrouter/z-ai/glm-4.5", "openrouter"),
    "glm-45-hybrid-thinking": ModelSpec("openrouter/z-ai/glm-4.5", "openrouter"),
    "glm-45-base-logprobs": ModelSpec(
        "zai-org/GLM-4.5-Base",
        "vllm_base_model_logprobs",
        accepts_system_message=False,
    ),
    "llama-31-8b-instruct-openrouter": ModelSpec(
        "openrouter/meta-llama/llama-3.1-8b-instruct",
        "openrouter",
    ),
    "ministral-3b-2512-openrouter": ModelSpec(
        "openrouter/mistralai/ministral-3b-2512",
        "openrouter",
    ),
    "mistral-small-2603-openrouter-thinking": ModelSpec(
        "openrouter/mistralai/mistral-small-2603",
        "openrouter",
    ),
}


def model_name_for_key(model_key: str) -> str | None:
    spec = MODEL_SPECS.get(model_key)
    if spec is not None:
        return spec.model_name
    cfg = MODEL_CONFIGS.get(model_key)
    return cfg.model_name_full if cfg is not None else None


class LiteLLMAgent:
    """Async LiteLLM wrapper with retry, usage, and reasoning trace logging."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        extra_body: dict[str, Any] | None = None,
        enable_cache: bool = False,
    ):
        try:
            import litellm
            from litellm import acompletion
        except ImportError as exc:
            raise ImportError(
                "LiteLLM is required for live API runs. Install the project with "
                "`pip install -e .` or install `litellm` in this environment."
            ) from exc

        litellm.drop_params = True
        self._acompletion = acompletion
        self._litellm_bad_request_error = litellm.BadRequestError

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter
        self.extra_body = extra_body or {}
        self.enable_cache = enable_cache
        self.usage_log: list[dict[str, Any]] = []
        self.reasoning_log: list[dict[str, Any]] = []
        self.retry_counts: dict[str, int] = {"timeouts": 0, "errors": 0}

    def _messages_for_call(self, message: list[dict[str, Any]]) -> list[dict[str, Any]]:
        call_messages = message
        if not self.enable_cache or not call_messages:
            return call_messages

        call_messages = [dict(m) for m in call_messages]
        model_lower = self.model.lower()
        is_anthropic = "claude" in model_lower or "anthropic" in model_lower
        if is_anthropic:
            for msg in call_messages:
                if msg.get("role") != "system":
                    continue
                content = msg.get("content")
                cache_control = {"type": "ephemeral", "ttl": "1h"}
                if isinstance(content, str):
                    msg["content"] = [
                        {"type": "text", "text": content, "cache_control": cache_control}
                    ]
                elif content and isinstance(content, list) and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = cache_control
                break
        elif call_messages[0].get("role") == "system":
            call_messages[0]["cache_control"] = {"type": "ephemeral"}
        return call_messages

    def _completion_kwargs(
        self,
        message: list[dict[str, Any]],
        timeout: float,
    ) -> dict[str, Any]:
        model_str = self.model.split("/", 1)[-1]
        is_gpt5_family = (
            model_str.startswith("gpt-5")
            or model_str.startswith("o1")
            or model_str.startswith("o3")
        )
        is_opus_47_family = "opus-4-7" in self.model.lower()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_for_call(message),
            "timeout": timeout,
        }
        if is_gpt5_family:
            kwargs["max_completion_tokens"] = self.max_tokens
        elif is_opus_47_family:
            kwargs["max_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _log_usage(self, completion_res: Any) -> None:
        try:
            usage = getattr(completion_res, "usage", None)
            usage_entry = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": None,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "openai_cached_tokens": None,
            }
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                usage_entry["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                usage_entry["openai_cached_tokens"] = getattr(prompt_details, "cached_tokens", None)
            self.usage_log.append(usage_entry)
        except Exception:
            return

    def _log_reasoning(self, completion_res: Any, message_idx: int, attempt: int, content: str) -> None:
        try:
            msg = completion_res.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            self.reasoning_log.append(
                {
                    "message_idx": message_idx,
                    "attempt": attempt,
                    "content": content,
                    "reasoning": reasoning,
                }
            )
        except Exception:
            return

    async def async_completions(
        self,
        messages: list[list[dict[str, Any]]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> list[str | None]:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results: dict[int, str | None] = {}

        async def process_message(message_idx: int) -> None:
            message = messages[message_idx]
            current_timeout = float(kwargs.get("timeout", kwargs.get("base_timeout", self.base_timeout)))
            retry_delay = self.base_delay
            response: str | None = None

            for attempt in range(self.max_retries):
                async with semaphore:
                    try:
                        completion_res = await self._acompletion(
                            **self._completion_kwargs(message, current_timeout)
                        )
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1
                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx} after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                        else:
                            current_timeout *= 2.0
                        continue
                    except self._litellm_bad_request_error as exc:
                        counts["errors"] += 1
                        if verbose:
                            print(
                                f"[Error] Bad request for message index {message_idx}: {exc}. "
                                "Not retrying."
                            )
                        response = None
                        break
                    except Exception as exc:
                        counts["errors"] += 1
                        if verbose:
                            print(
                                f"[Error] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}: {exc}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                        else:
                            sleep_for = retry_delay + (random.uniform(0, 1) if self.use_jitter else 0)
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)
                        continue

                try:
                    content = completion_res.choices[0].message.content
                except (IndexError, AttributeError):
                    content = None

                if content is None or content == "":
                    counts["errors"] += 1
                    if verbose:
                        print(
                            f"[Empty content] Attempt {attempt + 1}/{self.max_retries} "
                            f"for message index {message_idx}."
                        )
                    if attempt == self.max_retries - 1:
                        response = None
                    else:
                        sleep_for = retry_delay + (random.uniform(0, 1) if self.use_jitter else 0)
                        await asyncio.sleep(sleep_for)
                        retry_delay = min(retry_delay * 2.0, self.max_delay)
                    continue

                response = content.strip()
                self._log_usage(completion_res)
                self._log_reasoning(completion_res, message_idx, attempt, content)
                break

            results[message_idx] = response

        tasks = [process_message(i) for i in range(len(messages))]
        if verbose:
            total = len(tasks)
            completed = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                completed += 1
                if completed == total or completed % 50 == 0:
                    print(f"LLM calls completed: {completed}/{total}")
        else:
            await asyncio.gather(*tasks)

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        self.retry_counts["timeouts"] += counts["timeouts"]
        self.retry_counts["errors"] += counts["errors"]
        return [results[i] for i in range(len(messages))]


class VLLMLogprobAgent:
    """vLLM-backed forced-choice scorer for self-hosted base models.

    This agent returns ``{"A": p_a, "B": p_b}`` distributions instead of sampled
    text. The experiment runner's ``uses_logits`` branch consumes that shape.
    """

    uses_logits = True

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
        **_: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = False
        self.enable_cache = False
        self.usage_log: list[dict[str, Any]] = []
        self.reasoning_log: list[dict[str, Any]] = []
        self.retry_counts: dict[str, int] = {"timeouts": 0, "errors": 0}
        self._llm = None
        self._sampling_params = None
        self._token_id_a: int | None = None
        self._token_id_b: int | None = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM, torch, and transformers are required for self-hosted "
                "logprob models. Use the HF Jobs image or install the GPU stack."
            ) from exc

        cache_dir = os.environ.get("HF_HOME")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            cache_dir=cache_dir,
        )
        a_ids = tokenizer.encode(" A", add_special_tokens=False)
        b_ids = tokenizer.encode(" B", add_special_tokens=False)
        if not a_ids or not b_ids:
            raise RuntimeError("Could not resolve tokenizer ids for forced-choice labels A/B.")
        self._token_id_a = a_ids[0]
        self._token_id_b = b_ids[0]

        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "enable_prefix_caching": True,
        }
        if cache_dir:
            llm_kwargs["download_dir"] = cache_dir
        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    @staticmethod
    def _prompt_from_messages(message: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in message:
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        return "\n\n".join(p for p in parts if p)

    def _score_from_top_logprobs(self, top_logprobs: dict) -> dict[str, float]:
        assert self._token_id_a is not None
        assert self._token_id_b is not None
        lp_a_obj = top_logprobs.get(self._token_id_a)
        lp_b_obj = top_logprobs.get(self._token_id_b)
        score_a = lp_a_obj.logprob if lp_a_obj else float("-inf")
        score_b = lp_b_obj.logprob if lp_b_obj else float("-inf")
        if score_a == float("-inf") and score_b == float("-inf"):
            return {"A": 0.5, "B": 0.5}

        mx = max(score_a, score_b)
        ea = math.exp(score_a - mx) if score_a != float("-inf") else 0.0
        eb = math.exp(score_b - mx) if score_b != float("-inf") else 0.0
        total = ea + eb
        if total == 0:
            return {"A": 0.5, "B": 0.5}
        return {"A": ea / total, "B": eb / total}

    async def async_completions(
        self,
        messages: list[list[dict[str, Any]]],
        verbose: bool = True,
        **_: Any,
    ) -> list[dict[str, float] | None]:
        del verbose
        self._ensure_loaded()

        from llm_coherence.runtime.logprob_prompts import FEW_SHOT_PROMPT_LOGPROBS

        assert self._llm is not None
        assert self._sampling_params is not None
        prompts = [
            f"{FEW_SHOT_PROMPT_LOGPROBS}{self._prompt_from_messages(message)}\n\nAnswer:"
            for message in messages
        ]
        outputs = self._llm.generate(prompts, self._sampling_params)
        results: list[dict[str, float] | None] = []
        for output in outputs:
            logprobs_per_pos = output.outputs[0].logprobs
            if not logprobs_per_pos or logprobs_per_pos[0] is None:
                results.append({"A": 0.5, "B": 0.5})
            else:
                results.append(self._score_from_top_logprobs(logprobs_per_pos[0]))
        return results


def create_agent(
    model_key: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    concurrency_limit: int = 50,
    trust_remote_code: bool = True,
    extra_body: dict[str, Any] | None = None,
    enable_cache: bool = False,
    **kwargs: Any,
) -> LiteLLMAgent:
    """Create a LiteLLM-backed API agent from an llm_coherence model key."""
    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        raise ValueError(f"Unknown model key: {model_key}")
    if spec.model_type == "vllm_base_model_logprobs":
        return VLLMLogprobAgent(
            model=spec.model_name,
            temperature=temperature,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    if spec.model_type not in API_KEY_ENV_BY_TYPE:
        raise ValueError(
            f"Model {model_key!r} is configured as {spec.model_type!r}. "
            "The lightweight runtime supports API-backed models only; use the "
            "local validation runner for self-hosted logprob models."
        )

    ensure_api_key_env(spec.model_type)

    cfg = MODEL_CONFIGS.get(model_key)
    resolved_extra_body = extra_body if extra_body is not None else (cfg.extra_body if cfg else None)
    resolved_enable_cache = enable_cache or (cfg.enable_cache if cfg else False)

    return LiteLLMAgent(
        model=spec.model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency_limit=concurrency_limit,
        accepts_system_message=spec.accepts_system_message,
        base_timeout=float(kwargs.get("base_timeout", cfg.base_timeout if cfg else 5.0)),
        extra_body=resolved_extra_body,
        enable_cache=resolved_enable_cache,
    )
