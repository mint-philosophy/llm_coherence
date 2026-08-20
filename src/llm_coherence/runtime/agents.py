"""Lightweight API agent layer for llm_coherence experiment runs."""

from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from llm_coherence.config import MODEL_CONFIGS, canonical_model_key
from llm_coherence.runtime.api_keys import API_KEY_ENV_BY_TYPE, ensure_api_key_env
from llm_coherence.runtime.forced_choice_logprobs import (
    ForcedChoiceScoringError,
    normalized_choice_probabilities,
    resolve_choice_token_ids,
    vllm_load_kwargs_from_env,
)


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
    "gpt-56-sol": ModelSpec("openai/gpt-5.6-sol", "openai"),
    "gpt-56-sol-thinking": ModelSpec("openai/gpt-5.6-sol", "openai"),
    "gpt-56-terra": ModelSpec("openai/gpt-5.6-terra", "openai"),
    "gpt-56-terra-thinking": ModelSpec("openai/gpt-5.6-terra", "openai"),
    "gpt-56-luna": ModelSpec("openai/gpt-5.6-luna", "openai"),
    "gpt-56-luna-thinking": ModelSpec("openai/gpt-5.6-luna", "openai"),
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
    "qwen25-05b-instruct-smoke": ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct",
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
    "kimi-k2-openrouter": ModelSpec(
        "openrouter/moonshotai/kimi-k2",
        "openrouter",
    ),
    "kimi-k2-openrouter-thinking": ModelSpec(
        "openrouter/moonshotai/kimi-k2-thinking",
        "openrouter",
    ),
    "kimi-k3-openrouter-thinking-low": ModelSpec(
        "openrouter/moonshotai/kimi-k3",
        "openrouter",
    ),
    "kimi-k3-openrouter-thinking-medium": ModelSpec(
        "openrouter/moonshotai/kimi-k3",
        "openrouter",
    ),
    "kimi-k3-openrouter-thinking-high": ModelSpec(
        "openrouter/moonshotai/kimi-k3",
        "openrouter",
    ),
    "qwen-37-flash-openrouter": ModelSpec(
        "openrouter/qwen/qwen3.7-flash",
        "openrouter",
    ),
    "qwen-37-flash-openrouter-thinking": ModelSpec(
        "openrouter/qwen/qwen3.7-flash",
        "openrouter",
    ),
    "qwen-37-max-openrouter": ModelSpec(
        "openrouter/qwen/qwen3.7-max-20260520",
        "openrouter",
    ),
    "qwen-37-max-openrouter-thinking": ModelSpec(
        "openrouter/qwen/qwen3.7-max-20260520",
        "openrouter",
    ),
}


_TOKEN_CAP_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "max_completion_tokens",
}

_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_token_capped_finish_reason(reason: Any) -> bool:
    """Return whether a provider finish reason denotes the fixed token cap."""
    if reason is None:
        return False
    return str(reason).strip().lower() in _TOKEN_CAP_FINISH_REASONS


def is_retryable_transport_exception(exc: BaseException) -> bool:
    """Classify retryable infrastructure failures with no model outcome.

    LiteLLM may wrap ``httpx`` transport exceptions, so inspect the exception
    chain as well as the outer SDK exception. Transient HTTP statuses such as
    429 and 5xx are safe to retry because the provider returned no model
    choice. Bad requests, model refusals, token caps, empty responses, and
    parse failures remain non-retryable experimental outcomes.
    """
    transport_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "Timeout",
        "TimeoutError",
        "TransportError",
        "WriteError",
        "WriteTimeout",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (asyncio.TimeoutError, TimeoutError)):
            return True
        try:
            import httpx

            if isinstance(current, httpx.TransportError):
                return True
            if isinstance(current, httpx.HTTPStatusError):
                if current.response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                    return True
        except ImportError:
            pass
        module = type(current).__module__.lower()
        status_code = getattr(current, "status_code", None)
        if status_code is None:
            response = getattr(current, "response", None)
            status_code = getattr(response, "status_code", None)
        try:
            if int(status_code) in _RETRYABLE_HTTP_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass
        if (
            type(current).__name__ == "RateLimitError"
            and any(name in module for name in ("litellm", "openai", "anthropic"))
        ):
            return True
        if (
            type(current).__name__ in transport_names
            and any(name in module for name in ("httpx", "httpcore", "litellm", "openai"))
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _exception_http_status_code(exc: BaseException) -> int | None:
    """Return a wrapped HTTP status code when an SDK exposes one."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        if status_code is None:
            response = getattr(current, "response", None)
            status_code = getattr(response, "status_code", None)
        try:
            return int(status_code)
        except (TypeError, ValueError):
            pass
        if type(current).__name__ == "RateLimitError":
            return 429
        current = current.__cause__ or current.__context__
    return None


def _exception_retry_after_seconds(exc: BaseException) -> float | None:
    """Return Retry-After from a wrapped response when available."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return max(float(retry_after), 0.0)
                except (TypeError, ValueError):
                    pass
        current = current.__cause__ or current.__context__
    return None


def _completion_finish_reason(completion_res: Any) -> str | None:
    try:
        choice = completion_res.choices[0]
    except (IndexError, AttributeError, TypeError):
        return None
    reason = getattr(choice, "finish_reason", None)
    if reason is None:
        reason = getattr(choice, "native_finish_reason", None)
    return str(reason) if reason is not None else None


def model_name_for_key(model_key: str) -> str | None:
    resolved_key = canonical_model_key(model_key)
    spec = MODEL_SPECS.get(resolved_key)
    if spec is not None:
        return spec.model_name
    cfg = MODEL_CONFIGS.get(resolved_key)
    return cfg.model_name_full if cfg is not None else None


async def close_api_async_clients() -> None:
    """Close LiteLLM's cached async HTTP clients after a CLI live run.

    LiteLLM caches provider clients across requests for connection reuse. The
    cache must be closed before ``asyncio.run`` tears down its event loop or
    aiohttp reports an unclosed client session at process exit.
    """
    try:
        import litellm
    except ImportError:
        return

    close_clients = getattr(litellm, "close_litellm_async_clients", None)
    if close_clients is None:
        return
    try:
        await close_clients()
    except Exception:
        # Cleanup must not mask a completed experiment or its original error.
        return


class AsyncRequestLimiter:
    """Run-wide cap and optional smoothing for live API request attempts.

    ``LiteLLMAgent.async_completions`` is called independently by every
    concurrently running ladder.  A semaphore created inside that method only
    limits one batch, so outer ladder concurrency can otherwise multiply the
    actual number of in-flight HTTP requests.  Sharing one instance of this
    limiter across agents makes the configured cap apply to the whole Python
    process, including retry attempts.

    ``requests_per_second`` controls request *starts*, not completions.  Starts
    are evenly spaced while a concurrency slot is held, avoiding a new burst
    immediately after a provider window clears.
    """

    def __init__(
        self,
        max_concurrency: int,
        requests_per_second: float | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if requests_per_second is not None and requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0 when set")

        self.max_concurrency = int(max_concurrency)
        self.requests_per_second = (
            float(requests_per_second)
            if requests_per_second is not None
            else None
        )
        self._minimum_start_interval = (
            1.0 / self.requests_per_second
            if self.requests_per_second is not None
            else 0.0
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._start_lock = asyncio.Lock()
        self._next_start = 0.0
        self._active = 0
        self._peak_active = 0
        self._attempts_started = 0
        self._attempts_completed = 0
        self._first_start: float | None = None
        self._last_start: float | None = None
        self._concurrency_wait_seconds = 0.0
        self._pacing_wait_seconds = 0.0

    @asynccontextmanager
    async def slot(self):
        """Acquire one run-wide request slot and apply start-rate smoothing."""
        queued_at = time.monotonic()
        await self._semaphore.acquire()
        acquired_at = time.monotonic()
        self._concurrency_wait_seconds += acquired_at - queued_at
        active = False
        try:
            if self._minimum_start_interval:
                async with self._start_lock:
                    now = time.monotonic()
                    scheduled = max(now, self._next_start)
                    delay = scheduled - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                        self._pacing_wait_seconds += delay
                    started_at = time.monotonic()
                    # Do not "catch up" with a burst if the event loop wakes
                    # later than scheduled.
                    self._next_start = (
                        max(scheduled, started_at) + self._minimum_start_interval
                    )
            else:
                started_at = time.monotonic()

            self._attempts_started += 1
            self._first_start = self._first_start or started_at
            self._last_start = started_at
            self._active += 1
            active = True
            self._peak_active = max(self._peak_active, self._active)
            yield
        finally:
            if active:
                self._active -= 1
                self._attempts_completed += 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int | float | None]:
        """Return diagnostics without resetting the limiter."""
        observed_rate: float | None = None
        if (
            self._attempts_started > 1
            and self._first_start is not None
            and self._last_start is not None
            and self._last_start > self._first_start
        ):
            observed_rate = (
                (self._attempts_started - 1)
                / (self._last_start - self._first_start)
            )
        return {
            "max_concurrency": self.max_concurrency,
            "requests_per_second": self.requests_per_second,
            "attempts_started": self._attempts_started,
            "attempts_completed": self._attempts_completed,
            "in_flight": self._active,
            "peak_in_flight": self._peak_active,
            "observed_start_rate": observed_rate,
            "concurrency_wait_seconds": self._concurrency_wait_seconds,
            "pacing_wait_seconds": self._pacing_wait_seconds,
        }


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
        retry_transport_only: bool = False,
        request_limiter: AsyncRequestLimiter | None = None,
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
        # LiteLLM prints its provider-help URL directly to stdout whenever an
        # internal best-effort provider lookup misses (for example while
        # calculating response cost).  Suppress that diagnostic banner while
        # preserving the exception itself and our normal error accounting.
        litellm.suppress_debug_info = True
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
        self.retry_transport_only = retry_transport_only
        self.request_limiter = request_limiter
        self.usage_log: list[dict[str, Any]] = []
        self.reasoning_log: list[dict[str, Any]] = []
        self.last_completion_outcomes: list[dict[str, Any]] = []
        self.retry_counts: dict[str, int] = {
            "timeouts": 0,
            "errors": 0,
            "transport_retries": 0,
            "transport_failures": 0,
            "non_transport_errors": 0,
            "token_capped": 0,
            "empty_responses": 0,
        }

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
        extra_body = dict(self.extra_body or {})
        # OpenRouter returns usage.cost only when usage.include is true.
        if self.model.lower().startswith("openrouter/"):
            usage_opt = dict(extra_body.get("usage") or {})
            usage_opt.setdefault("include", True)
            extra_body["usage"] = usage_opt
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _log_usage(self, completion_res: Any) -> None:
        try:
            from llm_coherence.runtime.usage_cost import (
                infer_provider,
                usage_cost_breakdown,
            )

            usage = getattr(completion_res, "usage", None)
            fields = usage_cost_breakdown(
                usage,
                provider=infer_provider(self.model),
                model_id=self.model,
            )
            # LiteLLM sometimes exposes USD on the response, not usage.
            if fields.get("cost") is None:
                hidden = getattr(completion_res, "_hidden_params", None) or {}
                response_cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
                if isinstance(response_cost, (int, float)):
                    fields["cost"] = float(response_cost)
                    fields["cost_usd"] = float(response_cost)
                    fields["cost_source"] = "provider_reported"
                    fields["pricing_source"] = "litellm._hidden_params.response_cost"

            self.usage_log.append(
                {
                    "prompt_tokens": fields.get("prompt_tokens"),
                    "completion_tokens": fields.get("completion_tokens"),
                    "reasoning_tokens": fields.get("reasoning_tokens") or None,
                    "cache_creation_input_tokens": fields.get("cache_creation_input_tokens") or None,
                    "cache_read_input_tokens": fields.get("cache_read_input_tokens") or None,
                    "openai_cached_tokens": fields.get("openai_cached_tokens") or None,
                    "cost_usd": fields.get("cost_usd"),
                    "cost_source": fields.get("cost_source"),
                    "pricing_source": fields.get("pricing_source"),
                }
            )
        except Exception:
            return

    def _log_reasoning(self, completion_res: Any, message_idx: int, attempt: int, content: str) -> None:
        try:
            msg = completion_res.choices[0].message
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or getattr(msg, "reasoning_details", None)
            )
            if reasoning in (None, "", [], {}):
                return
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
        counts = {
            "timeouts": 0,
            "errors": 0,
            "transport_retries": 0,
            "transport_failures": 0,
            "non_transport_errors": 0,
            "token_capped": 0,
            "empty_responses": 0,
        }
        results: dict[int, str | None] = {}
        outcomes: dict[int, dict[str, Any]] = {}

        async def process_message(message_idx: int) -> None:
            message = messages[message_idx]
            current_timeout = float(kwargs.get("timeout", kwargs.get("base_timeout", self.base_timeout)))
            retry_delay = self.base_delay
            response: str | None = None
            outcome: dict[str, Any] = {
                "message_idx": message_idx,
                "status": "unknown",
                "finish_reason": None,
                "raw_response": None,
                "error": None,
                "attempts": 0,
                "transport_retries": 0,
            }

            for attempt in range(self.max_retries):
                outcome["attempts"] = attempt + 1
                completion_res = None
                try:
                    async with semaphore:
                        if self.request_limiter is None:
                            completion_res = await self._acompletion(
                                **self._completion_kwargs(message, current_timeout)
                            )
                        else:
                            async with self.request_limiter.slot():
                                completion_res = await self._acompletion(
                                    **self._completion_kwargs(message, current_timeout)
                                )
                except Exception as exc:
                    counts["errors"] += 1
                    transport = is_retryable_transport_exception(exc)
                    status_code = _exception_http_status_code(exc)
                    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                        counts["timeouts"] += 1
                    if transport:
                        can_retry = attempt < self.max_retries - 1
                    elif self.retry_transport_only or isinstance(
                        exc, self._litellm_bad_request_error
                    ):
                        can_retry = False
                    else:
                        # Preserve legacy behavior for callers that have not
                        # opted into the experiment's transport-only policy.
                        can_retry = attempt < self.max_retries - 1

                    if verbose:
                        label = (
                            "Retryable infrastructure"
                            if transport
                            else "Non-retryable provider"
                        )
                        suffix = "retrying" if can_retry else "not retrying"
                        print(
                            f"[{label} error] Attempt {attempt + 1}/{self.max_retries} "
                            f"for message index {message_idx}: {exc} ({suffix})."
                        )
                    if can_retry:
                        if transport:
                            counts["transport_retries"] += 1
                            outcome["transport_retries"] += 1
                        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                            current_timeout *= 2.0
                        else:
                            retry_after = _exception_retry_after_seconds(exc)
                            if retry_after is not None:
                                sleep_for = retry_after
                            elif status_code == 429:
                                # OpenRouter's shared upstream pools frequently
                                # omit Retry-After. Let the rolling window clear.
                                sleep_for = max(
                                    retry_delay,
                                    10.0 * (attempt + 1),
                                )
                            else:
                                sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)
                        continue

                    if transport:
                        counts["transport_failures"] += 1
                        outcome["status"] = "transport_failure"
                    else:
                        counts["non_transport_errors"] += 1
                        outcome["status"] = "provider_error"
                    outcome["error"] = str(exc)
                    response = None
                    break

                try:
                    content = completion_res.choices[0].message.content
                except (IndexError, AttributeError):
                    content = None

                finish_reason = _completion_finish_reason(completion_res)
                outcome["finish_reason"] = finish_reason
                outcome["raw_response"] = content
                # Every provider response is billable and must be retained in
                # usage/reasoning accounting, including capped/empty outcomes.
                self._log_usage(completion_res)
                self._log_reasoning(completion_res, message_idx, attempt, content or "")

                if is_token_capped_finish_reason(finish_reason):
                    counts["token_capped"] += 1
                    outcome["status"] = "token_capped"
                    if self.retry_transport_only:
                        response = None
                    elif content:
                        response = content.strip()
                    break

                if content is None or content == "":
                    counts["errors"] += 1
                    counts["empty_responses"] += 1
                    if verbose:
                        print(
                            f"[Empty content] Attempt {attempt + 1}/{self.max_retries} "
                            f"for message index {message_idx}; "
                            + ("not retrying." if self.retry_transport_only else "retrying if possible.")
                        )
                    if self.retry_transport_only or attempt == self.max_retries - 1:
                        response = None
                        outcome["status"] = "empty_response"
                    else:
                        sleep_for = retry_delay + (random.uniform(0, 1) if self.use_jitter else 0)
                        await asyncio.sleep(sleep_for)
                        retry_delay = min(retry_delay * 2.0, self.max_delay)
                        continue
                    break

                response = content.strip()
                outcome["status"] = "completed"
                break

            results[message_idx] = response
            outcomes[message_idx] = outcome

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

        for key, value in counts.items():
            self.retry_counts[key] = self.retry_counts.get(key, 0) + value
        self.last_completion_outcomes = [outcomes[i] for i in range(len(messages))]
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

        model_source = os.environ.get("LLM_COHERENCE_VLLM_MODEL", self.model)
        cache_dir = os.environ.get("HF_HOME")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=self.trust_remote_code,
            cache_dir=cache_dir,
        )
        self._token_id_a, self._token_id_b = resolve_choice_token_ids(tokenizer)

        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        llm_kwargs: dict[str, Any] = {
            "model": model_source,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "enable_prefix_caching": True,
        }
        if cache_dir:
            llm_kwargs["download_dir"] = cache_dir
        llm_kwargs.update(vllm_load_kwargs_from_env())
        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=2,
            allowed_token_ids=[self._token_id_a, self._token_id_b],
        )

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
        return normalized_choice_probabilities(
            top_logprobs,
            self._token_id_a,
            self._token_id_b,
        )

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
        if len(outputs) != len(prompts):
            raise ForcedChoiceScoringError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts."
            )
        results: list[dict[str, float] | None] = []
        for index, output in enumerate(outputs):
            logprobs_per_pos = output.outputs[0].logprobs
            if not logprobs_per_pos or logprobs_per_pos[0] is None:
                raise ForcedChoiceScoringError(
                    f"vLLM returned no next-token logprobs for prompt index {index}."
                )
            results.append(self._score_from_top_logprobs(logprobs_per_pos[0]))
        return results


def create_agent(
    model_key: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    concurrency_limit: int = 50,
    max_retries: int = 5,
    trust_remote_code: bool = True,
    extra_body: dict[str, Any] | None = None,
    enable_cache: bool = False,
    retry_transport_only: bool = False,
    request_limiter: AsyncRequestLimiter | None = None,
    **kwargs: Any,
) -> LiteLLMAgent:
    """Create a LiteLLM-backed API agent from an llm_coherence model key."""
    resolved_key = canonical_model_key(model_key)
    spec = MODEL_SPECS.get(resolved_key)
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

    cfg = MODEL_CONFIGS.get(resolved_key)
    resolved_extra_body = extra_body if extra_body is not None else (cfg.extra_body if cfg else None)
    resolved_enable_cache = enable_cache or (cfg.enable_cache if cfg else False)

    return LiteLLMAgent(
        model=spec.model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency_limit=concurrency_limit,
        accepts_system_message=spec.accepts_system_message,
        max_retries=max_retries,
        base_timeout=float(kwargs.get("base_timeout", cfg.base_timeout if cfg else 5.0)),
        extra_body=resolved_extra_body,
        enable_cache=resolved_enable_cache,
        retry_transport_only=retry_transport_only,
        request_limiter=request_limiter,
    )
