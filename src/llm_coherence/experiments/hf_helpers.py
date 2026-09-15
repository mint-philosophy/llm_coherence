"""
Shared Hugging Face router helper utilities (for 10a + 10b live inference).

This centralizes:
- HF token loading
- SSL/cert env fix
- retry-after / fatal-error classification
- HF router model-name resolution (openrouter/moonshot aliases)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from llm_coherence.paths import API_KEYS_DIR
from llm_coherence.runtime.agents import model_name_for_key

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_HF_PROVIDER = "featherless-ai"
DEFAULT_HF_BILL_TO = "MINTLABJHUANU"

FATAL_STATUS = {401, 402, 403}
FATAL_TEXT = (
    "payment required",
    "insufficient credits",
    "exceeded your monthly included credits",
    "exceeded your included credits",
    "exceeded your spending limit",
)

# Map openrouter model-name (as returned by model_name_for_key) -> HF Hub model id.
HF_HUB_MODEL_ALIASES: dict[str, str] = {
    "openrouter/moonshotai/kimi-k2-thinking": "moonshotai/Kimi-K2-Thinking",
    "openrouter/moonshotai/kimi-k2": "moonshotai/Kimi-K2-Instruct",
    # Also accept the raw provider model name that some pipelines expose
    # (e.g. from MODELS[cfg.model_key][0]).
    "moonshotai/kimi-k2-thinking": "moonshotai/Kimi-K2-Thinking",
    "moonshotai/kimi-k2": "moonshotai/Kimi-K2-Instruct",
}


def fix_ssl_env() -> None:
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        cert = os.environ.get(var)
        if cert and not Path(cert).is_file():
            os.environ.pop(var, None)
    try:
        import certifi

        ca = certifi.where()
    except Exception:
        ca = ""
    if ca and Path(ca).is_file():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            if not os.environ.get(var):
                os.environ[var] = ca


def load_hf_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = (os.environ.get(env_name) or "").strip()
        if token:
            return token
    for name in ("hf_token", "hf_token.txt", "api_key_huggingface.txt"):
        key_path = API_KEYS_DIR / name
        if key_path.is_file():
            token = key_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    try:
        from huggingface_hub import get_token

        token = (get_token() or "").strip()
        if token:
            return token
    except Exception:
        pass
    raise SystemExit(f"No Hugging Face token. Set HF_TOKEN or create {API_KEYS_DIR / 'hf_token'}.")


def is_fatal_error(status: int | None, text: str) -> bool:
    if status is not None and status < 400:
        return False
    if status in FATAL_STATUS:
        return True
    lower = text.lower()
    return any(needle in lower for needle in FATAL_TEXT)


def retry_after_seconds(resp, fallback: float = 15.0) -> float:
    ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if ra:
        try:
            return min(max(float(ra), 1.0), 300.0)
        except ValueError:
            pass

    rl = resp.headers.get("RateLimit") or resp.headers.get("ratelimit") or ""
    match = re.search(r"[;,]t=(\d+(?:\.\d+)?)", rl)
    if match:
        return min(max(float(match.group(1)), 1.0), 300.0)

    return min(max(fallback, 1.0), 300.0)


def resolve_hf_hub_model_id(model_key: str, *, override: str | None = None) -> str:
    """Resolve HF Hub model id for a given llm-coherence `model_key`."""
    if override:
        return override
    api_model = model_name_for_key(model_key)
    if not api_model:
        raise SystemExit(f"Cannot resolve HF hub model for {model_key!r}: unknown model key.")
    # Exact match first (case-sensitive).
    if api_model in HF_HUB_MODEL_ALIASES:
        return HF_HUB_MODEL_ALIASES[api_model]

    # Case-insensitive match next (covers router-returned casing differences).
    for k, v in HF_HUB_MODEL_ALIASES.items():
        if k.lower() == api_model.lower():
            return v

    # If model came in as openrouter/<org>/<model>, try stripping the prefix.
    if api_model.startswith("openrouter/"):
        stripped = api_model[len("openrouter/") :]
        if stripped in HF_HUB_MODEL_ALIASES:
            return HF_HUB_MODEL_ALIASES[stripped]
        for k, v in HF_HUB_MODEL_ALIASES.items():
            if k.lower() == stripped.lower():
                return v

        # As a last resort, accept the stripped id as-is. HF routers generally
        # normalize casing, and this keeps the function usable for additional
        # models without constantly extending HF_HUB_MODEL_ALIASES.
        return stripped

    return api_model


def resolve_hf_router_model(
    model_key: str, *, hf_model_override: str | None = None, hf_provider: str = DEFAULT_HF_PROVIDER
) -> str:
    hub_id = resolve_hf_hub_model_id(model_key, override=hf_model_override)
    return f"{hub_id}:{hf_provider}"


# HF Hub ids -> OpenRouter catalog slugs (for live $/M rates when HF usage has no cost).
_HF_HUB_TO_OPENROUTER_SLUG: dict[str, str] = {
    "moonshotai/Kimi-K2": "moonshotai/kimi-k2",
    "moonshotai/Kimi-K2-Instruct": "moonshotai/kimi-k2",
    "moonshotai/Kimi-K2-Instruct-0905": "moonshotai/kimi-k2",
    "moonshotai/Kimi-K2-Thinking": "moonshotai/kimi-k2-thinking",
}


def openrouter_slug_for_hf_router_model(router_model: str) -> str:
    """Map `hub_id:provider` to an OpenRouter model id for live rate lookup."""
    hub_id = (router_model or "").split(":", 1)[0].strip()
    if hub_id in _HF_HUB_TO_OPENROUTER_SLUG:
        return _HF_HUB_TO_OPENROUTER_SLUG[hub_id]
    for k, v in _HF_HUB_TO_OPENROUTER_SLUG.items():
        if k.lower() == hub_id.lower():
            return v
    return hub_id.lower()
