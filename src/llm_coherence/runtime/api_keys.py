"""Load provider API keys from environment variables or ``api_keys/*.txt`` files."""

from __future__ import annotations

import os
from pathlib import Path

from llm_coherence.paths import API_KEYS_DIR

API_KEY_ENV_BY_TYPE: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gdm": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "togetherai": "TOGETHER_AI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

KEY_PREFIX_BY_TYPE: dict[str, str] = {
    "openai": "sk-",
    "anthropic": "sk-ant-",
    "openrouter": "sk-or-",
}


def api_key_path(provider: str) -> Path:
    """Default ``api_keys/api_key_<provider>.txt`` path for a provider."""
    return API_KEYS_DIR / f"api_key_{provider}.txt"


def _env_key_looks_real(provider: str, key: str) -> bool:
    key = key.strip()
    prefix = KEY_PREFIX_BY_TYPE.get(provider)
    if not key:
        return False
    if prefix is None:
        return len(key) >= 32
    return key.startswith(prefix) and len(key) >= 40


def load_api_key(provider: str, *, key_path: Path | None = None) -> str | None:
    """Return a provider API key from env (if set) or a key file."""
    env_var = API_KEY_ENV_BY_TYPE.get(provider)
    if env_var:
        env_key = (os.environ.get(env_var) or "").strip()
        if env_key and _env_key_looks_real(provider, env_key):
            return env_key

    path = key_path if key_path is not None else api_key_path(provider)
    if path.is_file():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def require_api_key(provider: str, *, key_path: Path | None = None) -> str:
    """Like :func:`load_api_key`, but raise when no key is available."""
    key = load_api_key(provider, key_path=key_path)
    if key is not None:
        return key
    env_var = API_KEY_ENV_BY_TYPE.get(provider, f"{provider.upper()}_API_KEY")
    path = key_path if key_path is not None else api_key_path(provider)
    raise ValueError(
        f"No API key found for {provider!r}. Set {env_var} or create {path}."
    )


def ensure_api_key_env(provider: str) -> str:
    """Load a key and export it to the provider env var for LiteLLM clients."""
    key = require_api_key(provider)
    env_var = API_KEY_ENV_BY_TYPE[provider]
    os.environ[env_var] = key
    return key
