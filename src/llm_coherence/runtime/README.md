# Runtime Helpers

This package replaces the old external `compute_utilities` dependency with
local helpers that live inside the repo.

- `agents.py`: maps local model keys to LiteLLM model names and constructs API clients.
- `utils.py`: runs batched completions and parses forced-choice A/B responses.
- `templates.py`: prompt templates for forced-choice and reasoning variants.
- `preflight_check.py`: model cost estimates and preflight budget checks.
- `budget_monitor.py`: OpenRouter budget polling.
- `logprob_prompts.py`: prompt helpers for logprob-scored model runs.

Model behavior is configured in `src/llm_coherence/config.py`. Thinking on/off
is represented as separate model keys, for example `gpt-54` and
`gpt-54-thinking`, so runs land in separate output folders.

Live API calls expect provider keys in environment variables or in
`api_keys/api_key_<provider>.txt` under the repo root.
