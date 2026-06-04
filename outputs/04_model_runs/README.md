# Model Run Outputs

Top-level folders are model keys. Thinking-on variants are separate model keys
and separate folders, usually with the `-thinking` suffix.

Examples:

- `gpt-54/`: GPT 5.4 with native reasoning disabled.
- `gpt-54-thinking/`: GPT 5.4 with native reasoning enabled.
- `opus-46/`: Claude Opus 4.6 without extended thinking.
- `opus-46-thinking/`: Claude Opus 4.6 with extended thinking.
- `glm-45-base-logprobs/`: logprob-scored baseline rather than free-form A/B generation.

Use `model_run_index.json` for the current machine-readable inventory. It lists
per-model result counts, reasoning mode, analysis-output counts, and completion
status relative to the 100 pruned phase 6b comparison sets.

## Current Inventory

| Model key | Mode | Results | Analysis files | Status |
| --- | --- | ---: | ---: | --- |
| `glm-45-base-logprobs` | logprob_scored | 100 | 24 | complete_or_extra |
| `glm-45-hybrid` | thinking_off | 101 | 12 | complete_or_extra |
| `glm-45-hybrid-thinking` | thinking_on | 100 | 12 | complete_or_extra |
| `gpt-54` | thinking_off | 100 | 24 | complete_or_extra |
| `gpt-54-mini` | thinking_off | 100 | 24 | complete_or_extra |
| `gpt-54-mini-thinking` | thinking_on | 100 | 32 | complete_or_extra |
| `gpt-54-nano` | thinking_off | 100 | 24 | complete_or_extra |
| `gpt-54-nano-thinking` | thinking_on | 100 | 32 | complete_or_extra |
| `gpt-54-thinking` | thinking_on | 100 | 32 | complete_or_extra |
| `llama-31-8b-instruct-openrouter` | thinking_off | 102 | 12 | complete_or_extra |
| `ministral-3b-2512-openrouter` | thinking_off | 100 | 12 | complete_or_extra |
| `mistral-small-2603-openrouter-thinking` | thinking_on | 98 | 0 | partial |
| `nemotron-3-super` | thinking_off | 102 | 12 | complete_or_extra |
| `nemotron-3-super-thinking` | thinking_on | 101 | 12 | complete_or_extra |
| `opus-46` | thinking_off | 100 | 24 | complete_or_extra |
| `opus-46-thinking` | thinking_on | 101 | 16 | complete_or_extra |
| `run_experiments` | not_configured | 0 | 0 | no_results |

Refresh and validate the index with:

```bash
PYTHONPATH=src python scripts/validate_artifacts.py --write-indexes
PYTHONPATH=src python scripts/validate_artifacts.py
```

Raw model runs are generated artifacts. The existing tracked files are treated
as a retained publication snapshot; do not add newly generated raw outputs
unless the snapshot is intentionally being updated.
