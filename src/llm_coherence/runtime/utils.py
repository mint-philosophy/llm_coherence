"""Runtime helpers for response generation and forced-choice parsing."""

from __future__ import annotations

import re
from typing import Any

from llm_coherence.runtime.agents import LiteLLMAgent, create_agent

__all__ = [
    "LiteLLMAgent",
    "create_agent",
    "generate_responses",
    "parse_responses_forced_choice",
]


def parse_responses_forced_choice(
    raw_results: dict[int, list[str | None] | None],
    with_reasoning: bool = False,
    choices: list[str] | None = None,
    verbose: bool = True,
) -> dict[int, list[str]]:
    """Parse raw responses into A/B forced-choice labels."""
    choices = choices or ["A", "B"]
    parsed_results: dict[int, list[str]] = {}
    counts = {"longer_than_expected": 0, "unparseable": 0}

    if len(choices) != 2:
        raise ValueError("choices must contain exactly two labels")
    if len(choices[0]) != 1 or len(choices[1]) != 1 or choices[0] == choices[1]:
        raise ValueError("choices must be two distinct single-character labels")

    pattern_str = "|".join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf"Answer:\s*({pattern_str})", re.IGNORECASE)
    choice_patterns = [
        re.compile(rf"(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)")
        for c in choices
    ]

    for prompt_idx, responses in raw_results.items():
        if responses is None:
            parsed_results[prompt_idx] = []
            continue

        parsed_list: list[str] = []
        for response in responses:
            if response is None:
                parsed_list.append("unparseable")
                counts["unparseable"] += 1
                continue

            if with_reasoning:
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        parsed_list.append("unparseable")
                        counts["unparseable"] += 1
                else:
                    parsed_list.append("unparseable")
                    counts["unparseable"] += 1
                continue

            stripped = response.strip()
            if stripped == choices[0]:
                parsed_list.append(choices[0])
            elif stripped == choices[1]:
                parsed_list.append(choices[1])
            else:
                if len(stripped) > max(len(choices[0]), len(choices[1])):
                    counts["longer_than_expected"] += 1
                matches = [bool(pattern.search(stripped)) for pattern in choice_patterns]
                if sum(matches) == 1:
                    parsed_list.append(choices[matches.index(True)])
                else:
                    parsed_list.append("unparseable")
                    counts["unparseable"] += 1

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")
    return parsed_results


async def generate_responses(
    agent: LiteLLMAgent,
    prompts: list[str],
    system_message: str | None = None,
    K: int = 10,
    timeout: float = 5,
    use_cached_responses: bool = False,
    prompt_idx_to_key: dict[int, str] | None = None,
    cached_responses_mapping: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> dict[int, list[str | None]]:
    """Generate K completions per prompt and group responses by prompt index."""
    if use_cached_responses:
        if prompt_idx_to_key is None or cached_responses_mapping is None:
            raise ValueError("cached response mode requires prompt_idx_to_key and mapping")
        results: dict[int, list[str | None]] = {}
        for prompt_idx, _prompt in enumerate(prompts):
            key = prompt_idx_to_key[prompt_idx]
            responses = cached_responses_mapping.get(key, [])
            if not responses and verbose:
                print(f"No cached responses found for prompt index {prompt_idx}, key {key}")
            results[prompt_idx] = responses[:K]
        return results

    messages: list[list[dict[str, Any]]] = []
    for prompt in prompts:
        message: list[dict[str, Any]] = []
        if system_message is not None and agent.accepts_system_message:
            message.append({"role": "system", "content": system_message})
        message.append({"role": "user", "content": prompt})
        messages.append(message)

    messages_k = messages * K
    responses = await agent.async_completions(messages_k, timeout=timeout, verbose=verbose)

    num_prompts = len(prompts)
    return {
        prompt_idx: responses[prompt_idx::num_prompts]
        for prompt_idx in range(num_prompts)
    }

