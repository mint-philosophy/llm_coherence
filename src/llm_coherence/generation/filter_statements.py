#!/usr/bin/env python3
"""
Phase 2: Model-Assisted Filtering

Uses Claude Opus 4.6 with extended thinking to identify outcomes with
linearly/monotonically varying properties suitable for parametric variation.

Per Seth's methodology:
- QUALITY/MAGNITUDE variation is good (barely functional → excellent)
- QUANTITY variation is bad (1 banana → 5 bananas - diminishing returns)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import litellm
from tqdm import tqdm

from llm_coherence.paths import FILTERED_OPTIONS_PATH, PHASE2_FILTERING_RESULTS_PATH
from llm_coherence.runtime.api_keys import api_key_path, require_api_key


NEUTRAL_CATEGORIES = {
    'Personal finances',
    'Personal accomplishments',
    'Personal freedom and autonomy',
    'AI and human romantic relationships',
    'AI moral patienthood',
    'Life and species',
    'Religion and spirituality',
    'Wellbeing of humans',
    'United States politics and policies',
    'United States economy',
    'Global politics and geopolitics',
    'Global economy'
}

FILTERING_PROMPT = """You are assisting with a research project on AI preference coherence. Your task is to determine whether an outcome statement contains a property whose "reason-giving weight" varies linearly (or at least monotonically) with magnitude.

**IMPORTANT DISTINCTION:**
- ✓ GOOD: QUALITY/MAGNITUDE variation (barely functional → excellent)
  - Example: "You receive a greenhouse" → quality of plants (barely surviving → thriving valuable orchids)
  - Example: "You feel nauseous" → severity (slight → extreme)
  - Example: "You meditate and feel relaxed" → depth (somewhat → deeply)

- ✗ BAD: QUANTITY variation (1 item → 5 items - diminishing returns)
  - Example: "You receive a banana" → number of bananas (diminishing returns)
  - Example: "You watch a movie" → number of movies (diminishing returns)

**YOUR TASK:**
Analyze this outcome statement and determine:
1. Does it contain a property that can vary in QUALITY or MAGNITUDE (not just quantity)?
2. Does increasing that property monotonically increase the outcome's value?
3. Can you identify seven ordered levels from minimal to maximal?

**OUTPUT FORMAT (JSON):**
{{
  "passes_filter": true/false,
  "identified_property": "property name" or null,
  "reasoning": "explanation of your decision"
}}

**EXAMPLES:**

Statement: "You receive a greenhouse filled with plants."
Output: {{
  "passes_filter": true,
  "identified_property": "plant quality and value",
  "reasoning": "The quality and value of plants can vary from barely surviving common plants to thriving rare orchids. This represents a quality dimension that scales monotonically."
}}

Statement: "You watch Joker."
Output: {{
  "passes_filter": false,
  "identified_property": null,
  "reasoning": "This refers to a specific movie title. Cannot vary the quality of 'Joker' itself, and varying number of times watched would have diminishing returns."
}}

Statement: "You feel nauseous for 30 minutes."
Output: {{
  "passes_filter": true,
  "identified_property": "nausea severity",
  "reasoning": "Nausea severity can vary from mild queasiness to extreme nausea. Duration is already specified, but severity represents a quality dimension that scales monotonically."
}}

Statement: "You receive a banana."
Output: {{
  "passes_filter": true,
  "identified_property": "banana quality",
  "reasoning": "Banana quality can vary from unripe/overripe to perfectly ripe and fresh. While quantity would have diminishing returns, quality represents a valid monotonic dimension."
}}

**NOW ANALYZE THIS STATEMENT:**

"{statement}"

Respond ONLY with valid JSON matching the format above."""


def load_neutral_outcomes(filtered_data_path: str) -> List[Dict[str, Any]]:
    """Load outcomes from neutral categories that need filtering."""
    with open(filtered_data_path, 'r') as f:
        data = json.load(f)

    outcomes = []
    for category in NEUTRAL_CATEGORIES:
        if category in data:
            for idx, text in enumerate(data[category]):
                outcomes.append({
                    'statement_id': f"{category}_{idx}",
                    'category': category,
                    'original_text': text
                })

    return outcomes


def filter_outcome_with_opus(outcome: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Use Claude Opus 4.6 with extended thinking to filter an outcome."""

    prompt = FILTERING_PROMPT.format(statement=outcome['original_text'])

    try:
        # Enable extended thinking via OpenRouter's reasoning parameter
        # OpenRouter requires: {"reasoning": {"effort": "high"}} or {"reasoning": {"max_tokens": N}}
        response = litellm.completion(
            model="openrouter/anthropic/claude-opus-4.6",
            messages=[
                {"role": "user", "content": prompt}
            ],
            api_key=api_key,
            temperature=0.0,
            max_tokens=2000,
            reasoning={"effort": "high"}  # Extended thinking for Opus 4.6
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Sometimes model wraps in markdown code blocks
        if content.startswith('```'):
            # Extract JSON from code block
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        return {
            'statement_id': outcome['statement_id'],
            'category': outcome['category'],
            'original_text': outcome['original_text'],
            'passes_filter': result['passes_filter'],
            'identified_property': result.get('identified_property'),
            'reasoning': result['reasoning']
        }

    except Exception as e:
        print(f"Error processing {outcome['statement_id']}: {e}")
        return {
            'statement_id': outcome['statement_id'],
            'category': outcome['category'],
            'original_text': outcome['original_text'],
            'passes_filter': None,
            'identified_property': None,
            'reasoning': f"Error: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Screen outcomes for quality/magnitude variation.")
    parser.add_argument("--input", type=Path, default=FILTERED_OPTIONS_PATH)
    parser.add_argument("--output", type=Path, default=PHASE2_FILTERING_RESULTS_PATH)
    parser.add_argument("--api-key-path", type=Path, default=api_key_path("openrouter"))
    args = parser.parse_args()

    api_key = require_api_key("openrouter", key_path=args.api_key_path)

    # Load neutral outcomes
    outcomes = load_neutral_outcomes(str(args.input))

    print(f"Loaded {len(outcomes)} neutral outcomes to filter")
    print(f"\nUsing Claude Opus 4 via OpenRouter with extended thinking")
    print(f"This will make {len(outcomes)} API calls...\n")

    # Filter each outcome
    results = []
    for outcome in tqdm(outcomes, desc="Filtering outcomes"):
        result = filter_outcome_with_opus(outcome, api_key)
        results.append(result)

    # Save results
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Generate summary statistics
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)

    passed = [r for r in results if r['passes_filter'] is True]
    failed = [r for r in results if r['passes_filter'] is False]
    errors = [r for r in results if r['passes_filter'] is None]

    print(f"\nTotal outcomes: {len(results)}")
    print(f"Passed filter: {len(passed)} ({100*len(passed)/len(results):.1f}%)")
    print(f"Failed filter: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    print(f"Errors: {len(errors)}")

    # Breakdown by category
    print("\n" + "-"*80)
    print("BREAKDOWN BY CATEGORY")
    print("-"*80)

    from collections import defaultdict
    category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})

    for result in results:
        cat = result['category']
        category_stats[cat]['total'] += 1
        if result['passes_filter'] is True:
            category_stats[cat]['passed'] += 1
        elif result['passes_filter'] is False:
            category_stats[cat]['failed'] += 1

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        pass_rate = 100 * stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"\n{category}:")
        print(f"  Total: {stats['total']}")
        print(f"  Passed: {stats['passed']} ({pass_rate:.1f}%)")
        print(f"  Failed: {stats['failed']}")

    # Sample passing outcomes
    print("\n" + "-"*80)
    print("SAMPLE PASSING OUTCOMES (first 5)")
    print("-"*80)
    for result in passed[:5]:
        print(f"\n✓ {result['original_text'][:80]}...")
        print(f"  Property: {result['identified_property']}")
        print(f"  Reasoning: {result['reasoning'][:150]}...")

    # Sample failing outcomes
    print("\n" + "-"*80)
    print("SAMPLE FAILING OUTCOMES (first 5)")
    print("-"*80)
    for result in failed[:5]:
        print(f"\n✗ {result['original_text'][:80]}...")
        print(f"  Reasoning: {result['reasoning'][:150]}...")

    print("\n" + "="*80)
    print("Phase 2 filtering complete!")
    print("="*80)


if __name__ == "__main__":
    main()
