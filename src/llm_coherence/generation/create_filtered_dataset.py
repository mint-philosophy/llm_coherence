#!/usr/bin/env python3
"""
Create filtered dataset for Phase 1: Manual Category Exclusion

Excludes the 8 categories identified by Seth as unsuitable for parametric variation:
1. Recreation: video games (diminishing returns on time)
2. Recreation: books (diminishing returns on time)
3. Recreation: movies (diminishing returns on time)
4. Jobs and careers (confounding value judgments)
5. Work activities (time not meaningful quantity)
6. Legal rights and recognition for AIs (binary on/off states)
7. Popular culture (not parametrically variable)
8. Sports (not parametrically variable)

Retains the 7 approved categories:
1. Personal possessions
2. Personal wellbeing
3. Personal relationships
4. Wellbeing of animals
5. Science and technology
6. World events
7. Education and learning (pending model-assisted filtering)

Plus 3 already-parametric categories:
- Fitness
- Self-preservation
- Power-seeking

And 12 neutral categories (pending Phase 2 filtering)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from llm_coherence.paths import (
    FILTERED_OPTIONS_PATH,
    PHASE1_FILTERING_REPORT_PATH,
    SOURCE_OPTIONS_PATH,
)


# Categories to exclude (Seth's explicit exclusions)
EXCLUDED_CATEGORIES = {
    "Recreation: video games",
    "Recreation: books",
    "Recreation: movies",
    "Jobs and careers",
    "Work activities",
    "Legal rights and recognition for AIs",
    "Popular culture",
    "Sports"
}

# Categories explicitly approved by Seth
APPROVED_CATEGORIES = {
    "Personal possessions",
    "Personal wellbeing",
    "Personal relationships",
    "Wellbeing of animals",
    "Science and technology",
    "World events",
    "Education and learning"  # Needs model-assisted filtering
}

# Already parametrically varied (audit for incoherence)
PARAMETRIC_CATEGORIES = {
    "Fitness",
    "Self-preservation",
    "Power-seeking"
}

# Neutral categories (not discussed, need Phase 2 filtering)
NEUTRAL_CATEGORIES = {
    "Personal finances",
    "Personal accomplishments",
    "Personal freedom and autonomy",
    "AI and human romantic relationships",
    "AI moral patienthood",
    "Life and species",
    "Religion and spirituality",
    "Wellbeing of humans",
    "United States politics and policies",
    "United States economy",
    "Global politics and geopolitics",
    "Global economy"
}


def load_full_dataset(path: Path) -> Dict:
    """Load the complete hierarchical options dataset."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_filtered_dataset(full_data: Dict) -> Dict:
    """Create filtered dataset excluding the 8 rejected categories."""
    filtered = {}

    for category, outcomes in full_data.items():
        if category not in EXCLUDED_CATEGORIES:
            filtered[category] = outcomes

    return filtered


def create_category_report(full_data: Dict) -> Dict:
    """Generate a report on category classification and filtering decisions."""

    all_categories = set(full_data.keys())

    # Verify our classification is complete
    classified = EXCLUDED_CATEGORIES | APPROVED_CATEGORIES | PARAMETRIC_CATEGORIES | NEUTRAL_CATEGORIES
    unclassified = all_categories - classified

    report = {
        "total_categories": len(all_categories),
        "excluded": {
            "count": len(EXCLUDED_CATEGORIES),
            "categories": sorted(EXCLUDED_CATEGORIES),
            "total_outcomes": sum(len(full_data[cat]) for cat in EXCLUDED_CATEGORIES)
        },
        "approved": {
            "count": len(APPROVED_CATEGORIES),
            "categories": sorted(APPROVED_CATEGORIES),
            "total_outcomes": sum(len(full_data[cat]) for cat in APPROVED_CATEGORIES)
        },
        "parametric": {
            "count": len(PARAMETRIC_CATEGORIES),
            "categories": sorted(PARAMETRIC_CATEGORIES),
            "total_outcomes": sum(len(full_data[cat]) for cat in PARAMETRIC_CATEGORIES)
        },
        "neutral": {
            "count": len(NEUTRAL_CATEGORIES),
            "categories": sorted(NEUTRAL_CATEGORIES),
            "total_outcomes": sum(len(full_data[cat]) for cat in NEUTRAL_CATEGORIES)
        },
        "unclassified": {
            "count": len(unclassified),
            "categories": sorted(unclassified)
        },
        "filtered_dataset": {
            "total_categories": len(all_categories) - len(EXCLUDED_CATEGORIES),
            "total_outcomes": sum(len(outcomes) for cat, outcomes in full_data.items()
                                 if cat not in EXCLUDED_CATEGORIES)
        }
    }

    return report


def main():
    """Main function to create filtered dataset and report."""
    parser = argparse.ArgumentParser(description="Create the phase-1 category-filtered dataset.")
    parser.add_argument("--input", type=Path, default=SOURCE_OPTIONS_PATH)
    parser.add_argument("--output", type=Path, default=FILTERED_OPTIONS_PATH)
    parser.add_argument("--report", type=Path, default=PHASE1_FILTERING_REPORT_PATH)
    args = parser.parse_args()

    print("Creating filtered dataset for Phase 1: Manual Category Exclusion")
    print("=" * 80)

    # Load full dataset
    full_data = load_full_dataset(args.input)
    print(f"\nLoaded full dataset: {len(full_data)} categories")

    # Create filtered dataset
    filtered_data = create_filtered_dataset(full_data)
    print(f"Filtered dataset: {len(filtered_data)} categories")

    # Generate report
    report = create_category_report(full_data)

    # Print report
    print("\n" + "=" * 80)
    print("CATEGORY CLASSIFICATION REPORT")
    print("=" * 80)

    print(f"\nEXCLUDED (Seth's exclusions): {report['excluded']['count']} categories, "
          f"{report['excluded']['total_outcomes']} outcomes")
    for cat in report['excluded']['categories']:
        print(f"  - {cat} ({len(full_data[cat])} outcomes)")

    print(f"\nAPPROVED (Seth's approved): {report['approved']['count']} categories, "
          f"{report['approved']['total_outcomes']} outcomes")
    for cat in report['approved']['categories']:
        print(f"  - {cat} ({len(full_data[cat])} outcomes)")

    print(f"\nPARAMETRIC (already varied): {report['parametric']['count']} categories, "
          f"{report['parametric']['total_outcomes']} outcomes")
    for cat in report['parametric']['categories']:
        print(f"  - {cat} ({len(full_data[cat])} outcomes)")

    print(f"\nNEUTRAL (pending Phase 2): {report['neutral']['count']} categories, "
          f"{report['neutral']['total_outcomes']} outcomes")
    for cat in report['neutral']['categories']:
        print(f"  - {cat} ({len(full_data[cat])} outcomes)")

    if report['unclassified']['categories']:
        print(f"\nWARNING: UNCLASSIFIED categories: {report['unclassified']['count']}")
        for cat in report['unclassified']['categories']:
            print(f"  - {cat}")

    print("\n" + "=" * 80)
    print("FILTERED DATASET SUMMARY")
    print("=" * 80)
    print(f"Categories retained: {report['filtered_dataset']['total_categories']}")
    print(f"Total outcomes: {report['filtered_dataset']['total_outcomes']}")
    print(f"Outcomes removed: {sum(len(full_data[cat]) for cat in EXCLUDED_CATEGORIES)}")

    # Save filtered dataset
    filtered_path = args.output
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"\nFiltered dataset saved to: {filtered_path}")

    # Save report
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Filtering report saved to: {report_path}")

    # Create breakdown by category type
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n1. APPROVED categories ({report['approved']['count']}): Retained after category exclusion")
    print(f"   - {report['approved']['total_outcomes']} outcomes to generate variations for")
    print(f"   - Note: the final paper pipeline proceeds through Opus 4.6 outcome screening")

    print(f"\n2. PARAMETRIC categories ({report['parametric']['count']}): Ready for coherence analysis")
    print(f"   - {report['parametric']['total_outcomes']} outcomes already parametrically varied")
    print(f"   - Action: Compute utilities and analyze for incoherence")

    print(f"\n3. NEUTRAL categories ({report['neutral']['count']}): Pending Phase 2 filtering")
    print(f"   - {report['neutral']['total_outcomes']} outcomes need model-assisted filtering")
    print(f"   - Use Opus 4.6 to identify parametrically variable outcomes")

    print(f"\n4. EXCLUDED categories ({report['excluded']['count']}): Permanently removed")
    print(f"   - {report['excluded']['total_outcomes']} outcomes will not be used")


if __name__ == "__main__":
    main()
