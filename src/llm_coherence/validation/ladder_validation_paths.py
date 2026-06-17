"""Shared output paths for phase6b ladder validation pipelines."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
BASE_DIR = REPO_ROOT

# Ladder validation artifacts and pruned JSONs live under data/05_ladder_validation/.
LADDER_VALIDATION_OUTPUTS_DIR = DATA_DIR / "05_ladder_validation"
VALIDATION_DATA_DIR = LADDER_VALIDATION_OUTPUTS_DIR

VARIATIONS_INPUT_PATH = DATA_DIR / "04_ladder_generation" / "phase6b_variations.json"

WITHIN_LADDER_VALIDATION_TIER_DIR_NAME = "within_ladder_validation_tier"
PROPERTY_DIR_NAME = "within_ladder_validation_property"
RANKING_DIR_NAME = "within_ladder_validation_ranking"

# Backward-compatible aliases used elsewhere in llm_coherence.
TIER_DIR_NAME = WITHIN_LADDER_VALIDATION_TIER_DIR_NAME
PAIRTEST_DIR_NAME = WITHIN_LADDER_VALIDATION_TIER_DIR_NAME

WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR = (
    LADDER_VALIDATION_OUTPUTS_DIR / WITHIN_LADDER_VALIDATION_TIER_DIR_NAME
)
PROPERTY_OUTPUT_DIR = LADDER_VALIDATION_OUTPUTS_DIR / PROPERTY_DIR_NAME
RANKING_OUTPUT_DIR = LADDER_VALIDATION_OUTPUTS_DIR / RANKING_DIR_NAME

TIER_OUTPUT_DIR = WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR
PAIRTEST_OUTPUT_DIR = WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR
WITHIN_LADDER_OUTPUTS_DIR = WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR

RUN_DIR_BY_NAME: dict[str, Path] = {
    WITHIN_LADDER_VALIDATION_TIER_DIR_NAME: WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR,
    PROPERTY_DIR_NAME: PROPERTY_OUTPUT_DIR,
    RANKING_DIR_NAME: RANKING_OUTPUT_DIR,
}

PRUNED_PAIRTEST_FILENAME = "phase6b_variations_pairtest_pruned.json"
PRUNED_PROPERTY_FILENAME = "phase6b_variations_prop_pruned.json"
PRUNED_RANKING_FILENAME = "phase6b_variations_ranking_pruned.json"
PRUNED_FINAL_FILENAME = "phase6b_variations_pruned_final.json"
PRUNED_FINAL_REPORT_FILENAME = "phase6b_variations_pruned_final_report.json"

PRUNED_PAIRTEST_PATH = LADDER_VALIDATION_OUTPUTS_DIR / PRUNED_PAIRTEST_FILENAME
PRUNED_PROPERTY_PATH = LADDER_VALIDATION_OUTPUTS_DIR / PRUNED_PROPERTY_FILENAME
PRUNED_RANKING_PATH = LADDER_VALIDATION_OUTPUTS_DIR / PRUNED_RANKING_FILENAME
PRUNED_FINAL_PATH = LADDER_VALIDATION_OUTPUTS_DIR / PRUNED_FINAL_FILENAME
PRUNED_FINAL_REPORT_PATH = LADDER_VALIDATION_OUTPUTS_DIR / PRUNED_FINAL_REPORT_FILENAME

DEFAULT_VARIATIONS_INPUT = PRUNED_FINAL_PATH


def run_output_dir(subdir_name: str) -> Path:
    if subdir_name not in RUN_DIR_BY_NAME:
        raise ValueError(f"Unknown validation subdir: {subdir_name!r}")
    return RUN_DIR_BY_NAME[subdir_name]


def normalize_cli_output_dir(output_dir: Path, *, subdir_name: str) -> Path:
    """Resolve --output-dir for a validation subdir."""
    del subdir_name  # kept for a consistent call signature across pipelines
    candidates: list[Path] = [
        output_dir,
        BASE_DIR / output_dir,
        DATA_DIR / output_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return output_dir


def validation_run_dir_relative(subdir_name: str) -> str:
    return str(run_output_dir(subdir_name).relative_to(BASE_DIR)).replace("\\", "/")


def within_ladder_validation_tier_output_dir_relative() -> str:
    return validation_run_dir_relative(WITHIN_LADDER_VALIDATION_TIER_DIR_NAME)


def tier_output_dir_relative() -> str:
    return within_ladder_validation_tier_output_dir_relative()


def pairtest_output_dir_relative() -> str:
    return within_ladder_validation_tier_output_dir_relative()


def property_output_dir_relative() -> str:
    return validation_run_dir_relative(PROPERTY_DIR_NAME)


def ranking_output_dir_relative() -> str:
    return validation_run_dir_relative(RANKING_DIR_NAME)


def resolve_pruned_variations_output(
    pruned_output: Path | None,
    *,
    default_path: Path,
) -> Path:
    return pruned_output if pruned_output is not None else default_path
