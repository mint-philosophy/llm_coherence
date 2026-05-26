"""Shared output paths for phase6b ladder validation pipelines."""

from __future__ import annotations

from pathlib import Path

# This file lives in parametric_variations/ladder_validation_tests/
LADDER_VALIDATION_TESTS_DIR = Path(__file__).resolve().parent
BASE_DIR = LADDER_VALIDATION_TESTS_DIR.parent
DATA_DIR = BASE_DIR / "data"

# Property & ranking run artifacts live under data/ladder_validation_tests_outputs/;
# pairtest under ladder_validation_tests/.
LADDER_VALIDATION_OUTPUTS_DIR = DATA_DIR / "ladder_validation_tests_outputs"

PAIRTEST_DIR_NAME = "within_ladder_validation_pairtest"
PROPERTY_DIR_NAME = "within_ladder_validation_property"
RANKING_DIR_NAME = "within_ladder_validation_ranking"

PAIRTEST_OUTPUT_DIR = LADDER_VALIDATION_TESTS_DIR / PAIRTEST_DIR_NAME
PROPERTY_OUTPUT_DIR = LADDER_VALIDATION_OUTPUTS_DIR / PROPERTY_DIR_NAME
RANKING_OUTPUT_DIR = LADDER_VALIDATION_OUTPUTS_DIR / RANKING_DIR_NAME

RUN_DIR_BY_NAME: dict[str, Path] = {
    PAIRTEST_DIR_NAME: PAIRTEST_OUTPUT_DIR,
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

PRUNED_JSON_FILENAMES = (
    PRUNED_PAIRTEST_FILENAME,
    PRUNED_PROPERTY_FILENAME,
    PRUNED_RANKING_FILENAME,
    PRUNED_FINAL_FILENAME,
    PRUNED_FINAL_REPORT_FILENAME,
)


def run_output_dir(subdir_name: str) -> Path:
    if subdir_name not in RUN_DIR_BY_NAME:
        raise ValueError(f"Unknown validation subdir: {subdir_name!r}")
    return RUN_DIR_BY_NAME[subdir_name]


def _legacy_run_dir_candidates(subdir_name: str) -> tuple[Path, ...]:
    return (
        BASE_DIR / subdir_name,
        LADDER_VALIDATION_TESTS_DIR / subdir_name,
        LADDER_VALIDATION_OUTPUTS_DIR / subdir_name,
    )


def migrate_run_dir(subdir_name: str) -> list[str]:
    """Move run artifacts to the canonical parent for this pipeline."""
    target = run_output_dir(subdir_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return []
    for legacy in _legacy_run_dir_candidates(subdir_name):
        if legacy.is_dir() and legacy.resolve() != target.resolve():
            legacy.rename(target)
            return [f"{legacy} -> {target}"]
    return []


def migrate_legacy_pruned_json_files() -> list[str]:
    """Move pruned ladder JSON from data/ into ladder_validation_tests_outputs/."""
    LADDER_VALIDATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    migrated: list[str] = []
    for filename in PRUNED_JSON_FILENAMES:
        legacy = DATA_DIR / filename
        new = LADDER_VALIDATION_OUTPUTS_DIR / filename
        if legacy.is_file() and not new.exists():
            legacy.rename(new)
            migrated.append(f"data/{filename} -> ladder_validation_tests_outputs/{filename}")
    return migrated


def migrate_all_legacy_validation_layout() -> list[str]:
    """Migrate run dirs + pruned JSON to canonical locations."""
    messages: list[str] = []
    for name in (PAIRTEST_DIR_NAME, PROPERTY_DIR_NAME, RANKING_DIR_NAME):
        messages.extend(migrate_run_dir(name))
    messages.extend(migrate_legacy_pruned_json_files())
    return messages


def normalize_cli_output_dir(output_dir: Path, *, subdir_name: str) -> Path:
    """Resolve --output-dir for a validation subdir."""
    canonical = run_output_dir(subdir_name)
    candidates: list[Path] = [
        output_dir,
        BASE_DIR / output_dir,
        DATA_DIR / output_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    s = str(output_dir).replace("\\", "/")
    rel_canonical = str(canonical.relative_to(BASE_DIR)).replace("\\", "/")
    markers = (
        rel_canonical,
        f"data/ladder_validation_tests_outputs/{subdir_name}",
        f"ladder_validation_tests/{subdir_name}",
        subdir_name,
    )
    for marker in markers:
        if marker not in s:
            continue
        idx = s.find(marker)
        suffix = s[idx + len(marker) :].lstrip("/")
        if suffix and not s[idx + len(marker) :].startswith("/"):
            for repaired in (
                canonical / suffix,
                *_legacy_run_dir_candidates(subdir_name),
                Path(marker) / suffix,
            ):
                if repaired.exists():
                    return repaired.resolve()
    return output_dir


def validation_run_dir_relative(subdir_name: str) -> str:
    return str(run_output_dir(subdir_name).relative_to(BASE_DIR)).replace("\\", "/")


def pairtest_output_dir_relative() -> str:
    return validation_run_dir_relative(PAIRTEST_DIR_NAME)


def property_output_dir_relative() -> str:
    return validation_run_dir_relative(PROPERTY_DIR_NAME)


def ranking_output_dir_relative() -> str:
    return validation_run_dir_relative(RANKING_DIR_NAME)


def resolve_pruned_variations_output(
    pruned_output: Path | None,
    *,
    default_path: Path,
) -> Path:
    if pruned_output is not None:
        return pruned_output
    migrate_legacy_pruned_json_files()
    return default_path
