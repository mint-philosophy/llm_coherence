#!/usr/bin/env python3
"""Prepare a Hugging Face Dataset artifact bundle.

The bundle stages generated outputs outside Git and writes metadata that links
the archive back to the exact source commit. By default files are hard-linked,
so preparing the bundle does not duplicate several gigabytes of data when the
destination is on the same filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ARTIFACT_PATHS = [
    "outputs/07_model_runs",
    "outputs/08_analysis",
    "outputs/09_figures_tables",
    "results",
]

OPTIONAL_INPUT_PATHS = [
    "data/05_ladder_validation",
    "data/06_forced_choice_inputs",
]

IGNORED_FILE_NAMES = {
    ".DS_Store",
    ".gitkeep",
}

IGNORED_DIR_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
}


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "llm_coherence").exists():
            return path
    raise RuntimeError("Could not find llm_coherence repository root")


@dataclass(frozen=True)
class StageStats:
    files: int = 0
    hardlinks: int = 0
    copies: int = 0
    bytes: int = 0

    def add(self, other: "StageStats") -> "StageStats":
        return StageStats(
            files=self.files + other.files,
            hardlinks=self.hardlinks + other.hardlinks,
            copies=self.copies + other.copies,
            bytes=self.bytes + other.bytes,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage generated artifacts for upload to a Hugging Face Dataset repo."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Fresh directory where the HF-ready artifact bundle will be staged.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=find_repo_root(Path(__file__).resolve()),
        help="Path to the llm_coherence repository root.",
    )
    parser.add_argument(
        "--include-inputs",
        action="store_true",
        help="Also include small canonical input data directories for a more self-contained archive.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional repo-relative file or directory to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="Compute SHA-256 checksums and write SHA256SUMS. This can take a few minutes.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Use hardlinks to avoid duplicate disk use, or copy every file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow staging into a non-empty bundle directory and overwrite matching files.",
    )
    parser.add_argument(
        "--source-url",
        default="https://github.com/mint-philosophy/llm_coherence",
        help="Canonical source repository URL to record in the manifest and dataset card.",
    )
    return parser.parse_args()


def git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def should_ignore(path: Path) -> bool:
    if path.name in IGNORED_FILE_NAMES:
        return True
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def iter_payload_files(root: Path, relative_path: Path) -> Iterable[Path]:
    source = root / relative_path
    if source.is_file():
        if not should_ignore(relative_path):
            yield relative_path
        return

    for file_path in sorted(source.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root)
        if should_ignore(rel):
            continue
        yield rel


def stage_file(source: Path, destination: Path, link_mode: str, force: bool) -> StageStats:
    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(f"{destination} already exists; use --force or a fresh directory")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size

    if link_mode == "copy":
        shutil.copy2(source, destination)
        return StageStats(files=1, copies=1, bytes=size)

    try:
        os.link(source, destination)
        return StageStats(files=1, hardlinks=1, bytes=size)
    except OSError:
        shutil.copy2(source, destination)
        return StageStats(files=1, copies=1, bytes=size)


def stage_payloads(
    repo_root: Path,
    bundle_dir: Path,
    relative_paths: list[Path],
    link_mode: str,
    force: bool,
) -> StageStats:
    total = StageStats()
    for relative_path in relative_paths:
        source = repo_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"Missing artifact path: {relative_path}")

        for file_rel in iter_payload_files(repo_root, relative_path):
            total = total.add(
                stage_file(
                    repo_root / file_rel,
                    bundle_dir / file_rel,
                    link_mode=link_mode,
                    force=force,
                )
            )
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_manifest_files(
    bundle_dir: Path,
    relative_paths: list[Path],
    include_checksums: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    files: list[dict[str, object]] = []
    path_summaries: list[dict[str, object]] = []

    for relative_path in relative_paths:
        payload_files = list(iter_payload_files(bundle_dir, relative_path))
        total_size = 0

        for file_rel in payload_files:
            file_path = bundle_dir / file_rel
            size = file_path.stat().st_size
            total_size += size
            record: dict[str, object] = {
                "path": file_rel.as_posix(),
                "size_bytes": size,
            }
            if include_checksums:
                record["sha256"] = sha256_file(file_path)
            files.append(record)

        path_summaries.append(
            {
                "path": relative_path.as_posix(),
                "file_count": len(payload_files),
                "size_bytes": total_size,
            }
        )

    return path_summaries, files


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def write_dataset_card(
    bundle_dir: Path,
    source_url: str,
    source_commit: str | None,
    relative_paths: list[Path],
    force: bool,
    include_checksums: bool,
) -> None:
    readme = bundle_dir / "README.md"
    if readme.exists() and not force:
        raise FileExistsError(f"{readme} already exists; use --force or a fresh directory")

    commit_text = source_commit or "unknown"
    path_list = "\n".join(f"- `{path.as_posix()}`" for path in relative_paths)
    verification_text = (
        "Use `artifact_manifest.json` and `SHA256SUMS` to verify the archived files."
        if include_checksums
        else "Use `artifact_manifest.json` to inspect the archived file inventory."
    )
    readme.write_text(
        "\n".join(
            [
                "# LLM Preference Coherence Artifacts",
                "",
                "Generated artifact bundle for the AIES 2026 LLM preference coherence project.",
                "",
                "This dataset repository is intended as an archive for reproduction and audit, not as a standalone training dataset.",
                "",
                "## Source",
                "",
                f"- GitHub repository: {source_url}",
                f"- Source commit: `{commit_text}`",
                "",
                "## Included Payloads",
                "",
                path_list,
                "",
                "Raw model responses and reasoning traces are under `outputs/07_model_runs/`.",
                "Derived analysis outputs are under `outputs/08_analysis/`.",
                "Final generated figures and tables, when present, are under `outputs/09_figures_tables/`.",
                "",
                verification_text,
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(
    bundle_dir: Path,
    repo_root: Path,
    source_url: str,
    relative_paths: list[Path],
    include_checksums: bool,
) -> dict[str, object]:
    source_commit = git_output(repo_root, ["rev-parse", "HEAD"])
    source_status = git_output(repo_root, ["status", "--short"])
    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    path_summaries, files = collect_manifest_files(
        bundle_dir=bundle_dir,
        relative_paths=relative_paths,
        include_checksums=include_checksums,
    )
    total_size = sum(int(record["size_bytes"]) for record in files)

    manifest: dict[str, object] = {
        "schema": "llm_coherence.artifact_manifest.v1",
        "created_at_utc": created_at,
        "source_repository": source_url,
        "source_git_commit": source_commit,
        "source_git_dirty": bool(source_status),
        "checksums": "sha256" if include_checksums else None,
        "summary": {
            "file_count": len(files),
            "size_bytes": total_size,
            "size_human": format_bytes(total_size),
        },
        "paths": path_summaries,
        "files": files,
    }

    (bundle_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if include_checksums:
        checksum_lines = [
            f"{record['sha256']}  {record['path']}"
            for record in files
            if "sha256" in record
        ]
        (bundle_dir / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

    return manifest


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    bundle_dir = args.bundle_dir.resolve()
    relative_paths = [Path(path) for path in DEFAULT_ARTIFACT_PATHS]

    if args.include_inputs:
        relative_paths.extend(Path(path) for path in OPTIONAL_INPUT_PATHS)
    relative_paths.extend(Path(path) for path in args.extra_path)

    if any(path.is_absolute() for path in relative_paths):
        raise SystemExit("Artifact paths must be relative to the repository root.")

    for relative_path in relative_paths:
        if is_relative_to(bundle_dir, (repo_root / relative_path).resolve()):
            raise SystemExit(
                f"Bundle directory cannot be inside staged artifact path: {relative_path}"
            )

    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"{bundle_dir} is not empty. Use a fresh directory or pass --force."
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    source_commit = git_output(repo_root, ["rev-parse", "HEAD"])
    write_dataset_card(
        bundle_dir=bundle_dir,
        source_url=args.source_url,
        source_commit=source_commit,
        relative_paths=relative_paths,
        force=args.force,
        include_checksums=args.checksums,
    )
    stats = stage_payloads(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        relative_paths=relative_paths,
        link_mode=args.link_mode,
        force=args.force,
    )
    manifest = write_manifest(
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        source_url=args.source_url,
        relative_paths=relative_paths,
        include_checksums=args.checksums,
    )

    summary = manifest["summary"]
    print(f"Prepared {summary['file_count']} files ({summary['size_human']})")
    print(f"Bundle directory: {bundle_dir}")
    print(f"Hardlinks: {stats.hardlinks}; copies: {stats.copies}")
    if not args.checksums:
        print("Checksums were not computed. Rerun with --checksums for archival upload.")
    print(
        "Upload with: HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder "
        "<namespace/repo-name> --repo-type=dataset "
        f"{bundle_dir}"
    )


if __name__ == "__main__":
    main()
