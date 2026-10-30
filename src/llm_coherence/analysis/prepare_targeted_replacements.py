"""Build an offline, hash-verified Phase 6b replacement input bundle.

The plan's ``reruns`` records bind a model/ladder pair to the canonical input
and the original result, including SHA-256 hashes and original configuration.
This command validates those bindings and current model settings, copies exact
input bytes, and writes one scoped manifest and argument list per model.
It never instantiates an agent, submits a job, or changes existing results.

Usage::

    python -m llm_coherence.analysis.prepare_targeted_replacements \
        --plan targeted_replacement_plan.json --output-dir replacement_bundle
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from llm_coherence.analysis.rebuttal_inputs import normalize_id
from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.runtime.agents import is_local_logprob_model, model_name_for_key


_RUN_FIELDS = (
    "model_variant", "reasoning_mode", "num_trials", "include_flipped",
    "with_reasoning", "max_tokens", "temperature", "k_samples",
    "infrastructure", "gpu_type", "gpu_count", "quantization",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_json(record: dict, prefix: str) -> tuple[Path, bytes, dict]:
    path = Path(record[f"{prefix}_path"]).resolve()
    data = path.read_bytes()
    if _sha256(data) != record[f"{prefix}_sha256"]:
        raise ValueError(f"Hash mismatch for {prefix}: {path}")
    return path, data, json.loads(data.decode("utf-8"))


def _provider_model(name: str | None) -> str | None:
    # Older result writers omitted LiteLLM's routing prefix.
    return name.removeprefix("openrouter/") if name else name


def _validate_task(task: dict) -> dict:
    model_key = task["model_key"]
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")
    cfg = MODEL_CONFIGS[model_key]
    input_path, input_bytes, inputs = _verified_json(task, "canonical_input")
    _, _, source = _verified_json(task, "original_result")
    original_config = task["original_config"]
    source_config = source["config"]
    if any(source_config.get(k) != v for k, v in original_config.items()):
        raise ValueError(f"Plan does not match original result configuration: {model_key}")
    if source_config.get("model_key") != model_key:
        raise ValueError(f"Original result belongs to a different model: {model_key}")
    metadata = source.get("metadata") or {}
    if metadata.get("model_name_full") != task["original_provider_model"]:
        raise ValueError(f"Plan does not match original provider model: {model_key}")
    if _provider_model(model_name_for_key(model_key)) != _provider_model(task["original_provider_model"]):
        raise ValueError(f"Current provider model differs from original: {model_key}")
    if metadata.get("extra_body") != task["original_extra_body"]:
        raise ValueError(f"Plan does not match original provider settings: {model_key}")
    if cfg.extra_body != task["original_extra_body"]:
        raise ValueError(f"Current provider settings differ from original: {model_key}")

    settings = {field: original_config.get(field) for field in _RUN_FIELDS}
    if not settings["include_flipped"]:
        raise ValueError("Phase 6b replacement manifests require both prompt orders")
    for field in ("num_trials", "max_tokens", "k_samples"):
        if not isinstance(settings[field], int) or isinstance(settings[field], bool) or settings[field] < 1:
            raise ValueError(f"Invalid original {field}: {settings[field]!r}")
    system_message = metadata.get("system_message")
    if not isinstance(system_message, str):
        raise ValueError(f"Original system message is unavailable: {model_key}")
    if cfg.system_message is not None and cfg.system_message != system_message:
        raise ValueError(f"Current system message overrides the original: {model_key}")
    settings["system_message"] = system_message

    comparisons = inputs["comparisons"]
    n_tiers = inputs["n_tiers"]
    n_comparison_samples = inputs["n_comparison_statements"]
    if n_tiers < 1 or n_comparison_samples < 1:
        raise ValueError("Comparison dimensions must be positive")
    if normalize_id(inputs["variation_id"]) != task["ladder_id"]:
        raise ValueError(f"Input belongs to another ladder: {input_path}")
    if input_path.name != f"{inputs['test_name']}_comparisons.json":
        raise ValueError(f"Input filename disagrees with its test name: {input_path}")
    pairs = Counter()
    for comp in comparisons:
        a, b = comp["outcome_a"], comp["outcome_b"]
        if not isinstance(a.get("text"), str) or not isinstance(b.get("text"), str):
            raise ValueError(f"Comparison is missing stimulus text: {input_path}")
        if normalize_id(a["variation_id"]) != task["ladder_id"]:
            raise ValueError(f"Comparison belongs to another ladder: {input_path}")
        pairs[(a["tier"], b["comparison_id"])] += 1
    expected = {(tier, comparator) for tier in range(1, n_tiers + 1)
                for comparator in range(n_comparison_samples)}
    if set(pairs) != expected or any(n != 1 for n in pairs.values()):
        raise ValueError(f"Input has missing, duplicate, or unexpected tier/comparator pairs: {input_path}")
    if inputs["num_comparisons"] != len(comparisons):
        raise ValueError(f"Input comparison count is inconsistent: {input_path}")
    logprob_mode = is_local_logprob_model(model_key)
    calls = len(comparisons) * 2 * (1 if logprob_mode else settings["num_trials"])
    if calls != task["expected_scoring_calls"]:
        raise ValueError(f"Planned call count differs from verified inputs: {model_key}")
    return {
        "task": task, "input_path": input_path, "input_bytes": input_bytes,
        "settings": settings, "n_tiers": n_tiers,
        "n_comparison_samples": n_comparison_samples,
        "scoring_calls": calls, "logprob_mode": logprob_mode,
    }


def prepare_bundle(plan_path: Path, output_dir: Path) -> dict:
    """Validate all tasks before writing a new, isolated input bundle."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new destination: {output_dir}")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    if not plan.get("reruns"):
        raise ValueError("Replacement plan has no reruns")
    groups = defaultdict(list)
    seen = set()
    for task in plan["reruns"]:
        identity = (task["model_key"], task["ladder_id"])
        if identity in seen:
            raise ValueError(f"Duplicate replacement task: {identity}")
        seen.add(identity)
        groups[identity[0]].append(_validate_task(task))
    for model_key, items in groups.items():
        first = items[0]
        for item in items[1:]:
            if any(item[key] != first[key] for key in ("settings", "n_tiers", "n_comparison_samples")):
                raise ValueError(f"Selected ladders have different run settings/dimensions: {model_key}")
        if len({item["input_path"].name for item in items}) != len(items):
            raise ValueError(f"Input filenames collide: {model_key}")

    output_dir.mkdir(parents=True)
    report = {
        "schema_version": 1, "plan_sha256": _sha256(plan_bytes),
        "model_count": len(groups), "replacement_ladders": len(seen),
        "base_scoring_calls": 0, "sampled_requests_before_retries": 0,
        "mode": "offline_preparation_only", "models": {},
    }
    for model_key, items in sorted(groups.items()):
        root = output_dir / model_key
        (root / "inputs").mkdir(parents=True)
        settings = items[0]["settings"]
        local = items[0]["logprob_mode"]
        manifest = {
            "n_tiers": items[0]["n_tiers"],
            "n_comparison_samples": items[0]["n_comparison_samples"],
            "variation_files": [], "replacement_sources": [],
        }
        for item in items:
            relative = Path("inputs") / item["input_path"].name
            (root / relative).write_bytes(item["input_bytes"])
            manifest["variation_files"].append(relative.as_posix())
            manifest["replacement_sources"].append(item["task"])
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        argv = [
            sys.executable, "-m", "llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment",
            "--model", model_key, "--manifest", str(root / "manifest.json"),
            "--data-dir", str(root), "--results-dir", str(root / "results"),
            "--checkpoints-dir", str(root / "checkpoints"),
            "--trials", str(settings["num_trials"]),
            "--max-tokens", str(settings["max_tokens"]),
            "--k-samples", str(settings["k_samples"]),
            "--system-message", settings["system_message"],
            "--max-concurrent", "1" if local else "3",
            "--max-retries", "5",
        ]
        for field in ("model_variant", "reasoning_mode", "temperature", "infrastructure",
                      "gpu_type", "gpu_count", "quantization"):
            if settings[field] is not None:
                argv.extend(["--" + field.replace("_", "-"), str(settings[field])])
        if settings["with_reasoning"]:
            argv.append("--with-reasoning")
        calls = sum(item["scoring_calls"] for item in items)
        entry = {
            "ladders": len(items), "scoring_calls_before_retries": calls,
            "scoring_method": "logprobs" if local else "sampled_choices",
            "settings": settings, "argv": argv,
            "extra_body": MODEL_CONFIGS[model_key].extra_body,
            "provider_model": model_name_for_key(model_key),
        }
        (root / "run_arguments.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        report["models"][model_key] = entry
        report["base_scoring_calls" if local else "sampled_requests_before_retries"] += calls
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_bundle(args.plan, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
