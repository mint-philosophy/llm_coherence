"""Bounded GPU validation: two tiny ladder runs through one pinned engine.

This executes model inference when invoked. The input JSON contains exactly two
comparison records. It checks CUDA devices, raw A/B evidence, saved results,
and engine reuse; it does not run the replacement manifest or upload results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import time

from llm_coherence.experiments.ladder_statement_pair.experiment_runner_tradeoff import run_experiment
from llm_coherence.runtime.agents import VLLMLogprobAgent, is_local_logprob_model, model_name_for_key


def validate_score_evidence(payload: dict) -> None:
    if payload["metadata"]["sampled_trials"] != 0 or payload["metadata"]["missing_response_count"] != 0:
        raise ValueError("Logprob runs must not claim sampled trials or missing sampled responses")
    for pref in payload["preferences"]:
        values = []
        for order, chosen in (("original", "A"), ("flipped", "B")):
            evidence = pref["logprob_evidence"][order]
            if len(evidence) != 1:
                raise ValueError("Expected one scoring record per prompt order")
            record = evidence[0]
            if record["token_ids"]["A"] == record["token_ids"]["B"]:
                raise ValueError("A/B token IDs must be distinct")
            scores = record["logprobs"]
            if not all(math.isfinite(scores[label]) for label in ("A", "B")):
                raise ValueError("Missing or non-finite A/B logprobs")
            shift = max(scores.values())
            weights = {label: math.exp(scores[label] - shift) for label in ("A", "B")}
            p = {label: weights[label] / sum(weights.values()) for label in ("A", "B")}
            for label in ("A", "B"):
                if not math.isclose(p[label], pref["order_probabilities"][order][label], abs_tol=1e-10):
                    raise ValueError("Saved probabilities disagree with returned logprobs")
            values.append(p[chosen])
        if pref["prob_prefer_a"] != round(sum(values) / 2, 4):
            raise ValueError("Saved preference disagrees with the flipped-order average")


async def validate_gpu(model_key: str, comparisons_path: Path, output_dir: Path, expected_gpus: int) -> dict:
    if not is_local_logprob_model(model_key):
        raise ValueError("GPU validation requires a self-hosted logprob model")
    comparisons = json.loads(comparisons_path.read_text(encoding="utf-8"))["comparisons"]
    if len(comparisons) != 2:
        raise ValueError("Provide exactly two comparisons for the bounded validation")
    if output_dir.exists():
        raise ValueError("Use a new validation output directory")
    owner = VLLMLogprobAgent(model_name_for_key(model_key), temperature=1.0)
    binding = owner.runtime_binding
    if not re.fullmatch(r"[0-9a-f]{40}", binding["model_revision"] or ""):
        raise ValueError("Pin LLM_COHERENCE_VLLM_REVISION to a model commit")
    if not re.fullmatch(r"[0-9a-f]{40}", binding["tokenizer_revision"] or ""):
        raise ValueError("Pin the tokenizer revision to a model commit")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_digest"] or ""):
        raise ValueError("Record LLM_COHERENCE_IMAGE_DIGEST for the actual job image")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} CUDA devices; found {torch.cuda.device_count()}")
    hardware = [{"name": torch.cuda.get_device_name(i),
                 "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                 "capability": list(torch.cuda.get_device_capability(i))}
                for i in range(expected_gpus)]
    output_dir.mkdir(parents=True)
    results, timings = [], []
    first_engine = None
    for i, comparison in enumerate(comparisons):
        name = f"gpu_validation_{i}"
        path = output_dir / f"{name}_comparisons.json"
        path.write_text(json.dumps({"comparisons": [comparison]}, ensure_ascii=False), encoding="utf-8")
        start = time.monotonic()
        result = await run_experiment(
            name, model_key, num_trials=10, include_flipped=True, with_reasoning=False,
            temperature=1.0, model_variant="base", max_tokens=10,
            data_dir=output_dir, comparison_path=path,
            results_dir=output_dir / "results", checkpoints_dir=output_dir / "checkpoints",
            shared_logprob_agent=owner, infrastructure="hf_jobs", gpu_count=expected_gpus,
            verbose=False,
        )
        timings.append(time.monotonic() - start)
        validate_score_evidence(result)
        if first_engine is not None and owner._llm is not first_engine:
            raise RuntimeError("The second ladder loaded a different engine")
        first_engine = owner._llm
        results.append(result["preferences"][0])
    report = {"status": "passed", "model_key": model_key, "runtime": binding,
              "hardware": hardware, "cuda_version": torch.version.cuda,
              "single_engine_reused": True, "scoring_calls": 4,
              "ladder_seconds": timings, "preferences": results}
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="glm-45-base-logprobs")
    parser.add_argument("--comparisons", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-gpus", type=int, default=8)
    args = parser.parse_args()
    report = asyncio.run(validate_gpu(args.model, args.comparisons, args.output_dir, args.expected_gpus))
    print("GPU_VALIDATION_RESULT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
