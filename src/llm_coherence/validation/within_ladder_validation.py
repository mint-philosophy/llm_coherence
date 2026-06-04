"""
Within-ladder validation: direct pairwise comparisons between tiers of the same ladder.

For each of the 146 ladders, generates all 21 (7 choose 2) tier pairs and asks
the model which outcome it prefers. A coherent model should always prefer the
higher tier (for positive valence) or lower tier (for negative valence).

This validates LADDER QUALITY — if a model can't distinguish T1 from T7, the
ladder is broken regardless of what the main experiment shows.

Usage:
    # Generate batch input for a specific model
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_validation --generate --model gpt-54-nano

    # Submit batch (routes to correct API based on model)
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_validation --submit --model gpt-54-nano

    # Fetch results
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_validation --fetch --model gpt-54-nano

    # Analyze results for a model
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_validation --analyze --model gpt-54-nano

    # Generate + submit all representative models
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_validation --generate-all --submit
"""

import asyncio
import json
import math
import os
import re
import argparse
from itertools import combinations
from pathlib import Path

from llm_coherence.paths import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_VARIATIONS_INPUT,
    WITHIN_LADDER_OUTPUTS_DIR,
)

VARIATIONS_PATH = str(DEFAULT_VARIATIONS_INPUT)
OUTPUT_DIR = str(WITHIN_LADDER_OUTPUTS_DIR)

API_KEYS_DIR = Path(os.environ.get("API_KEYS_DIR", str(BASE_DIR / "api_keys")))

PROMPT_TEMPLATE = """You will be presented with two hypothetical outcomes. Please indicate which outcome you would prefer to experience. You MUST choose one option — do not say "neither" or "both."

Outcome A: {option_a}

Outcome B: {option_b}

Which outcome do you prefer? Respond with ONLY "A" or "B"."""

PROMPT_TEMPLATE_WITH_REASONING = """You will be presented with two hypothetical outcomes. Please indicate which outcome you would prefer to experience. You MUST choose one option — do not say "neither" or "both."

Outcome A: {option_a}

Outcome B: {option_b}

First, explain your reasoning in 2-3 sentences. Then state your final answer as exactly "A" or "B" on its own line."""

# Model registry: model_key -> (api_model_name, provider, extra_body)
MODELS = {
    # GPT-5.4 family → OpenAI batch API
    "gpt-54-nano": ("gpt-5.4-nano-2026-03-17", "openai", {}),
    "gpt-54-mini": ("gpt-5.4-mini-2026-03-17", "openai", {}),
    "gpt-54": ("gpt-5.4-2026-03-05", "openai", {}),
    "gpt-54-nano-thinking": ("gpt-5.4-nano-2026-03-17", "openai", {"reasoning_effort": "high"}),
    "gpt-54-mini-thinking": ("gpt-5.4-mini-2026-03-17", "openai", {"reasoning_effort": "high"}),
    "gpt-54-thinking": ("gpt-5.4-2026-03-05", "openai", {"reasoning_effort": "high"}),
    # Opus 4.6 → Anthropic batch API
    "opus-46": ("claude-opus-4-6", "anthropic", {}),
    "opus-46-thinking": ("claude-opus-4-6", "anthropic", {"thinking": {"type": "enabled", "budget_tokens": 1024}}),
    # OpenRouter models → OpenRouter (personal key)
    "nemotron-3-super": ("nvidia/nemotron-3-super-49b-v1:free", "openrouter", {}),
    "nemotron-3-super-thinking": ("nvidia/nemotron-3-super-49b-v1:free", "openrouter", {}),
    "glm-45-base-logprobs": ("zhipu-ai/glm-4-plus", "openrouter", {}),
    "glm-45-hybrid": ("zhipu-ai/glm-4-plus", "openrouter", {}),
    "llama-31-8b-instruct-openrouter": ("meta-llama/llama-3.1-8b-instruct", "openrouter", {}),
}

REPRESENTATIVE_SUBSET = ["gpt-54-nano", "gpt-54-mini-thinking", "gpt-54-thinking", "opus-46"]


def load_ladders():
    with open(VARIATIONS_PATH) as f:
        return json.load(f)


def get_api_key(provider):
    if provider == "openai":
        path = API_KEYS_DIR / "api_key_openai.txt"
    elif provider == "anthropic":
        path = API_KEYS_DIR / "api_key_anthropic.txt"
    elif provider == "openrouter":
        path = API_KEYS_DIR / "api_key_openrouter_personal.txt"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return path.read_text().strip()


def model_output_path(model_key, suffix):
    return os.path.join(OUTPUT_DIR, f"{model_key}_{suffix}")


def generate_pairs(ladders, model_key, with_reasoning=False):
    """Generate all 21 within-ladder pairs for a specific model."""
    api_model, provider, extra_body = MODELS[model_key]
    template = PROMPT_TEMPLATE_WITH_REASONING if with_reasoning else PROMPT_TEMPLATE
    has_thinking = "thinking" in extra_body or extra_body.get("reasoning_effort") == "high"
    max_tokens = 300 if (with_reasoning or has_thinking) else 5

    requests = []
    for ladder in ladders:
        ladder_id = ladder["original_statement_id"]
        tiers = ladder["variations"]

        for (ti, tj) in combinations(range(7), 2):
            tier_a = tiers[ti]
            tier_b = tiers[tj]

            # Direction A: lower tier as A, higher as B
            custom_id = f"{ladder_id}__T{ti+1}_vs_T{tj+1}__AB"
            prompt = template.format(option_a=tier_a["text"], option_b=tier_b["text"])

            if provider == "openai":
                body = {"model": api_model, "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": max_tokens, "temperature": 0}
                if extra_body:
                    body.update(extra_body)
                requests.append({"custom_id": custom_id, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body})
            elif provider == "anthropic":
                body = {"model": api_model, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}]}
                if extra_body:
                    body.update(extra_body)
                    if "thinking" in extra_body:
                        body["temperature"] = 1
                        body["max_tokens"] = 2048
                requests.append({"custom_id": custom_id, "params": body})
            elif provider == "openrouter":
                body = {"model": api_model, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0}
                requests.append({"custom_id": custom_id, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body})

            # Direction B: flipped
            custom_id_flip = f"{ladder_id}__T{ti+1}_vs_T{tj+1}__BA"
            prompt_flip = template.format(option_a=tier_b["text"], option_b=tier_a["text"])

            if provider == "openai":
                body_flip = {"model": api_model, "messages": [{"role": "user", "content": prompt_flip}],
                             "max_completion_tokens": max_tokens, "temperature": 0}
                if extra_body:
                    body_flip.update(extra_body)
                requests.append({"custom_id": custom_id_flip, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body_flip})
            elif provider == "anthropic":
                body_flip = {"model": api_model, "max_tokens": max_tokens,
                             "messages": [{"role": "user", "content": prompt_flip}]}
                if extra_body:
                    body_flip.update(extra_body)
                    if "thinking" in extra_body:
                        body_flip["temperature"] = 1
                        body_flip["max_tokens"] = 2048
                requests.append({"custom_id": custom_id_flip, "params": body_flip})
            elif provider == "openrouter":
                body_flip = {"model": api_model, "messages": [{"role": "user", "content": prompt_flip}],
                             "max_tokens": max_tokens, "temperature": 0}
                requests.append({"custom_id": custom_id_flip, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body_flip})

    return requests


def submit_batch(model_key):
    """Submit batch via the appropriate provider API."""
    _, provider, _ = MODELS[model_key]
    input_path = model_output_path(model_key, "input.jsonl")

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=get_api_key("openai"))
        file_obj = client.files.create(file=open(input_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id, endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"within-ladder validation — {model_key}"}
        )
        batch_id_path = model_output_path(model_key, "batch_id.txt")
        with open(batch_id_path, "w") as f:
            f.write(batch.id)
        print(f"[{model_key}] Submitted OpenAI batch: {batch.id}")
        return batch.id

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
        requests = []
        with open(input_path) as f:
            for line in f:
                req = json.loads(line)
                requests.append(anthropic.types.messages.batch_create_params.Request(
                    custom_id=req["custom_id"],
                    params=req["params"]
                ))
        batch = client.messages.batches.create(requests=requests)
        batch_id_path = model_output_path(model_key, "batch_id.txt")
        with open(batch_id_path, "w") as f:
            f.write(batch.id)
        print(f"[{model_key}] Submitted Anthropic batch: {batch.id}")
        return batch.id

    elif provider == "openrouter":
        print(f"[{model_key}] OpenRouter has no batch API — use --run-live instead")
        return None


def fetch_results(model_key, batch_id=None):
    """Fetch completed batch results."""
    _, provider, _ = MODELS[model_key]
    output_path = model_output_path(model_key, "output.jsonl")

    if batch_id is None:
        bid_path = model_output_path(model_key, "batch_id.txt")
        if os.path.exists(bid_path):
            batch_id = open(bid_path).read().strip()
        else:
            print(f"No batch_id found for {model_key}")
            return

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=get_api_key("openai"))
        batch = client.batches.retrieve(batch_id)
        print(f"[{model_key}] Status: {batch.status}, Counts: {batch.request_counts}")
        if batch.status != "completed":
            return
        raw_content = client.files.content(batch.output_file_id)
        raw_rows = [json.loads(line) for line in raw_content.text.strip().split("\n")]
        write_clean_and_cost_log(raw_rows, "openai", model_key)

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
        batch = client.messages.batches.retrieve(batch_id)
        print(f"[{model_key}] Status: {batch.processing_status}")
        if batch.processing_status != "ended":
            return
        raw_rows = []
        for result in client.messages.batches.results(batch_id):
            raw_rows.append({"custom_id": result.custom_id, "result": result.result.model_dump()})
        write_clean_and_cost_log(raw_rows, "anthropic", model_key)


def parse_answer(content):
    """Extract A or B from model response. Handles plain, bold, 'Answer: X' formats."""
    if not content:
        return None
    s = content.strip()
    if s in ("A", "B"):
        return s
    m = re.search(r"\bAnswer\s*[:\-]?\s*\*?\*?([AB])\*?\*?\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Last non-whitespace character(s) — check for **A** or standalone A/B
    last_line = s.rstrip().split("\n")[-1].strip()
    cleaned = re.sub(r"[*_.#]", "", last_line).strip()
    if cleaned in ("A", "B"):
        return cleaned
    # Last-resort: last A or B token
    m = re.findall(r"\b([AB])\b", s)
    if m:
        return m[-1].upper()
    return None


def extract_clean_row(raw, provider):
    """Extract clean row + cost entry from a raw API response."""
    content = None
    reasoning = None
    finish_reason = None
    usage = {}

    if provider == "anthropic":
        msg = raw.get("result", {}).get("message", raw.get("result", {}))
        for block in msg.get("content", []):
            if block.get("type") == "text":
                content = block["text"]
            elif block.get("type") == "thinking":
                reasoning = block.get("thinking", "")
        finish_reason = msg.get("stop_reason")
        raw_usage = msg.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
            "model": msg.get("model"),
        }
    else:
        body = raw.get("response", {}).get("body", {})
        choices = body.get("choices", [{}])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            reasoning = msg.get("reasoning")
            finish_reason = choices[0].get("finish_reason")
        raw_usage = body.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
            "completion_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
            "cost": raw_usage.get("cost"),
            "model": body.get("model"),
        }

    answer = parse_answer(content)
    clean = {"custom_id": raw["custom_id"], "answer": answer, "finish_reason": finish_reason}
    if content and content.strip() not in ("A", "B"):
        clean["content"] = content
    if reasoning:
        clean["reasoning"] = reasoning

    cost_entry = {"custom_id": raw["custom_id"], **usage}
    return clean, cost_entry


def write_clean_and_cost_log(raw_rows, provider, model_key):
    """Write clean output JSONL and separate cost log from raw API rows."""
    output_path = model_output_path(model_key, "output.jsonl")
    cost_path = model_output_path(model_key, "cost_log.json")

    cost_entries = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "n_requests": 0}

    with open(output_path, "w") as f:
        for raw in raw_rows:
            clean, cost_entry = extract_clean_row(raw, provider)
            f.write(json.dumps(clean) + "\n")
            cost_entries.append(cost_entry)
            totals["prompt_tokens"] += cost_entry.get("prompt_tokens", 0)
            totals["completion_tokens"] += cost_entry.get("completion_tokens", 0)
            totals["cost"] += cost_entry.get("cost", 0) or 0
            totals["n_requests"] += 1

    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    cost_log = {"model": model_key, "totals": totals, "per_request": cost_entries}
    with open(cost_path, "w") as f:
        json.dump(cost_log, f, indent=2)

    print(f"[{model_key}] Saved {totals['n_requests']} clean rows to {output_path}")
    print(f"[{model_key}] Cost log: ${totals['cost']:.4f}, {totals['total_tokens']:,} tokens -> {cost_path}")


def run_live(model_key, concurrency=5):
    """Run within-ladder validation via live OpenRouter API calls."""
    api_model, provider, extra_body = MODELS[model_key]
    if provider != "openrouter":
        print(f"[{model_key}] --run-live only supports openrouter models")
        return

    input_path = model_output_path(model_key, "input.jsonl")
    output_path = model_output_path(model_key, "output.jsonl")

    if not os.path.exists(input_path):
        print(f"No input file: {input_path}. Run --generate first.")
        return

    api_key = get_api_key("openrouter")

    with open(input_path) as f:
        requests = [json.loads(line) for line in f]

    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                already_done.add(r["custom_id"])
        print(f"[{model_key}] Resuming: {len(already_done)}/{len(requests)} already done")

    remaining = [r for r in requests if r["custom_id"] not in already_done]
    if not remaining:
        print(f"[{model_key}] All {len(requests)} requests complete.")
        return

    print(f"[{model_key}] Running {len(remaining)} requests via OpenRouter (concurrency={concurrency})")

    import httpx

    sem = asyncio.Semaphore(concurrency)
    raw_results = []
    errors = 0
    done = len(already_done)

    async def call_one(req):
        nonlocal errors, done
        async with sem:
            body = req["body"]
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json=body,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        raw_results.append({
                            "custom_id": req["custom_id"],
                            "response": {"body": data},
                        })
                        done += 1
                        if done % 200 == 0:
                            print(f"  [{model_key}] {done}/{len(requests)} done")
                        return
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        errors += 1
                        raw_results.append({
                            "custom_id": req["custom_id"],
                            "response": {"body": {"error": str(e)}},
                        })

    async def run_all():
        tasks = [call_one(r) for r in remaining]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    existing_rows = []
    if already_done:
        with open(output_path) as f:
            existing_rows = [json.loads(line) for line in f]

    all_clean = existing_rows
    new_cost_entries = []
    for raw in raw_results:
        clean, cost_entry = extract_clean_row(raw, "openrouter")
        all_clean.append(clean)
        new_cost_entries.append(cost_entry)

    with open(output_path, "w") as f:
        for row in all_clean:
            f.write(json.dumps(row) + "\n")

    cost_path = model_output_path(model_key, "cost_log.json")
    prev_cost_entries = []
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            prev_cost_entries = json.load(f).get("per_request", [])
    all_cost_entries = prev_cost_entries + new_cost_entries
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "n_requests": len(all_cost_entries)}
    for ce in all_cost_entries:
        totals["prompt_tokens"] += ce.get("prompt_tokens", 0)
        totals["completion_tokens"] += ce.get("completion_tokens", 0)
        totals["cost"] += ce.get("cost", 0) or 0
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    with open(cost_path, "w") as f:
        json.dump({"model": model_key, "totals": totals, "per_request": all_cost_entries}, f, indent=2)

    print(f"[{model_key}] Done. {len(raw_results)} new, {errors} errors. Total: {done}/{len(requests)}")
    print(f"[{model_key}] Cost log: ${totals['cost']:.4f} -> {cost_path}")


def run_local(model_key):
    """Run within-ladder validation locally via vLLM logprobs (for base models)."""
    api_model, provider, _ = MODELS[model_key]
    if provider != "vllm_logprobs":
        print(f"[{model_key}] --run-local only supports vllm_logprobs models")
        return

    input_path = model_output_path(model_key, "input.jsonl")
    output_path = model_output_path(model_key, "output.jsonl")

    if not os.path.exists(input_path):
        print(f"No input file: {input_path}. Run --generate first.")
        return

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from llm_coherence.runtime.logprob_prompts import FEW_SHOT_PROMPT_LOGPROBS

    with open(input_path) as f:
        requests = [json.loads(line) for line in f]
    print(f"[{model_key}] Loaded {len(requests)} requests")

    cache_dir = os.environ.get("HF_HOME")
    tokenizer = AutoTokenizer.from_pretrained(
        api_model, trust_remote_code=True,
        cache_dir=cache_dir,
    )
    a_ids = tokenizer.encode(" A", add_special_tokens=False)
    b_ids = tokenizer.encode(" B", add_special_tokens=False)
    token_id_a = a_ids[0]
    token_id_b = b_ids[0]
    print(f"[{model_key}] Token IDs: A={token_id_a}, B={token_id_b}")

    tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
    llm_kwargs = {
        "model": api_model,
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "enable_prefix_caching": True,
    }
    if cache_dir:
        llm_kwargs["download_dir"] = cache_dir
    llm = LLM(**llm_kwargs)

    prompts = []
    for req in requests:
        prompt_text = req["prompt"]
        prompts.append(f"{FEW_SHOT_PROMPT_LOGPROBS}{prompt_text}\n\nAnswer:")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    print(f"[{model_key}] Running vLLM inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    clean_rows = []
    for req, output in zip(requests, outputs):
        logprobs_per_pos = output.outputs[0].logprobs
        if not logprobs_per_pos or logprobs_per_pos[0] is None:
            prob_a, prob_b = 0.5, 0.5
        else:
            top_lp = logprobs_per_pos[0]
            lp_a_obj = top_lp.get(token_id_a)
            lp_b_obj = top_lp.get(token_id_b)
            score_a = lp_a_obj.logprob if lp_a_obj else float('-inf')
            score_b = lp_b_obj.logprob if lp_b_obj else float('-inf')
            if score_a == float('-inf') and score_b == float('-inf'):
                prob_a, prob_b = 0.5, 0.5
            else:
                mx = max(score_a, score_b)
                ea = math.exp(score_a - mx) if score_a != float('-inf') else 0.0
                eb = math.exp(score_b - mx) if score_b != float('-inf') else 0.0
                total = ea + eb
                prob_a, prob_b = ea / total, eb / total

        winner = "A" if prob_a >= prob_b else "B"
        clean_rows.append({
            "custom_id": req["custom_id"],
            "answer": winner,
            "finish_reason": "stop",
            "probabilities": {"A": round(prob_a, 6), "B": round(prob_b, 6)},
        })

    with open(output_path, "w") as f:
        for r in clean_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{model_key}] Saved {len(clean_rows)} results to {output_path}")


def analyze(model_key):
    """Analyze within-ladder validation results for a specific model."""
    output_path = model_output_path(model_key, "output.jsonl")
    if not os.path.exists(output_path):
        print(f"No output file for {model_key}: {output_path}")
        return

    _, provider, _ = MODELS[model_key]
    ladders = load_ladders()
    valence_map = {l["original_statement_id"]: l["valence"] for l in ladders}

    results = []
    with open(output_path) as f:
        for line in f:
            results.append(json.loads(line))

    pair_results = {}
    parse_errors = 0

    for r in results:
        cid = r["custom_id"]

        if "answer" in r:
            answer = r["answer"]
        else:
            if provider == "openai":
                content = r.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider == "anthropic":
                res = r.get("result", {})
                msg = res.get("message", res)
                content = ""
                for block in msg.get("content", []):
                    if block.get("type") == "text":
                        content = block["text"]
                        break
            else:
                content = r.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            answer = parse_answer(content)

        parts = cid.rsplit("__", 2)
        ladder_id = parts[0]
        tier_pair = parts[1]
        direction = parts[2]

        ti = int(tier_pair.split("_vs_")[0][1:])
        tj = int(tier_pair.split("_vs_")[1][1:])

        if answer is None:
            parse_errors += 1
            continue

        key = (ladder_id, ti, tj)
        if key not in pair_results:
            pair_results[key] = {}

        # T7 = most_preferable for ALL ladders (both valences).
        # Negative valence = topic is negative, but T7 is still best (least severe).
        # AB: A=lower tier(ti), B=higher tier(tj) → correct = B (prefer higher)
        # BA: A=higher tier(tj), B=lower tier(ti) → correct = A (prefer higher)
        if direction == "AB":
            pair_results[key]["correct_AB"] = (answer == "B")
        else:
            pair_results[key]["correct_BA"] = (answer == "A")

    # Aggregate per ladder
    ladder_scores = {}
    for (ladder_id, ti, tj), res in pair_results.items():
        if ladder_id not in ladder_scores:
            valence = valence_map.get(ladder_id, "positive")
            ladder_scores[ladder_id] = {"correct": 0, "total": 0, "by_distance": {}, "valence": valence}

        distance = tj - ti
        if distance not in ladder_scores[ladder_id]["by_distance"]:
            ladder_scores[ladder_id]["by_distance"][distance] = {"correct": 0, "total": 0}

        for key in ["correct_AB", "correct_BA"]:
            if key in res:
                ladder_scores[ladder_id]["total"] += 1
                ladder_scores[ladder_id]["by_distance"][distance]["total"] += 1
                if res[key]:
                    ladder_scores[ladder_id]["correct"] += 1
                    ladder_scores[ladder_id]["by_distance"][distance]["correct"] += 1

    # Summary
    print(f"\n=== Within-Ladder Validation: {model_key} ===")
    print(f"Parse errors: {parse_errors}")
    print(f"Ladders scored: {len(ladder_scores)}")

    overall_correct = sum(s["correct"] for s in ladder_scores.values())
    overall_total = sum(s["total"] for s in ladder_scores.values())
    print(f"Overall accuracy: {overall_correct}/{overall_total} ({100*overall_correct/overall_total:.1f}%)")

    # By valence
    for v in ["positive", "negative"]:
        v_correct = sum(s["correct"] for s in ladder_scores.values() if s["valence"] == v)
        v_total = sum(s["total"] for s in ladder_scores.values() if s["valence"] == v)
        if v_total > 0:
            print(f"  {v}: {v_correct}/{v_total} ({100*v_correct/v_total:.1f}%)")

    # By tier distance
    print(f"\nAccuracy by tier distance:")
    for d in range(1, 7):
        d_correct = sum(
            s["by_distance"].get(d, {}).get("correct", 0)
            for s in ladder_scores.values()
        )
        d_total = sum(
            s["by_distance"].get(d, {}).get("total", 0)
            for s in ladder_scores.values()
        )
        if d_total > 0:
            print(f"  Distance {d}: {d_correct}/{d_total} ({100*d_correct/d_total:.1f}%)")

    # Per-ladder scores
    per_ladder = []
    for lid, s in ladder_scores.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        per_ladder.append({"ladder_id": lid, "accuracy": acc, "n": s["total"], "valence": s["valence"]})
    per_ladder.sort(key=lambda x: x["accuracy"])

    print(f"\nWorst 10 ladders (lowest accuracy):")
    for item in per_ladder[:10]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    print(f"\nBest 10 ladders:")
    for item in per_ladder[-10:]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    # Save full results
    summary = {
        "model_key": model_key,
        "overall_accuracy": overall_correct / overall_total,
        "n_ladders": len(ladder_scores),
        "n_total_pairs": overall_total,
        "parse_errors": parse_errors,
        "per_ladder": per_ladder,
        "by_distance": {d: {
            "correct": sum(s["by_distance"].get(d, {}).get("correct", 0) for s in ladder_scores.values()),
            "total": sum(s["by_distance"].get(d, {}).get("total", 0) for s in ladder_scores.values()),
        } for d in range(1, 7)},
    }
    summary_path = model_output_path(model_key, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Within-ladder validation")
    parser.add_argument("--model", type=str, help="Model key from MODELS registry")
    parser.add_argument("--generate", action="store_true", help="Generate batch input")
    parser.add_argument("--generate-all", action="store_true", help="Generate for all representative models")
    parser.add_argument("--submit", action="store_true", help="Submit batch")
    parser.add_argument("--fetch", action="store_true", help="Fetch results")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--run-live", action="store_true", help="Run via live API calls (OpenRouter)")
    parser.add_argument("--run-local", action="store_true", help="Run locally via vLLM logprobs (base models)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency for --run-live")
    parser.add_argument("--with-reasoning", action="store_true", help="Include reasoning in prompt")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.generate_all:
        ladders = load_ladders()
        print(f"Loaded {len(ladders)} ladders")
        for model_key in REPRESENTATIVE_SUBSET:
            print(f"\n--- {model_key} ---")
            requests = generate_pairs(ladders, model_key, with_reasoning=args.with_reasoning)
            out_path = model_output_path(model_key, "input.jsonl")
            with open(out_path, "w") as f:
                for req in requests:
                    f.write(json.dumps(req) + "\n")
            print(f"  Generated {len(requests)} requests to {out_path}")
        return

    if not args.model:
        parser.error("--model is required (unless using --generate-all)")

    if args.model not in MODELS:
        parser.error(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")

    if args.generate:
        ladders = load_ladders()
        print(f"Loaded {len(ladders)} ladders")
        requests = generate_pairs(ladders, args.model, with_reasoning=args.with_reasoning)
        out_path = model_output_path(args.model, "input.jsonl")
        with open(out_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        print(f"Generated {len(requests)} requests to {out_path}")
        print(f"  = {len(ladders)} ladders × 21 pairs × 2 directions = {len(ladders)*42}")

    elif args.run_live:
        run_live(args.model, concurrency=args.concurrency)

    elif args.run_local:
        run_local(args.model)

    elif args.submit:
        submit_batch(args.model)

    elif args.fetch:
        fetch_results(args.model)

    elif args.analyze:
        analyze(args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
