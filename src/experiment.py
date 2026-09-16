from __future__ import annotations

import random
from dataclasses import asdict
from statistics import mean
from typing import Any

from .dataset import GraphWalkSample
from .evaluator import extract_final_answer, score_set
from .io_utils import *
from .prompts import (
    answer_feedback_parts,
    baseline_parts,
    question_first_parts,
    re2_parts,
    random_trace_parts,
    serialize_traces,
    trace_append_parts,
    trace_as_state_parts,
    ANSWER_FORMAT,
)
from .provider import OpenAICompatibleClient
from tqdm import tqdm

METRIC_KEYS = ["em", "precision", "recall", "f1"]


def score_response(text: str, gold: list[str]) -> dict[str, Any]:
    pred, malformed = extract_final_answer(text)
    return {**score_set(pred, gold), "prediction": pred, "malformed": malformed}


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    return {k: mean(float(r["metrics"][k]) for r in records) for k in METRIC_KEYS}


def serialize_answer_feedback(answers: list[str]) -> str:
    return "\n\n".join(f"[Answer {i}]\n{a.strip()}" for i, a in enumerate(answers, 1))


def majority_answer(first_runs: list[dict[str, Any]]) -> list[str]:
    # Majority@5 is defined over the parsed answer set. Ties are broken
    # deterministically by first appearance.
    counts: dict[tuple[str, ...], int] = {}
    order: list[tuple[str, ...]] = []
    for run in first_runs:
        key = tuple(sorted(run["metrics"]["prediction"]))
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1
    best = max(order, key=lambda k: (counts[k], -order.index(k)))
    return list(best)


def _call_repeated(
    client: OpenAICompatibleClient,
    prompt: str,
    gold: list[str],
    repeats: int,
    condition: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i in tqdm(range(repeats), desc=f"[second-pass {condition}]",):
        result = client.generate(prompt)
        results.append(
            {
                "repeat": i + 1,
                "text": result.text,
                "reasoning": result.reasoning,
                "metrics": score_response(result.text, gold),
                "elapsed_s": result.elapsed_s,
                "usage": result.usage,
                "condition": condition,
            }
        )
    return results


def generate_source_runs(
    sample: GraphWalkSample,
    client: OpenAICompatibleClient,
    source_repeats: int,
    source_temperature: float | None,
) -> list[dict[str, Any]]:
    prompt = "\n\n".join(baseline_parts(sample.prompt))
    runs: list[dict[str, Any]] = []
    for i in range(source_repeats):
        result = client.generate(prompt, temperature=source_temperature)
        if not result.reasoning.strip():
            raise RuntimeError(
                f"Sample {sample.row_id}: source repeat {i + 1} returned no observable reasoning. "
                "Enable reasoning and expose reasoning via reasoning_split/reasoning_content."
            )
        runs.append(
            {
                "repeat": i + 1,
                "text": result.text,
                "answer_text": result.text,
                "reasoning": result.reasoning,
                "metrics": score_response(result.text, sample.answer_nodes),
                "elapsed_s": result.elapsed_s,
                "usage": result.usage,
            }
        )
    return runs


def run_sample_second_pass(
    sample: GraphWalkSample,
    client: OpenAICompatibleClient,
    source_runs: list[dict[str, Any]],
    *,
    second_pass_repeats: int,
    trace_max_chars: int,
    conditions: list[str],
    random_trace_block: str | None,
) -> dict[str, Any]:
    traces = [r["reasoning"] for r in source_runs]
    trace_block = serialize_traces(traces, trace_max_chars)
    outputs: dict[str, Any] = {
        "first_pass": {
            "repeats": source_runs,
            "mean_metrics": mean_metrics(source_runs),
        }
    }

    builders = {
        "trace_append": lambda: trace_append_parts(sample.prompt, trace_block),
        "trace_as_state": lambda: trace_as_state_parts(sample.prompt, trace_block),
        "re2": lambda: re2_parts(sample.prompt),
        "question_first": lambda: question_first_parts(sample.prompt),
        "answer_feedback": lambda: answer_feedback_parts(sample.prompt, [r["answer_text"] for r in source_runs]),
        "random_trace": lambda: random_trace_parts(sample.prompt, random_trace_block) if random_trace_block else None,
    }

    for condition in conditions:
        if condition == "first_pass":
            continue
        if condition == "majority_at5":
            pred = majority_answer(source_runs)
            m = score_set(pred, sample.answer_nodes)
            outputs[condition] = {
                "repeats": [{"repeat": 1, "text": "", "reasoning": "", "metrics": {**m, "prediction": pred, "malformed": False}}],
                "mean_metrics": m,
                "non_generation": True,
            }
            continue
        if condition == "oracle_at5":
            best = max((r["metrics"] for r in source_runs), key=lambda m: (m["f1"], m["em"]))
            outputs[condition] = {
                "repeats": [{"repeat": 1, "text": "", "reasoning": "", "metrics": best}],
                "mean_metrics": {k: float(best[k]) for k in METRIC_KEYS},
                "non_generation": True,
            }
            continue
        if condition == "trace_only":
            prompt = f"{trace_block}\n\n{ANSWER_FORMAT}"
        else:
            if condition not in builders:
                raise ValueError(f"Unsupported condition: {condition}")
            parts = builders[condition]()
            if parts is None:
                outputs[condition] = {"skipped": True, "reason": "No donor trace available."}
                continue
            prompt = "\n\n".join(p.rstrip() for p in parts)

        second_runs = _call_repeated(
            client, prompt, sample.answer_nodes, second_pass_repeats, condition
        )
        outputs[condition] = {
            "repeats": second_runs,
            "mean_metrics": mean_metrics(second_runs),
            "prompt_chars": len(prompt),
        }

    return {
        "sample": asdict(sample),
        "trace": {
            "source_count": len(source_runs),
            "max_chars": trace_max_chars,
            "serialized_chars": len(trace_block),
            "source_trace_chars": [len(r["reasoning"]) for r in source_runs],
            "block": trace_block,
        },
        "outputs": outputs,
    }


def run_dataset(
    samples: list[GraphWalkSample],
    client: OpenAICompatibleClient,
    out_path: str,
    *,
    source_repeats: int,
    second_pass_repeats: int,
    trace_max_chars: int,
    conditions: list[str],
    source_temperature: float | None,
    seed: int,
) -> None:
    # Stage 1: all source runs first. This is necessary for a faithful
    # Random Trace control: its donor must come from another problem in the
    # same subtask, not merely from earlier samples.
    
    processed_samples_path = f'{out_path}/processed_samples.jsonl'
    processed_samples = get_datas(processed_samples_path, mode='r') if Path(processed_samples_path).exists() else []
    md5_2_samples = {s.get('md5'): s for s in processed_samples}
    
    source_cache: dict[int, tuple[GraphWalkSample, list[dict[str, Any]], str]] = {}
    for idx, sample in tqdm(enumerate(samples), desc="[source runs]"):
        sample_md5 = md5(sample.prompt)
        if sample_md5 in md5_2_samples:
            runs = md5_2_samples[sample_md5]['runs']
            block = md5_2_samples[sample_md5]['block']
        else:
            runs = generate_source_runs(sample, client, source_repeats, source_temperature)
            block = serialize_traces([r["reasoning"] for r in runs], trace_max_chars)
            save_datas(processed_samples_path, [{'md5': sample_md5, 'runs': runs, 'block': block, 'sample': asdict(sample)}], mode='a')
        source_cache[idx] = (sample, runs, block)
        print(
            f"[source {idx + 1}/{len(samples)}] sample={sample.row_id} chars={sample.prompt_chars} "
            f"F1={mean_metrics(runs)['f1']:.4f} trace_chars={len(block)}",
            flush=True,
        )

    # Deterministic donor permutation, guaranteeing a different problem.
    indices = list(source_cache.keys())
    rng = random.Random(seed)
    shuffled = indices[:]
    rng.shuffle(shuffled)
    donors: dict[int, int] = {}
    if len(shuffled) > 1:
        for pos, idx in enumerate(shuffled):
            donors[idx] = shuffled[(pos + 1) % len(shuffled)]

    processed_experiments_path = f'{out_path}/processed_experiments.jsonl'
    processed_experiments = get_datas(processed_experiments_path, mode='r') if Path(processed_experiments_path).exists() else []
    md5_2_experiments = {s.get('md5'): s for s in processed_experiments}
    
    # Stage 2: fresh second-pass repeats per condition, using the same T per sample.
    for idx in indices:
        sample, runs, _ = source_cache[idx]
        donor_idx = donors.get(idx)
        donor_block = source_cache[donor_idx][2] if donor_idx is not None else None
        sample_md5 = md5(sample.prompt)
        if sample_md5 in md5_2_experiments:
            result = md5_2_experiments[sample_md5]['result']
        else:
            result = run_sample_second_pass(
                sample,
                client,
                runs,
                second_pass_repeats=second_pass_repeats,
                trace_max_chars=trace_max_chars,
                conditions=conditions,
                random_trace_block=donor_block,
            )
            result["random_trace_donor_row_id"] = source_cache[donor_idx][0].row_id if donor_idx is not None else None
            save_datas(processed_experiments_path, [{'md5': sample_md5, 'result': result, 'sample': asdict(sample)}], mode='a')
        print(f"[second-pass {idx + 1}/{len(samples)}] sample={sample.row_id}", flush=True)
