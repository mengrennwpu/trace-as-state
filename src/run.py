from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, load_model_config
from src.dataset import load_graphwalks
from src.experiment import run_dataset
from src.provider import OpenAICompatibleClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--task", choices=["parents", "bfs"], default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--target-prompt-chars", type=int, default=None)
    p.add_argument("--bin-tolerance", type=int, default=None)
    p.add_argument("--source-traces", type=int, default=None)
    p.add_argument("--second-pass-repeats", type=int, default=None)
    p.add_argument("--trace-max-chars", type=int, default=None)
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--output-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out

    cfg = load_config(cfg_path)
    model = load_model_config(cfg)
    d = cfg.get("dataset", {})
    e = cfg.get("experiment", {})

    task = args.task or d.get("task", "parents")
    split = args.split or d.get("split", "train")
    max_samples = args.max_samples if args.max_samples is not None else e.get("max_samples", 100)
    target_chars = args.target_prompt_chars if args.target_prompt_chars is not None else e.get("target_prompt_chars", 1_000_000)
    tolerance = args.bin_tolerance if args.bin_tolerance is not None else e.get("bin_tolerance", 200_000)
    source_traces = args.source_traces if args.source_traces is not None else e.get("source_traces", 5)
    second_repeats = args.second_pass_repeats if args.second_pass_repeats is not None else e.get("second_pass_repeats", 1)
    trace_max = args.trace_max_chars if args.trace_max_chars is not None else e.get("trace_max_chars", 50_000)
    conditions = args.conditions or e.get("conditions", ["first_pass", "trace_append", "trace_as_state"])
    seed = int(e.get("seed", 42))

    samples = load_graphwalks(
        task=task,
        split=split,
        max_samples=max_samples,
        target_chars=int(target_chars),
        bin_tolerance=int(tolerance),
        seed=seed,
    )

    # Safety diagnostics before expensive API calls.
    print("Configuration:")
    print(f"  dataset       = openai/graphwalks/{split}")
    print(f"  task          = {task}")
    print(f"  samples       = {len(samples)}")
    print(f"  prompt_chars  = [{int(target_chars)-int(tolerance)}, {int(target_chars)+int(tolerance)}]")
    print(f"  source traces = {source_traces}")
    print(f"  2nd-pass reps = {second_repeats}")
    print(f"  trace cap     = {trace_max} chars/trace")
    print(f"  model         = {model.model}")
    print(f"  temperature   = {model.temperature}")
    print(f"  top_p         = {model.top_p}")
    print(f"  max_output    = {model.max_output_tokens}")
    print(f"  conditions    = {conditions}")

    out.mkdir(parents=True, exist_ok=True)

    client = OpenAICompatibleClient(model)
    try:
        run_dataset(
            samples,
            client,
            str(out),
            source_repeats=int(source_traces),
            second_pass_repeats=int(second_repeats),
            trace_max_chars=int(trace_max),
            conditions=list(conditions),
            source_temperature=model.temperature,
            seed=seed,
        )
    finally:
        client.close()
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
