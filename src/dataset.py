from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class GraphWalkSample:
    row_id: int
    prompt: str
    answer_nodes: list[str]
    prompt_chars: int
    problem_type: str


def _to_sample(idx: int, row: dict[str, Any]) -> GraphWalkSample:
    return GraphWalkSample(
        row_id=idx,
        prompt=str(row["prompt"]),
        answer_nodes=[str(x).strip() for x in (row.get("answer_nodes") or row.get("answer") or [])],
        prompt_chars=int(row.get("prompt_chars", len(str(row["prompt"])))),
        problem_type=str(row["problem_type"]),
    )


def load_graphwalks(
    task: str = "parents",
    split: str = "train",
    max_samples: int | None = 200,
    *,
    target_chars: int = 262_144,
    bin_tolerance: int = 16_384,
    seed: int = 42,
    shuffle: bool = True,
) -> list[GraphWalkSample]:
    """Load GraphWalks and select the 256K-length neighborhood.

    The paper names the target subset "GraphWalks 256K" but does not state a
    character-level numeric interval in the paper. Since the public dataset
    exposes prompt_chars, this implementation uses a configurable nearest-bin
    selection centered at 262,144 characters by default.
    """
    ds = load_dataset("openai/graphwalks", split=split)
    rows = []
    lower = target_chars - bin_tolerance
    upper = target_chars + bin_tolerance

    for idx, row in enumerate(ds):
        if str(row["problem_type"]) != task:
            continue
        chars = int(row["prompt_chars"])
        if lower <= chars <= upper:
            rows.append(_to_sample(idx, row))

    if not rows:
        # Give a useful diagnostic instead of silently using the wrong length.
        candidates = [_to_sample(i, r) for i, r in enumerate(ds) if str(r["problem_type"]) == task]
        nearest = sorted(candidates, key=lambda x: abs(x.prompt_chars - target_chars))[:10]
        detail = ", ".join(f"{x.prompt_chars}" for x in nearest)
        raise RuntimeError(
            f"No {task!r} samples in [{lower}, {upper}] chars. Nearest prompt_chars: {detail}"
        )

    if shuffle:
        random.Random(seed).shuffle(rows)
    else:
        rows.sort(key=lambda x: (x.prompt_chars, x.row_id))

    if max_samples is not None:
        if len(rows) < max_samples:
            raise RuntimeError(
                f"Requested {max_samples} samples, but selected 256K bin contains only {len(rows)} "
                f"samples. Increase bin_tolerance or lower max_samples."
            )
        rows = rows[:max_samples]
    return rows
