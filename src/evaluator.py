from __future__ import annotations

import re
from typing import Iterable


def extract_final_answer(response: str | None) -> tuple[list[str], bool]:
    """Follow the GraphWalks released scorer: inspect only the final line."""
    if response is None:
        return [], True
    line = response.splitlines()[-1].strip() if response.splitlines() else ""
    if "Final Answer:" not in line:
        return [], True
    match = re.search(r"Final Answer:\s*\[(.*)\]", line)
    if not match:
        return [], True
    body = match.group(1).strip()
    if body == "":
        return [], False
    values = [x.strip() for x in body.split(",") if x.strip()]
    return values, False


def score_set(pred: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    pred_set = {str(x).strip() for x in pred if str(x).strip()}
    gold_set = {str(x).strip() for x in gold if str(x).strip()}
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else (1.0 if not pred_set and not gold_set else 0.0)
    em = 1.0 if pred_set == gold_set else 0.0
    return {"em": em, "precision": precision, "recall": recall, "f1": f1}
