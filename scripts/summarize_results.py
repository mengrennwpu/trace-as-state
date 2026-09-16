from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ["em", "precision", "recall", "f1"]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Trace as State JSONL results")
    p.add_argument("--path", default="data/processed_experiments.jsonl", help="Path to JSONL results file")
    args = p.parse_args()
    rows = list(load_rows(Path(args.path)))
    if not rows:
        raise SystemExit("No result rows found.")

    conditions = sorted({c for row in rows for c in row.get('result', {}).get("outputs", {})})
    print("condition\tEM\tPrecision\tRecall\tF1\tn")
    for condition in conditions:
        sample_metrics = []
        for row in rows:
            out = row.get('result', {}).get("outputs", {}).get(condition)
            if not out or out.get("skipped"):
                continue
            mm = out.get("mean_metrics")
            if mm is not None:
                sample_metrics.append(mm)
        if not sample_metrics:
            continue
        vals = {m: mean(float(x[m]) for x in sample_metrics) for m in METRICS}
        print(
            f"{condition}\t{vals['em']:.4f}\t{vals['precision']:.4f}\t"
            f"{vals['recall']:.4f}\t{vals['f1']:.4f}\t{len(sample_metrics)}"
        )

    # Guardrail to catch the exact failure seen in the previous experiment.
    for condition in conditions:
        for row in rows:
            out = row.get("outputs", {}).get(condition)
            if not out or out.get("skipped") or not out.get("repeats"):
                continue
            for rep in out["repeats"]:
                m = rep.get("metrics", {})
                pval, rval, fval = float(m.get("precision", 0)), float(m.get("recall", 0)), float(m.get("f1", 0))
                expected = 2 * pval * rval / (pval + rval) if pval + rval > 0 else (1.0 if pval == rval == 0 and not m.get("malformed") else 0.0)
                if abs(expected - fval) > 1e-9:
                    print(
                        f"WARNING: inconsistent F1 in sample={row['sample']['row_id']} "
                        f"condition={condition} repeat={rep.get('repeat')}: P={pval}, R={rval}, F1={fval}, expected={expected}"
                    )
                    return


if __name__ == "__main__":
    main()
