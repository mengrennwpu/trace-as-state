> **Language / 语言:** [English](README.md) · [中文](README.zh.md)

---

# Trace as State — GraphWalks reproduction

A lightweight reproduction scaffold for **Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers** (Xu Zou, Jie Tang, arXiv:2609.02702, v1).

- Paper: [https://alphaxiv.org/abs/2609.02702](https://alphaxiv.org/abs/2609.02702)
- arXiv: [2609.02702](https://arxiv.org/abs/2609.02702)

The core experiment replaces one-shot chain-of-thought with a serialized bundle of independent reasoning traces placed at different positions in the prompt:

| Name            | Template         | Notes                                                                                  |
| --------------- | ---------------- | -------------------------------------------------------------------------------------- |
| First Pass      | `[x, q]`       | Paper baseline. No trace, plain single-shot.                                           |
| Trace Append    | `[x, T, q]`    | Trace placed after the graph, before the question.                                     |
| Trace as State  | `[T, x, q]`    | Trace placed at the front as a conditional state.                                      |
| Re2             | `[x, q, x, q]` | Re-Reading control: graph + question repeated.                                         |
| Question First  | `[q, x, q]`    | Question placed before the graph.                                                      |
| Answer Feedback | `[F, x, q]`    | First-pass answers fed back as feedback.                                               |
| Random Trace    | `[T', x, q]`   | Trace from**another problem in the same task** as a donor.                       |
| Trace Only      | `[T]`          | Trace alone, no graph, no question.                                                    |
| Majority@5      | —               | Majority vote over the**parsed answer sets** of the source runs (no model call). |
| Oracle@5        | —               | Pick the source run with the highest F1 (no model call).                               |

`T` is built from `source_traces` (paper default: 5) independent reasoning traces; each trace is truncated to `trace_max_chars` (paper default: 50 000) characters.

> This implementation is intentionally **provider-agnostic**: it works with any chat-completions-compatible endpoint that exposes the reasoning text via `reasoning_content` / `reasoning` / a configurable field. Many hosted APIs hide raw CoT by default — **without observable reasoning text, this trace-as-state setup cannot be reproduced faithfully**.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `datasets`, `httpx`, `python-dotenv`, `pandas`, `matplotlib`, `pyyaml`.

---

## 2. Configure the model

Copy `.env.example` to `.env` and fill in:

```text
API_BASE_URL=https://your-openai-compatible-endpoint/v1
API_KEY=...
MODEL=your-reasoning-model
```

Example: running against the MiniMax M3 reasoning endpoint:

```text
API_BASE_URL=https://api.minimax.cn/v1
API_KEY=sk-...
MODEL=MiniMax-M3
REASONING_FIELD=reasoning_content
TIMEOUT_SECONDS=1800
```

The cached `data/processed_*.jsonl` files shipped with this repo were produced with this exact setup.

Optional fields (also overridable via YAML `model.*`; not exposed on CLI):

```text
REASONING_FIELD=reasoning_content
TEMPERATURE=0.6
TOP_P=0.95
MAX_OUTPUT_TOKENS=
TIMEOUT_SECONDS=600
```

If the endpoint stores reasoning text in a different JSON key, change `REASONING_FIELD`. For extra request-body fields (e.g. `thinking: {type: adaptive}`, `reasoning_split: true`), put them in `configs/default.yaml` under `model.extra_body`.

---

## 3. Workflow

### 3.1 Smoke test (3 samples)

```bash
python -m src.run --task parents --max-samples 3 --target-prompt-chars 50000 --bin-tolerance 20000 --source-traces 2
```

> `--max-prompt-chars` and `--traces-per-sample` were typos from older README versions; see §5 for the real CLI.

### 3.2 Paper 256K neighbourhood first run

The paper evaluates on GraphWalks 256K. `dataset.load_graphwalks` centres a window at 262 144 characters with half-width `bin_tolerance`, then samples rows whose `prompt_chars` falls inside that window (with `shuffle=True, seed=42`). Start with a smaller dev run:

```bash
python -m src.run \
  --task parents \
  --max-samples 50 \
  --target-prompt-chars 262144 \
  --bin-tolerance 16384 \
  --source-traces 5 \
  --second-pass-repeats 2 \
  --conditions first_pass trace_append trace_as_state re2 question_first answer_feedback random_trace
```

By default the run writes two intermediate JSONL files under `--output-dir` (default `data/`):

- `processed_samples.jsonl` — source runs and the trace block per sample (keyed by `md5(prompt)`).
- `processed_experiments.jsonl` — final per-condition results per sample (also keyed by `md5(prompt)`).

Re-running on the same samples will hit the cache and skip duplicate model calls.

### 3.3 Makefile shortcuts

```bash
make smoke     # 3-sample smoke test
make repro     # 50 samples + 7 generation conditions
make summary   # summarise data/processed_experiments.jsonl
```

---

## 4. Config file: `configs/default.yaml`

```yaml
dataset:
  name: openai/graphwalks
  split: train
  task: parents         # parents | bfs

experiment:
  target_prompt_chars: 300000     # bin centre
  bin_tolerance: 180000           # half-width: [target-tol, target+tol]
  max_samples: 150
  source_traces: 5                # how many first-pass traces are stitched into T
  second_pass_repeats: 2          # repeats per condition (for mean metrics)
  trace_max_chars: 50000          # per-trace character cap
  seed: 42
  conditions:
    - first_pass
    - trace_append
    - trace_as_state

model:
  temperature: 1.0
  top_p: 0.95
  max_output_tokens: null
  timeout_seconds: 1800
  extra_body:
    reasoning_split: true
    thinking:
      type: adaptive
```

**Priority: CLI > YAML > code defaults.** `src.run` chains `parse_args → load_config → load_model_config`, where every experiment/model field has a YAML fallback.

---

## 5. Full CLI reference

```
python -m src.run
  --config <yaml>          # default configs/default.yaml
  --task <parents|bfs>
  --split <split name>     # default train
  --max-samples <N>
  --target-prompt-chars <N>     # bin centre (NOT a prompt cap)
  --bin-tolerance <N>           # bin half-width
  --source-traces <N>           # how many first-pass traces go into T
  --second-pass-repeats <N>     # repeats per condition
  --trace-max-chars <N>         # per-trace character cap (chars, not tokens)
  --conditions <c1 c2 ...>      # see §6
  --output-dir <dir>            # default data/
```

Before any API call, the CLI prints a diagnostic block (sample count, character range, source/2nd-pass repeat counts, trace cap, model/temperature/top_p/max_output, condition list) so you can sanity-check what was actually selected.

---

## 6. Conditions at a glance

All conditions are implemented in `experiment.run_sample_second_pass`:

| Condition           | Type                 |    Re-invokes model    | Description                                                                                                                                                             |
| ------------------- | -------------------- | :---------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `first_pass`      | baseline             | no (reuses source runs) | Mean metrics over source runs.                                                                                                                                          |
| `trace_append`    | paper main           |           yes           | `[x, T, q]`                                                                                                                                                           |
| `trace_as_state`  | paper main           |           yes           | `[T, x, q]`                                                                                                                                                           |
| `re2`             | control              |           yes           | `[x, q, x, q]`, no trace, just prompt repetition.                                                                                                                     |
| `question_first`  | control              |           yes           | `[q, x, q]`                                                                                                                                                           |
| `answer_feedback` | control              |           yes           | `[F, x, q]`, F stitched from the source runs' answer text.                                                                                                            |
| `random_trace`    | control              |           yes           | `[T', x, q]`, T' from another sample in the same `task`. The donor comes from a `seed`-driven cyclic permutation to guarantee it differs from the current sample. |
| `trace_only`      | ablation             |           yes           | Trace alone, no graph, no question.                                                                                                                                     |
| `majority_at5`    | upper-bound analysis |           no           | Majority vote over the**parsed answer sets** of the source runs (ties broken by first appearance).                                                                |
| `oracle_at5`      | upper-bound analysis |           no           | Picks the source run with the highest F1.                                                                                                                               |

`first_pass`, `majority_at5`, and `oracle_at5` are tagged with `non_generation: true` in the result and do not contribute to second-pass token spend.

---

## 7. Prompt assembly (`src/prompts.py`)

`split_graphwalks_prompt` splits a GraphWalks prompt at the **last** `Operation:` boundary:

- `x`: graph + context (everything before the question)
- `q`: the trailing question starting at `Operation:`

`serialize_traces` stitches the N traces as:

```
<TRACE_PREAMBLE>

<trace_start>

[Trace 1]
<trace 1 text (truncated to trace_max_chars + "...")>

[Trace 2]
...

</trace_end>
```

Each condition appends `ANSWER_FORMAT` at the very end:

> Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.

---

## 8. Evaluation (`src/evaluator.py`)

Mirrors the GraphWalks official scorer:

1. Inspect only the **last line** of the response.
2. Extract the set via the regex `Final Answer:\s*\[(.*)\]`. Missing or malformed → `malformed=True` (scores zero).
3. Compute `em` (set equality), `precision` / `recall` / `f1` (set semantics).
4. All four metrics are written to each record's `metrics` field; condition-level results use `mean_metrics` for the mean.

`scripts/summarize_results.py`:

- Lists EM / P / R / F1 / n per condition.
- Runs an **F1-consistency guardrail**: recomputes expected F1 from P and R, compares against the stored F1, and emits a `WARNING` if the gap exceeds 1e-9. (This caught a real P/R/F1 mismatch in a previous reproduction.)

---

## 9. Provider adapter (`src/provider.py`)

`OpenAICompatibleClient` POSTs directly to `{base_url}/chat/completions`:

- `system` uses `GRAPHWALKS_SYSTEM`; `user` content is the full prompt assembled for the condition.
- `temperature` / `top_p` / `max_completion_tokens` come from `ModelConfig`; `source_temperature` defaults to the configured temperature.
- `extra_body` (from YAML) is merged into the payload, e.g. `thinking: {type: adaptive}`, `reasoning_split: true`.
- Reasoning extraction order: `message[<REASONING_FIELD>]` → `message["reasoning_content"]` → `message["reasoning"]` → `raw["reasoning_content"]` → parse `content` blocks whose `type` contains `"reason"`.
- 3 retries on `httpx.RequestError`; any remaining failure raises `RuntimeError` (never silently swallowed).

If any source run returns an empty `reasoning`, the run aborts immediately — **no reasoning, no T to stitch** — with a hint to enable reasoning and expose the reasoning field.

---

## 10. Architecture

```text
                 openai/graphwalks
                        │
                        ▼
              load_graphwalks(task, split)
                  │ filter by prompt_chars
                  │ in [target-tol, target+tol]
                  │ (seed=42 shuffle)
                        │
                        ▼
   ┌────────────── Stage 1: source runs ──────────────┐
   │  For each sample:                                │
   │    md5(prompt) hit on processed_samples.jsonl?   │
   │      ├── yes → reuse runs / block                │
   │      └── no  → run baseline N times → stitch T   │
   │                                                  │
   │  Deterministic cyclic permutation over the same  │
   │  task → donors[idx]                              │
   └──────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────── Stage 2: per-condition second pass ──────────┐
   │  For each sample + each condition:                     │
   │    md5(prompt) hit on processed_experiments.jsonl?     │
   │      ├── yes → reuse result                            │
   │      └── no  → _call_repeated × second_pass_repeats    │
   │                                                      │
   │  conditions:                                          │
   │    first_pass / majority_at5 / oracle_at5 → no LLM    │
   │    trace_append / trace_as_state / re2 / question_first│
   │    answer_feedback / random_trace / trace_only → LLM   │
   │      └─ random_trace uses donor's block, others own T  │
   └──────────────────────────────────────────────────────┘
                        │
                        ▼
                data/processed_*.jsonl
                        │
                        ▼
              scripts/summarize_results.py
                  → EM / P / R / F1 table + F1 guardrail
```

---

## 11. Reproduction notes

> The cached `data/processed_samples.jsonl` (150 samples) and `data/processed_experiments.jsonl` (150 samples, 3 conditions: `first_pass` / `trace_append` / `trace_as_state`) shipped with this repo were produced with **MiniMax-M3** via `https://api.minimax.cn/v1`, `temperature=1.0`, `top_p=0.95`, `source_traces=5`, `second_pass_repeats=2`, `trace_max_chars=50_000`, seed `42`. To reproduce the same numbers end-to-end, point your `.env` at the same provider/model (see §2 example).

1. **Same T for both placements.** Trace Append and Trace as State share the exact same `source_traces`; only the placement differs.
2. **Random Trace donor must be a different problem.** The implementation uses a seed-driven cyclic permutation so the donor is always another sample in the same task. Don't swap it for a same-sample historical trace.
3. **GraphWalks historical ground-truth bug.** Use the current Hugging Face revision, not the 2025 snapshot.
4. **50 000 is characters, not tokens.** `trace_max_chars` truncates with `len(text) > max_chars`.
5. **Provider prompts and API settings aren't portable.** The paper's exact provider prompts/API may not transfer cleanly to other providers; this repo locks only the experimental logic and keeps the model adapter configurable.
6. **Paper-grade numbers need paper-grade setup.** Same provider / model / reasoning settings / tokenizer / context length / benchmark subset. This repo reproduces the **experimental logic**, not the published numbers.
7. **Resumable.** Finished samples are persisted by `md5(prompt)` to JSONL; re-running skips them automatically. Delete the corresponding row to force a rerun.
8. **`majority_at5` / `oracle_at5` are analytical.** They are tagged `non_generation: true` and add no second-pass token cost.

---

## 12. Directory layout

```text
src/
  run.py            # CLI entry point
  config.py         # YAML + .env → ModelConfig
  dataset.py        # GraphWalks loading + length binning
  prompts.py        # prompt split, trace serialisation, condition assembly
  provider.py       # OpenAI-compatible chat/completions client
  experiment.py     # source runs + per-condition second pass
  evaluator.py      # Final Answer extraction + set scoring
  io_utils.py       # JSONL + md5
configs/
  default.yaml      # default experiment + model config
scripts/
  inspect_dataset.py
  summarize_results.py
tests/
  test_core.py      # unit tests for evaluator / prompts
data/
  processed_samples.jsonl     # Stage 1 cache: source runs + trace block
  processed_experiments.jsonl # Stage 2 cache: per-condition results
```

---

## 13. Tests

```bash
pytest -q
```

`tests/test_core.py` covers: `extract_final_answer` (normal / empty set), `score_set` (P/R/F1), `serialize_traces` (truncation), `split_graphwalks_prompt` (x/q split).

---

## 14. Output data schema

`src.run` writes two JSONL files under `--output-dir` (default `data/`). Both are append-only and keyed by `md5(prompt)` so re-runs can resume from cache.

### 14.1 `processed_samples.jsonl` (Stage 1)

One record per sample, written right after the source runs finish. Schema:

```json
{
  "md5": "e49fe902951b793fcfe133c40c16fc8b",
  "sample": { /* GraphWalkSample asdict(): row_id, prompt, answer_nodes, prompt_chars, problem_type */ },
  "runs": [
    {
      "repeat": 1,
      "text": "Final Answer: [55b37c5c27]",
      "answer_text": "Final Answer: [55b37c5c27]",
      "reasoning": "The operation is to find the parents of ...",
      "metrics": {
        "em": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "prediction": ["55b37c5c27"],
        "malformed": false
      },
      "elapsed_s": 11.94,
      "usage": {
        "total_tokens": 257325,
        "prompt_tokens": 256938,
        "completion_tokens": 387,
        "completion_tokens_details": { "reasoning_tokens": 0 },
        "prompt_tokens_details": { "cached_tokens": 0 }
      }
    }
  ],
  "block": "<TRACE_PREAMBLE>\n\n<trace_start>\n\n[Trace 1]\n...\n</trace_end>"
}
```

Field notes:

- `md5` — `md5(sample.prompt)`. The cache key.
- `sample` — the original `GraphWalkSample` (graph text + `Operation:` block + `answer_nodes` + `prompt_chars`).
- `runs` — `source_traces` independent first-pass calls. Each `repeat` carries:
  - `text` — the model's visible answer text (the `Final Answer: [...]` line, possibly with surrounding output).
  - `answer_text` — kept equal to `text`; preserved for `answer_feedback` condition assembly.
  - `reasoning` — the raw CoT / chain-of-thought string returned by the model. **This is what gets stitched into `T` for downstream conditions.** Empty `reasoning` aborts the run.
  - `metrics` — `extract_final_answer` + `score_set` result for this single repeat.
  - `elapsed_s` — wall-clock seconds for the API call.
  - `usage` — the endpoint's `usage` payload verbatim; useful for cost accounting.
- `block` — the fully-serialised `T` (preamble + `<trace_start>` + `[Trace i]` blocks + `</trace_end>`), already truncated to `trace_max_chars` per trace. Stored so Stage 2 doesn't need to recompute.

### 14.2 `processed_experiments.jsonl` (Stage 2)

One record per sample, written after every condition finishes. Schema:

```json
{
  "md5": "e49fe902951b793fcfe133c40c16fc8b",
  "sample": { /* GraphWalkSample asdict() */ },
  "result": {
    "sample": { /* same shape as above */ },
    "trace": {
      "source_count": 5,
      "max_chars": 50000,
      "serialized_chars": 213044,
      "source_trace_chars": [41820, 41510, 41100, 42033, 41802],
      "block": "..."
    },
    "outputs": {
      "first_pass":      { "repeats": [...], "mean_metrics": { "em":..., "precision":..., "recall":..., "f1":... } },
      "trace_append":    { "repeats": [...], "mean_metrics": {...}, "prompt_chars": 447247 },
      "trace_as_state":  { "repeats": [...], "mean_metrics": {...}, "prompt_chars": 213044 },
      "re2":             { "repeats": [...], "mean_metrics": {...}, "prompt_chars": ... },
      "question_first":  { ... },
      "answer_feedback": { ... },
      "random_trace":    { ... },
      "trace_only":      { ... },
      "majority_at5":    { "repeats": [{...}], "mean_metrics": {...}, "non_generation": true },
      "oracle_at5":      { "repeats": [{...}], "mean_metrics": {...}, "non_generation": true }
    },
    "random_trace_donor_row_id": 1024
  }
}
```

Per-condition record shape:

| Field                    | Meaning                                                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repeats`              | List of per-call records. Same schema as Stage-1 `runs[i]`: `repeat`, `text`, `reasoning`, `metrics`, `elapsed_s`, `usage`, `condition`. Empty `text`/`reasoning` for non-generation conditions. |
| `mean_metrics`         | `mean(em/precision/recall/f1)` across `repeats`. The number to compare across conditions.                                                                                                                        |
| `prompt_chars`         | Length of the prompt actually sent to the model for this condition. Useful for sanity-checking trace placement.                                                                                                      |
| `non_generation`       | Only present for `first_pass`, `majority_at5`, `oracle_at5` — true means no second-pass LLM call was made.                                                                                                    |
| `skipped` + `reason` | Only present if a condition could not run (e.g.`random_trace` with no donor available).                                                                                                                            |

Top-level helpers:

- `trace.source_count` — number of source traces stitched into `T` (= `len(runs)`).
- `trace.max_chars` — per-trace character cap that was applied.
- `trace.serialized_chars` — actual length of `block` after truncation.
- `trace.source_trace_chars` — pre-truncation reasoning length per source run.
- `random_trace_donor_row_id` — the `row_id` of the sample whose `T` was used for the `random_trace` condition (a different sample in the same `task`, picked by the `seed`-driven cyclic permutation). `null` if there was only one sample.

### 14.3 Reading the data

Quick peek:

```bash
head -n 1 data/processed_experiments.jsonl | python -m json.tool
```

Per-condition EM / P / R / F1 / n table:

```bash
python scripts/summarize_results.py --path data/processed_experiments.jsonl
```

The summariser also runs an F1-consistency guardrail (§8) and warns when a stored F1 disagrees with the P/R-derived value.

To rerun a single sample, delete its `md5` row from both files; the next run will rebuild it from scratch.
