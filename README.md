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
```

---

## 13. Tests

```bash
pytest -q
```

`tests/test_core.py` covers: `extract_final_answer` (normal / empty set), `score_set` (P/R/F1), `serialize_traces` (truncation), `split_graphwalks_prompt` (x/q split).
