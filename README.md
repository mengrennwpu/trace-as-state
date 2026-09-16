# Trace as State - GraphWalks reproduction

A lightweight reproduction scaffold for **Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers** (Xu Zou, Jie Tang, arXiv:2609.02702, v1).

The core experiment is:

- First Pass: `[x, q]`
- Trace Append: `[x, T, q]`
- Trace as State: `[T, x, q]`

where `T` is serialized from `n_traces` independent reasoning traces generated on the same task. The paper uses 5 runs and truncates each trace to 50,000 characters in its reported realization.

This repo is intentionally provider-agnostic. It expects a chat-completions-compatible endpoint returning a reasoning field such as `reasoning_content`, `reasoning`, or a configurable field. This matters because many hosted APIs do not expose raw chain-of-thought; without exposed reasoning text, the exact paper setup cannot be reproduced.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Configure model

Copy `.env.example` to `.env` and fill in:

```text
API_BASE_URL=https://your-openai-compatible-endpoint/v1
API_KEY=...
MODEL=your-reasoning-model
```

Optional:

```text
REASONING_FIELD=reasoning_content
TEMPERATURE=0.6
```

If your endpoint uses a different JSON key for reasoning, set `REASONING_FIELD` accordingly.

## 3. Smoke test with 3 samples

```bash
python -m src.run --task parents --max-samples 3 --max-prompt-chars 50000 --traces-per-sample 2
```

## 4. Recommended first reproduction

The paper evaluates GraphWalks at 256K. For an inexpensive development run, start smaller and scale up later:

```bash
python -m src.run \
  --task parents \
  --max-samples 50 \
  --max-prompt-chars 128000 \
  --traces-per-sample 5 \
  --conditions first_pass trace_append trace_as_state re2 question_first answer_feedback
```

Outputs are JSONL files under `results/`.

## 5. Evaluation

GraphWalks requires a terminal `Final Answer: [...]`. The evaluator follows the dataset card's extraction/scoring logic closely:

- exact match on the predicted set
- set precision / recall / F1
- malformed final answer counts as failure

## 6. Important reproduction notes

1. **Reuse the exact same five traces** for Trace Append and Trace as State. Only placement changes.
2. Keep the question at the end for the placement experiments: `[x,T,q]` vs `[T,x,q]`.
3. GraphWalks has had a historical ground-truth bug; use the current Hugging Face dataset revision rather than the original 2025 snapshot.
4. The 50,000 value is in **characters**, not tokens.
5. The paper's exact provider prompts/API settings may not be fully portable across other providers. This implementation keeps the experimental logic fixed while making the model adapter configurable.
6. For exact paper-level results, use the same provider/model, reasoning settings, tokenizer/context limits, and benchmark subset described by the paper.

## 7. Architecture

```text
GraphWalks
   |
   +--> First Pass x N --> traces.jsonl
   |                        |
   |                        +--> serialize --> T
   |
   +--> First Pass ----------> baseline answer
   |
   +--> [x, T, q] ----------> Trace Append
   |
   +--> [T, x, q] ----------> Trace as State
   |
   +--> controls ------------> Re2 / Question First / Answer Feedback / Random Trace
                                  |
                                  v
                            EM / Precision / Recall / F1
```
