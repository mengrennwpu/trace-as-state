> **Language / 语言:** [English](README.md) · [中文](README.zh.md)

---

# Trace as State — GraphWalks 复现

**Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers**（Xu Zou, Jie Tang, arXiv:2609.02702, v1）的轻量级复现脚手架。

- 论文地址：[https://alphaxiv.org/abs/2609.02702](https://alphaxiv.org/abs/2609.02702)
- arXiv：[2609.02702](https://arxiv.org/abs/2609.02702)

核心实验：用一束独立的 reasoning trace 替换单次链式思考，并把它们放在 prompt 中的不同位置。

| 名称            | 模板             | 说明                                                                        |
| --------------- | ---------------- | --------------------------------------------------------------------------- |
| First Pass      | `[x, q]`       | 论文基线。无 trace，只跑一次问答。                                          |
| Trace Append    | `[x, T, q]`    | 把 trace 放在 graph 之后、问题之前。                                        |
| Trace as State  | `[T, x, q]`    | 把 trace 放在最前面，作为条件状态。                                         |
| Re2             | `[x, q, x, q]` | Re-Reading：把 graph + 问题重复一遍。                                       |
| Question First  | `[q, x, q]`    | 把问题前置。                                                                |
| Answer Feedback | `[F, x, q]`    | 用首轮 answer 列表作为反馈。                                                |
| Random Trace    | `[T', x, q]`   | 同子任务里**另一道题**的 trace 作为 donor。                           |
| Trace Only      | `[T]`          | 只看 trace，没有 graph 也没有问题。                                         |
| Majority@5      | —               | 在 source runs 上对**解析后的答案集合**做多数投票（无需再调用模型）。 |
| Oracle@5        | —               | 取 source runs 里 F1 最高的答案（无需再调用模型）。                         |

`T` 由 `source_traces`（论文默认 5）条独立 reasoning trace 拼接而成，每条 trace 截断到 `trace_max_chars`（论文默认 50 000）字符。

> 这份实现刻意做成 **provider-agnostic**：只要端点兼容 chat-completions、并把推理文本暴露为 `reasoning_content` / `reasoning` / 可配置的字段即可。许多托管 API 默认不暴露原始 CoT，**没有 reasoning 文本就无法做 trace-as-state 复现**。

---

## 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`datasets`、`httpx`、`python-dotenv`、`pandas`、`matplotlib`、`pyyaml`。

---

## 2. 配置模型

复制 `.env.example` 为 `.env` 并填写：

```text
API_BASE_URL=https://your-openai-compatible-endpoint/v1
API_KEY=...
MODEL=your-reasoning-model
```

示例：对接 MiniMax M3 推理端点：

```text
API_BASE_URL=https://api.minimax.cn/v1
API_KEY=sk-...
MODEL=MiniMax-M3
REASONING_FIELD=reasoning_content
TIMEOUT_SECONDS=1800
```

仓库自带的 `data/processed_*.jsonl` 就是用这套配置跑出来的。

可选字段（也都可通过 YAML 的 `model.*` 覆盖；CLI 不暴露）：

```text
REASONING_FIELD=reasoning_content
TEMPERATURE=0.6
TOP_P=0.95
MAX_OUTPUT_TOKENS=
TIMEOUT_SECONDS=600
```

如果端点把推理文本放在别的 JSON key（例如某些中转站），把 `REASONING_FIELD` 改掉即可；如果端点需要额外的请求体（例如 `thinking: {type: adaptive}`、`reasoning_split: true`），写到 `configs/default.yaml` 的 `model.extra_body` 即可。

---

## 3. 跑通流程

### 3.1 冒烟测试（3 条样本）

```bash
python -m src.run --task parents --max-samples 3 --target-prompt-chars 50000 --bin-tolerance 20000 --source-traces 2
```

> `--max-prompt-chars` / `--traces-per-sample` 是 README 旧版本的笔误；实际 CLI 见第 5 节。

### 3.2 论文 256K 邻域首跑

论文评估在 GraphWalks 256K。`dataset.load_graphwalks` 默认中心 262 144 字符、半宽 `bin_tolerance`，从 `prompt_chars` 落入该窗口的样本里挑选（`shuffle=True, seed=42`）。先做小一点的开发跑：

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

默认会把两份中间产物写到 `--output-dir`（默认 `data/`）下：

- `processed_samples.jsonl` —— 每个 sample 的 source runs 与 trace block（按 `md5(prompt)` 断点）。
- `processed_experiments.jsonl` —— 每个 sample 在所选 condition 上的最终结果（同样按 `md5(prompt)` 断点）。

下次再跑完全相同的样本会自动跳过、不会重复调用模型。

### 3.3 Makefile 等价命令

```bash
make smoke     # 冒烟
make repro     # 50 条样本 + 7 个 generation condition
make summary   # 把 data/processed_experiments.jsonl 汇总成表格
```

---

## 4. 配置文件：`configs/default.yaml`

```yaml
dataset:
  name: openai/graphwalks
  split: train
  task: parents         # parents | bfs

experiment:
  target_prompt_chars: 300000     # 长度分箱中心
  bin_tolerance: 180000           # 半宽：[target-tol, target+tol]
  max_samples: 150
  source_traces: 5                # 每条样本跑几次首轮 → 拼成 T
  second_pass_repeats: 2          # 每个 condition 的复跑次数（用于算均值）
  trace_max_chars: 50000          # 单条 trace 的字符上限
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

优先级：**CLI > YAML > 代码默认值**。`src.run` 的 `parse_args → load_config → load_model_config` 链路完全覆盖了 YAML 中的实验/模型字段。

---

## 5. CLI 全量参数

```
python -m src.run
  --config <yaml>          # 默认 configs/default.yaml
  --task <parents|bfs>
  --split <split name>     # 默认 train
  --max-samples <N>
  --target-prompt-chars <N>     # 长度分箱中心（不是 prompt 上限）
  --bin-tolerance <N>           # 长度分箱半宽
  --source-traces <N>           # T 由多少条首轮 trace 拼接
  --second-pass-repeats <N>     # 每个 condition 的复跑次数
  --trace-max-chars <N>         # 单条 trace 字符上限（截断，不是 token）
  --conditions <c1 c2 ...>      # 详见第 6 节
  --output-dir <dir>            # 默认 data/
```

启动时会先打印一份诊断（样本数、字符区间、source/2nd-pass 复跑数、trace cap、模型/temperature/top_p/max_output、condition 列表），可以一眼看出分箱选中了多少样本。

---

## 6. Conditions 一览

`experiment.run_sample_second_pass` 中实现的全部条件：

| Condition           | 类型       |       是否再调用模型       | 说明                                                                                                                  |
| ------------------- | ---------- | :------------------------: | --------------------------------------------------------------------------------------------------------------------- |
| `first_pass`      | baseline   | 否（直接复用 source runs） | source runs 的均值作为基线。                                                                                          |
| `trace_append`    | 论文主条件 |             是             | `[x, T, q]`                                                                                                         |
| `trace_as_state`  | 论文主条件 |             是             | `[T, x, q]`                                                                                                         |
| `re2`             | 控制组     |             是             | `[x, q, x, q]`，无 trace，仅靠 prompt 重复。                                                                        |
| `question_first`  | 控制组     |             是             | `[q, x, q]`                                                                                                         |
| `answer_feedback` | 控制组     |             是             | `[F, x, q]`，F 由 source runs 的 answer 文本拼成。                                                                  |
| `random_trace`    | 控制组     |             是             | `[T', x, q]`，T' 是同一 `task` 中**另一道题**的 trace。Donor 由 `seed` 决定的循环置换保证与当前样本不同。 |
| `trace_only`      | 消融       |             是             | 只看 trace，不给 graph 和问题。                                                                                       |
| `majority_at5`    | 上界分析   |             否             | 对 source runs 的解析后答案集合做多数投票（ties 按首次出现顺序破平）。                                                |
| `oracle_at5`      | 上界分析   |             否             | 直接选 source runs 里 F1 最大的那个。                                                                                 |

`first_pass`、`majority_at5`、`oracle_at5` 在结果里会用 `non_generation: true` 标记，且不计入第二阶段 token 开销。

---

## 7. Prompt 拼接逻辑（`src/prompts.py`）

`split_graphwalks_prompt` 把 GraphWalks 的 prompt 在最后一个 `Operation:` 处切成两部分：

- `x`：graph 与上下文（不含问题）
- `q`：`Operation:` 之后的最终问题

`serialize_traces` 把 N 条 trace 拼成：

```
<TRACE_PREAMBLE>

<trace_start>

[Trace 1]
<trace 1 文本（截断到 trace_max_chars + "..."）>

[Trace 2]
...

</trace_end>
```

每种 condition 的拼接顺序如第 1 节表格所示。所有 condition 末尾都追加 `ANSWER_FORMAT`：

> Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.

---

## 8. 评估指标（`src/evaluator.py`）

严格沿用 GraphWalks 官方 scorer：

1. 只看响应的**最后一行**。
2. 用正则 `Final Answer:\s*\[(.*)\]` 抽取集合；找不到或格式错误 → `malformed=True`（记 0 分）。
3. 计算 `em`（集合相等）、`precision` / `recall` / `f1`（集合语义）。
4. `em` 和 `precision` / `recall` / `f1` 都会被写入每条 record 的 `metrics` 字段；条件级用 `mean_metrics` 取均值。

`scripts/summarize_results.py` 会：

- 按 condition 列出 EM / P / R / F1 / n；
- **做一次 F1 一致性 guardrail**：用 P、R 反推 expected F1，与记录里写的 F1 对比，差值 > 1e-9 就 `WARNING` 报警（捕获过去一轮复现里出现过的 P/R/F1 不一致 bug）。

---

## 9. Provider 适配（`src/provider.py`）

`OpenAICompatibleClient` 直接 POST 到 `{base_url}/chat/completions`：

- `system` 用 `GRAPHWALKS_SYSTEM`；`user` 内容是各 condition 拼好的整段 prompt。
- `temperature` / `top_p` / `max_completion_tokens` 来自 `ModelConfig`；`source_temperature` 默认与 `temperature` 相同。
- `extra_body`（来自 YAML）会合并进 payload，例如 `thinking: {type: adaptive}`、`reasoning_split: true`。
- reasoning 抽取顺序：`message[<REASONING_FIELD>]` → `message["reasoning_content"]` → `message["reasoning"]` → `raw["reasoning_content"]` → 解析 content blocks 中 `type` 含 "reason" 的部分。
- 若 3 次重试都失败，会直接 raise RuntimeError，不静默吞错。

如果 `source_runs` 里某条 trace 的 `reasoning` 为空，会立即抛错（**没有 reasoning 就无法拼 T**），并提示开启 reasoning 并暴露 reasoning 字段。

---

## 10. 架构总览

```text
                 openai/graphwalks
                        │
                        ▼
              load_graphwalks(task, split)
                  │ 按 prompt_chars
                  │ 在 [target-tol, target+tol]
                  │ 区间内挑样本 (seed=42)
                        │
                        ▼
   ┌────────────── Stage 1: source runs ──────────────┐
   │  对每个 sample：                                  │
   │    md5(prompt) 命中 processed_samples.jsonl?      │
   │      ├── 是 → 复用 runs / block                   │
   │      └── 否 → 跑 N 次 baseline → 拼成 T          │
   │                                                  │
   │  同一 task 内做确定性循环置换 → donors[idx]       │
   └──────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────── Stage 2: per-condition second pass ──────────┐
   │  对每个 sample + 每个 condition：                       │
   │    md5(prompt) 命中 processed_experiments.jsonl?       │
   │      ├── 是 → 复用 result                              │
   │      └── 否 → _call_repeated × second_pass_repeats     │
   │                                                      │
   │  conditions:                                          │
   │    first_pass / majority_at5 / oracle_at5 → no LLM    │
   │    trace_append / trace_as_state / re2 / question_first│
   │    answer_feedback / random_trace / trace_only → LLM   │
   │      └─ random_trace 用 donor 的 block，其它用自己的   │
   └──────────────────────────────────────────────────────┘
                        │
                        ▼
                data/processed_*.jsonl
                        │
                        ▼
              scripts/summarize_results.py
                  → EM / P / R / F1 表 + F1 guardrail
```

---

## 11. 复现注意事项

> 仓库里附带的 `data/processed_samples.jsonl`（150 条样本）与 `data/processed_experiments.jsonl`（150 条样本、3 个 condition：`first_pass` / `trace_append` / `trace_as_state`）是用 **MiniMax-M3**（端点 `https://api.minimax.cn/v1`，`temperature=1.0`，`top_p=0.95`，`source_traces=5`，`second_pass_repeats=2`，`trace_max_chars=50_000`，seed `42`）跑出来的。要逐字复现同一份数据，把 `.env` 指到同一 provider/model 即可（见 §2 示例）。

1. **同一份 T**：Trace Append 和 Trace as State 共享同一组 `source_traces`；只改 placement，不重跑首轮。
2. **Random Trace 的 donor 必须异题**：实现里用 seed 控制的循环置换保证 donor 是同 task 内另一道题；不要换成同样本历史 trace。
3. **GraphWalks 数据集历史 ground-truth bug**：使用当前的 Hugging Face 修订版，不要锁 2025 旧版快照。
4. **50 000 是字符数，不是 token 数**：`trace_max_chars` 直接用 `len(text) > max_chars` 截断。
5. **API / 提示词不可直接移植**：论文具体 provider 的 prompt 与 API 设置不一定能照搬到其它 provider；本仓库只锁实验逻辑，模型适配层（`OpenAICompatibleClient` + YAML `model.extra_body`）保持可换。
6. **要拿到论文级数字**：同 provider / model / reasoning 设置 / tokenizer / 上下文长度 / 评测子集；本仓库复现的是**实验逻辑**，不是具体数字。
7. **断点续跑**：跑过的样本会按 `md5(prompt)` 写进 JSONL，再次跑会自动跳过；删掉对应行即可重跑。
8. **跳过 `majority_at5` / `oracle_at5`**：这两个是分析用的非生成条件，结果里 `non_generation: true`，不会被算进 token 开销。

---

## 12. 目录结构

```text
src/
  run.py            # CLI 入口
  config.py         # YAML + .env → ModelConfig
  dataset.py        # GraphWalks 加载与分箱
  prompts.py        # prompt 拆分、trace 序列化、condition 拼接
  provider.py       # OpenAI 兼容 chat/completions 客户端
  experiment.py     # source runs + 第二轮条件实验
  evaluator.py      # Final Answer 抽取与集合打分
  io_utils.py       # JSONL / md5
configs/
  default.yaml      # 默认实验 + 模型配置
scripts/
  inspect_dataset.py
  summarize_results.py
tests/
  test_core.py      # evaluator / prompts 的单元测试
data/
  processed_samples.jsonl     # Stage 1 缓存：source runs + trace block
  processed_experiments.jsonl # Stage 2 缓存：每个 condition 的结果
```

---

## 13. 单元测试

```bash
pytest -q
```

`tests/test_core.py` 覆盖：`extract_final_answer`（普通 / 空集）、`score_set`（P/R/F1）、`serialize_traces`（截断）、`split_graphwalks_prompt`（x/q 拆分）。

---

## 14. 输出数据 schema

`src.run` 在 `--output-dir`（默认 `data/`）下写两份 append-only 的 JSONL，都以 `md5(prompt)` 作为缓存 key，方便断点续跑。

### 14.1 `processed_samples.jsonl`（Stage 1）

每条样本一行，在 source runs 跑完后立即写入。结构：

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

字段说明：

- `md5` —— `md5(sample.prompt)`，缓存的主键。
- `sample` —— 原始 `GraphWalkSample`（graph 文本 + `Operation:` 块 + `answer_nodes` + `prompt_chars`）。
- `runs` —— `source_traces` 次独立首轮调用的结果。每条 `repeat` 包含：
  - `text` —— 模型可见的回答文本（一般就是 `Final Answer: [...]` 这一行）。
  - `answer_text` —— 与 `text` 相同，专门保留给 `answer_feedback` condition 拼接用。
  - `reasoning` —— 模型返回的原始 CoT / chain-of-thought 字符串。**就是后面要拼成 `T` 的内容；为空会让整次 run 直接报错。**
  - `metrics` —— 该次调用的 `extract_final_answer` + `score_set` 结果。
  - `elapsed_s` —— 该次 API 调用的墙钟秒数。
  - `usage` —— 端点返回的 `usage` 字段原样保留，便于算成本。
- `block` —— 已拼好的完整 `T`（preamble + `<trace_start>` + `[Trace i]` 块 + `</trace_end>`），每条 trace 已经按 `trace_max_chars` 截断。**存下来是为了让 Stage 2 不必重算。**

### 14.2 `processed_experiments.jsonl`（Stage 2）

每条样本一行，在该样本所有 condition 跑完后写入。结构：

```json
{
  "md5": "e49fe902951b793fcfe133c40c16fc8b",
  "sample": { /* GraphWalkSample asdict() */ },
  "result": {
    "sample": { /* 同上 */ },
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

每个 condition 的字段含义：

| 字段                     | 含义                                                                                                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repeats`              | 每次模型调用的结果列表，schema 与 Stage 1 的 `runs[i]` 一致：`repeat` / `text` / `reasoning` / `metrics` / `elapsed_s` / `usage` / `condition`。非生成类 condition 的 `text` / `reasoning` 为空。 |
| `mean_metrics`         | 对该 condition 所有 `repeats` 取 `em/precision/recall/f1` 的均值，是跨 condition 对比的主指标。                                                                                                                   |
| `prompt_chars`         | 该 condition 实际发给模型的 prompt 长度，方便核对 trace 放置是否正确。                                                                                                                                                |
| `non_generation`       | 仅出现在 `first_pass`、`majority_at5`、`oracle_at5`，值为 `true` 表示这次没有真正调用模型。                                                                                                                   |
| `skipped` + `reason` | 仅在 condition 跑不了时出现（例如 `random_trace` 找不到 donor）。                                                                                                                                                   |

顶层辅助字段：

- `trace.source_count` —— 拼成 `T` 的 source trace 数（= `len(runs)`）。
- `trace.max_chars` —— 单条 trace 应用过的字符上限。
- `trace.serialized_chars` —— 截断后 `block` 的实际长度。
- `trace.source_trace_chars` —— 每条 source trace 截断前的 reasoning 长度。
- `random_trace_donor_row_id` —— `random_trace` condition 用的 donor 样本的 `row_id`（同 `task` 内另一道题，由 `seed` 控制的循环置换选出来）。只有一个样本时为 `null`。

### 14.3 读取方式

快速浏览一条记录：

```bash
head -n 1 data/processed_experiments.jsonl | python -m json.tool
```

按 condition 汇总 EM / P / R / F1 / n：

```bash
python scripts/summarize_results.py --path data/processed_experiments.jsonl
```

汇总脚本还会跑一次 F1 一致性 guardrail（第 8 节），发现存的 F1 与从 P/R 反推的值不一致就会 `WARNING` 报警。

要重跑某条样本，把它的 `md5` 行从两份 JSONL 里删掉，下次运行就会从头重建。
