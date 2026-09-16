from __future__ import annotations

GRAPHWALKS_SYSTEM = (
    "You solve directed-graph algorithm problems. Use only the graph and operation in the user message. "
    "Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. "
    "Do not include any text before or after that line."
)

ANSWER_FORMAT = (
    "Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set."
)

TRACE_PREAMBLE = (
    "Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. "
    "They may contain mistakes. Use them only as scratchpad hints, and verify against the graph."
)


def serialize_traces(traces: list[str], max_chars: int = 50_000) -> str:
    blocks = []
    for i, trace in enumerate(traces, 1):
        text = (trace or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        blocks.append(f"[Trace {i}]\n{text}")
    return (f"{TRACE_PREAMBLE}\n\n<trace_start>\n" + "\n\n".join(blocks) + "\n</trace_end>").strip()


def split_graphwalks_prompt(prompt: str) -> tuple[str, str]:
    pos = prompt.rfind("Operation:")
    if pos < 0:
        raise ValueError("GraphWalks prompt does not contain a final Operation: block")
    return prompt[:pos].rstrip(), prompt[pos:].strip()


def baseline_parts(prompt: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [x, q, ANSWER_FORMAT]


def trace_as_state_parts(prompt: str, trace_block: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [trace_block, x, q, ANSWER_FORMAT]


def trace_append_parts(prompt: str, trace_block: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [x, trace_block, q, ANSWER_FORMAT]


def re2_parts(prompt: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [x, q, x, q, ANSWER_FORMAT]


def question_first_parts(prompt: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [q, x, q, ANSWER_FORMAT]


def answer_feedback_parts(prompt: str, answers: list[str]) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    feedback = "\n\n".join(f"[Answer {i}]\n{a.strip()}" for i, a in enumerate(answers, 1))
    return [feedback, x, q, ANSWER_FORMAT]


def random_trace_parts(prompt: str, trace_block: str) -> list[str]:
    x, q = split_graphwalks_prompt(prompt)
    return [trace_block, x, q, ANSWER_FORMAT]
