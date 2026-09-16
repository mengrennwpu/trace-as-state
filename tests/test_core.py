from src.evaluator import extract_final_answer, score_set
from src.prompts import serialize_traces, split_graphwalks_prompt


def test_extract_answer():
    pred, bad = extract_final_answer("thinking\nFinal Answer: [abc, def]")
    assert pred == ["abc", "def"]
    assert not bad


def test_extract_empty():
    pred, bad = extract_final_answer("Final Answer: []")
    assert pred == []
    assert not bad


def test_score():
    s = score_set(["a", "b"], ["b", "c"])
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["f1"] == 0.5


def test_serialize():
    s = serialize_traces(["abc", "xyz"], max_chars=2)
    assert "[Trace 1]" in s
    assert "ab..." in s


def test_split():
    context, question = split_graphwalks_prompt("A graph\nOperation:\nFind")
    assert context == "A graph"
    assert question.startswith("Operation:")
