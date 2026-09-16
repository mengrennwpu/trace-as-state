from __future__ import annotations

from src.dataset import load_graphwalks

samples = load_graphwalks(task="parents", max_samples=10, max_prompt_chars=128000)
for s in samples:
    print(s.row_id, s.prompt_chars, s.answer_nodes, s.prompt[-160:].replace("\n", " "))
