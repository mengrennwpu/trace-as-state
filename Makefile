smoke:
	python -m src.run --task parents --max-samples 3 --max-prompt-chars 50000 --traces-per-sample 2 --conditions first_pass trace_append trace_as_state

repro:
	python -m src.run --task parents --max-samples 50 --max-prompt-chars 128000 --traces-per-sample 5 --conditions first_pass trace_append trace_as_state re2 question_first answer_feedback random_trace

summary:
	python scripts/summarize_results.py results/run.jsonl
