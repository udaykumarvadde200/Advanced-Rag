# S5 - Generation evaluation and failure attribution
**Depends on:** S1 (and S4 if the judge is ready). **Size:** medium.
**Goal:** understand answer quality: gold-context vs retrieved-context, per-question-type results, why answers fail, and which model/prompt to keep.

**Claude reads only:** HANDOFF.md, PROJECT_GOAL.md, this file; `cli.py` (cmd_eval), `rag/llm.py`, `rag/metrics.py`, `rag/pipeline.py`.

## Tasks
1. `eval/run_generation.py` + `python cli.py gen`: run each question with (a) **gold context** (the evidence chunks) and (b) **retrieved context**; the gap = retrieval's cost.
2. **Failure attribution** for every wrong answer: chunking (evidence not intact) / retrieval miss / ranking (found but not top-k) / generation error (context had it) / abstention error / metric artifact. Table + counts per question type.
3. Better metrics, keeping the old ones: normalised correctness (number words, units, alternates) next to the strict keyword check; citation-aware groundedness (strip `[n]` and doc ids); judge-based faithfulness if S4 exists.
4. Compare models (`llama3.2:3b`, `qwen2.5:3b`, optionally a 7B) and 2-3 prompt variants (including a short few-shot); k = 2/4/6; context ordering. Record tokens, latency, estimated cost per model.
5. Abstention precision/recall including the near-miss unanswerables. Look at the three known failures (q020, q033, q035) and fix only what is a real prompt or pipeline problem.
6. Final table to `docs/results/S5_generation.md`.

**User runs:** `ollama pull` for each model, `python cli.py gen` per model; pastes summaries.
## Done when
Failure-attribution table, the gold-vs-retrieved gap, and a chosen model/prompt with numbers; HANDOFF updated; tag `s5`.
