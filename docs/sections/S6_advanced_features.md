# S6 - Advanced features, each behind a config switch and ablated
**Depends on:** S1, S5 (S3 for the regression re-run). **Size:** large: split into two conversations if needed (6a: reranker + routing, 6b: calculator + stretch).
**Goal:** add advanced RAG features and *measure what each buys*, so the claim is "feature X changed metric Y by Z at W ms".

**Claude reads only:** HANDOFF.md, PROJECT_GOAL.md, this file; `rag/pipeline.py`, `rag/retrieval.py`, `rag/llm.py`, `eval/run_sweeps.py`, `configs/ollama.yaml`.

## Features
1. **Hybrid retrieval** is the default decided in S1.
2. **Reranker:** evaluate the Ollama pointwise reranker (and the lexical one) on the paraphrase and multi-hop slices: quality gain vs added latency (one LLM call per candidate). Keep only if the gain beats the cost.
3. **Metadata routing:** classify the query into department / document type (rules first, optional small-LLM classifier) and pre-filter chunks, like the ACL filter. Measure wrong-document rate before/after (especially Berlin vs main remote policy, archived vs current leave policy).
4. **Calculator tool:** the LLM extracts operands, Python computes with a safe evaluator (no `eval`). Target: arithmetic questions such as q073 (cab fare + travel allowance).
5. Stretch only if time remains: citation verification + abstention guard, parent-child retrieval.

## Tasks
Implement each as a config switch; offline unit tests; ablation table (add/remove each feature) with deltas in hit@k, MRR, correctness by type, p50/p95 latency, tokens; re-run the S3 regression suite; a verdict per feature (keep / drop).
## Done when
`docs/results/S6_ablations.md` with verdicts; HANDOFF updated; tag `s6`.
