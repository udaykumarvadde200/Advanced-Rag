# S1 - Real baseline on the 82-question set

Run on the user's machine (Windows, RTX 4050 6 GB, 16 GB RAM), 2026-11-16: `python cli.py eval --out results/baseline_v2.json`.
Config (`configs/ollama.yaml`, unchanged in S1 so far): recursive-300 chunks, hybrid (BM25 + dense, RRF), k=4, no reranker, `llama3.2:3b` (temperature 0, seed 42), `nomic-embed-text` without task prefixes.
Data: synthetic "Acme-Fake Corp", 30 docs, 82 golden questions = 69 answerable (52 factual, 9 multi-hop, 8 paraphrase) + 13 unanswerable. Single run.

## Results
| Metric | Value | n | 95% interval (Wilson, counts only) |
|---|---|---|---|
| hit@4 | 1.000 | 69 | - |
| MRR | 0.949 | 69 | see `S1_sweeps.md` (bootstrap) |
| doc_recall | 1.000 | 69 | - |
| correctness (keyword match) | 0.855 (59/69) | 69 | 0.753 - 0.919 |
| - factual | 0.865 (45/52) | 52 | 0.747 - 0.933 |
| - multi-hop | 0.667 (6/9) | 9 | 0.354 - 0.879 |
| - paraphrase | 1.000 (8/8) | 8 | 0.676 - 1.000 |
| abstention accuracy | 1.000 (13/13) | 13 | 0.772 - 1.000 |
| groundedness (token-overlap proxy) | 0.680 | 69 | crude metric, see limits |

Application: p50 latency 399 ms, p95 828 ms (mean 432 ms); stage p50: embed_query 26 ms, rank 0.13 ms, generate 377 ms. Tokens per question: 270 in, 25 out. Estimated cost at the reference price: $0.005 for all 82 questions. Wall time 49.5 s.

## What this says
- Retrieval is not the bottleneck: every answerable question had its evidence in the top 4 (hit@4 = 1.0, doc_recall = 1.0). All 10 incorrect answers are therefore generation-side or metric-side (the 3B model, or the strict keyword check).
- The weak spot is multi-hop (6/9), as expected for a 3B model that must combine two chunks. With n=9 the interval is very wide (0.35 - 0.88): this is a signal to investigate in S5, not a measured rate.
- Latency is dominated by generation (about 95% of p50); retrieval costs about 26 ms, almost all of it the query embedding.
- Consistency check: the eval MRR (0.949, k=4) equals the sweep's hybrid / recursive-300 / no-reranker MRR (0.949), so the eval and the sweep harness agree.

## Limits (do not over-read)
- The earlier 40-question numbers (correctness 0.914 on 35 answerable) are NOT comparable: the set changed (42 harder questions added, 9 multi-hop, 13 abstention). The lower correctness is not a regression.
- Correctness is a keyword match (strict, can mark a right answer wrong); groundedness is a token-overlap proxy that counts citation text as ungrounded. Both are replaced by validated judging in S4/S5.
- Single run. Quality was identical across repeated runs at temperature 0 / seed 42 on the old set, so run-to-run noise is small, but the sampling noise from n=69 is large (intervals above).
- The corpus is small and synthetic, so retrieval numbers are near ceiling and will overstate real-world performance.
- The 10 wrong questions were not listed in the pasted summary (they are in `results/baseline_v2.json`); failure attribution is S5's job.
