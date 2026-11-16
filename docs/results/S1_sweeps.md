# S1 - Retrieval sweeps and the default config decision

Run on the user's machine, 2026-11-16: `python cli.py sweep` -> 60 configs (6 chunkers x {bm25, dense, hybrid} x {no reranker, lexical reranker}; dense/hybrid with `nomic-embed-text` plain and with task prefixes).
n = 69 answerable questions; one question moves MRR by at most 0.5/69 = 0.007 (rank 2 -> 1) or 0.014 (rank missed -> 1). Metrics come from one ranked pool of 10, no generation involved.
Statistics (added in S1): 95% bootstrap interval of MRR, and for every row won/lost questions and a PAIRED bootstrap interval of the MRR difference versus the best BM25 row (fixed-200-ov40, BM25, no reranker; 2000 resamples, seed 0). `*` would mark an interval excluding 0.
Full 60-row output: `docs/results/S1_sweeps_full.md` (a copy of `results/sweeps.md`).

## Headline
**No row differs from the best BM25 row by more than noise.** Not one of the 59 comparison rows has a `*`. The numerically best rows are dense (MRR 0.981-0.993, 5 questions better / 1-2 worse than BM25, interval about [0.000, +0.07]),
but an exact sign test on those win/loss counts gives p = 0.22 (5/1) and 0.45 (5/2). With 69 questions a row must win 6 and lose 0, or win 8 and lose 1, to reach p < 0.05; nothing came close. Treat all differences below as "no evidence", not as "no difference".

## Retriever x chunker (plain nomic-embed-text, no reranker), versus best BM25 row
| chunking | retriever | MRR [95% CI] | won/lost | d_mrr [95% CI] | p50 ms |
|---|---|---|---|---|---|
| fixed-200-ov40 | bm25 | 0.959 [0.923, 0.993] | (reference) | - | 0.03 |
| fixed-200-ov40 | dense | 0.981 [0.952, 1.000] | 5/2 | +0.022 [-0.023, +0.068] | 27.8 |
| fixed-200-ov40 | hybrid | 0.971 [0.942, 0.993] | 3/1 | +0.012 [-0.014, +0.039] | 27.9 |
| fixed-400-ov80 | bm25 | 0.947 [0.908, 0.978] | 4/5 | -0.013 [-0.051, +0.029] | 0.02 |
| fixed-400-ov80 | dense | 0.983 [0.957, 1.000] | 5/2 | +0.024 [-0.019, +0.070] | 27.8 |
| fixed-400-ov80 | hybrid | 0.957 [0.920, 0.986] | 5/5 | -0.003 [-0.043, +0.041] | 27.8 |
| recursive-200 | bm25 | 0.935 [0.885, 0.977] | 2/6 | -0.024 [-0.065, +0.012] | 0.03 |
| recursive-200 | dense | 0.960 [0.924, 0.993] | 4/4 | +0.001 [-0.047, +0.049] | 27.8 |
| recursive-200 | hybrid | 0.942 [0.906, 0.978] | 3/5 | -0.017 [-0.058, +0.019] | 27.9 |
| recursive-300 (= recursive-500) | bm25 | 0.945 [0.900, 0.983] | 3/5 | -0.014 [-0.055, +0.029] | 0.03 |
| recursive-300 (= recursive-500) | dense | 0.960 [0.924, 0.993] | 4/4 | +0.001 [-0.047, +0.049] | 27.8 |
| recursive-300 (= recursive-500) | hybrid | 0.949 [0.913, 0.978] | 4/5 | -0.010 [-0.051, +0.030] | 27.9 |
| sentence | bm25 | 0.940 [0.889, 0.986] | 3/6 | -0.020 [-0.063, +0.024] | 0.05 |
| sentence | dense | 0.959 [0.920, 0.990] | 3/5 | -0.000 [-0.039, +0.040] | 27.8 |
| sentence | hybrid | 0.964 [0.920, 0.993] | 4/4 | +0.004 [-0.039, +0.049] | 27.9 |

## Answers to the S1 questions
1. **Does dense/hybrid beat BM25?** Dense is numerically ahead of the same chunker's BM25 row in all 6 chunkers (+0.015 to +0.036 MRR, a few questions' worth), and hybrid is mostly in between (on `sentence` it is the highest of the three), but no difference is distinguishable from noise. hit@3 is 0.971-1.0 for BM25 and 0.986-1.0 for dense/hybrid, i.e. at most 1-2 questions apart (an earlier sandbox BM25-only run located BM25's weakness on paraphrase questions, n=8; per-type values are in `results/sweeps.json` as `hit3_by_type`). With 8 paraphrase questions this cannot be established, so the paraphrase hypothesis remains **unproven**.
2. **Does any chunker matter?** Not detectably. Chunk integrity is 1.0 for all six (documents are at most 446 characters), `recursive-300` and `recursive-500` produce identical chunks (59), so the corpus cannot stress chunking. The only visible pattern is that the smallest chunks (`sentence`, 106 chunks, and `recursive-200`) are lowest under BM25 (MRR 0.930-0.940) and are the only rows where evidence recall at 5 drops below 1.0 (BM25 0.971 / 0.986, hybrid on `sentence` 0.986): plausibly because multi-hop questions need several chunks and small chunks split the evidence, but it is within noise.
3. **Lexical reranker:** changes MRR by -0.014 to +0.012 per row (one question); worst on `sentence`. No evidence of benefit; costs about 0.1 ms. Not adopted.
4. **nomic task prefixes (`search_query: ` / `search_document: `):** 24 paired comparisons (fewer independent ones: `recursive-300` and `-500` are identical), none significant. 13 rows moved +0.002 to +0.014 (best: hybrid recursive, 2 questions better, 0 worse, interval [0.000, +0.036], sign test p = 0.5), 10 did not change at all, 1 moved -0.003 (one question). Never meaningfully harmful, never proven useful; cost +1.9 ms per query (27.8 -> 29.7 ms). **Decision: leave OFF.**
5. **Latency:** BM25 0.02-0.13 ms; dense/hybrid about 28 ms (the query embedding) = about 7% of the end-to-end p50 of 399 ms, which is dominated by generation (377 ms).

## Decision: default retrieval config unchanged
`recursive-300`, `hybrid`, k = 4, no reranker, `nomic-embed-text` without prefixes (written as comments in `configs/ollama.yaml`). Why:
- There is no evidence that any other config is better, so there is no reason to move. Taking the best of 60 rows (dense, 0.993) would be selecting on noise: 60 configs were scored on the same 69 questions, so the maximum is optimistic.
- Hybrid keeps lexical matching for exact terms and identifiers, which dense embeddings can miss; here it costs nothing measurable (hit@3 = 1.0).
- It is the config `baseline_v2` was measured with (MRR 0.949, identical to this sweep's row), so S2-S6 and the S3 regression baseline stay consistent.
- Re-open when the corpus is larger or harder (S7): that is where chunking, reranking and dense-vs-lexical can actually separate.

## Limits
- Small synthetic corpus, near-ceiling retrieval (hit@5 = 1.0 in almost every row): the sweep mostly shows what *cannot* be distinguished here.
- The paired bootstrap interval degenerates when only 1-3 questions differ (lower bound exactly 0.000 without `*`). Use the won/lost counts and an exact sign test for such rows (done by hand above); the S3 regression gate should use the same paired logic and state its minimum detectable change (about 6 net questions at n = 69).
- Rows share the same 69 questions, so they are not independent; no multiple-comparison correction was applied (none was needed, as nothing was significant).
