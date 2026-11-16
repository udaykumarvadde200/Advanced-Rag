# HANDOFF (living document)  -  read after PROJECT_GOAL.md

## UPDATE RULES (whoever finishes a task must do all of this before ending the conversation)
1. Set the section/task row in the **Status table**.
2. Append an entry to the **Completed log** (date, what, files touched, how verified, key numbers). Summarise: put long tables in `docs/results/<section>_*.md` (tracked), not here.
3. Refresh **Current state** (numbers, config, commands that changed).
4. Rewrite **Next** (one short list, in order).
5. Add any new problem to **Open issues**; delete ones that are fixed.
6. End the reply with the git commands: `git add -A`, a commit message, and a tag (`s1`, `s2`, ...). Give the user the updated zip (`rag-eval.zip`, includes `.git`).
Keep this file under ~150 lines.

## Starting a new conversation (copy-paste this as the first message)
> I'm continuing a project. Attached: rag-eval.zip (current repo). Unzip it, read HANDOFF.md, then PROJECT_GOAL.md, then docs/sections/<SECTION>.md, and work ONLY on that section.
> Read other files only if the brief lists them. Follow the UPDATE RULES in HANDOFF.md as each task finishes. My data/summary output from my machine is below: <paste only summary blocks, not full logs>.

To keep context small: one section per conversation, upload the zip (do not paste code), paste only summary blocks, and end by updating HANDOFF.md.

## Status table
| Section | Name | Status |
|---|---|---|
| S0 | Foundation (data, pipeline, Ollama runtime, CLI, tests) | DONE (tag s0-foundation) |
| S1 | Real baseline + retrieval sweeps | DONE (tag s1 pending: user runs the git commands) |
| S2 | Safety suite | NOT STARTED |
| S3 | Regression gate | NOT STARTED |
| S4 | LLM judge + judge validation | NOT STARTED |
| S5 | Generation evaluation + failure attribution | NOT STARTED |
| S6 | Advanced features + ablations | NOT STARTED |
| S7 | Dataset decision (optional) | NOT STARTED (FinanceBench was discussed; on hold) |
| S8 | Final report + polish | NOT STARTED |

## Current state
- User environment: Windows PowerShell, project at `C:\Users\uday2\Downloads\rag-eval\rag-eval`, venv `.venv`, Ollama installed, models pulled: `llama3.2:3b`, `nomic-embed-text` (both fit on the GPU together: 2.6 GB + 335 MB).
- Commands (run from project root, venv active): `python build_data.py` | `python cli.py check` | `python cli.py ask "..." --role employee` | `python cli.py eval --out results/x.json [--limit N]` |
  `python cli.py sweep [--offline] [--retrievers bm25,dense,hybrid]` | `python -m unittest` (expect 18 tests, all OK).
- Config: `configs/ollama.yaml` (recursive-300, hybrid, k=4, no reranker, llama3.2:3b, temperature 0, seed 42, `127.0.0.1` host). Sweep grid: `configs/sweeps.yaml`.
- Data (synthetic "Acme-Fake Corp"): 30 docs with access tags, 82 golden questions (52 factual, 8 paraphrase, 9 multi-hop, 13 unanswerable) in `data/golden/golden.jsonl`, 23 safety cases in `data/safety/safety.json`.
  q001-q040 ids are stable; q041-q082 added later. Results on the old 40-question set are NOT comparable with the 82-question set.
- Architecture: `rag/` = chunking, embeddings (Ollama + disk cache for chunk vectors; query vectors never cached), retrieval (BM25, dense, hybrid RRF, ACL applied as a pre-filter before ranking, lexical + Ollama rerankers),
  llm (OllamaLLM; MockLLM incl. naive vulnerable mode), pipeline, metrics, telemetry (stage latency, tokens, estimated cost at a reference price). `eval/run_sweeps.py` = retrieval sweeps. `cli.py` = entry point.
- REAL numbers from the user's machine, old 40-question set, llama3.2:3b + nomic-embed-text, hybrid k=4: hit@4 1.0, MRR 0.986, doc_recall 1.0, correctness 0.914 (32/35), groundedness proxy 0.655 (crude),
  abstention 1.0 (n=5), p50 latency 385 ms (embed_query 40 ms, generate 347 ms), 15.8 s wall for 40 questions. Quality identical across repeated runs (temperature 0, seed 42).
- Wrong answers then: q020 (kitchen time; the doc also holds a planted injection) -> "I don't know"; q033 and q035 (multi-hop/comparison) -> "I don't know" / evasive. All three are generation errors (retrieval had hit).
- S1 RESULT (details: `docs/results/S1_baseline.md`, `S1_sweeps.md`, `S1_sweeps_full.md`): real baseline on 82 questions: hit@4 1.0, MRR 0.949, correctness 0.855 (59/69; multi-hop 6/9), abstention 13/13, p50 399 ms, est. cost $0.005. Sweep of 60 configs: NO difference vs the best BM25 row is significant (no `*`; best sign-test p = 0.22); dense numerically ahead (MRR up to 0.993) but that is selecting among 60 noisy rows. nomic prefixes: no proven gain (13 up, 10 same, 1 down, all within noise), left OFF. Lexical reranker: no evidence. Chunking not stressed (docs <= 446 chars; recursive-300 == recursive-500).
- DEFAULT retrieval config UNCHANGED (recursive-300, hybrid, k=4, no reranker, no prefixes); reasons recorded as comments in `configs/ollama.yaml`. Minimum detectable difference at n=69: about 6 net questions (won 6/lost 0, or 8/1).
- Sandbox-only (deterministic, valid): BM25-only sweep on the 82-question set (n=69 answerable): MRR 0.935-0.959 across 6 chunkers, differences ~1 question (not significant); chunk integrity = 1.0 for all chunkers
  (documents too short to stress chunking); paraphrase questions are BM25's weak spot (hit@3 0.75-0.875, n=8). Hypothesis (untested): dense/hybrid improves paraphrase.

## Pending user actions
1. Copy the S1 closing files into the project (HANDOFF.md, configs/ollama.yaml, docs/results/S1_sweeps.md), `Copy-Item results\sweeps.md docs\results\S1_sweeps_full.md`, `python -m unittest` (18 OK), commit, `git tag s1`.
2. Optional: pushing to GitHub only after `git diff --stat origin/main 3033c56` was empty (history was rewritten locally); never `git pull`.

## Completed log (newest last)
### 2026-11-15 - S0 Foundation (tag s0-foundation, plus the docs commit that follows it)
- Built synthetic corpus, golden/safety sets, RAG pipeline, Ollama-only runtime (client with retries and clear errors, cached chunk embeddings, optional LLM reranker), CLI (check/ask/eval/sweep), retrieval sweep harness, 11 tests.
- Verified: unit tests against a stand-in Ollama server in the sandbox; eval run on the user's machine with real Ollama (numbers above).
- Fixes found by real runs: (1) `localhost` added ~2 s per request on the user's Windows machine; default is now 127.0.0.1 (embed_query 2084 -> 22 ms, generate 2696 -> 215 ms in a single warm `ask`);
  (2) query embeddings were cached, which hid latency; now never cached, with timing split into embed_query / rank / generate; (3) UTF-8 file handling for Windows; (4) eval does a warm-up call first.

### 2026-11-16 - S1 (partial, conversation 1: the user's run outputs were NOT in the message, only the commands)
- Task 1 NOT confirmable: no unittest line / `by_type` eval block / `results/sweeps.md` was pasted. Interpretation (tasks 2, 6) is blocked until they are.
- Task 5 DONE: `chunk_fixed` rewritten as a plain sliding window (`rag/chunking.py`). Old logic was correct for size >= 2*overlap (so the 200/40 and 400/80 sweep chunks are byte-identical on all 30 docs; earlier sweep numbers stay valid) but emitted a redundant tail chunk when size < 2*overlap and silently used step 1 when overlap >= size. Now raises ValueError for overlap outside [0, size). Verified against a reference over 6000 random cases; new test fails on the old code.
- Task 3 DONE: `eval/run_sweeps.py` now stores per-question reciprocal rank (`rr`) in each row and adds `add_stats`: 95% bootstrap CI of MRR (`mrr_lo/hi`), and versus the best BM25 row: `won`/`lost` question counts, `d_mrr` with a PAIRED bootstrap CI (`d_lo/d_hi`) and `sig` (CI excludes 0). Stdlib `random`, seed 0, 2000 resamples. `results/sweeps.md` has new columns `mrr_ci`, `won/lost`, `d_mrr [95% CI]` (`*` = real). Treat only `*` rows as real differences.
- Task 4 CODE DONE (decision pending real numbers): `OllamaEmbedder` takes `query_prefix` / `doc_prefix` (config keys on any embedder; applied before the cache lookup so variants never share cached vectors; name becomes `<model>+prefix`). `configs/sweeps.yaml` now lists nomic-embed-text twice (plain first = reference, then with `search_query: ` / `search_document: `). `results/sweeps.md` gets a section "Embedder comparison" (paired won/lost + CI of the prefixed row vs the plain row per chunking/retriever/reranker). Keep the prefixes only if `*` rows show a gain; otherwise leave `configs/ollama.yaml` unchanged. Also: BM25 rows are now emitted once, not once per embedder.
- Task 7 (second embedder) needs no code: uncomment `mxbai-embed-large` / `all-minilm` in `configs/sweeps.yaml` after `ollama pull`; it is compared with the first embedder automatically.
- Tests: 11 -> 18 (bootstrap, paired CI, won/lost + significance, chunk_fixed properties, prefix + cache separation, embedder comparison, BM25 dedupe). All OK in the sandbox. Sandbox sweeps with the hash embedder / stand-in server are plumbing only, never model results.

### 2026-11-16 - S1 (conversation 1, continued): user's runs arrived
- Task 1 CONFIRMED on the user's machine: `Ran 18 tests ... OK`, eval summary has `by_type`. (First attempt failed only because the venv was not active and Ollama was not running; not code problems.)
- REAL baseline, 82-question set, hybrid k=4 recursive-300, llama3.2:3b + nomic-embed-text (details + intervals in `docs/results/S1_baseline.md`): hit@4 1.0, MRR 0.949, doc_recall 1.0, correctness 0.855 (59/69; factual 45/52, multi-hop 6/9, paraphrase 8/8), abstention 13/13, groundedness proxy 0.68, p50 399 ms (embed_query 26, generate 377), p95 828 ms, est. cost $0.005, 49.5 s. Retrieval is at ceiling; the 10 wrong answers are generation/metric side. Not comparable with the old 40-question numbers.
- Sweep ran (60 configs: 36 plain-nomic + 24 prefixed). Console-only first look (NO intervals yet, so NOT conclusions): dense MRR 0.959-0.986 vs BM25 0.935-0.959 in every chunker; hybrid in between (0.942-0.978); lexical reranker +0.01 on dense for most chunkers but lower on `sentence`; prefixed nomic changes MRR by 0.000 to +0.015 (looks like noise). Needs `sweeps.md` for won/lost and paired CIs.

### 2026-11-16 - S1 closed (same conversation)
- Interpreted the user's `results/sweeps.md` (60 configs): see S1 RESULT in Current state. Wrote `docs/results/S1_sweeps.md` (condensed table, answers, decision, limits); the user copies the full table to `docs/results/S1_sweeps_full.md`. Tasks 2, 4, 6 done; task 7 (second embedder) skipped: no extra model pulled, switch exists (uncomment in `configs/sweeps.yaml`).
- Verified: 18 tests OK on the user's machine (sandbox too); sweep/eval run on the user's machine; sign tests computed by hand from won/lost counts (python, exact binomial).
- Found: the paired bootstrap interval degenerates (lower bound exactly 0, no `*`) when only 1-3 questions differ; S3 should add an exact sign test / minimum-detectable-change.

## Next
1. Start conversation S2 (`docs/sections/S2_safety.md`): safety suite, ACL violations = 0 as a hard gate, proof the tests detect failures.
2. Then S3 regression gate (use paired won/lost + sign test; state the minimum detectable change), S4 judge, S5 generation (where the 10 wrong answers and multi-hop 6/9 get analysed), S6, S8.

## Open issues / known gaps
- Correctness = keyword match (strict; q035 miss is partly the metric). Groundedness = token-overlap proxy (counts citation text as ungrounded). Both are crude until S4/S5.
- The synthetic corpus is small and easy; chunking cannot be stressed on it and retrieval is near ceiling (S1 confirmed). State this in the final report; S7 may change the corpus.
- Paraphrase advantage of dense/hybrid is unproven (n=8). Paired bootstrap CI is degenerate for 1-3 discordant questions: rely on won/lost + sign test.
- OllamaReranker costs one LLM call per candidate chunk (slow); off by default.
- Stand-in server and MockLLM prove plumbing only.
- Git: local main (3033c56, 359f102) and origin/main (fadbacf, 3e1eeb4) are the SAME two commits with different hashes (history rewritten locally, probably author reset). Remote: github.com/udaykumarvadde200/Advanced-Rag. Do not `git pull`; reconcile with a checked force-push (see conversation) or ignore the remote.
- Windows: `core.autocrlf=true`, so LF/CRLF warnings on `git add` are harmless. Venv must be active and `ollama serve` running before eval/sweep.

## Key decisions (do not reverse without asking the user)
Ollama only; no DeepTeam; judge model != generator and run sequentially; deterministic checks before LLM judging; ACL as a pre-filter; temperature 0 / seed 42;
query embeddings uncached when measuring latency; synthetic data for now (FinanceBench / lecture-transcript chatbot discussed, decided later in S7).

## Git
Repo is initialised (branch `main`). Convention: one commit per finished task, a tag per finished section (`s0-foundation`, `s1`, ...). `results/` is git-ignored (regenerate); frozen regression baselines go in `baselines/`; final tables go in `docs/results/`.
Commits so far (local hashes): 359f102 S0 foundation, 3033c56 docs, 49a7b40 S1 partial, next: S1 closed (tag s1).
