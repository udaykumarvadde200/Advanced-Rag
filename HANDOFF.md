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
| S1 | Real baseline + retrieval sweeps | CODE READY, waiting for user runs (see Pending user actions) |
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
  `python cli.py sweep [--offline] [--retrievers bm25,dense,hybrid]` | `python -m unittest` (expect 11 tests, all OK).
- Config: `configs/ollama.yaml` (recursive-300, hybrid, k=4, no reranker, llama3.2:3b, temperature 0, seed 42, `127.0.0.1` host). Sweep grid: `configs/sweeps.yaml`.
- Data (synthetic "Acme-Fake Corp"): 30 docs with access tags, 82 golden questions (52 factual, 8 paraphrase, 9 multi-hop, 13 unanswerable) in `data/golden/golden.jsonl`, 23 safety cases in `data/safety/safety.json`.
  q001-q040 ids are stable; q041-q082 added later. Results on the old 40-question set are NOT comparable with the 82-question set.
- Architecture: `rag/` = chunking, embeddings (Ollama + disk cache for chunk vectors; query vectors never cached), retrieval (BM25, dense, hybrid RRF, ACL applied as a pre-filter before ranking, lexical + Ollama rerankers),
  llm (OllamaLLM; MockLLM incl. naive vulnerable mode), pipeline, metrics, telemetry (stage latency, tokens, estimated cost at a reference price). `eval/run_sweeps.py` = retrieval sweeps. `cli.py` = entry point.
- REAL numbers from the user's machine, old 40-question set, llama3.2:3b + nomic-embed-text, hybrid k=4: hit@4 1.0, MRR 0.986, doc_recall 1.0, correctness 0.914 (32/35), groundedness proxy 0.655 (crude),
  abstention 1.0 (n=5), p50 latency 385 ms (embed_query 40 ms, generate 347 ms), 15.8 s wall for 40 questions. Quality identical across repeated runs (temperature 0, seed 42).
- Wrong answers then: q020 (kitchen time; the doc also holds a planted injection) -> "I don't know"; q033 and q035 (multi-hop/comparison) -> "I don't know" / evasive. All three are generation errors (retrieval had hit).
- Sandbox-only (deterministic, valid): BM25-only sweep on the 82-question set (n=69 answerable): MRR 0.935-0.959 across 6 chunkers, differences ~1 question (not significant); chunk integrity = 1.0 for all chunkers
  (documents too short to stress chunking); paraphrase questions are BM25's weak spot (hit@3 0.75-0.875, n=8). Hypothesis (untested): dense/hybrid improves paraphrase.

## Pending user actions (S1 cannot finish without these)
1. Copy the latest zip over the project folder; `python -m unittest` must say `Ran 11 tests ... OK`.
2. `python build_data.py`, then `python cli.py eval --out results/baseline_v2.json` and `python cli.py sweep`.
3. Paste into the S1 conversation: the eval summary block (it must contain a `by_type` line) and the contents of `results/sweeps.md`.

## Completed log (newest last)
### 2026-11-15 - S0 Foundation (tag s0-foundation, plus the docs commit that follows it)
- Built synthetic corpus, golden/safety sets, RAG pipeline, Ollama-only runtime (client with retries and clear errors, cached chunk embeddings, optional LLM reranker), CLI (check/ask/eval/sweep), retrieval sweep harness, 11 tests.
- Verified: unit tests against a stand-in Ollama server in the sandbox; eval run on the user's machine with real Ollama (numbers above).
- Fixes found by real runs: (1) `localhost` added ~2 s per request on the user's Windows machine; default is now 127.0.0.1 (embed_query 2084 -> 22 ms, generate 2696 -> 215 ms in a single warm `ask`);
  (2) query embeddings were cached, which hid latency; now never cached, with timing split into embed_query / rank / generate; (3) UTF-8 file handling for Windows; (4) eval does a warm-up call first.

## Next
1. User does the Pending actions; then start conversation S1 (`docs/sections/S1_baseline_and_sweeps.md`).
2. Then S2 safety, S3 regression, S4 judge, S5 generation (see PROJECT_GOAL.md section map).

## Open issues / known gaps
- Correctness = keyword match (strict; q035 miss is partly the metric). Groundedness = token-overlap proxy (counts citation text as ungrounded). Both are crude until S4/S5.
- The synthetic corpus is small and easy; chunking cannot be stressed on it. State this in the final report.
- `chunk_fixed` has an odd end-of-text break condition: recheck in S1 when chunkers are compared.
- nomic-embed-text works better with task prefixes (`search_query:` / `search_document:`); not added yet (test in S1).
- OllamaReranker costs one LLM call per candidate chunk (slow); off by default.
- Stand-in server and MockLLM prove plumbing only.
- Git author in the first commits is a placeholder ("RAG Eval Project"); see "Git" below to reset it.

## Key decisions (do not reverse without asking the user)
Ollama only; no DeepTeam; judge model != generator and run sequentially; deterministic checks before LLM judging; ACL as a pre-filter; temperature 0 / seed 42;
query embeddings uncached when measuring latency; synthetic data for now (FinanceBench / lecture-transcript chatbot discussed, decided later in S7).

## Git
Repo is initialised (branch `main`). Convention: one commit per finished task, a tag per finished section (`s0-foundation`, `s1`, ...). `results/` is git-ignored (regenerate); frozen regression baselines go in `baselines/`; final tables go in `docs/results/`.
Commits so far: (1) S0 foundation, (2) docs: PROJECT_GOAL, HANDOFF, section briefs.
