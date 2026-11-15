# PROGRESS / HANDOFF (read this first)

> Handoff protocol: the user is on a 7-day deadline using a rate-limited Claude plan. Each Claude session
> builds as much as possible, keeps THIS file current, and the user pastes it into the next session.
> The next Claude must continue from "NEXT", update this file as it goes, and regenerate it near its limit.
> The user asked for Claude to do the ENTIRE project, not just advise.

## Goal
Evaluate every component, the workflow, and the application of an advanced RAG system, including safety,
cost/latency tracking and regression testing. **Ollama is the only runtime** (generation, embeddings, optional reranker, judge).
DeepEval (Ollama judge) for LLM-judged metrics later; LangSmith optional (traces go to its cloud). **No DeepTeam.**

## Decisions so far (conversation summary)
- User hardware: 16 GB RAM, RTX 4050 laptop (6 GB VRAM). Use a 3-4B generator for sweeps; 7-8B only for a few final runs; run generator and judge sequentially (`keep_alive: 0` to unload).
- Recruiter-facing target project discussed: **FinanceBench (150-question open sample, Hugging Face PatronusAI/financebench + GitHub patronus-ai/financebench PDFs) at "Option C" scope**:
  hybrid retrieval, reranker, metadata routing (company/year), calculator tool; safety suite; regression gate with a deliberately bad change caught.
  Subset ~40-50 questions over 8-10 filings. License is disputed between sources (HF card says CC BY-NC 4.0): user must check. Fallback: lecture-transcript doubt-solving chatbot.
- **The user said "don't start" the FinanceBench build, and asked FIRST to convert the existing synthetic project into an Ollama-runnable one. That is DONE (below). Do NOT begin FinanceBench/advanced features until the user says so.**
- User's idea of a CampusX-style project: a RAG chatbot where the user adds lecture transcripts and asks doubts about a playlist (not built).
- Workflow the user asked for: whenever a Claude session nears its limit, it updates this file; the user pastes it into the next session.

## Sandbox constraints (Claude's sandbox, not the user's machine)
No network, no Ollama, no model downloads. numpy/pandas/pyyaml/requests installed; no pytest. Everything Ollama-related is tested only against `tests/fake_ollama.py`
(a stand-in HTTP server, NOT a model), so no real quality numbers exist yet. Shell: `mkdir -p a/{b,c}` does not brace-expand here.

## Layout (rag-eval/)
- `build_data.py` -> data/corpus (16 synthetic docs w/ access tags), data/golden/golden.jsonl (40 Qs), data/safety/safety.json (23 cases)
- `cli.py` `check | ask | eval` (eval writes results/*.json with hit@k, MRR, doc_recall, correctness, groundedness, abstention accuracy, latency p50/p95, tokens, est. cost)
- `configs/ollama.yaml` (real run: llama3.2:3b + nomic-embed-text), `configs/baseline.yaml` (offline stand-ins: hash embedder + MockLLM)
- `rag/` ollama_client.py (HTTP client, retries, clear errors), embeddings.py (OllamaEmbedder, batched + disk cache; HashEmbedder for tests), retrieval.py (BM25, dense, hybrid RRF, ACL pre-filter, LexicalReranker, OllamaReranker), llm.py (OllamaLLM, MockLLM incl. naive=True vulnerable mode), chunking.py, pipeline.py, metrics.py, telemetry.py
- `tests/` fake_ollama.py + test_ollama_path.py (7 tests, all passing: `python -m unittest -v`)
- `requirements.txt` (numpy, pyyaml, requests), `README.md`

## REAL RESULTS (user's machine: Windows, 16 GB RAM, RTX 4050 6 GB; llama3.2:3b + nomic-embed-text via Ollama; configs/ollama.yaml)
Full golden set, first real run (results/baseline.json is on the USER's machine, not in the sandbox):
- 35 answerable + 5 unanswerable. hit@4 = 1.0, MRR = 0.986, doc_recall = 1.0, correctness = 0.914 (32/35; 3 failures NOT yet inspected),
  groundedness proxy = 0.657 (crude; known to undercount, e.g. citation text like "hr-01"), abstention accuracy = 1.0 (only 5 cases, so weak evidence).
- Latency p50 4.70 s, p95 5.06 s, mean 4.2 s; wall 168 s for 40 questions. Tokens in/out mean 274 / 23. `ollama ps`: both models 100% GPU simultaneously (2.6 GB + 335 MB).
- FINDING: q001-q010 took ~2.5 s but q011+ ~4.7 s. The first 10 query embeddings were cached from the earlier `--limit 10` run, so an uncached Ollama embedding call costs ~2 s on this machine
  (cause unexplained). Query embeddings are now NEVER cached (patched), so latency numbers are honest; timing is split into embed_query / rank / generate.
- RESOLVED (measured on user's machine): the ~2 s per-call overhead was `localhost`. Using 127.0.0.1: embed_query 2084 -> 22 ms, generate 2696 -> 215 ms per question
  (single `ask`, warm). Likely Windows IPv6-first lookup, mechanism unproven. Default host is now 127.0.0.1 and any "localhost" is rewritten to it. Eval now does a warm-up call first.
  Quality metrics were byte-identical across two runs (temperature 0, seed 42), so generation looked deterministic on this machine (n=2 runs).
- The 3 wrong answers (all GENERATION errors, retrieval had hit and doc_recall = 1.0): q020 (kitchen cleaning time, poisoned doc) -> "I don't know" (cause unverified: maybe the injection chunk
  in context confused the 3B model); q033 and q035 are multi-hop/comparison questions (q033-q035 are the 3 multi-hop items in build_data.py) -> "I don't know" and an evasive answer lacking "higher".
  So factual ~31/32 vs multi-hop 1/3 (tiny samples). q035's miss is partly the crude keyword metric.
- Caveat: the golden set is easy (33 chunks, single-sentence facts), so retrieval scores are at ceiling. Do not over-claim.

## Patch log (latest)
- UTF-8 encoding everywhere + closed files (Windows cp1252 default would break non-ASCII/Hindi/Telugu). 8 tests pass (added query-not-cached/stage-split test).

## Data v2 + retrieval sweeps (built this session; sweeps NOT yet run with real Ollama embeddings)
- build_data.py now also imports build_data_extra.py: 30 docs (14 added: near-duplicate Berlin/contractor/travel docs, a stale ARCHIVED leave policy, new topics) and 82 golden questions
  (52 factual, 8 paraphrase, 9 multi-hop, 13 unanswerable incl. near-miss). q001-q040 ids unchanged; q041-q082 are new. Golden items now have `evidences` (list) for multi-hop. Baseline results from the 40-question set are NOT comparable with the 82-question set.
- `python cli.py sweep [--offline] [--retrievers bm25,dense,hybrid]` -> results/sweeps.json + results/sweeps.md: chunking(6) x retriever(3) x embedder x reranker(2), retrieval only.
  Metrics at k=1/3/5 from one pool of 10 (reranker reorders that pool): hit, MRR, nDCG@5, evidence_recall@5, chunk `intact` rate, p50/p95 retrieval ms (real per-query embedding latency recorded), hit@3 by question type.
- metrics.retrieval_metrics rewritten for multi-evidence; 11 unit tests pass (`python -m unittest`).
- REAL BM25-only sweep (sandbox, deterministic, n=69 answerable): MRR 0.935-0.959, hit@3 0.971-1.0 across chunkers; differences are ~1 question, so NOT significant.
  chunk `intact` = 1.0 for every chunker (docs are tiny, so chunking never splits evidence: this corpus cannot stress chunking; FinanceBench/transcripts would).
  Paraphrase questions are BM25's weak spot: hit@3 0.75-0.875 (n=8) vs factual 1.0. HYPOTHESIS to test on the user's machine: dense/hybrid (nomic-embed-text) improves paraphrase. Lexical reranker did not help.
  Stale-document effect observed with the stand-in LLM: "How many days of annual leave..." now surfaces the archived 18-day policy.

## DONE
- [x] Synthetic data builders; pipeline modules; telemetry
- [x] Ollama conversion: client, cached embeddings, LLM, optional LLM reranker, CLI, config, README; 7 tests pass; CLI check/ask/eval verified against the stand-in server
- [x] Removed sentence-transformers/Hugging Face runtime dependency from defaults (HF cross-encoder reranker deleted; Ollama-only)

## NEXT
1. USER RUNS (Windows PowerShell, venv active, after copying the new zip over the project; `python -m unittest` should say 11 tests): 
   `python build_data.py` ; `python cli.py eval --out results/baseline_v2.json` (82 questions, ~40 s) ; `python cli.py sweep` (needs nomic-embed-text) ; paste the eval summary block and results/sweeps.md.
2. Claude: interpret real sweep (does hybrid/dense beat BM25 on paraphrase? is any difference > 1-2 questions?), pick a best config, write it into configs/ollama.yaml.
3. Safety suite (ACL zero-tolerance, injection canary, PII, jailbreak) on the real model + vulnerable MockLLM(naive=True) to prove detection; per-type/failure-attribution report for eval.
4. Regression gate (baseline JSON, per-metric tolerances, flipped questions, exit code; demo with enforce_acl=false / tiny chunks), then DeepEval + Ollama judge + judge-vs-human agreement (user labels ~30 answers), README results + findings.
5. Only then decide on FinanceBench / advanced features.

## Known gaps / caveats
- Correctness = keyword match; groundedness = token-overlap proxy (both deterministic, crude). LLM-judged metrics not yet implemented.
- `chunk_fixed` has an odd end-of-text break condition: re-check when chunking is evaluated.
- OllamaReranker makes one LLM call per chunk (slow); default config keeps reranker off.
- nomic-embed-text works better with task prefixes (search_query:/search_document:); not added yet.
