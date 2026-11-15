# PROJECT GOAL (stable: change only if the user changes the plan)

## One line
A local, Ollama-only RAG chatbot over a company knowledge base with role-based access, built to be **evaluated**: every component, the whole
workflow and the application, plus safety, cost/latency and regression testing. The deliverable is the **evidence**, not the chatbot.

## Why this framing
- Course framing (CampusX LLM Evals): evaluate the *application*, not the model; replace "vibe testing" with tests that are systematic, repeatable and based on clear criteria.
  A golden dataset, a success criterion, an evaluation method, run -> analyse -> improve -> re-run, and production failures fed back into the golden set.
- A common interview question is "how would you evaluate a RAG application?". The project must answer it with measured numbers.
- Recruiter-facing: a headline result with numbers, trustworthy ground truth, proof of rigour (judge-vs-human agreement, failure attribution, a regression gate that catches a deliberately bad change), honest limits, one-command reproduction.

## Scope
IN: components (chunking, embeddings, retriever, reranker, generator, judge), workflow (end-to-end, failure attribution, ablations), application (latency per stage, tokens, estimated cost),
safety (access control, prompt injection, PII, jailbreak, abstention), regression gate.
OUT (unless section S7 approves): multimodal/ColPali, GraphRAG, full agent frameworks, fine-tuning, DeepTeam (user excluded it), paid APIs.

## Hard constraints
- Free and local. **Ollama is the only runtime** (generation, embeddings, reranking, judging). Python deps: numpy, pyyaml, requests (+ deepeval from S4 on).
- User hardware: Windows laptop, 16 GB RAM, RTX 4050 with 6 GB VRAM -> 3-4B models by default; a 7-8B model only partially on GPU (slower). Judge model must differ from the generator; run them sequentially (`keep_alive`).
- Claude's sandbox has no network and no Ollama. Tests there use a stand-in server (tests/fake_ollama.py) that proves plumbing only. **Real quality numbers come only from the user's machine.** Never present stand-in output as a model result.
- LangSmith is optional (traces go to its cloud; synthetic data only). Prefer local JSON logs.

## Definition of done (all must be true, with numbers in docs/results/)
1. Component tables: chunking, embedder, retriever (BM25/dense/hybrid), reranker, generator (gold vs retrieved context), judge.
2. Workflow: end-to-end metrics, per-question-type breakdown, failure attribution, ablations.
3. Application: p50/p95 latency per stage, tokens, estimated cost, hardware recorded.
4. Safety: access-control violations = 0 (hard gate), injection / PII / jailbreak pass-rates, and proof the tests detect failures (vulnerable stand-in model, ACL switched off).
5. Judge validated against ~30 human labels (agreement, Cohen's kappa, run-to-run variance).
6. Regression gate: frozen baseline, per-metric tolerances, flipped-question report, exit code; a deliberate bad change fails it and a no-op passes.
7. README with headline results, limitations and one-command reproduction; repo clean; tests pass.

## Working principles
Deterministic checks first, LLM judge only where needed. Report sample sizes and say when a difference is within noise (one question is ~1.4 points on 70 questions).
Change one thing at a time. Keep every experiment config-driven. Record decisions and numbers in HANDOFF.md. Be honest about synthetic data and small n.

## Section map (one conversation per section; each brief is in docs/sections/)
| # | Section | Needs |
|---|---|---|
| S0 | Foundation: data, pipeline, Ollama, CLI, tests | DONE |
| S1 | Real baseline + retrieval sweeps, choose default config | user runs |
| S2 | Safety suite | S0 |
| S3 | Regression gate | S1 (+S2) |
| S4 | LLM judge (DeepEval + Ollama) and judge validation | S1 |
| S5 | Generation evaluation and failure attribution | S1 (+S4) |
| S6 | Advanced features (reranker, routing, calculator) with ablations | S1, S5 |
| S7 | Dataset decision (stay / FinanceBench / lecture transcripts), optional | S5 |
| S8 | Final report, diagram, reproduce script, polish | all |
Suggested order: S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> (S7) -> S8. S2, S3 and S4 are independent enough to swap if time is short.
