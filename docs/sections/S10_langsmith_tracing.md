# S10 - Optional LangSmith tracing (off by default)
**Depends on:** S8 (S9 first is better, so the chat app is traced too). **Size:** small.
Claude reads: HANDOFF.md, PROJECT_GOAL.md, this file, `rag/pipeline.py`, `rag/llm.py`, `rag/telemetry.py`, `rag/ollama_client.py`, `configs/ollama.yaml`.

## Why / the privacy and cost facts
LangSmith is a **cloud** service: questions, retrieved text and answers are uploaded to LangSmith's servers. Allowed here only because the data is synthetic (PROJECT_GOAL: "synthetic data only if cloud"). It needs an account and an API key; the free tier has a monthly trace limit and going over it can cost money (check LangSmith's current pricing page before enabling; do not assume the limit). Everything else in the project is free and local, so tracing must never be on by default.

## Tasks
1. A thin wrapper (`rag/tracing.py`) that records one trace per question with child spans: retrieve (ACL-filtered chunk ids and scores, not full text if avoidable), optional rerank, generate (model, tokens, latency), and the answer. Config `tracing: {enabled: false, project: rag-eval}`; the API key comes ONLY from an environment variable, never from a config file or the repo. The `langsmith` package is imported only when tracing is enabled (lazy import), and goes in an optional requirements file, not in the core requirements.
2. Off by default and safe: with tracing disabled there is zero network access and the requests sent to Ollama are identical (a test); a tracing failure (no key, no network, quota) must never break an answer or a regression run (log a warning and continue); the regression gate (`regress`) and `run_all.ps1` run with tracing off.
3. Redaction: do not send the API key or any non-synthetic data; a clear warning in the CLI/app when tracing is on ("traces are uploaded to LangSmith").
4. Tests offline with a fake tracer (no network): disabled = no calls, enabled = expected spans, tracer exception does not change the answer.
5. Docs: README section (how to enable, what is uploaded, cost warning), `docs/results/S10_tracing.md` (what was verified: only offline tests unless the user runs a real trace and pastes the project link or a screenshot), HANDOFF update, tag `s10`.
6. Free local alternative if the user prefers no cloud: Langfuse self-hosted (needs Docker; heavier setup). Mention, do not build both unless asked.

## Done when
With a key set and `tracing.enabled: true`, a question produces a trace with the spans above in the user's LangSmith project (verified by the user); with tracing off nothing changes and `regress --compare baselines/v1.json` still passes; tests green offline.
