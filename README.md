# RAG Evaluation Project (Ollama-only)

A small RAG chatbot over a synthetic company knowledge base with role-based access control, built to be **evaluated**
component by component. Everything runs locally through **Ollama** (generation, embeddings, optional reranker). No cloud, no API keys.

## Setup
```bash
# 1. Install Ollama (https://ollama.com), then pull models (sized for ~6 GB VRAM):
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# 2. Python deps
pip install -r requirements.txt
# 3. Build the synthetic data (corpus, golden set, safety set)
python build_data.py
```

## Run
```bash
python cli.py check                                   # server up? models pulled?
python cli.py ask "How many sick leave days do we get?" --role employee
python cli.py eval --limit 10                         # quick run on the golden set
python cli.py eval --out results/baseline.json        # full golden set (82 questions)
python cli.py sweep                                   # retrieval sweeps -> results/sweeps.md (needs nomic-embed-text)
python cli.py sweep --offline --retrievers bm25       # no Ollama needed
```
Switch models by editing `configs/ollama.yaml` (e.g. `llama3.2:3b` -> `qwen2.5:7b`). A 7-8B model may be partly offloaded to CPU on 6 GB VRAM: slower but workable.
`keep_alive: 0` unloads a model right after use, useful when later swapping generator and judge models.

## What `eval` reports
hit@k, MRR, document recall, answer correctness (keyword match), groundedness (deterministic proxy), abstention accuracy on unanswerable questions,
latency p50/p95 per stage (retrieve / rerank / generate), tokens in/out, and an *estimated* cost at a reference price (local models are free; the figure is for comparing configs only).

## Tests (no Ollama needed)
```bash
python -m unittest -v
```
These run the real Ollama client code against a stand-in HTTP server (`tests/fake_ollama.py`) that mimics `/api/tags`, `/api/chat` and `/api/embed`.
**The stand-in is not a model.** Passing tests prove the client, caching, error handling and pipeline wiring, not answer quality. Real quality numbers come only from running `cli.py eval` against real Ollama.

## Layout
```
build_data.py    synthetic corpus (30 docs), golden questions (82), safety cases; build_data_extra.py holds the harder docs/questions
eval/run_sweeps.py  chunking x retriever x embedder x reranker sweeps
cli.py           check | ask | eval
configs/         baseline.yaml (offline stand-ins), ollama.yaml (real run)
rag/             chunking, embeddings (Ollama + disk cache), retrieval (BM25, dense, hybrid RRF, ACL pre-filter, rerankers), llm, pipeline, metrics, telemetry
tests/           stand-in Ollama server + tests
PROGRESS.md      handoff notes: done / next / pending
```
