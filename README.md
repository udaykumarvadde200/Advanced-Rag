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
python cli.py safety                                  # safety suite + detection proof -> results/safety.json, results/safety.md (exit 1 = ACL leak, 2 = proof failed)
python cli.py safety --offline                        # same plumbing with stand-ins (NOT model results)
python cli.py regress --save-baseline baselines/v1.json    # freeze golden + safety results (refuses a baseline that leaks or breaks a floor)
python cli.py regress --compare baselines/v1.json          # exit 0 PASS / 1 FAIL (gate named) / 2 invalid comparison; report in results/regress.md
python cli.py regress --compare baselines/v1.json --set retriever.enforce_acl=false   # deliberate break: must FAIL on safety
python cli.py regress --noise-study 3                      # run-to-run variance (justifies configs/tolerances.yaml)
python cli.py regress --offline --compare <offline baseline>   # stand-ins, plumbing only
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
eval/run_safety.py  safety suite (ACL, injection, PII, jailbreak, abstention)
eval/regression.py  regression gate (baseline, tolerances, flipped questions, exit code); rules in configs/tolerances.yaml
cli.py           check | ask | eval | sweep | safety | regress | judge | gen
configs/         baseline.yaml (offline stand-ins), ollama.yaml (real run)
rag/             chunking, embeddings (Ollama + disk cache), retrieval (BM25, dense, hybrid RRF, ACL pre-filter, rerankers), llm, pipeline, metrics, telemetry
tests/           stand-in Ollama server + tests
PROJECT_GOAL.md  what we are building and when it is done
HANDOFF.md       living status: what is done, what is next (update after every task)
docs/sections/   one brief per work section (S1..S8); docs/results/ holds final result tables
```

## LLM judge (S4)
`pip install deepeval` (evaluation only), `ollama pull qwen2.5:7b`, then: `python cli.py judge --probe` (speed / VRAM / structured-output check), `python cli.py judge --make-labels` (30 real answers to label, guide in `data/labels/LABELING_GUIDE.md`),
`python cli.py judge --validate` (agreement, Cohen's kappa, 3-run variance), `python cli.py judge` (judge scores for the golden set). Settings: `configs/judge.yaml`. Details and results: `docs/results/S4_judge.md`.

## Generation study (S5)
`python cli.py gen` runs every golden question twice with the same generator: GOLD context (the evidence chunks) and RETRIEVED context; the gap is retrieval's cost. Each wrong answer is attributed to one cause
(chunking / retrieval miss / ranking / abstention error / generation error / metric artifact; rules in `eval/run_generation.py::classify`). Next to the strict keyword check it reports normalised correctness
(number words, units, currency, times, comparison synonyms) and citation-stripped groundedness. Report: `results/gen_<label>.md`; write-up: `docs/results/S5_generation.md`.
```
python cli.py gen                                   # reference config: llama3.2:3b, default prompt, k=4
python cli.py gen --ids q020,q033,q035              # print answers + attribution for the known failures
python cli.py gen --model qwen2.5:3b                # another model (ollama pull qwen2.5:3b first)
python cli.py gen --prompt fewshot --k 6 --order doc   # prompt variants: default | inference_ok | fewshot; order: rank | reverse | doc
python cli.py gen --grid [--skip-missing]           # one factor at a time (configs/generation.yaml) + paired comparison and adoption rules
python cli.py gen --merge results/gen_a.json results/gen_b.json   # compare saved runs (first = reference)
python cli.py gen --offline                         # hash embedder + MockLLM: plumbing only, never model quality
```


## Advanced features (S6)
Each feature is a config switch (default off) and is ablated by `python cli.py s6 --part ablate` (one regression snapshot per variant, fixed verdict rules in `configs/ablations.yaml`, report `results/s6_ablations.md`).
- `defense.sanitize` / `defense.spotlight`: injection defences (`rag/defense.py`); `python cli.py s6 --part sanitizer` checks the sanitiser on a held-out set (no Ollama).
- `routing.enabled` (+ `variants`, `department`): metadata pre-filter after the ACL filter (`configs/routing.yaml`); `python cli.py s6 --part routing [--offline]`.
- `calculator.enabled`: LLM extracts an expression, Python evaluates it safely (`rag/calculator.py`); `python cli.py s6 --part calculator`.
- `reranker.type: lexical | ollama`. Results and verdicts: `docs/results/S6_ablations.md`.
