# One-command reproduction (Linux/macOS/WSL). Windows: use run_all.ps1.
# make test      offline unit tests only (no Ollama)
# make offline   stand-in plumbing run (NOT model results)
# make all       real run: needs `ollama serve`, llama3.2:3b and nomic-embed-text pulled
PY ?= python

.PHONY: test offline all data
data:
	$(PY) build_data.py
test:
	$(PY) -m unittest
offline: data test
	$(PY) cli.py sweep --offline --retrievers bm25
	$(PY) cli.py safety --offline
	$(PY) cli.py eval --config configs/baseline.yaml --out results/offline_eval.json
all: data test
	$(PY) cli.py check
	$(PY) cli.py eval --out results/baseline.json
	$(PY) cli.py sweep
	$(PY) cli.py safety
	$(PY) cli.py regress --compare baselines/v1.json
