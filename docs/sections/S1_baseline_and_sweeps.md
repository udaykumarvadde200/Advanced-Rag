# S1 - Real baseline + retrieval sweeps
**Depends on:** S0 and the user's runs. **Size:** small conversation.
**Goal:** first trustworthy real numbers on the 82-question set, and a justified default retrieval config.

**User pastes into chat (summary blocks only):** (a) `python -m unittest` last line, (b) the `python cli.py eval --out results/baseline_v2.json` summary block (must have a `by_type` line), (c) full `results/sweeps.md`.
**Claude reads only:** HANDOFF.md, PROJECT_GOAL.md, this file; `eval/run_sweeps.py` and `configs/ollama.yaml` only if editing them.

## Tasks
1. Check the user ran the new code (11 tests, `by_type` line present). If not, stop and fix that first.
2. Interpret the sweeps: which differences exceed noise (about 69 answerable questions; one question is ~0.014)? Does dense/hybrid beat BM25 on paraphrase questions? Does any chunker matter? (Known: chunk integrity is 1.0 everywhere, so chunking is not stressed by this corpus.)
3. Add statistical honesty to `eval/run_sweeps.py`: per row, count questions won/lost vs the best BM25 row and a bootstrap 95% interval for MRR (stdlib `random`, fixed seed). Add tests.
4. Test nomic-embed-text task prefixes (`search_query: ` / `search_document: `) as a config switch; keep only if it helps beyond noise.
5. Recheck `chunk_fixed` end-of-text logic (see HANDOFF open issues).
6. Choose and write the default config in `configs/ollama.yaml`; record why.
7. Optional: second embedder row if the user pulled one (`mxbai-embed-large`, `all-minilm`).

## Done when
`docs/results/S1_sweeps.md` (final table + the decision) and `docs/results/S1_baseline.md` (eval summary) exist; HANDOFF updated; tag `s1`.
**Out of scope:** generation changes, safety, judge.
