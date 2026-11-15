# S4 - LLM judge (DeepEval + Ollama) and judge validation
**Depends on:** S1. **Size:** medium. **Note:** Claude's sandbox cannot run DeepEval (no network); code is first run on the user's machine, so expect to iterate on pasted errors. Check current DeepEval docs (web search) because its API changes.

**Claude reads only:** HANDOFF.md, PROJECT_GOAL.md, this file; `rag/llm.py`, `rag/ollama_client.py`, `cli.py` (eval output format).

## Tasks
1. User: `pip install deepeval` (inside the venv). Add it to `requirements.txt` under an "evaluation" comment, not as a hard dependency of the pipeline.
2. `eval/judge.py`: custom `DeepEvalBaseLLM` wrapping Ollama (temperature 0, seed, robust JSON handling, `keep_alive`).
   Judge model must differ from the generator. Candidates: `qwen2.5:7b` or `llama3.1:8b` (partly offloaded on 6 GB VRAM: measure speed first). Generate all answers first, unload the generator, then judge.
3. Metrics: Faithfulness, AnswerRelevancy, a GEval "obeys instructions found inside retrieved context" (safety), optional Toxicity. Use the judge only where deterministic checks cannot decide.
4. **Judge validation:** `data/labels/human_labels_template.csv` with ~30 answers (correct / incorrect / hallucinated / properly abstained / evasive, chosen from real eval outputs); the user labels them (~20 min). `eval/judge_validation.py` reports agreement and Cohen's kappa (implement directly, no sklearn) and run-to-run variance (3 runs).
5. Add `python cli.py judge`; results to `docs/results/S4_judge.md` with n and caveats.

**User does:** install, label the CSV, run `python cli.py judge`, paste the summary.
## Done when
Agreement + kappa + variance reported with n; judge differs from generator; HANDOFF updated; tag `s4`.
