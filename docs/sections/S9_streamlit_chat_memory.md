# S9 - Streamlit chat UI + short-term memory
**Depends on:** S0-S8 (done, tag v1.0). **Size:** medium. **Do this before S10 (tracing)** so traces can cover the chat app.
Claude reads: HANDOFF.md, PROJECT_GOAL.md, this file, `rag/pipeline.py`, `rag/llm.py`, `rag/retrieval.py`, `cli.py`, `configs/ollama.yaml`, `configs/tolerances.yaml`, `docs/results/S6_ablations.md` (switch + ablation pattern). Nothing else unless needed.

## Why
Today the "chatbot" is one command per question (`cli.py ask`): no chat window, no follow-ups. The project's point is evidence, so the UI and the memory must be **evaluated like every other feature**, not just added.

## Tasks
1. **Streamlit app** (`app/streamlit_app.py`, local only, `streamlit run app/streamlit_app.py`): chat window, role picker in the sidebar (guest / employee / hr / finance / eng; real authentication is out of scope, say so on the page), answer with citations/sources, per-answer latency and tokens, "I don't know" shown plainly, clear-chat button, memory on/off toggle. Streamlit goes in a separate `requirements-app.txt` (core, tests and CI must not need it; do not import it from `rag/` or `eval/`).
2. **Short-term memory** behind a config switch, default OFF: `memory: {enabled: false, turns: 3}`. Last N turns only (no long-term store). Preferred design: **rewrite the follow-up into a standalone question** with the same generator (e.g. "and for contractors?" -> "How many sick leave days do contractors get?"), then run the normal pipeline on the rewritten question. Keep the rewrite visible in the UI and in the result record.
3. **Hard rules for memory** (each needs a test): (a) the ACL pre-filter applies to the rewritten query on every turn, with the role of the current turn; (b) changing the role clears the history (no cross-role leakage through memory); (c) history is untrusted text, like retrieved chunks (an injection planted in an earlier turn or earlier answer must not be obeyed; sanitise/delimit it); (d) with `memory.enabled: false` the requests sent to Ollama are **identical** to today's (a test like `TestSameRequestsAsEval`), so `baselines/v1.json` stays valid; (e) the first turn of a chat is never rewritten.
4. **Multi-turn golden set** (`data/golden/multiturn.jsonl`, about 20-30 short conversations; write them BEFORE seeing results): pronoun/ellipsis follow-ups, topic switch (history must not pollute the new topic), a follow-up that needs the previous answer, role switch mid-chat, history-injection attempts. Report on the real model: retrieval hit@4 and answer correctness **with vs without memory** on the same conversations (paired won/lost + exact sign test), rewrite quality (a deterministic check: key entity present in the rewrite), extra latency (one more LLM call), and safety cases for rule 3.
5. **Ablation + verdict** with the S6 pattern (fixed rules written first, snapshot per variant, one reference run, state min detectable change). Adoption of `memory.enabled` as the default for the app is a separate, explicit decision; a changed default needs `regress --save-baseline baselines/v2.json`.
6. **Docs:** `docs/results/S9_chat_memory.md`, README section (how to run the app, what memory does and does not do), tests offline (stand-in server + MockLLM; Streamlit's `streamlit.testing.v1.AppTest` may be used for the page if it runs offline), HANDOFF update, git tag `s9`.

## Do not
- Add a vector/long-term memory, agents, or a cloud service. Change retrieval/ranking defaults. Loosen the safety gate. Report stand-in output as a model result.

## Done when
`streamlit run app/streamlit_app.py` gives a working local chat with roles and optional memory; memory-off requests equal the current ones (tested); rules 3a-3e have tests; the multi-turn comparison is measured on the real model with sample sizes and limits stated; `regress --compare baselines/v1.json` still PASSES with memory off.
