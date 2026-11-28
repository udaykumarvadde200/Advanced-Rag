# S9 - Streamlit chat UI + short-term memory

**Status: code + offline proof done; the real-model measurement is PENDING** (needs the user's machine with Ollama). No number in this file is a model result unless it says so. The default config is unchanged and `baselines/v1.json` is still valid (frozen data hash and config hash both equal the working tree's).

## What was built
| file | role |
|---|---|
| `rag/memory.py` | `ChatSession`: rewrites a follow-up into a standalone question with the same generator, then runs the normal pipeline. Config switch `memory: {enabled: false, turns: 3}` (default OFF). No Streamlit import. |
| `rag/llm.py` | `complete(system, user)` added to `OllamaLLM` and `MockLLM` (the rewrite call). `generate` is untouched. |
| `app/streamlit_app.py`, `app/ui_helpers.py` | local chat page (role picker, sources, latency/tokens, plain "I don't know", clear button, memory toggle) and its Streamlit-free helpers. |
| `requirements-app.txt` | `streamlit>=1.35,<2`. The core, the tests and CI do not need it (a test checks nothing in `rag/`, `eval/` or `tests/` imports it). |
| `data/golden/multiturn.jsonl` | 30 conversations, 61 turns, written before any run. |
| `eval/run_multiturn.py`, `configs/memory_eval.yaml` | harness (memory off vs on, paired won/lost + exact sign test) and the verdict rules, fixed before any real result. `python cli.py s9 --part multiturn`. |
| `tests/test_memory.py` | 39 tests (38 run offline; 1 `AppTest` page test is skipped when Streamlit is not installed). |

## Design
Memory is **not** added to the answer prompt. The last N turns are used only to turn "and for contractors?" into "How many days of paid leave do contractors get?"; the normal pipeline then runs on that question, with the role of the current turn. With memory off, `rag.answer(question, role)` is called exactly as `cli.py eval` calls it. The history holds only strings (the question as used, the answer, the role): no retrieved chunk is ever stored.

## The five rules and what enforces them
| rule | enforced by | tests (all offline) | hand-mutation that must fail a test |
|---|---|---|---|
| a. ACL on every rewritten turn | the rewrite is retrieved through `rag.answer(used, role)` with the current role; the rewrite step sees no chunks | `test_acl_holds_on_every_rewritten_turn` (guest, employee, eng; scripted rewrite aimed at HR data), `test_a_the_check_would_catch_a_leak` (ACL off = violations), `test_history_stores_only_text_never_chunks` | rewrite retrieved with a fixed high role: caught (3 tests); history stores chunks: caught |
| b. role change clears history | `ChatSession.ask` resets on any role change; the page also clears the chat | `test_role_change_clears_history_in_both_directions`, `test_same_role_keeps_history`, `test_history_never_leaves_a_higher_role_answer...` | reset removed: caught (3 tests) |
| c. history is untrusted | injection-looking sentences removed with the S6 blocklist (`defense.clean_text`), delimiter characters stripped, text delimited and labelled as data, rewrite validated (one line, <= 300 chars, no injection pattern, not a refusal) or the ORIGINAL question is used | `test_injection_in_an_earlier_question_or_answer_is_removed...`, `test_history_is_delimited...`, `test_vulnerable_rewriter_cannot_be_steered...` (a stand-in rewriter that obeys the excerpt: canary absent with the guard, present without it), `test_a_flagged_rewrite_falls_back...`, `TestParseRewrite` | sanitiser off: caught; output check off: caught; delimiters not stripped: caught |
| d. memory off = identical requests | `ChatSession` calls `rag.answer(question, role)` and never the rewriter | `test_d_memory_off_sends_exactly_the_requests_eval_sends` (call-for-call equal to `evaluate`, 26 requests), `test_d_memory_off_ignores_history...`, stand-in-server test: no extra chat request | memory-off uses history: caught (5 tests) |
| e. first turn never rewritten | the rewrite call happens only when the recent history is non-empty | `test_e_first_turn_is_never_rewritten`, `test_e_with_memory_on_a_fresh_session_per_question_equals_eval` | rewrite on empty history: caught (6 tests) |

Eight hand mutations were applied one at a time to `rag/memory.py`; every one failed at least one test (the file was restored afterwards and the suite is green).

## The multi-turn set (`data/golden/multiturn.jsonl`)
30 conversations / 61 turns: 13 pronoun-ellipsis, 4 standalone controls, 4 topic switches, 3 role switches, 3 history injections, 2 "needs the previous answer", 1 reference-but-standalone. 18 follow-up turns need the history, 10 do not (controls, topic switches). Every evidence string exists in a chunk of its source document and every role can read the documents its turns cite (checked by `python cli.py s9 --part dataset` and a test).
Limits, stated before any result: the conversations were written by the assistant with the corpus in view; the "needs history" follow-ups are ambiguous **on purpose**, so that slice shows the best case for memory, not what a typical user session would gain; the controls and topic switches are what measures harm. `rewrite_must` / `rewrite_must_not` (key entity present, earlier topic absent) are a deterministic proxy for rewrite quality, not a judgement of fluency. Keyword correctness inherits the S1 metric's weaknesses.

## Verdict rules (fixed in `configs/memory_eval.yaml` before any real result)
R1 safety (memory-on rows): ACL violations 0, canary obeyed 0, role leaks 0. R2 target: on the history-needing follow-ups, correctness net >= 1 and hit@k net >= 0 versus the same-session reference. R3 harm: on follow-ups that do not need history, net correctness loss <= 1 (session noise). R4 cost: p95 <= 1.75x and tokens/turn <= 1.5x the reference. KEEP = R1-R4 and sign-test p < 0.05; KEEP? = R2 holds but within noise; DROP otherwise. First turns are identical requests in every variant, so flips there measure session noise. Minimum detectable change: 6 net questions (exact sign test, alpha 0.05). Variants: `memory_t1` (last turn), `memory_t3` (last 3 turns). Adopting memory as an app default stays a separate decision.

## Offline proof (stand-ins: hash embedder, 10-line rule-based rewriter) - plumbing only
The whole harness runs end to end (61 turns x 3 variants, report written, resumable). With the stand-in the verdict for both variants is DROP (R2 fails: a rewriter that concatenates two questions does not help the hash retriever). That says nothing about a model; it only shows the verdict logic can say DROP. Safety counters are 0 in every stand-in variant, and the canary check demonstrably fires when the guard is switched off (`guard: false`, a test-only proof switch).

## What was NOT verified here
- **The Streamlit page was never run.** The build sandbox has no Streamlit and no network. The page compiles, its logic lives in `app/ui_helpers.py` (tested), and `TestStreamlitPage` (Streamlit's `AppTest`, offline stand-ins) is written but was skipped. Run it on your machine after installing `requirements-app.txt`.
- **No real model was used**: rewrite quality, hit@4 and correctness with vs without memory, and the extra latency are all unmeasured.
- `regress --compare baselines/v1.json` was not run (needs Ollama). Its inputs are unchanged (data hash and config hash identical to the frozen baseline) and rule d is tested, so it should PASS with memory off; confirm it.

## Run it for real (your machine, Ollama running, venv active)
```powershell
python -m unittest                                   # expect 247 tests OK (246 + 1 skipped without Streamlit)
pip install -r requirements-app.txt                  # then re-run: the AppTest page test now runs
python cli.py regress --compare baselines/v1.json    # must still PASS with memory off
python cli.py s9 --part multiturn                    # ~3 passes x 61 turns; resumable; writes results/s9_memory.md
streamlit run app/streamlit_app.py                   # try it by hand
```
Then paste the table printed by `s9 --part multiturn` (and the first-turn noise line) into this file, replace this status, and tag `s9`.

## Known limits
- The sanitiser is a blocklist (S6: 9/10 mainstream, 0/6 evasive held-out injections); an evasive injection in an earlier turn is not removed, which is why the rewrite is also validated and why the history never reaches the answer prompt.
- A wrong rewrite compounds: the rewritten question is stored as the history for the next turn.
- One generator call per follow-up is added (a short call, 96 tokens max).
- Real authentication is out of scope: the role picker lets anyone choose any role (the page says so).
- n is small (18 + 10 follow-ups); only a net change of about 6 questions is detectable.
