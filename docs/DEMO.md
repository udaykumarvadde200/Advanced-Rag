# Demo script (about 2 minutes, screen recording)

Prep (off camera): `ollama serve` running, venv active, models pulled, `python cli.py check` is green, terminal font large, `results\` cleared, one pre-recorded `regress` result in case the live gate takes too long (a full `regress --compare` is about 200 LLM calls: minutes).

| time | show | say (outline) |
|---|---|---|
| 0:00-0:15 | README headline table | "A local RAG chatbot over a company knowledge base with role-based access. The deliverable is the evidence: every claim has a sample size and a stated limit." |
| 0:15-0:40 | `python cli.py ask "How many sick leave days do we get?" --role employee` then the same with `--role guest` | Citations appear. Rehearse a restricted question for the guest role first and confirm what it answers before recording (the ACL result to quote is structural: access is a pre-filter before ranking, 0 violations in 1250 checks). |
| 0:40-1:00 | `python cli.py eval --limit 10` | hit@4, MRR, correctness, abstention, latency per stage: retrieval is at ceiling, generation is the weak part. |
| 1:00-1:30 | `python cli.py regress --compare baselines/v1.json --set retriever.enforce_acl=false` (or open the saved `results/regress.md`) | A deliberate break: exit code 1, gate = safety, ACL violations 0 -> 1233, 37 safety cases newly failing, while correctness did not move. A no-op exits 0. |
| 1:30-1:50 | `results/safety.md` or the safety table in the README | Injection 17/17 in the fixed set, but a held-out set shows the 3B model obeys 5/16 undefended: the honest number. |
| 1:50-2:00 | README "Limitations" | Synthetic corpus, small n, nothing in the ablations significant; what it can and cannot tell you. |

Do NOT claim: that the defences "work" (effects are small and not significant), that retrieval results transfer to real corpora, or that the judge is validated beyond rough correctness agreement.
