# S4 - LLM judge and judge validation

STATUS: code and offline tests done (2026-11-21). **No real judge numbers yet**: they need the user's machine (DeepEval + Ollama + human labels).
Every table below marked PENDING is filled from `python cli.py judge --validate` output; nothing here is a model result until then.

## What was built
- `eval/judge.py`: `OllamaJudge` (temperature 0, fixed seed, Ollama structured output via `format` = JSON schema, robust JSON extraction, one repair retry
  with a different seed, call cache so a long run can resume, warning counter for prompts near `num_ctx`), the DeepEval adapter (`DeepEvalBaseLLM` subclass,
  imported lazily), the metrics, the deterministic gate and the scoring run. `python cli.py judge` scores the golden set.
- `eval/judge_validation.py`: stratified selection of ~30 answers, label CSV, agreement, Cohen's kappa (stdlib) with bootstrap CI, threshold sweep, run-to-run variance.
- `configs/judge.yaml` (judge model, thresholds, runs), `data/labels/LABELING_GUIDE.md`, `tests/test_judge.py` (49 offline tests), `rag/ollama_client.py` (`chat(..., fmt=None)`).
- Judge model must differ from the generator (`check_distinct` raises on the same model, warns on the same family). Answers are generated first, the generator is unloaded, then the judge runs.

## Metrics and the deterministic gate (judge only where code cannot decide)
| metric | DeepEval | judged when | decided without the judge |
|---|---|---|---|
| faithfulness | `FaithfulnessMetric` | answer is not an abstention and there are contexts | abstention -> n/a |
| answer_relevancy | `AnswerRelevancyMetric` | same | abstention -> n/a |
| correctness (replaces keyword match) | `GEval` vs the reference answer | answerable, answered | abstained on answerable -> 0; answered an unanswerable -> 0; correct abstention -> n/a |
| injection_obedience (opt in) | `GEval` over the retrieval context | any non-abstention | abstention -> 1 |
| toxicity (opt in) | `ToxicityMetric` | non-abstention | abstention -> 0 |

## Validation design
- ~30 answers from the real pipeline, chosen from DETERMINISTIC signals only (never from judge output, which would be circular): 11 keyword-correct, 9 keyword-wrong,
  3 wrong abstentions, 5 correct abstentions, 2 answered-unanswerable (shortfalls filled from other strata; round-robin over question types; order shuffled).
  This is an error-enriched sample: it is NOT representative of the golden set's class balance, so agreement is not an accuracy estimate for the whole set.
- Human labels: correct / incorrect / hallucinated / properly_abstained / evasive (guide in `data/labels/LABELING_GUIDE.md`). Labeled by one person (the project owner): no inter-human agreement is available.
- Reported: (a) three binary comparisons (correctness, faithfulness, relevancy; targets and exclusions in `judge_validation.py` header), each with n, class counts, agreement (Wilson CI), kappa (bootstrap CI);
  (b) five-class agreement and kappa for a FIXED rule over judge scores plus the abstain check; (c) the old keyword metric on the same items; (d) 3-run variance (seeds 42/43/44) of judged items;
  (e) every human-vs-judge disagreement with the judge's reason.
- Thresholds (faithfulness 0.8, relevancy 0.7, correctness 0.5) were fixed before any result. The sweep is an in-sample diagnostic; do not pick the best row and call it validated.

## Results (PENDING: paste from `python cli.py judge --validate`)
| comparison | n | agreement [95% CI] | kappa [95% CI] |
|---|---|---|---|
| judge correctness vs human | PENDING | | |
| keyword correctness vs human (same items) | PENDING | | |
| judge faithfulness vs human (correct vs hallucinated) | PENDING | | |
| judge relevancy vs human (evasive vs rest) | PENDING | | |
| five-class | PENDING | | |
| run-to-run variance (3 runs) | PENDING | | |
Judge model, digest, DeepEval version, speed (`--probe`): PENDING.

## Caveats (true regardless of the numbers)
- n about 30 gives wide intervals; a rare class (hallucinated, evasive) with fewer than 5 examples makes its kappa unreliable (the report says so).
- One labeler, who also knows the pipeline. Synthetic corpus. The judge is a 7B-class local model; DeepEval's prompts were written for stronger models.
- At temperature 0 the seed has no effect: low variance mostly shows determinism, not robustness. `--temperature 0.3` gives a stress view.
- `injection_obedience` is implemented but NOT validated: its ground truth is the S2 canary outcomes (17 injection cases), which needs `eval/run_safety.py` internals (not in this section's brief). Do that first in S5.
- The regression baseline (`baselines/v1.json`) still uses keyword correctness / token-overlap groundedness. Re-freezing as v2 with judge metrics is a decision for after the validation numbers (a judge with kappa below about 0.4 should not gate anything).
