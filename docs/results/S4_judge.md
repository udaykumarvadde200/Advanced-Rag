# S4 - LLM judge and judge validation

STATUS: DONE (2026-11-21). Real validation recorded below; result tables come from `python cli.py judge --validate` output pasted by the user.

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

## Results (2026-11-21; judge qwen2.5:7b digest 845dbda0ea48, generator llama3.2:3b, DeepEval 4.3.4, temperature 0, seeds 42/43/44)
Speed from `--probe`: 28.6 tok/s warm, 82% of the model in VRAM, structured output OK. Full validation (30 items x 3 runs): about 14 min.
Human labels, n=30 (final): correct 16, incorrect 6, evasive 2, properly_abstained 6, **hallucinated 0**.

| comparison | n | agreement [95% CI] | kappa [95% CI] | usable? |
|---|---|---|---|---|
| judge correctness vs human (seed 42) | 24 (+16 / -8) | 0.79 [0.60, 0.91] | 0.55 [0.14, 0.88], moderate | rough only: wide CI |
| same, per run (seeds 42 / 43 / 44) | 24 | 0.79 / 0.83 / 0.83 | 0.55 / 0.65 / 0.65 | one item of 24 moves agreement by 0.04 |
| same, median of the 3 runs | 24 | 0.83 | 0.65 | rough only |
| keyword correctness vs human (same 24 items) | 24 | 0.92 | 0.82 | CIs in `results/judge_validation.md`; overlap with the judge's at this n |
| judge faithfulness vs human (correct vs hallucinated) | 16 (+16 / -0) | 0.81 | undefined (0.00 by construction) | NO: no hallucinated example exists |
| judge relevancy vs human (evasive vs rest) | 22 (+22 / -0) | 0.86 | undefined (0.00 by construction) | NO: no non-relevant example exists |
| five-class (fixed rule over judge scores + abstain check) | 30 | 0.77 [0.59, 0.88] | 0.65 [0.41, 0.85], substantial | rough only; see label-change note |

Variance over 3 runs (judged items, n=22): faithfulness mean range 0.030, max 0.67 (q035: 0.67 vs 0.00), 0 verdict flips; relevancy identical in all runs; **correctness mean range 0.136, max 0.50, 3 of 22 verdicts flip across seeds** (run means 0.568 / 0.532 / 0.532).

### Label change after seeing judge output (disclosure)
After the first validation run (five-class agreement 0.70, kappa 0.55) the labeler was shown the judge-vs-human disagreement table and corrected two labels that contradicted the labeling guide: q040 (unanswerable question, answer "I don't know": `evasive` -> `properly_abstained`) and q003 (answer gives a wrong number: `evasive` -> `incorrect`). Both corrections move toward the judge's verdicts, so the second five-class result (0.77 / 0.65) is not fully independent of the judge's output. Both versions are reported; correctness numbers did not change.

### What this does and does not show
- Correctness: moderate agreement with human labels (19 of 24 at seed 42; 20 of 24 with the median of 3 runs). The keyword metric agreed on 22 of 24. With n=24 these differences are 2 to 3 items and not statistically established, but the judge did not beat the keyword metric on this sample.
- The 5 correctness disagreements at seed 42 (judge reasons read from the report): two answers judged wrong because of "15,000" vs "15000" and extra source text although the criteria say to ignore them (q048, q068); two wrong answers accepted (q005, q071; q005 scored 0.7, 0.2, 0.4 across seeds); q072 is a genuine borderline. Cleaning citations and number formats before judging is a candidate fix, to be tested on NEW labels, not on these 30.
- Faithfulness and relevancy are NOT validated: no hallucinated and no non-relevant answer in the sample. The only observation: faithfulness (threshold 0.8) scored 3 of 16 human-correct answers as unfaithful (multi-claim answers, 0.33 to 0.67), i.e. false alarms.

### Recommendation (the decision is the project owner's)
Do not gate on judge metrics and do not re-freeze the regression baseline as v2. Keep the keyword/overlap metrics as the gate and use the judge as an advisory second opinion (its disagreements with the keyword metric are a review list). Validating faithfulness needs labeled answers that contain hallucinations; S5 (generation failures) is where to obtain and label them.

## Caveats (true regardless of the numbers)
- n about 30 gives wide intervals; a rare class (hallucinated, evasive) with fewer than 5 examples makes its kappa unreliable (the report says so).
- One labeler, who also knows the pipeline. Synthetic corpus. The judge is a 7B-class local model; DeepEval's prompts were written for stronger models.
- Earlier expectation that the seed has no effect at temperature 0 was WRONG for correctness: verdicts flipped on 3 of 22 items across seeds 42/43/44 (cause not investigated: DeepEval's GEval prompts or GPU/offload nondeterminism are both possible). Relevancy was identical across seeds.
- The human labels were made with Claude's per-row guidance on most rows (the labeler was stuck/tired; exact count not tracked), so they are not fully independent human judgments, and the one labeler also knows the pipeline. Read agreement as optimistic for any metric that resembles the guidance (reference matching).
- The 30 rows were chosen using the keyword metric (strata) and are error-enriched: not representative of the golden set.
- `injection_obedience` is implemented but NOT validated: its ground truth is the S2 canary outcomes (17 injection cases), which needs `eval/run_safety.py` internals (not in this section's brief). Do that first in S5.
- The regression baseline (`baselines/v1.json`) still uses keyword correctness / token-overlap groundedness. Re-freezing as v2 with judge metrics is a decision for after the validation numbers (a judge with kappa below about 0.4 should not gate anything).
