# S3 - Regression gate
**Depends on:** S1 (config decided), ideally S2 (safety JSON). **Size:** medium.
**Goal:** freeze a baseline, compare any later run against it with per-metric tolerances, list flipped questions, return a non-zero exit code on failure, and prove it catches a deliberately bad change.

**Claude reads only:** HANDOFF.md, PROJECT_GOAL.md, this file; `cli.py` (eval output format), `configs/*.yaml`, and the safety output format if S2 is done.

## Tasks
1. `eval/regression.py` + `python cli.py regress --save-baseline baselines/v1.json` and `--compare baselines/v1.json`.
   Baseline stores: per-question pass/fail, aggregate metrics (retrieval, generation, safety), p50/p95 latency, tokens, config hash, model names + Ollama digests (`/api/tags`), data hash, hardware note.
2. Tolerances in `configs/tolerances.yaml`: retrieval -0.02, generation -0.05, safety zero tolerance, p95 latency +25%, tokens +15%, plus absolute floors (so slow drift cannot pass).
3. Noise study: run the eval 3x and report run-to-run variance so the tolerances are justified (expected near zero at temperature 0, seed 42, but measure it).
4. Report: summary, metric table (baseline / new / delta / tolerance / status), flipped questions (pass->fail and fail->pass), config diff.
5. Deliberate-break demo: (a) no-op -> PASS, (b) `enforce_acl: false` -> FAIL on safety, (c) tiny chunks -> FAIL on retrieval, (d) model swap -> shows quality/latency change. Offline tests for the comparator.

**User runs:** save baseline v1, then compare for each break; pastes the reports.
## Done when
No-op passes and every deliberate break fails with the correct gate named; `baselines/v1.json` committed; `docs/results/S3_regression.md`; tag `s3`.
