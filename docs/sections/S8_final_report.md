# S8 - Final report and polish
**Depends on:** all earlier sections. **Size:** small-medium. Claude reads: HANDOFF.md, PROJECT_GOAL.md, this file, `docs/results/*.md`, README.md.

## Tasks
1. README: headline results (measured numbers only), the component/workflow/application tables, ablations, failure-attribution breakdown, the regression demo, safety table, judge-vs-human agreement.
2. Limitations section: synthetic data (or the chosen dataset), small n, local 3B models, stand-in vs real runs, judge noise.
3. Architecture diagram (mermaid or SVG) and an evaluation-flow diagram that follows the course workflow.
4. One-command reproduce script (`run_all.ps1` and/or Makefile): build data -> check -> eval -> sweep -> safety -> regress.
5. Optional tracing: LangSmith or Langfuse wrapper, off by default (synthetic data only if cloud).
6. Repo hygiene: pinned `requirements.txt`, LICENSE, `.gitignore` check, CI workflow that runs only the offline unit tests, remove dead code, make sure `python -m unittest` passes.
7. Demo script: a 2-minute screen-recording outline (ask a question, show citations, show a failing regression gate, show the safety table).

## Done when
A stranger can clone, follow the README and reproduce the tables; tag `v1.0`; HANDOFF marked complete.
