# S7 - Dataset decision (optional, time-boxed)
**Depends on:** S5. **Decide with the user first**: is the synthetic corpus credible enough for the goal? Time-box the whole section.

## Options
- **A. Stay synthetic** and say so clearly in the report (cheapest; weakest credibility).
- **B. FinanceBench subset** (public benchmark, real SEC filings, ready golden questions with page-level evidence; est. +25-35 h). Sources: Hugging Face `PatronusAI/financebench` (150 open examples) and GitHub `patronus-ai/financebench` (PDFs). **License is disputed between sources (the HF card says CC BY-NC 4.0): the user must read it.**
  Needs: a download script the user runs (Claude has no network), PyMuPDF parsing, page-level evidence loader, ~40-50 questions over 8-10 filings, metadata routing (company/year) and the calculator tool become central, slow local runs (long contexts on 6 GB VRAM). Re-run the S1-S5 harness through config.
- **C. Lecture-transcript doubt-solving chatbot** over public YouTube captions (est. +15-20 h): transcript cleaning (ASR noise, repetition loops), time-window chunking, timestamp citations, attendee-name privacy tests. Do not publish members-only transcripts.

## Tasks
Agree option -> write loader + golden subset + one working eval run -> update HANDOFF with what changed and what must be re-run.
## Done when
Decision recorded in HANDOFF; if B/C, one end-to-end eval run on the new data; tag `s7`.
