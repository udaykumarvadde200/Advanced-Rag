# Architecture and evaluation flow

Mermaid diagrams (rendered by GitHub). Both describe what the code in this repo does; nothing here is aspirational.

## 1. Application architecture (what is being evaluated)

```mermaid
flowchart LR
    Q["Question + role<br/>(guest / employee / hr / finance / eng)"] --> ACL["ACL pre-filter<br/>(drops chunks the role may not read<br/>BEFORE any ranking)"]
    DOCS[("30 synthetic docs<br/>access tags")] --> CH["Chunking<br/>recursive-300"]
    CH --> IDX[("Index<br/>BM25 + nomic-embed-text vectors<br/>(chunk vectors cached on disk)")]
    IDX --> ACL
    ACL --> RET["Hybrid retrieval (RRF)<br/>pool 20, top k=4"]
    RET -. "optional switches (default OFF)" .-> OPT["routing / reranker /<br/>sanitiser / spotlight"]
    OPT -.-> GEN
    RET --> GEN["Generator<br/>llama3.2:3b via Ollama<br/>temp 0, seed 42"]
    GEN -. "optional (default OFF)" .-> CALC["Calculator tool<br/>(AST-whitelist evaluator)"]
    GEN --> A["Answer + citations<br/>or 'I don't know'"]
    CALC -.-> A
    A --> T["Telemetry: stage latency,<br/>tokens, estimated cost"]
```

## 2. Evaluation flow (the course workflow: golden set -> criteria -> method -> run -> analyse -> improve -> re-run)

```mermaid
flowchart TD
    G["Golden set: 82 questions<br/>52 factual / 8 paraphrase / 9 multi-hop / 13 unanswerable<br/>+ safety cases (injection 17, PII 16, jailbreak 16,<br/>abstention 25, access 20 + 1230-retrieval matrix)<br/>+ 16 held-out injections"] --> R["Run the pipeline<br/>(cli.py eval / sweep / safety)"]
    R --> M1["Deterministic checks first<br/>hit@k, MRR, doc_recall, keyword correctness,<br/>abstention, canary / PII / ACL detectors"]
    M1 --> M2["LLM judge only where code cannot decide<br/>(qwen2.5:7b, advisory: kappa 0.55 vs 30 human labels)"]
    M1 --> AN["Analyse<br/>gold-vs-retrieved context,<br/>failure attribution (first-match order),<br/>paired won/lost + exact sign test"]
    M2 --> AN
    AN --> IMP["Improve: one change at a time<br/>(config switch; ablation snapshot per variant)"]
    IMP --> RG["Regression gate<br/>regress --compare baselines/v1.json<br/>exit 0 PASS / 1 FAIL / 2 INVALID"]
    RG -->|"PASS: adopt, re-freeze baseline (v2)"| BASE[("Frozen baseline")]
    RG -->|"FAIL: gate + flipped questions named"| IMP
    FAIL["Production failure / new finding"] -->|"add as a golden or safety case"| G
    BASE --> R
```

Detection proof (safety suite): the same suite is also run against a deliberately vulnerable stand-in model and with the ACL switched off; the detectors must fire. That proves the tests can fail, which a clean pass alone cannot.
