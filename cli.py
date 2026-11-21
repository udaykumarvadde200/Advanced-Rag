"""Command-line entry point.
  python cli.py check [--config configs/ollama.yaml]          verify Ollama is up and models are pulled
  python cli.py ask "question" --role employee [--config ...]  ask one question
  python cli.py eval [--config ...] [--limit N] [--out results/run.json]   run the golden set
  python cli.py sweep [--offline] [--retrievers bm25,dense,hybrid]          retrieval sweeps -> results/sweeps.md
  python cli.py regress --save-baseline baselines/v1.json | --compare baselines/v1.json | --noise-study 3   [--offline] [--no-safety] [--set a.b=c ...]   regression gate (exit 1 = a gate failed, 2 = invalid comparison)
  python cli.py safety [--offline] [--no-proof] [--out results/safety.json] [--md results/safety.md]   safety suite (exit 1 = ACL violation, 2 = detection proof failed)
"""
import argparse, json, os, sys, time
from rag.pipeline import RAG, load_cfg, load_golden, ROLES
from rag.metrics import retrieval_metrics, correctness, groundedness, mean
from rag.llm import is_abstain
from rag.telemetry import summarize_latency
from rag import ollama_client as oc

def needed_models(cfg):
    out = []
    if cfg["embedder"].get("type") == "ollama" and cfg["retriever"]["mode"] != "bm25": out.append(cfg["embedder"]["model"])
    if cfg["llm"].get("type") == "ollama": out.append(cfg["llm"]["model"])
    if cfg.get("reranker", {}).get("type") == "ollama": out.append(cfg["reranker"]["model"])
    return sorted(set(out))

def cmd_check(cfg):
    need = needed_models(cfg); print(f"Ollama host: {oc.DEFAULT_HOST}\nModels this config needs: {need or 'none (offline stand-ins)'}")
    if not need: return 0
    miss = oc.missing_models(need)
    if miss:
        print("MISSING:"); [print(f"  ollama pull {m}") for m in miss]; return 1
    print("OK: server reachable and all models are pulled."); return 0

def cmd_ask(cfg, q, role):
    r = RAG(cfg); o = r.answer(q, role)
    print(o["answer"]); print("\nSources:", ", ".join(c.id for c in o["contexts"])); print("Trace:", json.dumps(o["trace"]["stages_ms"]), "tokens", o["trace"]["tokens_in"], "/", o["trace"]["tokens_out"])

def cmd_eval(cfg, limit, out):
    from eval.run_golden import evaluate
    res = evaluate(cfg, limit, log=lambda m: print(m, flush=True)); summ = res["summary"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: json.dump(dict(summary=summ, rows=res["rows"]), f, indent=1, ensure_ascii=False)
    print("\n" + "=" * 50); [print(f"{k:22s} {round(v, 3) if isinstance(v, float) else v}") for k, v in summ.items() if k not in ("config", "latency")]
    print("latency", summ["latency"]); print("saved ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "ask", "eval", "sweep", "safety", "regress"]); ap.add_argument("question", nargs="?")
    ap.add_argument("--config", default="configs/ollama.yaml"); ap.add_argument("--role", default="employee", choices=list(ROLES))
    ap.add_argument("--limit", type=int); ap.add_argument("--sweep-config", default="configs/sweeps.yaml"); ap.add_argument("--sweep-out", default="results/sweeps")
    ap.add_argument("--offline", action="store_true", help="sweep with the hash embedder (no Ollama)"); ap.add_argument("--retrievers", help="comma list, e.g. bm25,hybrid"); ap.add_argument("--out", default=None); ap.add_argument("--md", default=None); ap.add_argument("--no-proof", action="store_true", help="safety: skip the detection-proof runs"); 
    ap.add_argument("--save-baseline"); ap.add_argument("--compare"); ap.add_argument("--noise-study", type=int, metavar="N"); ap.add_argument("--tolerances", default="configs/tolerances.yaml")
    ap.add_argument("--no-safety", action="store_true", help="regress: skip the safety suite (baseline and compare must both skip it)"); ap.add_argument("--hardware", help="regress: hardware note stored in the baseline")
    ap.add_argument("--set", dest="sets", action="append", help="regress: override a config key, e.g. --set retriever.enforce_acl=false (repeatable)"); ap.add_argument("--label", default=""); a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.cmd == "regress":
        from eval.regression import apply_overrides
        from rag.pipeline import deep_merge
        if a.offline: cfg = deep_merge(cfg, dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False)))
        cfg = apply_overrides(cfg, a.sets)
    try:
        if a.cmd == "check": sys.exit(cmd_check(cfg))
        if a.cmd == "sweep":
            from eval.run_sweeps import run_from_config
            sys.exit(run_from_config(a.sweep_config, a.sweep_out, a.offline, a.retrievers.split(",") if a.retrievers else None))
        if a.cmd in ("safety", "regress") and a.offline: miss = []
        else: miss = oc.missing_models(needed_models(cfg)) if needed_models(cfg) else []
        if miss: sys.exit("Missing Ollama models. Run:\n" + "\n".join(f"  ollama pull {m}" for m in miss))
        if a.cmd == "safety":
            from eval.run_safety import run_all, console_summary, save, exit_code
            res = run_all(cfg, offline=a.offline, proof=not a.no_proof, log=lambda m: None if m.startswith("  ") else print(m, flush=True))
            save(res, a.out or "results/safety.json", a.md or "results/safety.md"); console_summary(res); print("saved ->", a.out or "results/safety.json", "and", a.md or "results/safety.md"); sys.exit(exit_code(res))
        if a.cmd == "regress":
            from eval.regression import run_cli
            sys.exit(run_cli(cfg, a))
        if a.cmd == "ask": cmd_ask(cfg, a.question or sys.exit("give a question"), a.role)
        else: cmd_eval(cfg, a.limit, a.out or "results/run.json")
    except oc.OllamaError as e: sys.exit(f"Ollama error: {e}")
