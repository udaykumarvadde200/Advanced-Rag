"""S9: multi-turn evaluation of the short-term memory (rag/memory.py) on data/golden/multiturn.jsonl.
  python cli.py s9 --part multiturn [--offline] [--only memory_t3] [--force]     one pass per variant of configs/memory_eval.yaml -> results/memory/<name>.json + docs-ready markdown
  python cli.py s9 --part dataset                                                   deterministic check of the conversation file (no Ollama)
Per turn it records: the question actually used (the rewrite), retrieval hit@k, keyword correctness, abstention / leak on role-switch turns, ACL check on the returned
chunks, canary checks for history-injection conversations, latency and tokens. The verdict rules are in configs/memory_eval.yaml and were fixed before any real result.
Stand-in (offline) numbers prove plumbing only: the stand-in rewriter is a few lines of Python, not a model."""
import json, os, re, yaml
from rag.pipeline import RAG, ROLES, _read, deep_merge
from rag.memory import ChatSession, memory_cfg
from rag.metrics import retrieval_metrics, correctness, mean
from rag.llm import is_abstain
from rag.telemetry import pct
from eval.regression import sign_test_p, min_detectable, apply_overrides, save_json, save_text
from eval.run_safety import norm

def load_multiturn(path="data/golden/multiturn.jsonl"): return [json.loads(l) for l in _read(path).splitlines() if l.strip()]

# ---------------------------------------------------------------- deterministic checks
def rewrite_ok(text, turn):
    """Key entities present (`rewrite_must`: every group, alternatives written a|b) and no pollution from the earlier topic (`rewrite_must_not`). None if the turn has no expectation."""
    if not turn.get("rewrite_must") and not turn.get("rewrite_must_not"): return None
    s = text.lower()
    return all(any(a.strip().lower() in s for a in g.split("|")) for g in turn.get("rewrite_must", [])) and not any(x.lower() in s for x in turn.get("rewrite_must_not", []))

def run_conversations(cfg, convs=None, rag=None, log=lambda m: None):
    """One pass over the conversations. A fresh ChatSession per conversation; `cfg['memory']` decides whether the history is used."""
    rag = rag or RAG(cfg); convs = convs or load_multiturn(); k = cfg["retriever"].get("k", 4); rows = []
    rag.answer("warm up", "guest")      # throwaway call, as in cli.py eval: loads the models so load time does not pollute latency
    for c in convs:
        s = ChatSession(rag, cfg)
        for n, t in enumerate(c["turns"], 1):
            role = t.get("role", c["role"]); o = s.ask(t["question"], role); m = o["memory"]; tr = o["trace"]
            row = dict(conv=c["id"], turn=n, kind=c["kind"], role=role, question=t["question"], standalone=m["standalone"], used=m["used"], rewritten=m["rewritten"], fallback=m["fallback"],
                       role_reset=m["role_reset"], needs_history=bool(t["needs_history"]), answerable=t["answerable"], answer=o["answer"], doc_ids=[x.doc_id for x in o["contexts"]],
                       acl_ok=all(x.access in ROLES[role] for x in o["contexts"]), total_ms=tr["total_ms"], rewrite_ms=m["rewrite_ms"], tokens=tr["tokens_in"] + tr["tokens_out"])
            if t["answerable"]: row.update(retrieval_metrics(o["contexts"], t, k)); row["correct"] = correctness(o["answer"], t)
            else: row["abstained"] = float(is_abstain(o["answer"])); row["leak"] = float(any(norm(x) in norm(o["answer"]) for x in t.get("must_not_contain", [])))
            if m["used"]: row["rewrite_ok"] = rewrite_ok(m["standalone"], t)
            if c.get("canary"):    # the canary was planted in turn 1; it may not appear in a later rewrite or a later answer
                row["canary_in_answer"] = float(n > 1 and norm(c["canary"]) in norm(o["answer"])); row["canary_in_rewrite"] = float(n > 1 and norm(c["canary"]) in norm(m["standalone"]))
            rows.append(row); log(f"  {c['id']}.{n} {tr['total_ms']:.0f} ms  {'REWRITE: ' + m['standalone'] if m['rewritten'] else ''}")
    return rows

# ---------------------------------------------------------------- statistics
def slices(rows):
    ans = [r for r in rows if r["answerable"]]
    return dict(first=[r for r in ans if r["turn"] == 1], hist=[r for r in ans if r["turn"] > 1 and r["needs_history"]], nohist=[r for r in ans if r["turn"] > 1 and not r["needs_history"]])

def paired(ref, new, key, pred=lambda r: True):
    """Won / lost on a pass-fail key over the (conv, turn) pairs that satisfy pred (judged on the reference row). Exact sign test on the changed pairs."""
    nr = {(r["conv"], r["turn"]): r for r in new}; won = lost = tie = 0
    for r in ref:
        n = nr.get((r["conv"], r["turn"]))
        if n is None or key not in r or key not in n or not pred(r): continue
        a, b = r[key], n[key]
        if b > a: won += 1
        elif b < a: lost += 1
        else: tie += 1
    return dict(n=won + lost + tie, won=won, lost=lost, tie=tie, net=won - lost, p=round(sign_test_p(won, lost), 4))

def summarize(rows):
    S = slices(rows); rate = lambda rs, key: round(mean(r[key] for r in rs if key in r), 3) if any(key in r for r in rs) else None
    used = [r for r in rows if r["used"]]; chk = [r for r in used if r.get("rewrite_ok") is not None]; tot = [r["total_ms"] for r in rows]
    sw = [r for r in rows if not r["answerable"]]
    return dict(turns=len(rows), followups=sum(1 for r in rows if r["turn"] > 1),
                slices={k: dict(n=len(v), hit=rate(v, "hit"), mrr=rate(v, "mrr"), correct=rate(v, "correct")) for k, v in S.items()},
                rewrites=dict(calls=len(used), changed=sum(r["rewritten"] for r in used), fallbacks=sum(1 for r in used if r["fallback"]), checked=len(chk), ok=sum(bool(r["rewrite_ok"]) for r in chk),
                              mean_ms=round(mean(r["rewrite_ms"] for r in used), 2) if used else 0.0),
                latency=dict(p50_ms=round(pct(tot, 50), 2), p95_ms=round(pct(tot, 95), 2)), tokens_per_turn=round(mean(r["tokens"] for r in rows), 1),
                safety=dict(acl_violations=sum(1 for r in rows if not r["acl_ok"]), canary_obeyed=int(sum(r.get("canary_in_answer", 0) + r.get("canary_in_rewrite", 0) for r in rows)),
                            role_switch_turns=len(sw), role_switch_abstained=int(sum(r["abstained"] for r in sw)), role_leaks=int(sum(r["leak"] for r in sw))),
                by_kind={k: dict(n=sum(1 for r in rows if r["kind"] == k and r["answerable"]), correct=sum(int(r["correct"]) for r in rows if r["kind"] == k and r["answerable"])) for k in sorted({r["kind"] for r in rows})})

def verdict(name, ref, new, rules, offline=False):
    """ref/new = row lists. Rules from configs/memory_eval.yaml (fixed before any real result)."""
    R, T, H, C = rules, rules["target"], rules["harm"], rules["cost"]; sr, sn = summarize(ref), summarize(new); why = []
    h = lambda r: r["turn"] > 1 and r["needs_history"]; nh = lambda r: r["turn"] > 1 and not r["needs_history"]
    P = dict(hist_correct=paired(ref, new, "correct", h), hist_hit=paired(ref, new, "hit", h), nohist_correct=paired(ref, new, "correct", nh), nohist_hit=paired(ref, new, "hit", nh),
             first_correct=paired(ref, new, "correct", lambda r: r["turn"] == 1), first_hit=paired(ref, new, "hit", lambda r: r["turn"] == 1))
    sf = sn["safety"]; r1 = sf["acl_violations"] == 0 and sf["canary_obeyed"] == 0 and sf["role_leaks"] == 0
    if not r1: why.append(f"R1 FAIL: ACL violations {sf['acl_violations']}, canary obeyed {sf['canary_obeyed']}, role leaks {sf['role_leaks']}")
    r2 = P["hist_correct"]["net"] >= T["correct_net_min"] and P["hist_hit"]["net"] >= T["hit_net_min"]
    why.append(f"R2 {'ok' if r2 else 'FAIL'}: history-needed turns (n={P['hist_correct']['n']}) correctness {P['hist_correct']['won']} won / {P['hist_correct']['lost']} lost (p={P['hist_correct']['p']}), hit@k {P['hist_hit']['won']} / {P['hist_hit']['lost']}")
    r3 = -P["nohist_correct"]["net"] <= H["max_net_loss"]
    why.append(f"R3 {'ok' if r3 else 'FAIL'}: turns that do not need history (n={P['nohist_correct']['n']}) correctness {P['nohist_correct']['won']} won / {P['nohist_correct']['lost']} lost")
    pr = sn["latency"]["p95_ms"] / sr["latency"]["p95_ms"] if sr["latency"]["p95_ms"] else 1.0; tr_ = sn["tokens_per_turn"] / sr["tokens_per_turn"] if sr["tokens_per_turn"] else 1.0
    r4 = True if offline else (pr <= C["p95_ratio_max"] and tr_ <= C["tokens_ratio_max"])
    why.append(f"R4 {'skipped (stand-in latency is meaningless)' if offline else ('ok' if r4 else 'FAIL')}: p95 x{pr:.2f} (max {C['p95_ratio_max']}), tokens/turn x{tr_:.2f} (max {C['tokens_ratio_max']})")
    why.append(f"session noise: first turns are identical requests; flips there: correctness {P['first_correct']['won'] + P['first_correct']['lost']}, hit {P['first_hit']['won'] + P['first_hit']['lost']} of {P['first_correct']['n']}")
    sig = P["hist_correct"]["p"] < R["alpha"]
    v = "DROP" if not (r1 and r3 and r4 and r2) else ("KEEP" if sig else "KEEP?")
    return dict(name=name, verdict=v + (" (stand-in)" if offline else ""), why=why, paired=P, summary=sn)

# ---------------------------------------------------------------- ablation runner + report
def _spec(path="configs/memory_eval.yaml"): return yaml.safe_load(_read(path))

def run_memory_ablation(cfg0, only=None, force=False, offline=False, outdir="results/memory", log=print, convs=None, spec=None):
    spec = spec or _spec(); convs = convs or load_multiturn(); rag = RAG(cfg0); runs = {}
    variants = [dict(name=spec["reference"], set={"memory.enabled": False})] + spec["variants"]        # the reference always runs first (same session, same call history)
    for v in variants:
        if only and v["name"] not in only and v["name"] != spec["reference"]: continue
        path = f"{outdir}/{v['name']}.json"
        if os.path.exists(path) and not force:
            with open(path, encoding="utf-8") as f: runs[v["name"]] = json.load(f)["rows"]
            log(f"[{v['name']}] loaded {path}"); continue
        cfg = apply_overrides(cfg0, [f"{k}={json.dumps(val)}" for k, val in v["set"].items()]); log(f"[{v['name']}] running {len(convs)} conversations")
        rows = run_conversations(cfg, convs, rag, log=lambda m: None); runs[v["name"]] = rows
        meta = dict(variant=v["name"], offline=offline, memory=memory_cfg(cfg), n_conversations=len(convs))
        if not offline:
            from rag import ollama_client as oc; meta["ollama_version"] = oc.version()
        save_json(dict(meta=meta, rows=rows), path)
    ref = runs[spec["reference"]]; out = [dict(name=spec["reference"], verdict="reference", why=[], paired={}, summary=summarize(ref))]
    for v in spec["variants"]:
        if v["name"] in runs: out.append(verdict(v["name"], ref, runs[v["name"]], spec["rules"], offline))
    return out, runs

def memory_markdown(results, n_conv=30):
    f = lambda x: "-" if x is None else f"{x:.3f}"
    L = ["| variant | follow-ups needing history: correct | hit@k | follow-ups not needing it: correct | rewrites (changed / fallback) | rewrite ok | p50 / p95 ms | tokens/turn | ACL viol. | canary | role leaks | verdict |", "|" + "---|" * 12]
    for r in results:
        s = r["summary"]; sl = s["slices"]; rw = s["rewrites"]; sf = s["safety"]
        L.append(f"| {r['name']} | {f(sl['hist']['correct'])} (n={sl['hist']['n']}) | {f(sl['hist']['hit'])} | {f(sl['nohist']['correct'])} (n={sl['nohist']['n']}) | {rw['calls']} ({rw['changed']} / {rw['fallbacks']}) | "
                 f"{rw['ok']}/{rw['checked']} | {s['latency']['p50_ms']} / {s['latency']['p95_ms']} | {s['tokens_per_turn']} | {sf['acl_violations']} | {sf['canary_obeyed']} | {sf['role_leaks']} | {r['verdict']} |")
    L += ["", f"Minimum detectable change (exact sign test, alpha 0.05): {min_detectable()} net questions out of the questions that changed. First turns (identical requests in every variant) measure session noise."]
    for r in results[1:]:
        L += ["", f"**{r['name']}**: " + r["verdict"]] + [f"- {w}" for w in r["why"]]
    kinds = sorted({k for r in results for k in r["summary"]["by_kind"]})
    L += ["", "Correct / answerable turns by conversation kind (all turns, first turns included):", "", "| kind | " + " | ".join(r["name"] for r in results) + " |", "|" + "---|" * (len(results) + 1)]
    for k in kinds: L.append(f"| {k} | " + " | ".join(f"{r['summary']['by_kind'][k]['correct']}/{r['summary']['by_kind'][k]['n']}" for r in results) + " |")
    return "\n".join(L)

def dataset_report(path="data/golden/multiturn.jsonl"):
    from rag.pipeline import load_docs
    from rag.chunking import chunk_docs
    convs = load_multiturn(path); docs = {d["id"]: d for d in load_docs()}; chunks = chunk_docs(list(docs.values()), "recursive", 300, 0); problems = []
    for c in convs:
        for n, t in enumerate(c["turns"], 1):
            role = t.get("role", c["role"])
            if role not in ROLES: problems.append((c["id"], n, "unknown role", role))
            for d in t["source_docs"]:
                if d not in docs: problems.append((c["id"], n, "unknown doc", d))
                elif docs[d]["access"] not in ROLES.get(role, ()): problems.append((c["id"], n, "role cannot read its own source doc", d))
            for e in t["evidences"]:
                if not any(e.lower() in ch.text.lower() for ch in chunks if ch.doc_id in t["source_docs"]): problems.append((c["id"], n, "evidence not in a chunk of the source docs", e))
    return dict(conversations=len(convs), turns=sum(len(c["turns"]) for c in convs), kinds={k: sum(1 for c in convs if c["kind"] == k) for k in sorted({c["kind"] for c in convs})},
                followups_needing_history=sum(1 for c in convs for t in c["turns"][1:] if t["answerable"] and t["needs_history"]),
                followups_not_needing_history=sum(1 for c in convs for t in c["turns"][1:] if t["answerable"] and not t["needs_history"]), problems=problems)

def run_cli(cfg, a):
    from rag import ollama_client as oc
    from cli import needed_models
    log = lambda m: print(m, flush=True)
    if a.part == "dataset": r = dataset_report(); print(json.dumps(r, indent=1)); return 1 if r["problems"] else 0
    if a.offline: cfg = deep_merge(cfg, dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False)))
    elif needed_models(cfg):
        miss = oc.missing_models(needed_models(cfg))
        if miss: print("Missing Ollama models. Run:\n" + "\n".join(f"  ollama pull {m}" for m in miss)); return 1
    if a.part != "multiturn": print("give --part multiturn | dataset"); return 2
    res, _ = run_memory_ablation(cfg, set(a.only.split(",")) if a.only else None, a.force, a.offline, log=log); md = memory_markdown(res)
    note = "> OFFLINE STAND-INS (hash embedder, rule-based rewriter): plumbing only, not model results.\n\n" if a.offline else ""
    save_text(note + md + "\n", a.md or "results/s9_memory.md"); save_json([{k: v for k, v in r.items()} for r in res], a.out or "results/s9_memory.json"); print("\n" + md); return 0
