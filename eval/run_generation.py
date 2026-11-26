"""S5 - generation evaluation and failure attribution.   python cli.py gen [--model M --prompt P --k N --order O] | --grid | --merge a.json b.json

For every golden question, two runs with the SAME generator: (a) GOLD context = the chunks that hold the evidence, (b) RETRIEVED context = what the pipeline retrieves.
The gap between them is retrieval's cost. Every wrong answer of the retrieved run is attributed to ONE cause by a fixed, deterministic rule (`classify`).
Sweeps change one factor at a time (model / prompt / k / context order) and compare each variant with the reference by paired won/lost + exact sign test.
Offline (`--offline`) uses the hash embedder + MockLLM: it proves the plumbing only, never model quality."""
import functools, json, math, os, re, time
from rag.pipeline import RAG, load_golden, load_docs, deep_merge, ROLES
from rag.chunking import Chunk
from rag.metrics import (retrieval_metrics, correctness, correctness_norm, groundedness, groundedness_clean, ungrounded_tokens, is_soft_abstain, mean)
from rag.llm import is_abstain
from rag.prompts import get_system
from rag.telemetry import est_cost_usd
from rag.text import tokens

ORDERS = ("rank", "reverse", "doc")
INJECTION_RE = re.compile(r"ignore (all |the )?(previous|prior|above)|disregard (your|all|the)|system notice|you must now (reply|respond)|reveal your system prompt", re.I)
CATEGORIES = ("chunking", "retrieval_miss", "ranking", "abstention_error", "generation_error", "metric_artifact")

class InvalidComparison(Exception): pass

# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))

def sign_test_p(won, lost):
    """Exact two-sided sign test on the discordant pairs (ties are ignored)."""
    n = won + lost
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(won, lost) + 1)) / 2 ** n)

def pctl(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))], 1) if xs else 0.0

# ---------------------------------------------------------------- contexts
@functools.lru_cache(maxsize=1)
def _doc_text(): return {d["id"]: d["text"] for d in load_docs()}

def gold_context(chunks, g):
    """Gold context = for each evidence string, the chunk (of a source document when possible) that holds it.
    Returns (contexts, intact). intact=False when some evidence string is split across chunks: the source documents are then passed whole
    (the failure is a CHUNKING problem and `classify` says so). Unanswerable questions have no evidence -> empty context."""
    if not g["answerable"]: return [], True
    src, out, intact = set(g["source_docs"]), [], True
    for e in g["evidences"]:
        el = e.lower()
        hit = next((c for c in chunks if c.doc_id in src and el in c.text.lower()), None) or next((c for c in chunks if el in c.text.lower()), None)
        if hit is None: intact = False
        elif hit.id not in {x.id for x in out}: out.append(hit)
    if not intact:
        out = []
        for d in g["source_docs"]:
            cs = [c for c in chunks if c.doc_id == d]                       # the ORIGINAL document text (joining chunks would insert text inside a split evidence phrase)
            if cs: out.append(Chunk(id=f"{d}#all", doc_id=d, text=_doc_text().get(d) or "".join(c.text for c in cs), access=cs[0].access))
    return out, intact

def order_contexts(ctx, order):
    if order == "rank": return list(ctx)
    if order == "reverse": return list(reversed(ctx))
    if order == "doc": return sorted(ctx, key=lambda c: (c.doc_id, c.id))
    raise ValueError(f"unknown context order {order!r}; choose from {ORDERS}")

def _memo(rag, key, fn):
    m = rag.__dict__.setdefault("_gen_memo", {})
    if key not in m: m[key] = fn()
    return m[key]

def retrieved(rag, g): return _memo(rag, ("top", g["question"], g["role"]), lambda: rag.retrieve(g["question"], g["role"]))
def pool_list(rag, g):
    """First-stage candidates (depth = the config's pool): evidence found here but not in the top-k is a RANKING failure, absent here a retrieval miss."""
    d = rag.cfg["retriever"].get("pool", 20)
    return _memo(rag, ("pool", g["question"], g["role"]), lambda: rag.retriever.retrieve(g["question"], ROLES[g["role"]], k=d, pool=d))

def ev_found(chunks, g): return sum(any(e.lower() in c.text.lower() for c in chunks) for e in g["evidences"])
def ev_ranks(chunks, g): return [next((i + 1 for i, c in enumerate(chunks) if e.lower() in c.text.lower()), None) for e in g["evidences"]]

# ---------------------------------------------------------------- scoring one answer
def near_miss(question, top1, thr):
    q = set(tokens(question)); return bool(q and top1 is not None and len(q & set(tokens(top1.text))) / len(q) >= thr)

def score_row(g, text, ctx, k):
    a = dict(correct=correctness(text, g), correct_norm=correctness_norm(text, g)) if g["answerable"] else {}
    ug = ungrounded_tokens(text, ctx)
    return dict(id=g["id"], type=g["type"], answerable=g["answerable"], answer=text, abstained=float(is_abstain(text)), soft_abstained=float(is_soft_abstain(text)),
                grounded=groundedness(text, ctx), grounded_clean=groundedness_clean(text, ctx), ungrounded=ug[:8], ungrounded_digits=sum(t.isdigit() for t in ug), **a)

def gen_once(llm, system, q, ctx):
    t = time.perf_counter(); out = llm.generate(q, ctx, system=system); ms = (time.perf_counter() - t) * 1000
    return out["text"], dict(ms=round(ms, 1), tokens_in=out["tokens_in"], tokens_out=out["tokens_out"], cost=est_cost_usd(out["tokens_in"], out["tokens_out"]))

def run_mode(rag, G, mode, system, order, log=lambda m: None, near_thr=0.5):
    k = rag.cfg["retriever"].get("k", 4); rows = []
    gen_once(rag.llm, system, "warm up", retrieved(rag, dict(question="warm up", role="guest")) if mode == "retrieved" else [])   # same throwaway request as cli.py eval (guest, retrieved context)
    for i, g in enumerate(G, 1):
        if mode == "gold": got, intact = gold_context(rag.chunks, g)
        else: got, intact = retrieved(rag, g), None
        text, m = gen_once(rag.llm, system, g["question"], order_contexts(got, order))
        row = score_row(g, text, got, k); row.update(m)
        if mode == "gold": row["gold_intact"] = intact
        else:
            if g["answerable"]: row.update(retrieval_metrics(got, g, k))
            else: row["near_miss"] = near_miss(g["question"], got[0] if got else None, near_thr)
            row["injection_in_context"] = any(INJECTION_RE.search(c.text) for c in got)
        rows.append(row); log(f"  [{mode} {i}/{len(G)}] {g['id']} {m['ms']:.0f} ms")
    return rows

# ---------------------------------------------------------------- failure attribution
def classify(f, hall_below=0.5):
    """One cause per failure, first matching rule wins (a fixed order, so counts add up):
    chunking (evidence not intact in ANY chunk) > retrieval_miss (evidence not even in the first-stage pool) > ranking (in the pool, not in the top-k)
    > metric_artifact (strict keyword check says wrong, normalised check says right; or a soft abstention the strict detector missed)
    > abstention_error (answerable but abstained / unanswerable but answered) > generation_error (the context held the evidence and the answer is still wrong).
    Returns None when the answer is fine, else (category, subtype)."""
    if not f["answerable"]:
        if f["abstained"]: return None
        return ("metric_artifact", "soft_abstain") if f["soft_abstained"] else ("abstention_error", "answered_unanswerable")
    if f["correct"]: return None
    if not f["intact"]: return ("chunking", "evidence_split")
    if f["ev_top"] < f["n_evid"]: return ("ranking", "in_pool_not_in_top_k") if f["ev_pool"] == f["n_evid"] else ("retrieval_miss", "not_in_pool")
    if f["correct_norm"]: return ("metric_artifact", "strict_keyword")
    if f["abstained"] or f["soft_abstained"]: return ("abstention_error", "false_abstain")
    if f["ungrounded_digits"] or f["grounded_clean"] < hall_below: return ("generation_error", "hallucination_candidate")
    return ("generation_error", "incomplete_or_misread")

def attribute(rag, G, ret_rows, gold_rows=None, hall_below=0.5):
    out = []; gold = {r["id"]: r for r in gold_rows or []}; by = {r["id"]: r for r in ret_rows}
    for g in G:
        r = by[g["id"]]; _, intact = gold_context(rag.chunks, g)
        top = retrieved(rag, g); pool = pool_list(rag, g)
        f = dict(answerable=g["answerable"], abstained=r["abstained"], soft_abstained=r["soft_abstained"], correct=r.get("correct"), correct_norm=r.get("correct_norm"),
                 intact=intact, ev_top=ev_found(top, g), ev_pool=ev_found(pool, g), n_evid=len(g["evidences"]), ungrounded_digits=r["ungrounded_digits"], grounded_clean=r["grounded_clean"])
        c = classify(f, hall_below)
        if c is None: continue
        gr = gold.get(g["id"]); gold_ok = None if gr is None else bool(gr["abstained"] if not g["answerable"] else gr["correct"])
        out.append(dict(id=g["id"], type=g["type"], category=c[0], subtype=c[1], gold_ok=gold_ok, injection_in_context=r.get("injection_in_context", False),
                        evidence_ranks_in_pool=ev_ranks(pool, g), near_miss=r.get("near_miss"), answer=r["answer"]))
    return out

def attribution_counts(att, types):
    t = {c: {ty: 0 for ty in types} for c in CATEGORIES}
    for a in att: t[a["category"]][a["type"]] += 1
    return t

# ---------------------------------------------------------------- summaries
def summarize(rows):
    A = [r for r in rows if r["answerable"]]; U = [r for r in rows if not r["answerable"]]
    tp = sum(r["abstained"] for r in U); fp = sum(r["abstained"] for r in A); stp = sum(r["soft_abstained"] for r in U); sfp = sum(r["soft_abstained"] for r in A)
    nm = [r for r in U if r.get("near_miss")]
    types = sorted({r["type"] for r in A}); ms = [r["ms"] for r in rows]
    s = dict(n_answerable=len(A), n_unanswerable=len(U), correct=mean(r["correct"] for r in A), correct_norm=mean(r["correct_norm"] for r in A),
             grounded=mean(r["grounded"] for r in A), grounded_clean=mean(r["grounded_clean"] for r in A),
             by_type={t: dict(n=sum(r["type"] == t for r in A), correct=int(sum(r["correct"] for r in A if r["type"] == t)), correct_norm=int(sum(r["correct_norm"] for r in A if r["type"] == t))) for t in types},
             abstention=dict(tp=int(tp), fp=int(fp), fn=int(len(U) - tp), precision=(tp / (tp + fp) if tp + fp else None), recall=(tp / len(U) if U else None), recall_ci=wilson(int(tp), len(U)),
                             soft_precision=(stp / (stp + sfp) if stp + sfp else None), soft_recall=(stp / len(U) if U else None),
                             near_miss_n=len(nm), near_miss_recall=(sum(r["abstained"] for r in nm) / len(nm) if nm else None),
                             other_n=len(U) - len(nm) if any("near_miss" in r for r in U) else None,
                             other_recall=(sum(r["abstained"] for r in U if "near_miss" in r and not r["near_miss"]) / (len(U) - len(nm)) if any("near_miss" in r for r in U) and len(U) > len(nm) else None)),
             tokens_in=mean(r["tokens_in"] for r in rows), tokens_out=mean(r["tokens_out"] for r in rows), gen_p50_ms=pctl(ms, 50), gen_p95_ms=pctl(ms, 95), est_cost_usd=sum(r["cost"] for r in rows))
    s["correct_ci"] = wilson(int(sum(r["correct"] for r in A)), len(A)); s["correct_norm_ci"] = wilson(int(sum(r["correct_norm"] for r in A)), len(A))
    if A and "hit" in A[0]: s.update(hit=mean(r["hit"] for r in A), mrr=mean(r["mrr"] for r in A), evidence_recall=mean(r["evidence_recall"] for r in A))
    return s

def gap_table(ret, gold):
    g_by = {r["id"]: r for r in gold}; types = sorted({r["type"] for r in ret if r["answerable"]}); out = {}
    for t in types + ["ALL"]:
        P = [r for r in ret if r["answerable"] and (t == "ALL" or r["type"] == t)]
        out[t] = dict(n=len(P), gold=int(sum(g_by[r["id"]]["correct"] for r in P)), retrieved=int(sum(r["correct"] for r in P)), gold_norm=int(sum(g_by[r["id"]]["correct_norm"] for r in P)),
                      retrieved_norm=int(sum(r["correct_norm"] for r in P)), retrieval_cost=sum(1 for r in P if g_by[r["id"]]["correct"] and not r["correct"]),
                      retrieval_luck=sum(1 for r in P if r["correct"] and not g_by[r["id"]]["correct"]))
    return out

# ---------------------------------------------------------------- running configs
def models_needed(cfg):
    out = []
    if cfg["embedder"].get("type") == "ollama" and cfg["retriever"]["mode"] != "bm25": out.append(cfg["embedder"]["model"])
    if cfg["llm"].get("type") == "ollama": out.append(cfg["llm"]["model"])
    if cfg.get("reranker", {}).get("type") == "ollama": out.append(cfg["reranker"]["model"])
    return sorted(set(out))

def spec_label(s): return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{s['model']}_{s['prompt']}_k{s['k']}_{s['order']}")

def variant_cfg(cfg, spec, offline=False, ref_model=None):
    v = deep_merge(cfg, dict(retriever=dict(k=spec["k"])))
    if offline:
        if ref_model and spec["model"] != ref_model: v["llm"] = dict(type="mock", naive=False, threshold=0.7)   # stand-in "other model": plumbing only
    else: v["llm"] = dict(v["llm"], model=spec["model"])
    return v

def run_config(cfg, spec, G, modes, log=lambda m: None, att_cfg=None, rag=None):
    att_cfg = att_cfg or {}; rag = rag or RAG(cfg); system = get_system(spec["prompt"]); res = {}
    for mode in ("retrieved", "gold"):                                       # retrieved FIRST: every variant's retrieved run then has the same request history as cli.py eval
        if mode in modes:
            log(f"{spec_label(spec)}: {mode} context ({len(G)} questions)")
            rows = run_mode(rag, G, mode, system, spec["order"], log, att_cfg.get("near_miss_overlap", 0.5)); res[mode] = dict(rows=rows, summary=summarize(rows))
    run = dict(label=spec_label(spec), spec=dict(spec), modes=res)
    if "retrieved" in res:
        run["attribution"] = attribute(rag, G, res["retrieved"]["rows"], res["gold"]["rows"] if "gold" in res else None, att_cfg.get("hallucination_grounded_below", 0.5))
    if "gold" in res and "retrieved" in res: run["gap"] = gap_table(res["retrieved"]["rows"], res["gold"]["rows"])
    return run

# ---------------------------------------------------------------- comparing variants
def _pairs(ref_rows, rows, key, ids=None):
    r = {x["id"]: x for x in ref_rows}; won = lost = 0
    for x in rows:
        if x["id"] not in r or (ids is not None and x["id"] not in ids): continue
        a, b = r[x["id"]][key], x[key]
        won += (b > a); lost += (b < a)
    return int(won), int(lost)

def compare(ref, run, known=(), rules=None):
    """Paired comparison of one variant with the reference (retrieved-context run) + the adoption rules R1-R5 of configs/generation.yaml."""
    R, V = ref["modes"]["retrieved"], run["modes"]["retrieved"]
    if [r["id"] for r in R["rows"]] != [r["id"] for r in V["rows"]]: raise InvalidComparison("the two runs cover different questions")
    A = {r["id"] for r in R["rows"] if r["answerable"]}; U = {r["id"] for r in R["rows"] if not r["answerable"]}; rest = A - set(known)
    sw, sl = _pairs(R["rows"], V["rows"], "correct", A); nw, nl = _pairs(R["rows"], V["rows"], "correct_norm", A); aw, al = _pairs(R["rows"], V["rows"], "abstained", U)
    rw, rl = _pairs(R["rows"], V["rows"], "correct", rest); kw, kl = _pairs(R["rows"], V["rows"], "correct", set(known) & A)
    p = sign_test_p(sw, sl); lat = (V["summary"]["gen_p95_ms"] / R["summary"]["gen_p95_ms"]) if R["summary"]["gen_p95_ms"] else 1.0
    c = dict(label=run["label"], spec=run["spec"], strict=dict(won=sw, lost=sl, p=p), norm=dict(won=nw, lost=nl), abstention=dict(won=aw, lost=al),
             rest=dict(won=rw, lost=rl), known=dict(won=kw, lost=kl), correct=V["summary"]["correct"], correct_norm=V["summary"]["correct_norm"],
             abst_recall=V["summary"]["abstention"]["recall"], p95_ratio=round(lat, 2), tokens_in=V["summary"]["tokens_in"], cost=V["summary"]["est_cost_usd"])
    if "gold" in ref["modes"] and "gold" in run["modes"]:
        gw, gl = _pairs(ref["modes"]["gold"]["rows"], run["modes"]["gold"]["rows"], "correct", A); c["gold"] = dict(won=gw, lost=gl, p=sign_test_p(gw, gl), correct=run["modes"]["gold"]["summary"]["correct"])
    failed = [n for n, ok in (("R1 strict net>0 and p<0.05", sw > sl and p < 0.05), ("R2 normalised net>0", nw > nl), ("R3 abstention net>=-1", aw - al >= -1),
                              ("R4 net>=0 outside the known failures", rw - rl >= 0), ("R5 p95 latency <=1.5x", lat <= 1.5)) if not ok]
    c["verdict"] = "ADOPTION CANDIDATE (still run `cli.py regress --compare`)" if not failed else "keep reference: " + "; ".join(failed)
    return c

def check_comparable(runs):
    base = runs[0]
    for r in runs[1:]:
        for m in set(base["modes"]) & set(r["modes"]):
            if [x["id"] for x in base["modes"][m]["rows"]] != [x["id"] for x in r["modes"][m]["rows"]]: raise InvalidComparison(f"{r['label']}: different question set than {base['label']}")
    if len({bool(r.get("offline")) for r in runs}) > 1: raise InvalidComparison("offline stand-in runs and real runs cannot be compared")

# ---------------------------------------------------------------- reports
def _f(x, d=3): return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def _frac(k, n): return f"{k}/{n} ({k / n:.3f})" if n else "-"
OFFLINE_WARN = "> **OFFLINE STAND-IN RUN (hash embedder + MockLLM): this checks the plumbing only. None of these numbers describes a real model.**\n"

def md_run(run, offline=False):
    L = [f"# S5 generation: {run['label']}", ""] + ([OFFLINE_WARN] if offline else [])
    L += [f"Spec: model `{run['spec']['model']}`, prompt `{run['spec']['prompt']}`, k {run['spec']['k']}, context order `{run['spec']['order']}`.", ""]
    ret = run["modes"].get("retrieved"); gold = run["modes"].get("gold")
    if "gap" in run:
        L += ["## A. Gold context vs retrieved context (answerable questions; the gap is retrieval's cost)", "",
              "| type | n | gold strict | retrieved strict | gold norm | retrieved norm | gold right, retrieved wrong | retrieved right, gold wrong |", "|---|---|---|---|---|---|---|---|"]
        for t, g in run["gap"].items(): L.append(f"| {t} | {g['n']} | {g['gold']} | {g['retrieved']} | {g['gold_norm']} | {g['retrieved_norm']} | {g['retrieval_cost']} | {g['retrieval_luck']} |")
        L += [""]
    if ret:
        types = sorted({r["type"] for r in ret["rows"]}); att = run.get("attribution", []); tab = attribution_counts(att, types)
        L += ["## B. Failure attribution (retrieved-context run; one cause per failure)", "", "| cause | " + " | ".join(types) + " | total |", "|---|" + "---|" * (len(types) + 1)]
        for c in CATEGORIES: L.append(f"| {c} | " + " | ".join(str(tab[c][t]) for t in types) + f" | {sum(tab[c].values())} |")
        L.append(f"| **all failures** | " + " | ".join(str(sum(tab[c][t] for c in CATEGORIES)) for t in types) + f" | {len(att)} |")
        ge = [a for a in att if a["category"] == "generation_error"]
        L += ["", f"Generation errors: {sum(1 for a in ge if a['gold_ok'] is False)} also fail with gold context (the model cannot do it), {sum(1 for a in ge if a['gold_ok'] is True)} pass with gold context (distractor-induced), "
              f"{sum(1 for a in ge if a['gold_ok'] is None)} untested. Subtypes: " + (", ".join(f"{s} {sum(1 for a in ge if a['subtype'] == s)}" for s in sorted({a['subtype'] for a in ge})) or "none") + ". "
              f"Failures with an injection phrase in the retrieved context: {sum(1 for a in att if a['injection_in_context'])}.", "",
              "## C. Every failure", "", "| id | type | cause | subtype | gold ok | injection in ctx | evidence rank in pool | answer |", "|---|---|---|---|---|---|---|---|"]
        for a in att: L.append(f"| {a['id']} | {a['type']} | {a['category']} | {a['subtype']} | {_f(a['gold_ok'])} | {a['injection_in_context']} | {a['evidence_ranks_in_pool']} | {a['answer'][:90].replace('|', '/').replace(chr(10), ' ')} |")
        L += [""]
    L += ["## D. Metrics", "", "| metric | " + " | ".join(m for m in ("gold", "retrieved") if m in run["modes"]) + " |", "|---|" + "---|" * len(run["modes"])]
    def row(name, fn): L.append(f"| {name} | " + " | ".join(fn(run["modes"][m]["summary"]) for m in ("gold", "retrieved") if m in run["modes"]) + " |")
    row("correctness, strict keywords", lambda s: f"{_frac(round(s['correct'] * s['n_answerable']), s['n_answerable'])} CI {s['correct_ci']}")
    row("correctness, normalised", lambda s: f"{_frac(round(s['correct_norm'] * s['n_answerable']), s['n_answerable'])} CI {s['correct_norm_ci']}")
    row("groundedness (old proxy)", lambda s: _f(s["grounded"])); row("groundedness (citation-stripped)", lambda s: _f(s["grounded_clean"]))
    row("abstention recall (unanswerable)", lambda s: f"{s['abstention']['tp']}/{s['n_unanswerable']} CI {s['abstention']['recall_ci']}")
    row("abstention precision (abstentions that were right)", lambda s: _frac(s["abstention"]["tp"], s["abstention"]["tp"] + s["abstention"]["fp"]))
    row("abstained on an answerable question", lambda s: str(s["abstention"]["fp"]))
    row("abstention recall, soft detector", lambda s: _f(s["abstention"]["soft_recall"]))
    if ret:
        a = ret["summary"]["abstention"]; row("near-miss unanswerables: abstained", lambda s: _frac(round((s["abstention"]["near_miss_recall"] or 0) * s["abstention"]["near_miss_n"]), s["abstention"]["near_miss_n"]) if s["abstention"]["near_miss_n"] else "-")
        L.append("| retrieval hit / MRR / evidence recall | " + " | ".join(("-" if m == "gold" else f"{_f(run['modes'][m]['summary'].get('hit'))} / {_f(run['modes'][m]['summary'].get('mrr'))} / {_f(run['modes'][m]['summary'].get('evidence_recall'))}") for m in ("gold", "retrieved") if m in run["modes"]) + " |")
    row("tokens in / out (mean)", lambda s: f"{s['tokens_in']:.0f} / {s['tokens_out']:.0f}"); row("generate ms p50 / p95", lambda s: f"{s['gen_p50_ms']} / {s['gen_p95_ms']}"); row("est. cost, whole run (USD)", lambda s: f"{s['est_cost_usd']:.4f}")
    L += ["", "Near-miss = the top-1 retrieved chunk shares at least half of the question's content words (a proxy, not a label).", ""]
    if ret:
        diff = [r for r in ret["rows"] if r["answerable"] and r["correct"] != r["correct_norm"]]
        L += ["## E. Strict vs normalised correctness (metric artifacts)", ""] + ([f"- {r['id']}: strict {int(r['correct'])}, normalised {int(r['correct_norm'])}: {r['answer'][:110]!r}" for r in diff] or ["- no answer differs between the two checks"]) + [""]
    return "\n".join(L)

def md_compare(runs, comps, offline=False, known=()):
    ref = runs[0]; L = ["# S5 variant comparison (paired against the reference)", ""] + ([OFFLINE_WARN] if offline else [])
    L += [f"Reference: `{ref['label']}`. Known failures excluded from R4: {', '.join(known) or 'none'}. Smallest significant change at n=69 answerable: 6 won / 0 lost (exact sign test).", "",
          "| variant | factor | strict correct | won/lost | p | norm won/lost | abstention won/lost | outside known failures | gold-ctx won/lost | p95 x | tokens in | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        s = c["spec"]; fac = ", ".join(f"{k}={s[k]}" for k in ("model", "prompt", "k", "order") if s[k] != ref["spec"][k]) or "(reference)"
        g = c.get("gold"); L.append(f"| {c['label']} | {fac} | {_f(c['correct'])} | {c['strict']['won']}/{c['strict']['lost']}{' *' if c['strict']['p'] < 0.05 else ''} | {c['strict']['p']:.3f} | {c['norm']['won']}/{c['norm']['lost']} | "
                                    f"{c['abstention']['won']}/{c['abstention']['lost']} | {c['rest']['won']}/{c['rest']['lost']} | {(str(g['won']) + '/' + str(g['lost'])) if g else '-'} | {c['p95_ratio']} | {c['tokens_in']:.0f} | {c['verdict']} |")
    L += ["", "`*` = sign-test p < 0.05 (in EITHER direction: a starred variant can be significantly worse). The variants were designed after seeing the S1/S2 failures on this same golden set, so rule R4 (outside the known failures) is the fairer test.", ""]
    return "\n".join(L)

def console_summary(run):
    for m, d in run["modes"].items():
        s = d["summary"]; a = s["abstention"]
        print(f"[{m:9s}] strict {s['correct']:.3f}  norm {s['correct_norm']:.3f}  grounded {s['grounded']:.3f}->{s['grounded_clean']:.3f}  abstain recall {a['tp']}/{s['n_unanswerable']}  false abstain {a['fp']}  gen p50 {s['gen_p50_ms']} ms  tokens {s['tokens_in']:.0f}/{s['tokens_out']:.0f}")
    if "gap" in run: g = run["gap"]["ALL"]; print(f"gold vs retrieved (strict): {g['gold']} vs {g['retrieved']} of {g['n']}; retrieval cost {g['retrieval_cost']} questions")
    if "attribution" in run:
        n = {}
        for a in run["attribution"]: n[a["category"]] = n.get(a["category"], 0) + 1
        print("failures:", n or "none")

def show_cases(run, G, ids):
    by = {g["id"]: g for g in G}; att = {a["id"]: a for a in run.get("attribution", [])}
    for i in ids:
        g = by[i]; print(f"\n=== {i} [{g['type']}] role={g['role']}\nQ: {g['question']}\nreference: {g['reference']} | keywords: {g['keywords']} | evidence: {g['evidences']}")
        for m, d in run["modes"].items():
            r = next(x for x in d["rows"] if x["id"] == i)
            print(f"  {m:9s} -> {r['answer']!r}\n            strict {r.get('correct')} norm {r.get('correct_norm')} abstained {r['abstained']} ungrounded {r['ungrounded']}")
        print("  attribution:", (att[i]["category"], att[i]["subtype"], "injection in ctx" if att[i]["injection_in_context"] else "") if i in att else "ok")

# ---------------------------------------------------------------- CLI
def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, ensure_ascii=False)

def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(text)

def load_runs(paths):
    runs, off = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f: d = json.load(f)
        for r in d["runs"]: runs.append(r); off.append(bool(d.get("offline")))
    for r, o in zip(runs, off): r["offline"] = o
    return runs

def sweep_specs(ref, factors):
    specs = [(dict(ref), ("gold", "retrieved"), "reference")]
    for fac, d in (factors or {}).items():
        for v in d["values"]: specs.append((dict(ref, **{fac: v}), tuple(d["modes"]), fac))
    return specs

def run_cli(cfg, a):
    import yaml
    from rag import ollama_client as oc
    log = lambda m: print(m, flush=True)
    gcfg = {}
    if os.path.exists(a.gen_config):
        with open(a.gen_config, encoding="utf-8") as f: gcfg = yaml.safe_load(f) or {}
    offline = bool(a.offline)
    if offline: cfg = deep_merge(cfg, dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False)))
    known = list(gcfg.get("known_failures", [])); att_cfg = gcfg.get("attribution", {})
    if a.merge:                                                                          # compare saved runs, no model needed
        runs = load_runs(a.merge)
        if a.label:
            runs.sort(key=lambda r: r["label"] != a.label)
            if runs[0]["label"] != a.label: raise SystemExit(f"--label {a.label!r} is not among the saved runs: {[r['label'] for r in runs]}")
        try: check_comparable(runs); comps = [compare(runs[0], r, known) for r in runs[1:] if "retrieved" in r["modes"]]
        except InvalidComparison as e: print("INVALID:", e); return 2
        md = md_compare(runs, comps, offline=runs[0].get("offline"), known=known); out = a.md or "results/gen_compare.md"; _write(out, md); print(md); print("saved ->", out); return 0
    G = load_golden(); G = [g for g in G if g["id"] in set(a.ids.split(","))] if a.ids else G
    if a.limit: G = G[:a.limit]
    ref = dict(model=cfg["llm"].get("model", "mock"), prompt="default", k=cfg["retriever"].get("k", 4), order="rank")
    if a.grid:
        ref.update(gcfg.get("reference", {}))
        if ref["model"] != cfg["llm"].get("model") and not offline: log(f"NOTE: the grid reference model ({ref['model']}) comes from {a.gen_config}; the default config uses {cfg['llm'].get('model')}")
    else:
        for key in ("model", "prompt", "k", "order"):
            if getattr(a, key, None) is not None: ref[key] = getattr(a, key)
    for s_ in [ref]: get_system(s_["prompt"]); order_contexts([], s_["order"])           # fail early on a typo
    only = {"both": ("gold", "retrieved"), "gold": ("gold",), "retrieved": ("retrieved",)}[a.modes or "both"]
    plan = sweep_specs(ref, gcfg.get("factors")) if a.grid else [(ref, ("gold", "retrieved"), "single")]
    plan = [(sp, tuple(m for m in md if m in only), fac) for sp, md, fac in plan]; plan = [x for x in plan if x[1]]
    if a.grid: plan.sort(key=lambda p: p[0]["model"] != ref["model"])                    # keep the reference model's runs together
    runs = []
    for spec, modes, fac in plan:
        vcfg = variant_cfg(cfg, spec, offline, ref["model"])
        if not offline:
            miss = oc.missing_models(models_needed(vcfg), vcfg["llm"].get("host") or vcfg["embedder"].get("host"))
            if miss:
                msg = "Missing Ollama models: " + ", ".join(miss) + "\n" + "\n".join(f"  ollama pull {m}" for m in miss)
                if a.skip_missing and fac != "reference": log("SKIPPED " + spec_label(spec) + "\n" + msg); continue
                print(msg); return 1
        runs.append(run_config(vcfg, spec, G, modes, log, att_cfg, rag=None)); runs[-1]["offline"] = offline; console_summary(runs[-1])
    if a.ids or a.show:
        for r in runs: print(f"\n##### {r['label']}"); show_cases(r, G, (a.ids.split(",") if a.ids else known))
    out = a.out or ("results/gen_sweep.json" if a.grid else f"results/gen_{runs[0]['label']}.json"); md_path = a.md or out[:-5] + ".md"
    _save(out, dict(kind="gen", offline=offline, n_questions=len(G), runs=runs))
    text = md_run(runs[0], offline)
    if a.grid and len(runs) > 1:
        comps = [compare(runs[0], r, known) for r in runs[1:] if "retrieved" in r["modes"]]; text += "\n\n" + md_compare(runs, comps, offline, known)
        for c in comps: print(f"{c['label']:42s} strict {c['strict']['won']}/{c['strict']['lost']} p={c['strict']['p']:.3f}  -> {c['verdict']}")
    _write(md_path, text); print("saved ->", out, "and", md_path); return 0
