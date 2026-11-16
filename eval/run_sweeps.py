"""Retrieval-component sweeps: chunking x retriever x embedder x reranker, no generation involved.
Metrics are computed at k = 1, 3, 5 from one ranked pool of POOL chunks (a reranker only reorders that pool), so rows are comparable."""
import copy, json, os, random, time
import yaml
from rag.pipeline import load_docs, load_golden, ROLES, _read
from rag.chunking import chunk_docs
from rag.embeddings import make_embedder
from rag.retrieval import Retriever, make_reranker
from rag.metrics import retrieval_metrics, mean, _evs
from rag.telemetry import Trace, pct
from rag import ollama_client as oc

POOL, KS = 10, (1, 3, 5)

class TimedEmbedder:
    """Embeds each distinct query once (so sweeps stay fast) but remembers its real embedding latency to report honest per-query timings."""
    def __init__(self, inner): self.inner, self.q, self.name = inner, {}, getattr(inner, "name", "emb")
    def embed(self, texts, cache=True):
        if cache: return self.inner.embed(texts, cache=True)
        key = texts[0]
        if key not in self.q:
            t = time.perf_counter(); v = self.inner.embed(texts, cache=False); self.q[key] = (v, (time.perf_counter() - t) * 1000)
        return self.q[key][0]

def variant(base, mode):
    r = copy.copy(base)
    if mode == "bm25": r.dense = None
    elif mode == "dense": r.bm25 = None
    return r

def intact_rate(chunks, G):
    """Chunking quality independent of the retriever: share of questions whose every evidence string survives whole inside some chunk."""
    texts = [c.text.lower() for c in chunks]
    return mean(all(any(e.lower() in t for t in texts) for e in _evs(g)) for g in G)

def bootstrap_ci(xs, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean (stdlib random, fixed seed -> deterministic)."""
    xs = list(xs); n = len(xs)
    if n == 0: return (0.0, 0.0)
    rng = random.Random(seed); ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (ms[int(n_boot * alpha / 2)], ms[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))])

def paired_ci(a, b, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap CI of mean(a - b) over the same questions (paired: resamples question indices, not rows)."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], n_boot, seed, alpha)

def add_stats(rows, n_boot=2000, seed=0):
    """Adds to every row: mrr_lo/mrr_hi (95% bootstrap CI of MRR), and versus the best BM25 row (highest MRR, then fastest):
    won / lost (questions where this row's reciprocal rank is higher / lower), d_mrr, d_lo, d_hi (paired CI of the MRR difference)
    and sig (True only if the paired CI excludes 0). Rows must carry `rr` = per-question reciprocal rank in a common order."""
    bm = [r for r in rows if r["retriever"] == "bm25"]; base = min(bm, key=lambda r: (-r["mrr"], r["p50_ms"])) if bm else None
    for r in rows:
        r["mrr_lo"], r["mrr_hi"] = (round(v, 3) for v in bootstrap_ci(r["rr"], n_boot, seed))
        if base is None: r.update(won=None, lost=None, d_mrr=None, d_lo=None, d_hi=None, sig=None, vs=None); continue
        lo, hi = paired_ci(r["rr"], base["rr"], n_boot, seed)
        r.update(vs=f"{base['chunking']}/{base['retriever']}/{base['reranker']}", won=sum(x > y for x, y in zip(r["rr"], base["rr"])),
                 lost=sum(x < y for x, y in zip(r["rr"], base["rr"])), d_mrr=round(mean(r["rr"]) - mean(base["rr"]), 3),
                 d_lo=round(lo, 3), d_hi=round(hi, 3), sig=bool(lo > 0 or hi < 0) and r is not base)
    return rows

def label(ch): return ch["strategy"] + (f"-{ch['size']}" if ch.get("size") else "") + (f"-ov{ch['overlap']}" if ch.get("overlap") else "")

def run_sweeps(sw, docs, golden, offline=False, retrievers=None, log=print):
    G = [g for g in golden if g["answerable"] and _evs(g)]; retrievers = retrievers or sw["retrievers"]
    need_dense = any(r in ("dense", "hybrid") for r in retrievers)
    embs = ([{"type": "hash"}] if offline else sw["embedders"]) if need_dense else [None]
    rows, types = [], sorted({g["type"] for g in G})
    for ei, ecfg in enumerate(embs):
        timed = TimedEmbedder(make_embedder(ecfg)) if ecfg else None
        ename = (getattr(timed.inner, "name", None) or ecfg.get("model") or ecfg.get("type")) if ecfg else "-"   # inner.name already says "+prefix" when prefixes are on
        for ch in sw["chunking"]:
            chunks = chunk_docs(docs, ch["strategy"], ch.get("size", 300), ch.get("overlap", 0))
            base = Retriever(chunks, timed, "hybrid" if need_dense else "bm25")
            intact, avg = intact_rate(chunks, G), mean(len(c.text) for c in chunks)
            for mode in retrievers:
                if mode == "bm25" and ei > 0: continue          # BM25 does not use the embedder: report it once, not once per embedder
                r = variant(base, mode)
                for rcfg in sw["rerankers"]:
                    rr = make_reranker(rcfg); per, lat = [], []
                    for g in G:
                        tr = Trace(); got = r.retrieve(g["question"], ROLES[g["role"]], k=POOL, pool=POOL, trace=tr)
                        if rr:
                            with tr.stage("rerank"): got = rr.rerank(g["question"], got, POOL)
                        emb_ms = timed.q[g["question"]][1] if (timed and mode != "bm25") else 0.0
                        lat.append(tr.stages.get("rank", 0) + tr.stages.get("rerank", 0) + emb_ms)
                        per.append(dict({k: retrieval_metrics(got, g, k) for k in KS}, type=g["type"]))
                    A = lambda key, k: round(mean(p[k][key] for p in per), 3)
                    row = dict(chunking=label(ch), retriever=mode, embedder=ename if mode != "bm25" else "-", reranker=rcfg.get("type", "none"),
                               n_chunks=len(chunks), rr=[p[5]["mrr"] for p in per], avg_chars=round(avg), intact=round(intact, 3), hit1=A("hit", 1), hit3=A("hit", 3), hit5=A("hit", 5),
                               mrr=A("mrr", 5), ndcg5=A("ndcg", 5), ev_recall5=A("evidence_recall", 5),
                               hit3_by_type={t: round(mean(p[3]["hit"] for p in per if p["type"] == t), 3) for t in types},
                               p50_ms=round(pct(lat, 50), 2), p95_ms=round(pct(lat, 95), 2), n_questions=len(G))
                    rows.append(row); log(f"{row['chunking']:18s} {mode:7s} {row['reranker']:8s} MRR={row['mrr']:.3f} hit@3={row['hit3']:.3f} p50={row['p50_ms']}ms")
    return add_stats(rows)

def embedder_comparison(rows, n_boot=2000, seed=0):
    """Paired comparison of every embedder against the FIRST embedder in the run, per identical (chunking, retriever, reranker) cell.
    Answers 'does this embedder / this prefix setting help beyond noise?' with the same won/lost + paired-CI logic as add_stats."""
    ref = next((r["embedder"] for r in rows if r["embedder"] != "-"), None); out = []
    for r in rows:
        if r["embedder"] in ("-", ref): continue
        m = next((x for x in rows if x["embedder"] == ref and all(x[k] == r[k] for k in ("chunking", "retriever", "reranker"))), None)
        if m is None: continue
        lo, hi = paired_ci(r["rr"], m["rr"], n_boot, seed)
        out.append(dict(chunking=r["chunking"], retriever=r["retriever"], reranker=r["reranker"], embedder=r["embedder"], ref=ref,
                        won=sum(x > y for x, y in zip(r["rr"], m["rr"])), lost=sum(x < y for x, y in zip(r["rr"], m["rr"])),
                        d_mrr=round(mean(r["rr"]) - mean(m["rr"]), 3), d_lo=round(lo, 3), d_hi=round(hi, 3), sig=bool(lo > 0 or hi < 0)))
    return out

def _cell(r, c):
    if c == "mrr_ci": return f"[{r['mrr_lo']:.3f}, {r['mrr_hi']:.3f}]"
    if c == "won/lost": return "-" if r.get("won") is None else f"{r['won']}/{r['lost']}"
    if c == "d_mrr [95% CI]": return "-" if r.get("d_mrr") is None else f"{r['d_mrr']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]" + (" *" if r.get("sig") else "")
    return str(r.get(c, "-"))

def to_markdown(rows):
    cols = ["chunking", "retriever", "embedder", "reranker", "n_chunks", "intact", "hit1", "hit3", "hit5", "mrr", "mrr_ci", "won/lost", "d_mrr [95% CI]", "ndcg5", "ev_recall5", "p50_ms", "p95_ms"]
    base = next((r["vs"] for r in rows if r.get("vs")), None); n = rows[0]["n_questions"] if rows else 0
    out = [f"n = {n} answerable questions (one question = {1 / n:.3f} of MRR). won/lost and d_mrr compare each row with the best BM25 row ({base or 'none in this run'}), question by question; "
           "`*` marks a paired 95% bootstrap interval that excludes 0 (the only differences to treat as real). 2000 resamples, seed 0.", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] if n else ["(no rows)"]
    for r in sorted(rows, key=lambda r: (-r["mrr"], r["p50_ms"])): out.append("| " + " | ".join(_cell(r, c) for c in cols) + " |")
    cmp = embedder_comparison(rows) if n else []
    if cmp:
        out += ["", f"### Embedder comparison (paired, versus `{cmp[0]['ref']}`, same chunking/retriever/reranker; `*` = interval excludes 0)", "",
                "| embedder | chunking | retriever | reranker | won/lost | d_mrr [95% CI] |", "|---|---|---|---|---|---|"]
        for c in cmp: out.append(f"| {c['embedder']} | {c['chunking']} | {c['retriever']} | {c['reranker']} | {c['won']}/{c['lost']} | {c['d_mrr']:+.3f} [{c['d_lo']:+.3f}, {c['d_hi']:+.3f}]" + (" *" if c["sig"] else "") + " |")
    return "\n".join(out)

def run_from_config(path, out_prefix, offline=False, retrievers=None):
    sw = yaml.safe_load(_read(path)); retrievers = retrievers or sw["retrievers"]
    need = []
    if not offline and any(r in ("dense", "hybrid") for r in retrievers): need += [e["model"] for e in sw["embedders"] if e.get("type") == "ollama"]
    need += [r["model"] for r in sw["rerankers"] if r.get("type") == "ollama"]
    miss = oc.missing_models(sorted(set(need))) if need else []
    if miss: print("Missing Ollama models. Run:\n" + "\n".join(f"  ollama pull {m}" for m in miss)); return 1
    rows = run_sweeps(sw, load_docs(), load_golden(), offline, retrievers)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(out_prefix + ".json", "w", encoding="utf-8") as f: json.dump(rows, f, indent=1)
    with open(out_prefix + ".md", "w", encoding="utf-8") as f: f.write(to_markdown(rows) + "\n")
    print(f"\nsaved -> {out_prefix}.json and {out_prefix}.md ({len(rows)} configs)"); return 0
