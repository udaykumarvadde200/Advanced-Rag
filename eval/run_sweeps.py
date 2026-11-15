"""Retrieval-component sweeps: chunking x retriever x embedder x reranker, no generation involved.
Metrics are computed at k = 1, 3, 5 from one ranked pool of POOL chunks (a reranker only reorders that pool), so rows are comparable."""
import copy, json, os, time
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

def label(ch): return ch["strategy"] + (f"-{ch['size']}" if ch.get("size") else "") + (f"-ov{ch['overlap']}" if ch.get("overlap") else "")

def run_sweeps(sw, docs, golden, offline=False, retrievers=None, log=print):
    G = [g for g in golden if g["answerable"] and _evs(g)]; retrievers = retrievers or sw["retrievers"]
    need_dense = any(r in ("dense", "hybrid") for r in retrievers)
    embs = ([{"type": "hash"}] if offline else sw["embedders"]) if need_dense else [None]
    rows, types = [], sorted({g["type"] for g in G})
    for ecfg in embs:
        timed = TimedEmbedder(make_embedder(ecfg)) if ecfg else None
        ename = (ecfg.get("model") or ecfg.get("type")) if ecfg else "-"
        for ch in sw["chunking"]:
            chunks = chunk_docs(docs, ch["strategy"], ch.get("size", 300), ch.get("overlap", 0))
            base = Retriever(chunks, timed, "hybrid" if need_dense else "bm25")
            intact, avg = intact_rate(chunks, G), mean(len(c.text) for c in chunks)
            for mode in retrievers:
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
                               n_chunks=len(chunks), avg_chars=round(avg), intact=round(intact, 3), hit1=A("hit", 1), hit3=A("hit", 3), hit5=A("hit", 5),
                               mrr=A("mrr", 5), ndcg5=A("ndcg", 5), ev_recall5=A("evidence_recall", 5),
                               hit3_by_type={t: round(mean(p[3]["hit"] for p in per if p["type"] == t), 3) for t in types},
                               p50_ms=round(pct(lat, 50), 2), p95_ms=round(pct(lat, 95), 2), n_questions=len(G))
                    rows.append(row); log(f"{row['chunking']:18s} {mode:7s} {row['reranker']:8s} MRR={row['mrr']:.3f} hit@3={row['hit3']:.3f} p50={row['p50_ms']}ms")
    return rows

def to_markdown(rows):
    cols = ["chunking", "retriever", "embedder", "reranker", "n_chunks", "intact", "hit1", "hit3", "hit5", "mrr", "ndcg5", "ev_recall5", "p50_ms", "p95_ms"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (-r["mrr"], r["p50_ms"])): out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
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
