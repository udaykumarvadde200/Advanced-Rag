import math
from .text import tokens

def _evs(g): return g.get("evidences") or ([g["evidence"]] if g.get("evidence") else [])
def chunk_relevant(chunk, g): return any(e.lower() in chunk.text.lower() for e in _evs(g))
def retrieval_metrics(chunks, g, k):
    """Binary relevance: a chunk is relevant if it contains any evidence string of the question.
    evidence_recall = share of the question's evidence strings found in the top-k (matters for multi-hop)."""
    top = chunks[:k]; ev = _evs(g); rel = [chunk_relevant(c, g) for c in top]
    rr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(ev), k)))
    found = sum(any(e.lower() in c.text.lower() for c in top) for e in ev)
    docs = {c.doc_id for c in top}; need = set(g["source_docs"])
    return dict(hit=float(any(rel)), mrr=rr, ndcg=min(1.0, dcg / ideal) if ideal else 0.0, evidence_recall=found / len(ev) if ev else 1.0,
                doc_recall=len(docs & need) / len(need) if need else 1.0, precision=sum(rel) / max(1, len(top)))
def correctness(answer, g):
    a = answer.lower(); return float(all(k.lower() in a for k in g["keywords"])) if g["keywords"] else 0.0
def groundedness(answer, contexts):
    """Deterministic faithfulness proxy: share of answer content tokens that appear in the retrieved context."""
    at = [t for t in tokens(answer) if not t.isdigit() or True]; ct = set(w for c in contexts for w in tokens(c.text))
    at = [t for t in at if not (t.startswith("[") )]
    return sum(t in ct for t in at) / len(at) if at else 1.0
def mean(xs): xs = list(xs); return sum(xs) / len(xs) if xs else 0.0
