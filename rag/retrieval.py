import math, numpy as np
from collections import Counter
from .text import tokens

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokens(c.text) for c in chunks]
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d)); n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + .5) / (f + .5)) for w, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]
    def scores(self, query):
        q, out = tokens(query), np.zeros(len(self.chunks))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for w in q:
                if w in tf: s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out[i] = s
        return out

class Dense:
    def __init__(self, chunks, embedder):
        self.chunks, self.emb = chunks, embedder
        self.M = embedder.embed([c.text for c in chunks])
    def query_vec(self, query): return self.emb.embed([query], cache=False)[0]
    def scores_from(self, qv): return self.M @ qv

def _rank(scores, allowed_idx, k):
    idx = [i for i in allowed_idx]
    idx.sort(key=lambda i: -scores[i])
    return idx[:k]

def rrf(rank_lists, k=60):
    sc = Counter()
    for rl in rank_lists:
        for r, i in enumerate(rl): sc[i] += 1.0 / (k + r + 1)
    return [i for i, _ in sc.most_common()]

class Retriever:
    """mode: bm25 | dense | hybrid. Access control is applied BEFORE ranking (pre-filter)."""
    def __init__(self, chunks, embedder=None, mode="hybrid", enforce_acl=True):
        self.chunks, self.mode, self.acl = chunks, mode, enforce_acl
        self.bm25 = BM25(chunks) if mode in ("bm25", "hybrid") else None
        self.dense = Dense(chunks, embedder) if mode in ("dense", "hybrid") else None
    def retrieve(self, query, roles, k=5, pool=20, trace=None):
        from .telemetry import Trace
        t = trace or Trace(); qv = None
        if self.dense:
            with t.stage("embed_query"): qv = self.dense.query_vec(query)
        with t.stage("rank"):
            allowed = [i for i, c in enumerate(self.chunks) if (not self.acl) or c.access in roles]
            lists = []
            if self.bm25: lists.append(_rank(self.bm25.scores(query), allowed, pool))
            if self.dense: lists.append(_rank(self.dense.scores_from(qv), allowed, pool))
            ranked = lists[0] if len(lists) == 1 else rrf(lists)
            return [self.chunks[i] for i in ranked[:k]] if k else [self.chunks[i] for i in ranked]

class LexicalReranker:
    """Offline stand-in reranker: query-term coverage + bigram overlap."""
    name = "lexical"
    def rerank(self, query, chunks, k):
        q = tokens(query); qb = set(zip(q, q[1:]))
        def s(c):
            t = tokens(c.text); tb = set(zip(t, t[1:]))
            return len(set(q) & set(t)) / max(1, len(set(q))) + 0.5 * len(qb & tb)
        return sorted(chunks, key=lambda c: -s(c))[:k]

class OllamaReranker:
    """Pointwise LLM reranker through Ollama: asks the model to rate each chunk 0-10. Slow (one call per chunk): use a small pool."""
    def __init__(self, model="llama3.2:3b", host=None, keep_alive="5m"):
        from . import ollama_client as oc
        self.oc, self.name, self.model, self.host, self.ka = oc, "ollama:" + model, model, host or oc.DEFAULT_HOST, keep_alive
    def _score(self, q, text):
        import re
        j = self.oc.chat(self.host, self.model, [dict(role="user", content=
            f"Rate how well the passage answers the question, from 0 (irrelevant) to 10 (directly answers). Reply with a single number only.\n\nQuestion: {q}\n\nPassage: {text}")],
            dict(temperature=0, seed=42, num_predict=4), self.ka)
        m = re.search(r"\d+(?:\.\d+)?", j["message"]["content"]); return float(m.group()) if m else 0.0
    def rerank(self, query, chunks, k):
        sc = [self._score(query, c.text) for c in chunks]
        return [c for _, c in sorted(zip(sc, chunks), key=lambda x: -x[0])][:k]

def make_reranker(cfg):
    if not cfg or cfg.get("type", "none") == "none": return None
    if cfg["type"] == "lexical": return LexicalReranker()
    if cfg["type"] == "ollama": return OllamaReranker(cfg["model"], cfg.get("host"), cfg.get("keep_alive", "5m"))
    raise ValueError(f"unknown reranker type '{cfg['type']}' (use none | lexical | ollama)")
