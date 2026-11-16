import hashlib, os, pickle, re, numpy as np
from .text import tokens
from . import ollama_client as oc

class HashEmbedder:
    """Offline, dependency-free embedder (feature hashing). Used only for tests / no-Ollama dry runs."""
    name = "hash"
    def __init__(self, dim=512): self.dim = dim
    def _h(self, s): return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    def embed(self, texts, cache=True):
        M = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            tk = tokens(t)
            for f in tk + [a + "_" + b for a, b in zip(tk, tk[1:])]: M[r, self._h(f) % self.dim] += 1.0
        n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
        return M / n

class EmbCache:
    """Disk cache keyed by sha256(text) per model, so re-running never re-embeds unchanged text."""
    def __init__(self, cache_dir, model):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "emb_" + re.sub(r"[^A-Za-z0-9_.-]", "_", model) + ".pkl")
        self.d, self.dirty = {}, False
        if os.path.exists(self.path):
            with open(self.path, "rb") as f: self.d = pickle.load(f)
    @staticmethod
    def key(t): return hashlib.sha256(t.encode()).hexdigest()
    def get(self, t): return self.d.get(self.key(t))
    def put(self, t, v): self.d[self.key(t)] = v; self.dirty = True
    def save(self):
        if self.dirty:
            with open(self.path, "wb") as f: pickle.dump(self.d, f)
            self.dirty = False

class OllamaEmbedder:
    """Local Ollama embeddings (e.g. nomic-embed-text, mxbai-embed-large, bge-m3). Batched + disk-cached.
    Optional task prefixes (nomic-embed-text was trained with 'search_document: ' / 'search_query: '): doc_prefix is added to texts embedded with
    cache=True (chunks), query_prefix to texts embedded with cache=False (queries) - the convention retrieval.py already follows. The prefix is applied
    BEFORE the cache lookup, so prefixed and unprefixed vectors never collide in the disk cache."""
    def __init__(self, model="nomic-embed-text", host=None, cache_dir=".cache", batch=32, query_prefix="", doc_prefix=""):
        self.name = self.model = model; self.host = host or oc.DEFAULT_HOST; self.batch = batch; self.query_prefix, self.doc_prefix = query_prefix or "", doc_prefix or ""
        if self.query_prefix or self.doc_prefix: self.name = model + "+prefix"
        self.cache = EmbCache(cache_dir, model); self.api_calls = 0
    def embed(self, texts, cache=True):
        """cache=False bypasses the disk cache (used for queries, so latency measurements stay honest)."""
        pre = self.doc_prefix if cache else self.query_prefix; texts = [pre + t for t in texts]
        vecs, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            v = self.cache.get(t) if cache else None
            if v is None: todo.append(i)
            else: vecs[i] = v
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]; out = oc.embed(self.host, self.model, [texts[i] for i in idx]); self.api_calls += 1
            for i, v in zip(idx, out):
                vecs[i] = np.asarray(v, dtype=np.float32)
                if cache: self.cache.put(texts[i], vecs[i])
        self.cache.save()
        M = np.stack(vecs); n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
        return M / n

def make_embedder(cfg):
    kind = cfg.get("type", "ollama")
    if kind == "hash": return HashEmbedder(cfg.get("dim", 512))
    if kind == "ollama": return OllamaEmbedder(cfg["model"], cfg.get("host"), cfg.get("cache_dir", ".cache"), cfg.get("batch", 32),
                                                  cfg.get("query_prefix", ""), cfg.get("doc_prefix", ""))
    raise ValueError(f"unknown embedder type '{kind}' (use 'ollama' or 'hash')")
