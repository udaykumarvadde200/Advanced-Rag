import json, glob, copy, yaml
from .chunking import chunk_docs
from .embeddings import make_embedder
from .retrieval import Retriever, make_reranker
from .llm import make_llm, build_prompt
from .telemetry import Trace, est_cost_usd
from .defense import defense_cfg, sanitize_chunks
from .routing import make_router
from .calculator import make_calc

ROLES = {"guest": {"public"}, "employee": {"public", "employee"}, "hr": {"public", "employee", "hr"},
         "finance": {"public", "employee", "finance"}, "eng": {"public", "employee", "eng"}}

def _read(path):
    with open(path, encoding="utf-8") as f: return f.read()
def load_docs(path="data/corpus"): return [json.loads(_read(f)) for f in sorted(glob.glob(f"{path}/*.json"))]
def load_golden(path="data/golden/golden.jsonl"): return [json.loads(l) for l in _read(path).splitlines() if l.strip()]
def load_cfg(path):
    base = yaml.safe_load(_read("configs/baseline.yaml"))
    if path.endswith("baseline.yaml"): return base
    return deep_merge(base, yaml.safe_load(_read(path)) or {})
def deep_merge(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items(): a[k] = deep_merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a

class RAG:
    def __init__(self, cfg, docs=None):
        self.cfg = cfg; docs = docs or load_docs()
        c = cfg["chunking"]; self.chunks = chunk_docs(docs, c["strategy"], c.get("size", 300), c.get("overlap", 0))
        self.embedder = make_embedder(cfg["embedder"]) if cfg["retriever"]["mode"] != "bm25" else None
        r = cfg["retriever"]
        self.retriever = Retriever(self.chunks, self.embedder, r["mode"], r.get("enforce_acl", True))
        self.reranker = make_reranker(cfg.get("reranker")); self.llm = make_llm(cfg["llm"])
        self.defense = defense_cfg(cfg); self.router = make_router(cfg.get("routing")); self.calc = make_calc(cfg.get("calculator"), self.llm)   # S6 switches, all off by default
    def retrieve(self, question, role, trace=None):
        r = self.cfg["retriever"]; k = r.get("k", 4); t = trace or Trace(); route = None
        if self.router:
            with t.stage("route"): route, self.last_route = self.router.predicate(question)
        got = self.retriever.retrieve(question, ROLES[role], k=(r.get("pool", 20) if self.reranker else k), pool=r.get("pool", 20), trace=t, route=route)
        if self.reranker:
            with t.stage("rerank"): got = self.reranker.rerank(question, got, k)
        return got
    def answer(self, question, role, contexts=None):
        t = Trace(); ctx = contexts if contexts is not None else self.retrieve(question, role, t)
        seen, info = ctx, {}
        if self.defense["sanitize"]:   # the model sees cleaned text; `contexts` in the result (and every retrieval metric) keeps the original chunks
            with t.stage("sanitize"): seen, info["sanitize"] = sanitize_chunks(ctx)
        g, tin, tout = None, 0, 0
        if self.calc and self.calc.triggers(question):
            with t.stage("calc"): cr = self.calc.run(question, seen)
            tin, tout = cr["tokens_in"], cr["tokens_out"]; info["calc"] = {k: cr.get(k) for k in ("used", "reason", "expression", "value")}
            if cr["used"]: g = dict(text=cr["text"], tokens_in=0, tokens_out=0, ollama={})
        if g is None:
            with t.stage("generate"): g = self.llm.generate(question, seen, spotlight=self.defense["spotlight"]) if self.defense["spotlight"] else self.llm.generate(question, seen)
        t.tokens_in, t.tokens_out = g["tokens_in"] + tin, g["tokens_out"] + tout
        d = t.as_dict()
        d["est_cost_usd"] = est_cost_usd(t.tokens_in, t.tokens_out); d["ollama"] = g.get("ollama", {})
        if info: d["s6"] = info
        return dict(answer=g["text"], contexts=ctx, trace=d)
