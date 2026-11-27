"""S6 metadata routing: rules classify the question, a pre-filter narrows the chunk set (after the ACL filter, never instead of it). See configs/routing.yaml."""
import re, yaml

class Router:
    def __init__(self, rules, variants=True, department=False):
        self.variants = [dict(r, _t=re.compile(r["doc_title"], re.I), _q=re.compile(r["unless_query"], re.I)) for r in rules.get("variants", [])] if variants else []
        self.depts = {n: dict(d, _k=re.compile(r"\b(?:" + d["keywords"] + r")", re.I)) for n, d in rules.get("departments", {}).items()} if department else {}
    def classify(self, question):
        """-> dict(excluded=[variant names to hide], dept=name or None). dept is None when zero or several departments match."""
        excl = [v["name"] for v in self.variants if not v["_q"].search(question)]
        hit = [n for n, d in self.depts.items() if d["_k"].search(question)]
        return dict(excluded=excl, dept=hit[0] if len(hit) == 1 else None, depts_matched=hit)
    def predicate(self, question):
        """Returns (chunk -> bool, decision dict). The predicate keeps a chunk unless its document is a hidden variant or outside the chosen department."""
        d = self.classify(question); hide = [v for v in self.variants if v["name"] in d["excluded"]]; dept = self.depts.get(d["dept"])
        def keep(c):
            title = (c.meta or {}).get("title", "")
            if any(v["_t"].search(title) for v in hide): return False
            if dept and not (any(c.doc_id.startswith(p + "-") for p in dept["prefixes"]) or c.doc_id in dept.get("extra_docs", [])): return False
            return True
        return keep, d

def make_router(cfg, path="configs/routing.yaml"):
    if not cfg or not cfg.get("enabled"): return None
    with open(path, encoding="utf-8") as f: rules = yaml.safe_load(f)
    return Router(rules, cfg.get("variants", True), cfg.get("department", False))
