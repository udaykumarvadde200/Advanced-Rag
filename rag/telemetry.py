import time, json, statistics
from contextlib import contextmanager

def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a backend reports no counts."""
    return max(1, len(text) // 4)

class Trace:
    """Collects per-stage latency (ms) and token counts for one query."""
    def __init__(self):
        self.stages, self.tokens_in, self.tokens_out = {}, 0, 0
    @contextmanager
    def stage(self, name):
        t = time.perf_counter()
        try: yield
        finally: self.stages[name] = self.stages.get(name, 0) + (time.perf_counter() - t) * 1000
    def total_ms(self): return sum(self.stages.values())
    def as_dict(self):
        return dict(stages_ms={k: round(v, 3) for k, v in self.stages.items()}, total_ms=round(self.total_ms(), 3),
                    tokens_in=self.tokens_in, tokens_out=self.tokens_out)

def pct(values, p):
    if not values: return 0.0
    v = sorted(values); k = min(len(v) - 1, int(round((p / 100) * (len(v) - 1))))
    return v[k]

def summarize_latency(traces):
    tot = [t["total_ms"] for t in traces]
    stages = {}
    for t in traces:
        for k, v in t["stages_ms"].items(): stages.setdefault(k, []).append(v)
    return dict(p50_ms=round(pct(tot, 50), 2), p95_ms=round(pct(tot, 95), 2), mean_ms=round(statistics.mean(tot), 2) if tot else 0,
                tokens_in_mean=round(statistics.mean([t["tokens_in"] for t in traces]), 1) if traces else 0,
                tokens_out_mean=round(statistics.mean([t["tokens_out"] for t in traces]), 1) if traces else 0,
                stage_p50_ms={k: round(pct(v, 50), 2) for k, v in stages.items()})

# Reference price (USD per 1M tokens) used ONLY to compare configs; local models are free.
REF_PRICE_IN, REF_PRICE_OUT = 0.15, 0.60
def est_cost_usd(tin, tout): return (tin * REF_PRICE_IN + tout * REF_PRICE_OUT) / 1e6
