"""Golden-set evaluation (shared by `cli.py eval` and the regression gate). Returns the summary, per-question rows and traces."""
import time
from rag.pipeline import RAG, load_golden
from rag.metrics import retrieval_metrics, correctness, groundedness, mean
from rag.llm import is_abstain
from rag.telemetry import summarize_latency

def evaluate(cfg, limit=None, log=lambda m: None, rag=None):
    t0 = time.time(); r = rag or RAG(cfg); G = load_golden()[:limit] if limit else load_golden(); k = cfg["retriever"].get("k", 4)
    r.answer("warm up", "guest")  # throwaway call: loads models so load time doesn't pollute latency stats
    rows, traces = [], []
    for i, g in enumerate(G, 1):
        o = r.answer(g["question"], g["role"]); traces.append(o["trace"]); row = dict(id=g["id"], type=g["type"], answer=o["answer"])
        if g["answerable"]:
            row.update(retrieval_metrics(o["contexts"], g, k)); row["correct"] = correctness(o["answer"], g); row["grounded"] = groundedness(o["answer"], o["contexts"])
        else: row["abstained"] = float(is_abstain(o["answer"]))
        rows.append(row); log(f"[{i}/{len(G)}] {g['id']} {o['trace']['total_ms']:.0f} ms")
    A = [x for x in rows if "hit" in x]; U = [x for x in rows if "abstained" in x]
    summ = dict(config=cfg, n_answerable=len(A), n_unanswerable=len(U), hit_at_k=mean(x["hit"] for x in A), mrr=mean(x["mrr"] for x in A),
                doc_recall=mean(x["doc_recall"] for x in A), correctness=mean(x["correct"] for x in A), groundedness=mean(x["grounded"] for x in A),
                abstention_accuracy=mean(x["abstained"] for x in U), latency=summarize_latency(traces),
                by_type={t: dict(n=sum(1 for x in A if x["type"] == t), correctness=round(mean(x["correct"] for x in A if x["type"] == t), 3)) for t in sorted({x["type"] for x in A})},
                est_cost_usd=sum(t["est_cost_usd"] for t in traces), wall_s=round(time.time() - t0, 1))
    return dict(summary=summ, rows=rows, traces=traces)
