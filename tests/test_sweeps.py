import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.pipeline import load_docs, load_golden
from rag.chunking import Chunk
from rag.metrics import retrieval_metrics
from eval.run_sweeps import run_sweeps, to_markdown

class T(unittest.TestCase):
    def test_multi_evidence_metrics(self):
        g = dict(evidences=["alpha fact", "beta fact"], evidence="alpha fact", source_docs=["a", "b"])
        ch = [Chunk("a#0", "a", "x alpha fact y", "public"), Chunk("c#0", "c", "noise", "public"), Chunk("b#0", "b", "beta fact", "public")]
        m1, m3 = retrieval_metrics(ch, g, 1), retrieval_metrics(ch, g, 3)
        self.assertEqual(m1["evidence_recall"], 0.5); self.assertEqual(m3["evidence_recall"], 1.0); self.assertEqual(m3["doc_recall"], 1.0)
        self.assertEqual(m1["mrr"], 1.0); self.assertLessEqual(m3["ndcg"], 1.0)
    def test_offline_sweep(self):
        sw = dict(chunking=[dict(strategy="fixed", size=200, overlap=40), dict(strategy="recursive", size=300)], retrievers=["bm25", "hybrid"],
                  embedders=[{"type": "hash"}], rerankers=[{"type": "none"}, {"type": "lexical"}])
        rows = run_sweeps(sw, load_docs(), load_golden(), offline=True, log=lambda *a: None)
        self.assertEqual(len(rows), 8)
        for r in rows:
            for k in ("hit1", "hit3", "hit5", "mrr", "ndcg5", "intact", "ev_recall5"): self.assertTrue(0 <= r[k] <= 1, (k, r))
            self.assertLessEqual(r["hit1"], r["hit3"]); self.assertLessEqual(r["hit3"], r["hit5"])
        self.assertIn("| chunking |", to_markdown(rows))
if __name__ == "__main__": unittest.main()
