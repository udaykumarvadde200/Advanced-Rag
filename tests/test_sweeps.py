import os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.pipeline import load_docs, load_golden
from rag.chunking import Chunk, chunk_fixed
from rag.metrics import retrieval_metrics
from eval.run_sweeps import run_sweeps, to_markdown, bootstrap_ci, paired_ci, add_stats, embedder_comparison
from rag import embeddings as emb
import tempfile

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
        n = rows[0]["n_questions"]
        for r in rows:  # stats are attached to every row and are internally consistent
            self.assertLessEqual(r["mrr_lo"], r["mrr"] + 1e-3); self.assertGreaterEqual(r["mrr_hi"], r["mrr"] - 1e-3)
            self.assertEqual(len(r["rr"]), n); self.assertLessEqual(r["won"] + r["lost"], n); self.assertLessEqual(r["d_lo"], r["d_hi"])
        self.assertIn("won/lost", to_markdown(rows)); self.assertIn("95% CI", to_markdown(rows))
    def test_bm25_rows_not_duplicated_per_embedder(self):
        sw = dict(chunking=[dict(strategy="recursive", size=300)], retrievers=["bm25", "hybrid"], rerankers=[{"type": "none"}], embedders=[{"type": "hash"}])
        import eval.run_sweeps as rs; orig = rs.make_embedder
        rs.make_embedder = lambda cfg: orig({"type": "hash", "dim": cfg.get("dim", 512)})
        try: rows = rs.run_sweeps(dict(sw, embedders=[{"type": "hash", "dim": 512}, {"type": "hash", "dim": 256}]), load_docs(), load_golden(), offline=False, log=lambda *a: None)
        finally: rs.make_embedder = orig
        self.assertEqual(sorted(r["retriever"] for r in rows), ["bm25", "hybrid", "hybrid"])
    def test_bootstrap_ci(self):
        self.assertEqual(bootstrap_ci([0.5] * 20), (0.5, 0.5))
        xs = [1, 0, 1, 1, 0, 1, 0.5, 1, 1, 0]; lo, hi = bootstrap_ci(xs)
        self.assertEqual((lo, hi), bootstrap_ci(xs))                                   # fixed seed -> deterministic
        self.assertLessEqual(lo, sum(xs) / len(xs)); self.assertGreaterEqual(hi, sum(xs) / len(xs)); self.assertGreater(hi - lo, 0)
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))
    def test_paired_ci(self):
        lo, hi = paired_ci([1.0] * 30, [0.5] * 30); self.assertGreater(lo, 0)              # consistent gain -> excludes 0
        self.assertEqual(paired_ci([0.7] * 10, [0.7] * 10), (0.0, 0.0))                  # identical -> exactly 0
        a = [1, 0] * 10; b = [0, 1] * 10; lo, hi = paired_ci(a, b); self.assertEqual(paired_ci(a, a), (0.0, 0.0))
    def test_add_stats_won_lost_and_significance(self):
        mk = lambda ch, ret, rr, p50: dict(chunking=ch, retriever=ret, reranker="none", rr=rr, mrr=sum(rr) / len(rr), p50_ms=p50, n_questions=len(rr), embedder="-" if ret == "bm25" else "e1")
        base, other, worse = mk("c1", "bm25", [1, 1, .5, .5], 1.0), mk("c1", "hybrid", [1, .5, 1, 1], 5.0), mk("c2", "bm25", [.5, .5, .5, .5], 1.0)
        add_stats([worse, other, base])
        self.assertEqual((base["won"], base["lost"], base["sig"]), (0, 0, False)); self.assertEqual(base["vs"], "c1/bm25/none")
        self.assertEqual((other["won"], other["lost"]), (2, 1)); self.assertEqual((worse["won"], worse["lost"]), (0, 2))
        self.assertAlmostEqual(other["d_mrr"], round(sum(other["rr"]) / 4 - sum(base["rr"]) / 4, 3))
        self.assertFalse(other["sig"])                                                   # 4 questions can never be significant
        big_base, big = mk("c", "bm25", [0.5] * 40, 1.0), mk("c", "dense", [1.0] * 40, 2.0)
        add_stats([big_base, big]); self.assertTrue(big["sig"]); self.assertEqual((big["won"], big["lost"]), (40, 0))
        only = [mk("c", "dense", [1, 0], 1.0)]; add_stats(only); self.assertIsNone(only[0]["won"]); self.assertIn("none in this run", to_markdown(only))
    def test_chunk_fixed_covers_text_without_redundant_tail(self):
        for L in (0, 1, 39, 40, 41, 199, 200, 201, 239, 240, 241, 399, 400, 1000):
            rng = random.Random(L); t = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(L))   # non-periodic, so substring checks are meaningful
            for size, ov in ((200, 40), (400, 80), (100, 0), (30, 20), (20, 15)):
                out = chunk_fixed(t, size, ov)
                self.assertEqual("".join(out[:1] + [c[ov:] for c in out[1:]]), t, (L, size, ov))          # every char kept, overlap exact
                self.assertTrue(all(len(c) <= size for c in out)); self.assertTrue(all(len(c) > ov for c in out[1:]), (L, size, ov))   # every later chunk adds new text (no redundant tail)
        self.assertEqual(len(chunk_fixed("x" * 200, 200, 40)), 1); self.assertEqual(len(chunk_fixed("x" * 201, 200, 40)), 2)
        for bad in ((10, 10), (10, 11), (10, -1)):
            with self.assertRaises(ValueError): chunk_fixed("abc", *bad)
    def test_embedder_prefixes_are_applied_and_cached_separately(self):
        sent = []; orig = emb.oc.embed
        emb.oc.embed = lambda host, model, texts: (sent.append(list(texts)) or [[float(len(t)), 1.0, 0.5] for t in texts])
        try:
            with tempfile.TemporaryDirectory() as d:
                plain = emb.make_embedder({"type": "ollama", "model": "m", "cache_dir": d})
                pre = emb.make_embedder({"type": "ollama", "model": "m", "cache_dir": d, "query_prefix": "Q: ", "doc_prefix": "D: "})
                self.assertEqual((plain.name, pre.name), ("m", "m+prefix"))
                plain.embed(["chunk"]); pre.embed(["chunk"]); pre.embed(["question"], cache=False)
                self.assertEqual(sent, [["chunk"], ["D: chunk"], ["Q: question"]])        # docs get doc_prefix, queries get query_prefix
                n = len(sent); pre.embed(["chunk"]); plain.embed(["chunk"])
                self.assertEqual(len(sent), n, "both variants must now come from the cache, never each other's vectors")
        finally: emb.oc.embed = orig
    def test_embedder_comparison_is_paired_and_per_cell(self):
        mk = lambda e, rr, ret="hybrid": dict(chunking="c", retriever=ret, reranker="none", embedder=e, rr=rr, mrr=sum(rr) / len(rr), p50_ms=1.0, n_questions=len(rr))
        rows = [mk("-", [1.0] * 40, "bm25"), mk("a", [0.5] * 40), mk("b", [1.0] * 40), mk("b", [0.2] * 40, "dense")]   # no "a" cell for dense -> skipped
        add_stats(rows); out = embedder_comparison(rows); self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["embedder"], out[0]["ref"], out[0]["won"], out[0]["lost"], out[0]["sig"]), ("b", "a", 40, 0, True))
        self.assertIn("Embedder comparison", to_markdown(rows)); self.assertEqual(embedder_comparison(rows[:2]), [])
if __name__ == "__main__": unittest.main()
