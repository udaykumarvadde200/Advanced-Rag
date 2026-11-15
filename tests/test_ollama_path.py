"""Exercises the real Ollama client code (HTTP, batching, caching, errors) against a stand-in server. Run: python -m unittest -v"""
import os, shutil, sys, tempfile, threading, unittest, requests
from http.server import HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fake_ollama import H, STATS
from rag import ollama_client as oc
from rag.pipeline import RAG, load_cfg, deep_merge

class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.srv = HTTPServer(("127.0.0.1", 0), H); c.host = f"http://127.0.0.1:{c.srv.server_port}"
        threading.Thread(target=c.srv.serve_forever, daemon=True).start(); c.tmp = tempfile.mkdtemp()
    @classmethod
    def tearDownClass(c): c.srv.shutdown(); shutil.rmtree(c.tmp, ignore_errors=True)
    def cfg(self, **over):
        base = load_cfg("configs/ollama.yaml")
        base["embedder"].update(model="fake-embed:latest", host=self.host, cache_dir=self.tmp); base["llm"].update(model="fake-chat:1b", host=self.host)
        return deep_merge(base, over)
    def test_localhost_mapped_to_ipv4(self):
        self.assertEqual(oc._norm("localhost:11434"), "http://127.0.0.1:11434"); self.assertEqual(oc._norm("http://localhost:11434"), "http://127.0.0.1:11434")
        self.assertEqual(oc._norm("10.0.0.5:11434"), "http://10.0.0.5:11434"); self.assertIn("127.0.0.1", oc.DEFAULT_HOST)
    def test_list_and_missing(self):
        self.assertIn("fake-chat:1b", oc.list_models(self.host)); self.assertEqual(oc.missing_models(["fake-chat:1b", "nope"], self.host), ["nope"])
    def test_model_not_found_has_pull_hint(self):
        with self.assertRaises(oc.OllamaError) as e: oc.chat(self.host, "missing", [dict(role="user", content="hi")])
        self.assertIn("ollama pull missing", str(e.exception))
    def test_unreachable_server(self):
        with self.assertRaises(oc.OllamaError) as e: oc.list_models("http://127.0.0.1:1")
        self.assertIn("ollama serve", str(e.exception))
    def test_end_to_end_answer_and_tokens(self):
        o = RAG(self.cfg()).answer("How many sick leave days are provided per year?", "employee")  # not affected by the archived-policy distractor
        self.assertIn("10", o["answer"]); self.assertGreater(o["trace"]["tokens_in"], 0); self.assertIn("generate", o["trace"]["stages_ms"])
        self.assertIn("eval_ms", o["trace"]["ollama"])
    def test_embedding_cache_prevents_reembedding(self):
        before = requests.get(self.host + "/stats").json()["embed"]; RAG(self.cfg())  # may hit cache from earlier tests
        mid = requests.get(self.host + "/stats").json()["embed"]; RAG(self.cfg()); after = requests.get(self.host + "/stats").json()["embed"]
        self.assertEqual(mid, after, "second build must not call /api/embed for already-cached chunks")
    def test_queries_not_cached_and_stages_split(self):
        r = RAG(self.cfg()); q = "What is the IT helpdesk extension?"; r.answer(q, "employee")
        a = requests.get(self.host + "/stats").json()["embed"]; o = r.answer(q, "employee"); b = requests.get(self.host + "/stats").json()["embed"]
        self.assertEqual(b - a, 1, "each query must hit /api/embed (no query cache)")
        self.assertIn("embed_query", o["trace"]["stages_ms"]); self.assertIn("rank", o["trace"]["stages_ms"])
    def test_acl_still_enforced_via_ollama(self):
        r = RAG(self.cfg()); o = r.answer("What are the salary bands and compensation?", "employee")
        self.assertTrue(all(c.doc_id != "hr-03" for c in o["contexts"]))
    def test_ollama_reranker_path(self):
        o = RAG(self.cfg(reranker=dict(type="ollama", model="fake-chat:1b", host=self.host))).answer("What is the IT helpdesk extension?", "employee")
        self.assertIn("rerank", o["trace"]["stages_ms"]); self.assertEqual(len(o["contexts"]), 4)
if __name__ == "__main__": unittest.main()
