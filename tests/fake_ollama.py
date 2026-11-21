"""Stand-in for the Ollama HTTP API (/api/tags, /api/chat, /api/embed) used ONLY to test the client code without Ollama.
It is NOT a real model: answers are extractive (MockLLM) and embeddings are hashed. Run: python tests/fake_ollama.py --port 11999"""
import argparse, json, re, sys, os
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.embeddings import HashEmbedder
from rag.llm import MockLLM
MODELS = ["fake-chat:1b", "fake-embed:latest"]; STATS = dict(chat=0, embed=0)
class C:  # minimal chunk stand-in
    def __init__(s, t, d): s.text, s.doc_id = t, d
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj): b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/api/tags": self._send(200, dict(models=[dict(name=m, digest="sha256:fake" + m.replace(":", "-")) for m in MODELS]))
        elif self.path == "/stats": self._send(200, STATS)
        else: self._send(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") not in MODELS and body.get("model") != "fake-embed": return self._send(404, dict(error=f"model '{body.get('model')}' not found"))
        if self.path == "/api/embed":
            STATS["embed"] += 1; self._send(200, dict(embeddings=HashEmbedder(64).embed(body["input"]).tolist()))
        elif self.path == "/api/chat":
            STATS["chat"] += 1; u = body["messages"][-1]["content"]
            if u.startswith("Rate how well"): txt = "7"
            else:
                q = re.search(r"Question: (.*)\nAnswer concisely", u).group(1)
                ctx = [C(m[1], m[0]) for m in re.findall(r"\[\d+\] \((.+?)\) (.*?)(?=\n\n\[|\n\nQuestion:)", u, flags=re.S)]
                txt = MockLLM().generate(q, ctx)["text"]
            self._send(200, dict(message=dict(role="assistant", content=txt), prompt_eval_count=len(u) // 4, eval_count=len(txt) // 4,
                                 total_duration=2_000_000, load_duration=500_000, prompt_eval_duration=800_000, eval_duration=700_000))
        else: self._send(404, {})
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=11999); HTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
