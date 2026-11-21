"""Thin client for the local Ollama HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md).
Host comes from the OLLAMA_HOST env var (default http://localhost:11434)."""
import os, time, requests

def _norm(h):
    """Add scheme and map 'localhost' to 127.0.0.1: on Windows, 'localhost' can try IPv6 first and add ~2 s per request (measured)."""
    h = h if h.startswith("http") else "http://" + h
    return h.replace("://localhost", "://127.0.0.1")
DEFAULT_HOST = _norm(os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))

class OllamaError(RuntimeError): pass

def _post(host, path, body, timeout=600, retries=2):
    host = _norm(host); last = None
    for i in range(retries + 1):
        try:
            r = requests.post(f"{host}{path}", json=body, timeout=timeout)
            if r.status_code == 404:
                m = body.get("model")
                raise OllamaError(f"Model '{m}' not found on {host}. Run: ollama pull {m}")
            r.raise_for_status(); return r.json()
        except requests.ConnectionError as e:
            raise OllamaError(f"Cannot reach Ollama at {host}. Start it with: ollama serve") from e
        except OllamaError: raise
        except requests.RequestException as e:
            last = e; time.sleep(1 + i)
    raise OllamaError(f"Ollama request {path} failed after {retries + 1} tries: {last}")

def list_models(host=None):
    host = _norm(host or DEFAULT_HOST)
    try: r = requests.get(f"{host}/api/tags", timeout=10); r.raise_for_status()
    except requests.ConnectionError as e: raise OllamaError(f"Cannot reach Ollama at {host}. Start it with: ollama serve") from e
    return [m["name"] for m in r.json().get("models", [])]

def model_digests(host=None):
    """{model name: digest} from /api/tags (used to pin the exact model files in a regression baseline)."""
    host = _norm(host or DEFAULT_HOST)
    try: r = requests.get(f"{host}/api/tags", timeout=10); r.raise_for_status()
    except requests.ConnectionError as e: raise OllamaError(f"Cannot reach Ollama at {host}. Start it with: ollama serve") from e
    return {m["name"]: m.get("digest", "") for m in r.json().get("models", [])}

def missing_models(needed, host=None):
    have = set(list_models(host)); out = []
    for m in needed:
        if m not in have and (m if ":" in m else m + ":latest") not in have: out.append(m)
    return out

def chat(host, model, messages, options=None, keep_alive="5m", timeout=600, fmt=None):
    """fmt: None (free text), "json", or a JSON-schema dict (Ollama structured outputs; the judge uses it)."""
    body = dict(model=model, messages=messages, stream=False, options=options or {}, keep_alive=keep_alive)
    if fmt is not None: body["format"] = fmt
    return _post(host, "/api/chat", body, timeout)

def embed(host, model, texts, timeout=300):
    return _post(host, "/api/embed", dict(model=model, input=list(texts)), timeout)["embeddings"]
