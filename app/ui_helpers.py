"""Streamlit-free helpers for app/streamlit_app.py, so the logic around the page can be tested offline (CI has no Streamlit)."""
import os
from rag.pipeline import load_cfg, deep_merge, ROLES

DEFAULT_CONFIG = "configs/ollama.yaml"
ROLE_CHOICES = list(ROLES)

def load_app_cfg(config=None, offline=None):
    """Config for the app. RAG_APP_CONFIG / RAG_APP_OFFLINE=1 (stand-ins: hash embedder + rule-based mock) can be set in the environment; the memory switch stays OFF unless the config says otherwise."""
    config = config or os.environ.get("RAG_APP_CONFIG", DEFAULT_CONFIG)
    offline = (os.environ.get("RAG_APP_OFFLINE") == "1") if offline is None else offline
    cfg = load_cfg(config)
    if offline: cfg = deep_merge(cfg, dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False)))
    return cfg, offline

def sources(contexts):
    """The passages the answer was based on, for the 'Sources' expander (document id + title + text; nothing else leaves the pipeline)."""
    return [dict(n=i + 1, doc_id=c.doc_id, title=(c.meta or {}).get("title", ""), access=c.access, text=c.text) for i, c in enumerate(contexts)]

def trace_caption(trace):
    st = trace.get("stages_ms", {}); parts = " | ".join(f"{k} {v:.0f}" for k, v in st.items())
    return f"{trace['total_ms']:.0f} ms ({parts}) | tokens in/out {trace['tokens_in']}/{trace['tokens_out']}"

def rewrite_caption(memory):
    """Shown under an answer whenever the memory rewrote the question or tried to; None when memory did nothing (first turn, memory off, role just changed)."""
    if not memory or not memory.get("used"): return None
    if memory.get("fallback"): return f"Memory tried to rewrite your question but the result was rejected ({memory['fallback']}); the original question was used."
    if memory.get("rewritten"): return "Searched as: " + memory["standalone"]
    return "Memory checked your question: it is already standalone."

def message_from_result(out):
    """One chat message (assistant) from a ChatSession.ask() result."""
    return dict(role="assistant", content=out["answer"], sources=sources(out["contexts"]), trace=trace_caption(out["trace"]), rewrite=rewrite_caption(out.get("memory")))
