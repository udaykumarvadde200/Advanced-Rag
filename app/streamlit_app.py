"""Local chat UI for the RAG assistant.   Run from the project root:   streamlit run app/streamlit_app.py
Needs `pip install -r requirements-app.txt` (Streamlit is NOT a dependency of the core, the tests or CI) and a running Ollama (or RAG_APP_OFFLINE=1 for stand-ins).
The page is a thin layer: retrieval, access control and memory live in rag/ (rag/memory.py has the rules and their tests)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.chdir(ROOT)                      # the pipeline reads data/ and configs/ relative to the project root

import streamlit as st
from app import ui_helpers as U
from rag.pipeline import RAG
from rag.memory import ChatSession
from rag import ollama_client as oc

st.set_page_config(page_title="Acme-Fake knowledge assistant", page_icon=":speech_balloon:")

@st.cache_resource(show_spinner="Loading the corpus and building the index ...")
def get_rag(config, offline):
    cfg, _ = U.load_app_cfg(config, offline); return RAG(cfg)

CONFIG = os.environ.get("RAG_APP_CONFIG", U.DEFAULT_CONFIG); OFFLINE = os.environ.get("RAG_APP_OFFLINE") == "1"
cfg, _ = U.load_app_cfg(CONFIG, OFFLINE); S = st.session_state

with st.sidebar:
    st.header("Settings")
    role = st.selectbox("Role", U.ROLE_CHOICES, index=U.ROLE_CHOICES.index("employee"), help="Decides which documents can be retrieved (access control is applied before ranking).")
    st.caption("This is a role PICKER for the demo. Real authentication is out of scope: anyone can choose any role.")
    memory_on = st.toggle("Short-term memory", value=False, help="Off by default. When on, a follow-up is rewritten into a standalone question using the last turns of THIS role's chat.")
    turns = st.slider("Turns remembered", 1, 5, 3, disabled=not memory_on)
    if st.button("Clear chat"): S.pop("chat", None); S.pop("messages", None)
    st.caption("Changing the role also clears the chat: nothing carries over between roles.")
    if OFFLINE: st.warning("OFFLINE STAND-INS (hash embedder + rule-based mock model): this tests the page, the answers are not model output.")

rag = get_rag(CONFIG, OFFLINE)
if "chat" not in S: S.chat, S.messages, S.role = ChatSession(rag, cfg), [], role
if role != S.role: S.chat.clear(); S.messages, S.role = [], role; st.toast("Role changed: the chat was cleared.")
S.chat.enabled, S.chat.turns = memory_on, turns

st.title("Company knowledge assistant")
st.caption(f"Role: **{role}** | memory: **{'on, last ' + str(turns) + ' turns' if memory_on else 'off'}** | answers come only from the documents this role may read; \"I don't know\" means nothing relevant was found.")

prompt = st.chat_input("Ask about leave, expenses, security, deployments ...")
if prompt:
    S.messages.append(dict(role="user", content=prompt))
    try:
        with st.spinner("Thinking ..."): S.messages.append(U.message_from_result(S.chat.ask(prompt, role)))
    except oc.OllamaError as e:
        S.messages.append(dict(role="assistant", content=f"Ollama error: {e}. Is `ollama serve` running and are the models pulled (`python cli.py check`)?", sources=[], trace="", rewrite=None))

for m in S.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            if m.get("rewrite"): st.caption(m["rewrite"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for s in m["sources"]: st.markdown(f"**[{s['n']}] {s['doc_id']}** - {s['title']} (access: {s['access']})\n\n{s['text']}")
            if m.get("trace"): st.caption(m["trace"])
