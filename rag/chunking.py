import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    id: str; doc_id: str; text: str; access: str; meta: dict = field(default_factory=dict)

def _sentences(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

def chunk_fixed(text, size=200, overlap=40):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size]); i += max(1, size - overlap)
        if i - overlap + size >= len(text) and out[-1] == text[i - size + overlap:]: break
    return out

def chunk_recursive(text, size=300, overlap=0):
    """Paragraph -> sentence packing up to `size` chars."""
    out = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        cur = ""
        for s in _sentences(para):
            if cur and len(cur) + len(s) + 1 > size: out.append(cur); cur = cur[-overlap:] + " " + s if overlap else s
            else: cur = (cur + " " + s).strip()
        if cur: out.append(cur)
    return out

def chunk_sentence(text, size=0, overlap=0):
    return [s for p in text.split("\n\n") for s in _sentences(p)]

STRATEGIES = {"fixed": chunk_fixed, "recursive": chunk_recursive, "sentence": chunk_sentence}

def chunk_docs(docs, strategy="recursive", size=300, overlap=0, add_title=True):
    fn, out = STRATEGIES[strategy], []
    for d in docs:
        for i, t in enumerate(fn(d["text"], size=size, overlap=overlap)):
            body = f"{d['title']}: {t}" if add_title else t
            out.append(Chunk(f"{d['id']}#{i}", d["id"], body, d["access"], {"title": d["title"]}))
    return out
