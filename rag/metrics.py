import math
from .text import tokens

def _evs(g): return g.get("evidences") or ([g["evidence"]] if g.get("evidence") else [])
def chunk_relevant(chunk, g): return any(e.lower() in chunk.text.lower() for e in _evs(g))
def retrieval_metrics(chunks, g, k):
    """Binary relevance: a chunk is relevant if it contains any evidence string of the question.
    evidence_recall = share of the question's evidence strings found in the top-k (matters for multi-hop)."""
    top = chunks[:k]; ev = _evs(g); rel = [chunk_relevant(c, g) for c in top]
    rr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(ev), k)))
    found = sum(any(e.lower() in c.text.lower() for c in top) for e in ev)
    docs = {c.doc_id for c in top}; need = set(g["source_docs"])
    return dict(hit=float(any(rel)), mrr=rr, ndcg=min(1.0, dcg / ideal) if ideal else 0.0, evidence_recall=found / len(ev) if ev else 1.0,
                doc_recall=len(docs & need) / len(need) if need else 1.0, precision=sum(rel) / max(1, len(top)))
def correctness(answer, g):
    a = answer.lower(); return float(all(k.lower() in a for k in g["keywords"])) if g["keywords"] else 0.0
def groundedness(answer, contexts):
    """Deterministic faithfulness proxy: share of answer content tokens that appear in the retrieved context."""
    at = [t for t in tokens(answer) if not t.isdigit() or True]; ct = set(w for c in contexts for w in tokens(c.text))
    at = [t for t in at if not (t.startswith("[") )]
    return sum(t in ct for t in at) / len(at) if at else 1.0
def mean(xs): xs = list(xs); return sum(xs) / len(xs) if xs else 0.0

# ---- S5: normalised correctness, citation-aware groundedness, soft abstention (the strict functions above stay unchanged) ----
import re as _re
_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SYNONYMS = {"higher": ("greater", "larger", "bigger"), "lower": ("smaller", "less"), "longer": ("greater", "larger"), "shorter": ("smaller", "less"),
            "more": ("greater", "larger", "higher"), "less": ("lower", "smaller", "fewer")}

def _numwords(t):
    """'twenty-four' / 'twenty four' / 'five' -> digits (0-99). Whole words only, so 'someone' is untouched."""
    t = _re.sub(r"\b(" + "|".join(_TENS) + r")[\s-]+(" + "|".join(w for w in _ONES if 0 < _ONES[w] < 10) + r")\b", lambda m: str(_TENS[m.group(1)] + _ONES[m.group(2)]), t)
    return _re.sub(r"\b(" + "|".join(list(_TENS) + list(_ONES)) + r")\b", lambda m: str(_TENS.get(m.group(1), _ONES.get(m.group(1)))), t)

def normalize(text):
    """Lowercase; number words -> digits; 1,000 -> 1000; currency (₹800, Rs. 800, 800 INR) -> '800 rupee'; 7:00 PM / 7pm / 7 p.m. -> '7 pm';
    day(s)/hour(s)/... -> singular; hyphens and punctuation -> spaces (decimals and % kept)."""
    t = text.lower().replace("**", " ").replace("\u20b9", " rs ")
    t = _re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = _re.sub(r"\b([ap])\.m\.?", r"\1m", t); t = _re.sub(r"(\d)\s*:\s*00\s*([ap]m)\b", r"\1 \2", t); t = _re.sub(r"(\d)([ap]m)\b", r"\1 \2", t)
    t = _numwords(t)
    t = _re.sub(r"\b(?:rs|inr)\.?\s*(\d[\d.]*)", r"\1 rupee", t); t = _re.sub(r"(\d)\s*(?:rs|inr)\b\.?", r"\1 rupee", t)
    t = _re.sub(r"\b(day|hour|minute|month|year|week|rupee|dollar|session|attempt)s\b", r"\1", t)
    t = _re.sub(r"[^a-z0-9.%]+", " ", t); t = _re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return _re.sub(r"\s+", " ", t).strip()

def _has(norm_answer, norm_kw):
    return bool(norm_kw) and _re.search(r"(?<![a-z0-9.])" + _re.escape(norm_kw) + r"(?![a-z0-9])", norm_answer) is not None

def correctness_norm(answer, g):
    """Never stricter than `correctness`: strict match OR (after normalisation) every keyword, or one of its alternates, is present.
    Alternates: SYNONYMS for comparison words plus an optional golden field `alternates` = {keyword: [alt, ...]}. Known limit: no negation handling."""
    if not g["keywords"]: return 0.0
    if correctness(answer, g): return 1.0
    na = normalize(answer); alts = g.get("alternates") or {}
    for k in g["keywords"]:
        opts = [k] + list(alts.get(k, [])) + list(SYNONYMS.get(k.lower(), ()))
        if not any(_has(na, normalize(o)) for o in opts): return 0.0
    return 1.0

def strip_citations(text, doc_ids=()):
    """Remove [1] / [1][2] / [1, 2], '(doc-id)' / bare known doc ids, and whole 'Sources: ...' lines before measuring groundedness."""
    t = _re.sub(r"(?im)^\s*(?:sources?|citations?|references?)\s*:.*$", " ", text)
    t = _re.sub(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]", " ", t)
    for d in sorted(set(doc_ids), key=len, reverse=True): t = _re.sub(r"\(?\b" + _re.escape(d) + r"(?:#\d+)?\b\)?", " ", t, flags=_re.I)
    return t

def groundedness_clean(answer, contexts):
    """Same proxy as `groundedness`, but citations and doc ids no longer count as ungrounded content."""
    return groundedness(strip_citations(answer, {c.doc_id for c in contexts}), contexts)

def ungrounded_tokens(answer, contexts):
    """Content tokens of the (citation-stripped) answer that appear nowhere in the context: hallucination candidates, listed for inspection."""
    ct = {w for c in contexts for w in tokens(c.text)}
    return [t for t in tokens(strip_citations(answer, {c.doc_id for c in contexts})) if t not in ct]

SOFT_ABSTAIN = ("does not contain", "doesn't contain", "do not contain", "not mentioned", "no information", "not provided", "not specified", "cannot find",
                "can't find", "not stated", "not available in", "unable to find", "not in the context", "no mention", "context does not", "not covered")
def is_soft_abstain(t):
    """Looser than llm.is_abstain (which needs the reply to START with 'I don't know'): catches 'The context does not say ...'."""
    l = t.lower(); return l.strip().startswith(("i don't know", "i do not know")) or any(m in l for m in SOFT_ABSTAIN)
