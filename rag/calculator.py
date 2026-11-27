"""S6 calculator tool. The LLM only EXTRACTS an arithmetic expression from the context (it does not do the arithmetic); Python evaluates it with a whitelist
AST walker (no eval/exec). Guards: the question must look like arithmetic (rules), every number in the expression must occur in the context or the
question, and anything unparseable or suspicious makes the tool decline, so the normal generator answers instead."""
import ast, math, operator as op, re

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
class CalcError(ValueError): pass

def safe_eval(expr, max_nodes=40):
    """Evaluate +, -, *, /, //, %, ** (small exponents) on number literals and parentheses. Anything else raises CalcError."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 200: raise CalcError("empty or too long")
    try: tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e: raise CalcError(f"syntax: {e.msg}")
    if sum(1 for _ in ast.walk(tree)) > max_nodes: raise CalcError("too many nodes")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN: return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 6 or abs(a) > 1e6): raise CalcError("power too large")
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0: raise CalcError("division by zero")
            return _BIN[type(n.op)](a, b)
        raise CalcError(f"not allowed: {type(n).__name__}")
    v = ev(tree)
    if not isinstance(v, (int, float)) or not math.isfinite(v): raise CalcError("not a finite number")
    return v

NUM = re.compile(r"\d+(?:\.\d+)?")
def numbers_in(text): return {float(x) for x in NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))}
def fmt(v): return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4f}".rstrip("0").rstrip(".")

TRIGGER = re.compile(r"\b(combined|total|sum|altogether|together|in all|difference|how much more|how many more|how much less|how many fewer|how many less|times|multipl\w*|product of|average)\b", re.I)
def looks_arithmetic(question): return bool(TRIGGER.search(question))

EXTRACT_SYSTEM = ("You extract arithmetic. Given a context and a question, write ONE arithmetic expression that answers the question using ONLY numbers that appear in the context or the question. "
                  "Use digits, + - * / and parentheses only. Do NOT compute the result. If the question cannot be answered by arithmetic on those numbers, set expression to null.")
SCHEMA = {"type": "object", "properties": {"expression": {"type": ["string", "null"]}, "unit": {"type": ["string", "null"]}}, "required": ["expression"]}

class OllamaExtractor:
    def __init__(self, model="llama3.2:3b", host=None, seed=42, keep_alive="5m"):
        from . import ollama_client as oc
        self.oc, self.model, self.host, self.seed, self.ka = oc, model, host or oc.DEFAULT_HOST, seed, keep_alive
    def __call__(self, question, contexts):
        import json
        ctx = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(contexts))
        j = self.oc.chat(self.host, self.model, [dict(role="system", content=EXTRACT_SYSTEM), dict(role="user", content=f"Context:\n{ctx}\n\nQuestion: {question}")],
                         dict(temperature=0, seed=self.seed, num_predict=80), self.ka, fmt=SCHEMA)
        try: d = json.loads(j["message"]["content"])
        except Exception: d = {}
        return dict(expression=d.get("expression") if isinstance(d, dict) else None, unit=d.get("unit") if isinstance(d, dict) else None,
                    tokens_in=j.get("prompt_eval_count", 0), tokens_out=j.get("eval_count", 0))

class CalcTool:
    def __init__(self, extractor, gate=True): self.extract, self.gate = extractor, gate
    def triggers(self, question): return looks_arithmetic(question) if self.gate else True
    def run(self, question, contexts):
        """-> dict(used=bool, reason, text?, expression?, value?, tokens_in, tokens_out). used=False means: let the normal generator answer."""
        e = self.extract(question, contexts); out = dict(used=False, expression=e.get("expression"), tokens_in=e.get("tokens_in", 0), tokens_out=e.get("tokens_out", 0))
        expr = e.get("expression")
        if not expr: return dict(out, reason="no expression")
        allowed = numbers_in(question + " " + " ".join(c.text for c in contexts)); used = {float(x) for x in NUM.findall(expr)}
        if not used or not used <= allowed: return dict(out, reason=f"operands not in context: {sorted(used - allowed)}")
        try: v = safe_eval(expr)
        except CalcError as ce: return dict(out, reason=f"rejected: {ce}")
        cites = "".join(f"[{i+1}]" for i, c in enumerate(contexts) if numbers_in(c.text) & used)
        unit = (e.get("unit") or "").strip()
        return dict(out, used=True, reason="ok", value=v, text=f"{fmt(v)}{(' ' + unit) if unit else ''} ({expr.strip()}) {cites}".strip())

def make_calc(cfg, llm=None):
    if not cfg or not cfg.get("enabled"): return None
    if cfg.get("extractor", "llm") == "none": return CalcTool(lambda q, c: dict(expression=None), cfg.get("gate", True))
    if hasattr(llm, "extract_expression"): return CalcTool(llm.extract_expression, cfg.get("gate", True))
    if getattr(llm, "model", None) and hasattr(llm, "oc"): return CalcTool(OllamaExtractor(cfg.get("model", llm.model), llm.host, 42, getattr(llm, "keep_alive", "5m")), cfg.get("gate", True))
    return CalcTool(lambda q, c: dict(expression=None), cfg.get("gate", True))   # MockLLM: no extractor -> the tool always declines
