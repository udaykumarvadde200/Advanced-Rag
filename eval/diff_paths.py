"""Which questions differ between two runs?   python -m eval.diff_paths A.json B.json [--run LABEL_SUBSTRING]
Accepts: a regression snapshot / baseline (`questions` dict, or `snapshot.questions`), a `cli.py eval` file (`rows`), or a `cli.py gen` file (`runs[...].modes.retrieved.rows`).
Prints every answerable question whose strict correctness or answer text differs. Used in S5 to explain 61/69 (gen) vs 59/69 (eval/baseline)."""
import json, sys

def load_rows(path, run=None):
    with open(path, encoding="utf-8") as f: d = json.load(f)
    if "runs" in d:
        r = next(x for x in d["runs"] if run is None or run in x["label"]); return {x["id"]: x for x in r["modes"]["retrieved"]["rows"]}, r["label"]
    if "snapshot" in d: d = d["snapshot"]
    if "questions" in d: return {k: dict(v, id=k) for k, v in d["questions"].items()}, "snapshot"
    if "rows" in d: return {x["id"]: x for x in d["rows"]}, "eval rows"
    raise SystemExit(f"{path}: no questions / rows / runs found")

def diff(a, b):
    out = []
    for i in sorted(set(a) & set(b)):
        x, y = a[i], b[i]
        if "correct" not in x or "correct" not in y: continue
        if x["correct"] != y["correct"] or x["answer"].strip() != y["answer"].strip(): out.append((i, x, y))
    return out

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]; run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else None
    if run: args.remove(run)
    (A, la), (B, lb) = load_rows(args[0]), load_rows(args[1], run)
    d = diff(A, B); ca = sum(x["correct"] for x in A.values() if "correct" in x); cb = sum(x["correct"] for x in B.values() if "correct" in x)
    print(f"A = {args[0]} ({la}): {ca:.0f} correct   B = {args[1]} ({lb}): {cb:.0f} correct   questions with a different answer or verdict: {len(d)}")
    for i, x, y in d: print(f"\n{i}: A correct={x['correct']}  B correct={y['correct']}\n  A: {x['answer'][:160]!r}\n  B: {y['answer'][:160]!r}")
