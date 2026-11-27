"""Regression gate (S3). Freeze a baseline, compare a later run against it, name the gate that failed.
  python cli.py regress --save-baseline baselines/v1.json     run golden eval + safety suite, freeze the result
  python cli.py regress --compare baselines/v1.json           run again, compare, print/save the report, exit code
  python cli.py regress --noise-study 3                       run N times, report run-to-run variance (justifies the tolerances)
Exit codes: 0 PASS | 1 FAIL (a gate failed; the report names it) | 2 comparison INVALID (data changed, offline/real mix, safety missing): nothing was judged.
Rules live in configs/tolerances.yaml. The comparator (`compare`) is a pure function of two snapshots, so it is tested offline."""
import hashlib, glob, json, math, os, platform, time, yaml
from datetime import datetime, timezone
from rag.pipeline import RAG, deep_merge, load_docs, load_golden
from rag import ollama_client as oc
from eval.run_golden import evaluate
from eval.run_safety import run_suite, load_safety

VERSION, EPS = 1, 1e-9
SAFETY_CATS = ("injection", "pii", "jailbreak", "abstention")
# aggregate metric -> (gate, per-question key in the snapshot)
HIB = {"hit_at_k": ("retrieval", "hit"), "mrr": ("retrieval", "mrr"), "doc_recall": ("retrieval", "doc_recall"),
       "correctness": ("generation", "correct"), "groundedness": ("generation", "grounded"), "abstention_accuracy": ("generation", "abstained")}
BINARY = ("hit", "correct", "abstained")  # per-question keys that are pass/fail

# ---------------------------------------------------------------- statistics
def sign_test_p(won, lost):
    """Exact two-sided sign test on the questions that changed (ties ignored). H0: a change is equally likely to be a win or a loss."""
    n = won + lost
    if n == 0: return 1.0
    k = min(won, lost); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def min_detectable(alpha=0.05):
    """Smallest number of questions lost (0 won) that a two-sided sign test calls significant."""
    k = 1
    while sign_test_p(0, k) > alpha: k += 1
    return k

# ---------------------------------------------------------------- hashing / metadata
def _file_bytes(path):
    with open(path, "rb") as f: return f.read().replace(b"\r\n", b"\n")   # Windows autocrlf must not change the hash
def config_hash(cfg): return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
def data_hash(root="."):
    files = sorted(glob.glob(f"{root}/data/corpus/*.json")) + [f"{root}/data/golden/golden.jsonl"] + sorted(glob.glob(f"{root}/data/safety/*.json")); h = hashlib.sha256()
    for f in files: h.update(os.path.basename(f).encode()); h.update(_file_bytes(f))
    return h.hexdigest()
def models_used(cfg, offline):
    from cli import needed_models
    need = needed_models(cfg)
    if offline: return {m: "stand-in" for m in need} or {"stand-ins": "hash embedder + mock LLM"}
    d = oc.model_digests(); return {m: d.get(m) or d.get(m + ":latest") or "unknown" for m in need}
def hardware_note(note=None):
    return note or f"{platform.platform()} | {platform.processor() or platform.machine()} | python {platform.python_version()}"
def apply_overrides(cfg, sets):
    """--set a.b.c=value (value parsed as YAML: false, 40, llama3.2:1b)."""
    cfg = deep_merge(cfg, {})
    for s in sets or []:
        key, _, val = s.partition("="); d = cfg; parts = key.split(".")
        for p in parts[:-1]: d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg
def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, f"{pre}{k}."))
        else: out[f"{pre}{k}"] = v
    return out
def config_diff(a, b):
    fa, fb = flatten(a), flatten(b)
    return [dict(key=k, baseline=fa.get(k, "(absent)"), new=fb.get(k, "(absent)")) for k in sorted(set(fa) | set(fb)) if fa.get(k, "(absent)") != fb.get(k, "(absent)")]

# ---------------------------------------------------------------- snapshot
def take_snapshot(cfg, offline=False, safety=True, label="", hardware=None, log=lambda m: None):
    """Run the golden eval (+ the safety suite without the detection-proof runs) and return one JSON-able snapshot."""
    t0 = time.time(); golden = load_golden(); gq = {g["id"]: g["question"] for g in golden}
    res = evaluate(cfg, log=log); s = res["summary"]; lat = s["latency"]
    questions = {}
    for r in res["rows"]:
        q = dict(type=r["type"], question=gq[r["id"]], answer=r["answer"][:200])
        for k in ("hit", "mrr", "doc_recall", "correct", "grounded", "abstained"):
            if k in r: q[k] = round(r[k], 6)
        questions[r["id"]] = q
    snap = dict(version=VERSION,
                meta=dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), label=label, offline=offline, config=cfg, config_hash=config_hash(cfg), data_hash=data_hash(),
                          models=models_used(cfg, offline), hardware=hardware_note(hardware), ollama_version="stand-in" if offline else oc.version(), wall_s=0),
                metrics=dict(n_answerable=s["n_answerable"], n_unanswerable=s["n_unanswerable"], hit_at_k=s["hit_at_k"], mrr=s["mrr"], doc_recall=s["doc_recall"],
                             correctness=s["correctness"], groundedness=s["groundedness"], abstention_accuracy=s["abstention_accuracy"], p50_ms=lat["p50_ms"], p95_ms=lat["p95_ms"],
                             tokens_in_mean=lat["tokens_in_mean"], tokens_out_mean=lat["tokens_out_mean"], tokens_per_query=round(lat["tokens_in_mean"] + lat["tokens_out_mean"], 1),
                             est_cost_usd=s["est_cost_usd"]),
                questions=questions, safety=None, safety_cases={})
    if safety:
        log("safety suite ..."); real = run_suite(cfg, load_safety(), load_docs(), golden, "regression", lambda m: None); c = real["categories"]
        snap["safety"] = dict(acl_violations=real["acl_violations"], categories={k: dict(n=c[k]["n"], passed=c[k]["passed"]) for k in SAFETY_CATS})
        for r in real["rows"]:
            if r["category"] == "injection" and not r["exposed"]: continue   # poison never reached the model: not tested, so not a pass
            snap["safety_cases"][r["id"]] = dict(category=r["category"], passed=bool(r["passed"]), question=r["question"], answer=(r.get("answer") or "")[:200])
    snap["meta"]["wall_s"] = round(time.time() - t0, 1)
    return snap

# ---------------------------------------------------------------- comparator (pure)
def load_tolerances(path="configs/tolerances.yaml"):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def paired(bq, nq, key):
    w = l = 0
    for i, b in bq.items():
        n = nq.get(i)
        if n is None or key not in b or key not in n: continue
        if n[key] > b[key] + EPS: w += 1
        elif n[key] < b[key] - EPS: l += 1
    return dict(won=w, lost=l, p=sign_test_p(w, l))

def _chk(gate, metric, base, new, rule, status, note=""):
    return dict(gate=gate, metric=metric, baseline=base, new=new, delta=None if base is None or new is None else new - base, rule=rule, status=status, note=note)

def flips(bq, nq):
    out = {}
    for gate, key, name in (("retrieval", "hit", "hit"), ("generation", "correct", "correct"), ("generation", "abstained", "abstained")):
        for i, b in bq.items():
            n = nq.get(i)
            if n is None or key not in b or key not in n or b[key] == n[key]: continue
            kind = "pass_to_fail" if b[key] > n[key] else "fail_to_pass"
            out.setdefault(gate, {"pass_to_fail": [], "fail_to_pass": []})[kind].append(dict(id=i, type=b["type"], question=b["question"], metric=name, old=b["answer"], new=n["answer"]))
    return out
def safety_flips(bs, ns):
    out = {"pass_to_fail": [], "fail_to_pass": []}
    for i, b in bs.items():
        n = ns.get(i)
        if n is None or b["passed"] == n["passed"]: continue
        out["pass_to_fail" if b["passed"] else "fail_to_pass"].append(dict(id=i, category=b["category"], question=b["question"], old=b["answer"], new=n["answer"]))
    return out

def compare(base, new, tol):
    """Judge `new` against `base` with the rules in `tol`. Returns dict(status PASS|FAIL|INVALID, exit_code, failed_gates, checks, flips, ...)."""
    bm, nm, bmeta, nmeta = base["metrics"], new["metrics"], base["meta"], new["meta"]; checks, warns, invalid = [], [], []
    floors_on = not nmeta["offline"]
    # --- the comparison must be like for like
    if base.get("version") != VERSION: invalid.append(f"baseline format version {base.get('version')} != {VERSION}")
    if bmeta["offline"] != nmeta["offline"]: invalid.append(f"baseline offline={bmeta['offline']} but this run offline={nmeta['offline']}: stand-in numbers are never comparable with model numbers")
    if bmeta["data_hash"] != nmeta["data_hash"]: invalid.append("data changed since the baseline was frozen (corpus / golden / safety files differ): per-question results are not comparable; re-freeze the baseline")
    if (base["safety"] is None) != (new["safety"] is None): invalid.append("safety suite present in only one of the two runs: a gate that skips safety is not a gate (do not use --no-safety on only one side)")
    if invalid:
        return dict(status="INVALID", exit_code=2, failed_gates=[], invalid=invalid, checks=[], flips={}, safety_flips={}, config_diff=config_diff(bmeta["config"], nmeta["config"]), warnings=[], notes=[])
    if bmeta["config_hash"] != nmeta["config_hash"]: warns.append("config differs from the baseline (see config diff)")
    if bmeta["models"] != nmeta["models"]: warns.append("model set/digests differ from the baseline: " + "; ".join(f"{m}: {bmeta['models'].get(m, '-')[:19]} -> {nmeta['models'].get(m, '-')[:19]}" for m in sorted(set(bmeta["models"]) | set(nmeta["models"])) if bmeta["models"].get(m) != nmeta["models"].get(m)))
    bv, nv = bmeta.get("ollama_version"), nmeta.get("ollama_version")
    if bv is None: warns.append("baseline has no Ollama version recorded (frozen before S6); a version change cannot be ruled out as a cause of differences")
    elif bv != nv: warns.append(f"Ollama version differs from the baseline: {bv} -> {nv}")
    if bmeta["hardware"] != nmeta["hardware"]: warns.append("hardware note differs: latency is not comparable across machines")
    if not floors_on: warns.append("OFFLINE stand-ins: absolute floors/ceilings are not applied; numbers prove the comparator, not a model")
    bq, nq = base["questions"], new["questions"]; notes = []
    # --- quality: retrieval + generation (higher is better)
    for gate in ("retrieval", "generation"):
        for metric, spec in tol[gate].items():
            _, key = HIB[metric]; b, n = bm[metric], nm[metric]; d = n - b; why = []
            nq_n = bm["n_unanswerable"] if metric == "abstention_accuracy" else bm["n_answerable"]
            if "tolerance" in spec and d < spec["tolerance"] - EPS: why.append(f"drop {d:+.3f} is below tolerance {spec['tolerance']:+.2f}")
            if floors_on and "floor" in spec and n < spec["floor"] - EPS: why.append(f"{n:.3f} is below the absolute floor {spec['floor']}")
            p = paired(bq, nq, key); rule = f"delta >= {spec.get('tolerance', 0):+.2f}" + (f"; floor {spec['floor']}" if floors_on and "floor" in spec else "")
            note = f"{p['won']} won / {p['lost']} lost, sign-test p={p['p']:.3f}; tolerance = {abs(spec.get('tolerance', 0)) * nq_n:.1f} questions of {nq_n}"
            checks.append(_chk(gate, metric, b, n, rule, "FAIL" if why else "PASS", ("; ".join(why) + " | " if why else "") + note))
    # --- safety: zero tolerance, counts
    st = tol["safety"]
    if base["safety"] is not None:
        a = new["safety"]["acl_violations"]; checks.append(_chk("safety", "acl_violations (hard gate)", base["safety"]["acl_violations"], a, f"must be <= {st['acl_violations']['max']}", "FAIL" if a > st["acl_violations"]["max"] else "PASS", "access control: always enforced"))
        sf = safety_flips(base["safety_cases"], new["safety_cases"]); k = len(sf["pass_to_fail"])
        checks.append(_chk("safety", "new failing cases", 0, k, f"must be <= {st['new_failing_cases']['max']}", "FAIL" if k > st["new_failing_cases"]["max"] else "PASS", "cases that passed in the baseline and fail now" + (": " + ", ".join(x["id"] for x in sf["pass_to_fail"][:8]) if k else "")))
        for c, spec in st["category_failures"].items():
            bf = base["safety"]["categories"][c]["n"] - base["safety"]["categories"][c]["passed"]; nf = new["safety"]["categories"][c]["n"] - new["safety"]["categories"][c]["passed"]; why = []
            if nf > bf + spec["allow_increase"]: why.append(f"{nf} failing vs baseline {bf} (+{spec['allow_increase']} allowed)")
            if floors_on and nf > spec["max"]: why.append(f"{nf} failing exceeds the absolute ceiling {spec['max']}")
            checks.append(_chk("safety", f"{c} failing cases (of {new['safety']['categories'][c]['n']})", bf, nf, f"<= baseline + {spec['allow_increase']}" + (f"; ceiling {spec['max']}" if floors_on else ""), "FAIL" if why else "PASS", "; ".join(why)))
    else: sf = {"pass_to_fail": [], "fail_to_pass": []}; warns.append("safety not run in either snapshot: safety gate NOT checked")
    # --- application: latency, tokens (lower is better, % rule + absolute slack + ceiling)
    for gate, metric, field in (("latency", "p95_ms", "p95_ms"), ("tokens", "tokens_per_query", "tokens_per_query")):
        spec = tol[gate][metric]; b, n = bm[field], nm[field]; why = []
        if n > b * (1 + spec["tolerance_pct"] / 100) + EPS and n - b > spec.get("min_abs", 0): why.append(f"+{(n / b - 1) * 100:.0f}% exceeds +{spec['tolerance_pct']}% (and more than {spec.get('min_abs', 0)} absolute)")
        if floors_on and "max" in spec and n > spec["max"]: why.append(f"{n:g} exceeds the absolute ceiling {spec['max']}")
        checks.append(_chk(gate, metric, b, n, f"<= +{spec['tolerance_pct']}%" + (f" (or < {spec['min_abs']} abs)" if spec.get("min_abs") else "") + (f"; ceiling {spec['max']}" if floors_on and "max" in spec else ""), "FAIL" if why else "PASS", "; ".join(why)))
    info = [dict(metric="p50_ms", baseline=bm["p50_ms"], new=nm["p50_ms"]), dict(metric="est_cost_usd", baseline=bm["est_cost_usd"], new=nm["est_cost_usd"])]
    failed = sorted({c["gate"] for c in checks if c["status"] == "FAIL"}, key=["retrieval", "generation", "safety", "latency", "tokens"].index)
    notes.append(f"Sign test: with n={bm['n_answerable']} answerable questions a loss only counts as statistically significant from {min_detectable()} lost questions (0 won) on; the gate is stricter on purpose (tolerance), because at temperature 0 the run-to-run noise is measured separately (noise study).")
    return dict(status="FAIL" if failed else "PASS", exit_code=1 if failed else 0, failed_gates=failed, invalid=[], checks=checks, info=info, flips=flips(bq, nq), safety_flips=sf,
                config_diff=config_diff(bmeta["config"], nmeta["config"]), warnings=warns, notes=notes, mdc_questions=min_detectable())

def check_baseline_ok(snap, tol):
    """A baseline may only be frozen if it passes the hard gate and the absolute floors (stand-ins: only the hard gate)."""
    probs = []
    if snap["safety"] and snap["safety"]["acl_violations"] > tol["safety"]["acl_violations"]["max"]: probs.append(f"ACL violations = {snap['safety']['acl_violations']} (must be 0)")
    if not snap["meta"]["offline"]:
        m = snap["metrics"]
        for gate in ("retrieval", "generation"):
            for k, spec in tol[gate].items():
                if "floor" in spec and m[k] < spec["floor"] - EPS: probs.append(f"{k} {m[k]:.3f} below floor {spec['floor']}")
        if snap["safety"]:
            for c, spec in tol["safety"]["category_failures"].items():
                f = snap["safety"]["categories"][c]["n"] - snap["safety"]["categories"][c]["passed"]
                if f > spec["max"]: probs.append(f"{c}: {f} failing cases exceed the ceiling {spec['max']}")
        for gate, k in (("latency", "p95_ms"), ("tokens", "tokens_per_query")):
            mx = tol[gate][k].get("max")
            if mx is not None and m[k] > mx: probs.append(f"{k} {m[k]:g} above ceiling {mx}")
    return probs

# ---------------------------------------------------------------- noise study
def noise_study(snaps, tol):
    """Run-to-run spread of the same config; compared with the tolerances. `snaps` = N snapshots of the same config."""
    names = ["hit_at_k", "mrr", "doc_recall", "correctness", "groundedness", "abstention_accuracy", "p50_ms", "p95_ms", "tokens_per_query"]; rows = []
    for m in names:
        v = [s["metrics"][m] for s in snaps]; t = None
        for g in ("retrieval", "generation", "latency", "tokens"):
            if m in tol.get(g, {}): t = tol[g][m]
        rows.append(dict(metric=m, values=v, range=max(v) - min(v), rel_range_pct=(max(v) - min(v)) / (sum(v) / len(v)) * 100 if sum(v) else 0.0, tolerance=t))
    srows = []
    if snaps[0]["safety"]:
        for c in SAFETY_CATS: f = [s["safety"]["categories"][c]["n"] - s["safety"]["categories"][c]["passed"] for s in snaps]; srows.append(dict(metric=f"{c} failing cases", values=f, range=max(f) - min(f)))
        a = [s["safety"]["acl_violations"] for s in snaps]; srows.append(dict(metric="acl_violations", values=a, range=max(a) - min(a)))
    unstable = []
    for i in snaps[0]["questions"]:
        for k in BINARY:
            vs = [s["questions"][i].get(k) for s in snaps]
            if None not in vs and len(set(vs)) > 1: unstable.append(dict(id=i, metric=k, values=vs))
    for i in snaps[0]["safety_cases"]:
        vs = [s["safety_cases"].get(i, {}).get("passed") for s in snaps]
        if len(set(vs)) > 1: unstable.append(dict(id=i, metric="safety", values=vs))
    return dict(runs=len(snaps), metrics=rows, safety=srows, unstable=unstable, wall_s=[s["meta"]["wall_s"] for s in snaps], offline=snaps[0]["meta"]["offline"])

# ---------------------------------------------------------------- reports
def _f(x): return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) and abs(x) < 100 else (f"{x:.1f}" if isinstance(x, float) else str(x)))
def _d(x): return "-" if x is None else (f"{x:+.3f}" if isinstance(x, float) and abs(x) < 100 else f"{x:+.1f}")

def to_markdown(cmp, base, new, base_path="baseline"):
    bm, nm = base["meta"], new["meta"]; L = [f"# Regression report: {cmp['status']}", ""]
    if nm["offline"]: L += ["> **OFFLINE STAND-INS: plumbing only.** The hash embedder and the mock LLM are not a model. Do not quote these numbers as model results.", ""]
    if cmp["status"] == "INVALID":
        L += ["**The comparison is invalid; nothing was judged (exit code 2):**", ""] + [f"- {x}" for x in cmp["invalid"]]
        return "\n".join(L) + "\n"
    L += [f"**{cmp['status']}**" + (f": failed gate(s): **{', '.join(cmp['failed_gates'])}**" if cmp["failed_gates"] else ": every gate is within its tolerance"), "",
          f"Baseline `{base_path}` ({bm['created']}, label '{bm['label']}', config {bm['config_hash'][:12]}) vs this run ({nm['created']}, label '{nm['label']}', config {nm['config_hash'][:12]}). Exit code {cmp['exit_code']}.", ""]
    if cmp["warnings"]: L += ["Warnings:", ""] + [f"- {w}" for w in cmp["warnings"]] + [""]
    L += ["## Gates", "", "| gate | metric | baseline | new | delta | rule | status | note |", "|---|---|---|---|---|---|---|---|"]
    for c in cmp["checks"]: L.append(f"| {c['gate']} | {c['metric']} | {_f(c['baseline'])} | {_f(c['new'])} | {_d(c['delta'])} | {c['rule']} | **{c['status']}** | {c['note']} |")
    L += ["", "Information only (not gated): " + "; ".join(f"{i['metric']} {_f(i['baseline'])} -> {_f(i['new'])}" for i in cmp["info"]), ""]
    L += ["## Flipped questions", ""]
    any_flip = False
    for gate in ("retrieval", "generation"):
        for kind, title in (("pass_to_fail", "pass -> FAIL"), ("fail_to_pass", "fail -> pass")):
            items = cmp["flips"].get(gate, {}).get(kind, [])
            if not items: continue
            any_flip = True; L += [f"**{gate}: {title} ({len(items)})**", "", "| id | type | check | question | baseline answer | new answer |", "|---|---|---|---|---|---|"]
            for x in items[:30]: L.append(f"| {x['id']} | {x['type']} | {x['metric']} | {x['question']} | {x['old'].replace(chr(10), ' ')[:90]} | {x['new'].replace(chr(10), ' ')[:90]} |")
            L.append("")
    for kind, title in (("pass_to_fail", "safety: pass -> FAIL"), ("fail_to_pass", "safety: fail -> pass")):
        items = cmp["safety_flips"][kind]
        if not items: continue
        any_flip = True; L += [f"**{title} ({len(items)})**", "", "| id | category | question | baseline answer | new answer |", "|---|---|---|---|---|"]
        for x in items[:30]: L.append(f"| {x['id']} | {x['category']} | {x['question']} | {x['old'].replace(chr(10), ' ')[:90]} | {x['new'].replace(chr(10), ' ')[:90]} |")
        L.append("")
    if not any_flip: L += ["None: every question and safety case has the same pass/fail outcome as in the baseline.", ""]
    L += ["## Config diff", ""]
    L += (["| key | baseline | new |", "|---|---|---|"] + [f"| {d['key']} | {d['baseline']} | {d['new']} |" for d in cmp["config_diff"]]) if cmp["config_diff"] else ["No difference (same config)."]
    L += ["", "## Models", "", "| model | baseline digest | new digest |", "|---|---|---|"] + [f"| {m} | {bm['models'].get(m, '-')[:19]} | {nm['models'].get(m, '-')[:19]} |" for m in sorted(set(bm["models"]) | set(nm["models"]))]
    L += [""] + cmp["notes"]
    return "\n".join(L) + "\n"

def console_summary(cmp):
    print("\n" + "=" * 60); print(f"REGRESSION: {cmp['status']}" + (f"   failed gate(s): {', '.join(cmp['failed_gates'])}" if cmp["failed_gates"] else ""))
    if cmp["status"] == "INVALID": [print("  -", x) for x in cmp["invalid"]]
    else:
        for c in cmp["checks"]: print(f"  [{c['status']:4s}] {c['gate']:10s} {c['metric']:32s} {_f(c['baseline']):>8s} -> {_f(c['new']):>8s}  {c['note'] if c['status'] == 'FAIL' else ''}")
        for w in cmp["warnings"]: print("  warning:", w)
        n = sum(len(v["pass_to_fail"]) for v in cmp["flips"].values()) + len(cmp["safety_flips"]["pass_to_fail"]); m = sum(len(v["fail_to_pass"]) for v in cmp["flips"].values()) + len(cmp["safety_flips"]["fail_to_pass"])
        print(f"  flipped: {n} pass->fail, {m} fail->pass; config keys changed: {len(cmp['config_diff'])}")
    print("exit code:", cmp["exit_code"])

def noise_markdown(ns, tol):
    L = [f"# Noise study: {ns['runs']} identical runs", ""]
    if ns["offline"]: L += ["> OFFLINE STAND-INS: deterministic by construction, so this only proves the plumbing.", ""]
    L += ["| metric | " + " | ".join(f"run {i + 1}" for i in range(ns["runs"])) + " | range | range % of mean | tolerance |", "|---|" + "---|" * ns["runs"] + "---|---|---|"]
    for r in ns["metrics"]:
        t = r["tolerance"]; ts = "-" if not t else (f"{t['tolerance']:+.2f}" if "tolerance" in t else f"+{t['tolerance_pct']}%")
        L.append(f"| {r['metric']} | " + " | ".join(_f(v) for v in r["values"]) + f" | {_f(r['range'])} | {r['rel_range_pct']:.1f}% | {ts} |")
    for r in ns["safety"]: L.append(f"| {r['metric']} | " + " | ".join(str(v) for v in r["values"]) + f" | {r['range']} | - | 0 |")
    L += ["", f"Questions / cases whose pass-fail outcome changed between runs: **{len(ns['unstable'])}**" + (" (" + ", ".join(f"{u['id']}:{u['metric']}" for u in ns["unstable"][:15]) + ")" if ns["unstable"] else ""), "", f"Wall time per run (s): {ns['wall_s']}"]
    return "\n".join(L) + "\n"

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, ensure_ascii=False)
def save_text(text, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(text)
def load_snapshot(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

# ---------------------------------------------------------------- CLI glue
def run_cli(cfg, a):
    tol = load_tolerances(a.tolerances); log = lambda m: print(m, flush=True)
    modes = [bool(a.save_baseline), bool(a.compare), bool(a.noise_study)]
    if sum(modes) != 1: print("give exactly one of --save-baseline PATH, --compare PATH, --noise-study N"); return 2
    kw = dict(offline=a.offline, safety=not a.no_safety, hardware=a.hardware)
    if a.noise_study:
        snaps = []
        for i in range(a.noise_study): log(f"--- noise run {i + 1}/{a.noise_study}"); snaps.append(take_snapshot(cfg, label=f"noise {i + 1}", log=lambda m: None, **kw))
        ns = noise_study(snaps, tol); out = a.out or "results/noise.json"; save_json(ns, out); save_text(noise_markdown(ns, tol), a.md or "results/noise.md")
        print(noise_markdown(ns, tol)); print("saved ->", out, "and", a.md or "results/noise.md"); return 0
    if a.save_baseline:
        snap = take_snapshot(cfg, label=a.label or "baseline", log=log, **kw); probs = check_baseline_ok(snap, tol)
        if probs: print("REFUSING to freeze this baseline:"); [print("  -", p) for p in probs]; save_json(snap, a.out or "results/regress_rejected.json"); return 1
        save_json(snap, a.save_baseline); m = snap["metrics"]
        print(f"baseline frozen -> {a.save_baseline}\n  config {snap['meta']['config_hash'][:12]}  data {snap['meta']['data_hash'][:12]}  models {snap['meta']['models']}\n  hit@k {m['hit_at_k']:.3f} mrr {m['mrr']:.3f} correctness {m['correctness']:.3f} abstention {m['abstention_accuracy']:.3f} p95 {m['p95_ms']} ms tokens/q {m['tokens_per_query']}")
        if snap["safety"]: print("  safety:", {k: f"{v['passed']}/{v['n']}" for k, v in snap["safety"]["categories"].items()}, "ACL violations", snap["safety"]["acl_violations"])
        return 0
    base = load_snapshot(a.compare)
    if not a.hardware: kw["hardware"] = base["meta"]["hardware"]; print("hardware note taken from the baseline (pass --hardware to state a different machine)")
    new = take_snapshot(cfg, label=a.label or "candidate", log=log, **kw); cmp = compare(base, new, tol)
    save_json(dict(comparison=cmp, snapshot=new), a.out or "results/regress.json"); save_text(to_markdown(cmp, base, new, a.compare), a.md or "results/regress.md")
    console_summary(cmp); print("saved ->", a.out or "results/regress.json", "and", a.md or "results/regress.md"); return cmp["exit_code"]
