"""Judge validation (S4): how far can the LLM judge be trusted? Compared with ~30 human labels (agreement, Cohen's kappa, no sklearn)
and with itself across 3 runs (variance).  Pure functions first, commands (`cmd_make_labels`, `cmd_validate`) at the bottom.

Human label scheme (see data/labels/LABELING_GUIDE.md):
  correct | incorrect | hallucinated | properly_abstained | evasive
Binary targets derived from it (assumptions, stated in docs/results/S4_judge.md):
  correctness : correct=1; incorrect, hallucinated, evasive=0; properly_abstained excluded
  faithfulness: correct=1; hallucinated=0; others excluded (an "incorrect" answer may or may not contradict the context)
  relevancy   : correct, incorrect, hallucinated=1 (on topic); evasive=0; properly_abstained excluded
The 5-class judge label is a FIXED rule over the judge scores and the abstain check (`judge_label`); thresholds come from configs/judge.yaml and are not
tuned on the human labels (the threshold sweep is an in-sample diagnostic only).
"""
import csv, json, math, os, random, re, sys, time
from collections import Counter

from rag.llm import is_abstain

LABELS = ("correct", "incorrect", "hallucinated", "properly_abstained", "evasive")
COLS = ["id", "type", "role", "question", "reference", "contexts", "answer", "human_label", "notes"]
CTX_SEP = " ||| "
QUOTAS = dict(A=11, B=9, C=3, D=5, E=2)    # A answerable+keyword-correct, B answerable+keyword-wrong, C answerable+abstained, D unanswerable+abstained, E unanswerable+answered


# ---------------------------------------------------------------- statistics (stdlib)
def agreement(a, b):
    if len(a) != len(b): raise ValueError("label lists differ in length")
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else None

def cohens_kappa(a, b):
    """Cohen's kappa for two raters over any labels. None when undefined (no items, or both raters use one identical label throughout)."""
    n = len(a)
    if n == 0: return None
    po = agreement(a, b); ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    return None if pe >= 1.0 else (po - pe) / (1 - pe)

def kappa_ci(a, b, reps=2000, seed=0):
    """95% percentile-bootstrap interval of kappa (resamples where kappa is undefined are skipped; None if fewer than half are usable)."""
    n = len(a); rng = random.Random(seed); ks = []
    if n == 0: return None
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]; k = cohens_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None: ks.append(k)
    if len(ks) < reps / 2: return None
    ks.sort(); return [ks[int(0.025 * len(ks))], ks[min(len(ks) - 1, int(0.975 * len(ks)))]]

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]

def kappa_band(k):
    if k is None: return "undefined"
    return "poor (worse than chance)" if k < 0 else "slight" if k <= .2 else "fair" if k <= .4 else "moderate" if k <= .6 else "substantial" if k <= .8 else "almost perfect"

def confusion(human, judge, labels):
    """rows = human label, columns = judge label (same order as `labels`)."""
    c = Counter(zip(human, judge)); return [[c[(h, j)] for j in labels] for h in labels]

def binary_stats(truth, pred, min_class=5):
    n = len(truth); pos = sum(truth); d = dict(n=n, human_pos=pos, human_neg=n - pos)
    if n == 0: d["note"] = "no items"; return d
    ok = sum(t == p for t, p in zip(truth, pred))
    d.update(agreement=ok / n, agreement_ci=wilson(ok, n), kappa=cohens_kappa(truth, pred), kappa_ci=kappa_ci(truth, pred),
             tp=sum(t == 1 and p == 1 for t, p in zip(truth, pred)), fp=sum(t == 0 and p == 1 for t, p in zip(truth, pred)),
             fn=sum(t == 1 and p == 0 for t, p in zip(truth, pred)), tn=sum(t == 0 and p == 0 for t, p in zip(truth, pred)))
    if min(pos, n - pos) < min_class: d["note"] = f"only {min(pos, n - pos)} example(s) of the rarer class: agreement and kappa are not reliable"
    return d

def threshold_sweep(truth, scores, grid=(0.3, 0.5, 0.7, 0.9)):
    out = []
    for t in grid:
        pred = [int(s >= t) for s in scores]; out.append(dict(threshold=t, agreement=agreement(truth, pred), kappa=cohens_kappa(truth, pred)))
    return out


# ---------------------------------------------------------------- label template
def _round_robin(pool, rng):
    by = {}
    for it in pool: by.setdefault(it["type"], []).append(it)
    for v in by.values(): rng.shuffle(v)
    order, keys = [], sorted(by)
    while any(by.values()):
        for k in keys:
            if by[k]: order.append(by[k].pop())
    return order

def stratum(it):
    ab = is_abstain(it["answer"])
    if it["answerable"]: return "C" if ab else ("A" if it.get("kw_correct") else "B")
    return "D" if ab else "E"

def select_for_labeling(items, seed=7, quotas=None):
    """~30 answers for the human, chosen by DETERMINISTIC signals only (never by the judge's output, which would make validation circular):
    quotas per stratum (see QUOTAS), round-robin over question types inside a stratum, shortfalls filled from the other strata, order shuffled
    so the human does not see grouped strata."""
    q = quotas or QUOTAS; rng = random.Random(seed); strata = {k: [] for k in q}
    for it in items: strata[stratum(it)].append(it)
    picked, left = [], []
    for k in q:
        order = _round_robin(strata[k], rng); picked += order[:q[k]]; left += order[q[k]:]
    short = sum(q.values()) - len(picked)
    if short > 0: rng.shuffle(left); picked += left[:short]
    rng.shuffle(picked); return picked

def norm_label(s): return re.sub(r"[\s-]+", "_", (s or "").strip().lower())

def read_labels(path):
    """-> (labeled rows with `label` and `contexts_list`, number of unlabeled rows). Raises on unknown labels (lists them)."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
    except FileNotFoundError: raise FileNotFoundError(f"{path} not found. Create it with: python cli.py judge --make-labels")
    miss = [c for c in COLS if rows and c not in rows[0]]
    if miss: raise ValueError(f"{path}: missing columns {miss}")
    ok, unl, bad = [], 0, []
    for r in rows:
        lab = norm_label(r.get("human_label"))
        if not lab: unl += 1
        elif lab not in LABELS: bad.append((r["id"], r["human_label"]))
        else: ok.append(dict(r, label=lab, contexts_list=[c.strip() for c in r["contexts"].split(CTX_SEP) if c.strip()]))
    if bad: raise ValueError("invalid human_label values " + ", ".join(f"{i}: {v!r}" for i, v in bad) + f". Allowed: {', '.join(LABELS)}")
    return ok, unl

def has_labels(path):
    try: return bool(read_labels(path)[0])
    except FileNotFoundError: return False
    except ValueError: return True        # something was typed: be conservative

def write_template(path, items, force=False):
    if os.path.exists(path) and has_labels(path) and not force:
        raise FileExistsError(f"{path} already contains human labels; refusing to overwrite (use --force to replace it)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:     # BOM so Excel opens it as UTF-8
        w = csv.DictWriter(f, COLS); w.writeheader()
        for it in items:
            w.writerow(dict(id=it["id"], type=it["type"], role=it["role"], question=it["question"], reference=it["reference"] if it["answerable"] else "(none: not in the knowledge base)",
                            contexts=CTX_SEP.join(it["contexts"]), answer=it["answer"], human_label="", notes=""))


# ---------------------------------------------------------------- targets and the fixed 5-class judge rule
def target_correct(label): return {"correct": 1, "incorrect": 0, "hallucinated": 0, "evasive": 0}.get(label)
def target_faithful(label): return {"correct": 1, "hallucinated": 0}.get(label)
def target_relevant(label): return {"correct": 1, "incorrect": 1, "hallucinated": 1, "evasive": 0}.get(label)

def judge_label(it, sc, thr):
    """Judge verdict in the human label space. sc = {metric: score or None}. Deterministic abstain check first; None when a needed score is missing (judge error)."""
    if is_abstain(it["answer"]): return "evasive" if it["answerable"] else "properly_abstained"
    c, r, f = sc.get("correctness"), sc.get("answer_relevancy"), sc.get("faithfulness")
    if it["answerable"]:
        if c is None: return None
        if c >= thr["correctness"]: return "correct"
    if r is None or f is None: return None
    if r < thr["answer_relevancy"]: return "evasive"
    if f < thr["faithfulness"]: return "hallucinated"
    return "incorrect" if it["answerable"] else "hallucinated"


# ---------------------------------------------------------------- report
def variance_report(runs, ids, names, thr):
    """Run-to-run variance over JUDGED items (gate decisions are deterministic and would dilute it)."""
    out = {}
    for n in names:
        ok = [i for i in ids if all(r[i][n]["judged"] and r[i][n]["score"] is not None for r in runs)]
        if not ok: continue
        rng_ = [max(r[i][n]["score"] for r in runs) - min(r[i][n]["score"] for r in runs) for i in ok]
        means = [sum(r[i][n]["score"] for i in ok) / len(ok) for r in runs]
        ver = [[int(r[i][n]["score"] >= thr[n]) for i in ok] for r in runs]
        out[n] = dict(n=len(ok), runs=len(runs), mean_range=sum(rng_) / len(ok), max_range=max(rng_), share_changed=sum(x > 1e-9 for x in rng_) / len(ok),
                      run_means=[round(m, 3) for m in means], mean_spread=max(means) - min(means),
                      verdict_flips=sum(len({v[k] for v in ver}) > 1 for k in range(len(ok))),
                      pairwise_kappa=[cohens_kappa(ver[a], ver[b]) for a in range(len(runs)) for b in range(a + 1, len(runs))])
    return out

def build_report(items, runs, thr, names):
    """items: labeled items (with `label`, `kw_correct`); runs: list of {id: {metric: record}} (runs[0] = production seed)."""
    main = runs[0]; sc = lambda it, n: main[it["id"]][n]["score"]; rep = dict(n_labeled=len(items), label_counts=dict(Counter(it["label"] for it in items)),
                                                                           thresholds=thr, metrics=names, judge_errors={n: sum(bool(main[it["id"]][n]["error"]) for it in items) for n in names}, binary={})
    for n, tf in dict(correctness=target_correct, faithfulness=target_faithful, answer_relevancy=target_relevant).items():
        if n not in names: continue
        pairs = [(tf(it["label"]), sc(it, n), it) for it in items]; pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        truth = [p[0] for p in pairs]; rep["binary"][n] = dict(stats=binary_stats(truth, [int(p[1] >= thr[n]) for p in pairs]), sweep=threshold_sweep(truth, [p[1] for p in pairs]), n_excluded=len(items) - len(pairs))
        if n == "correctness":                                       # judge vs the old keyword metric on the same answerable items
            sub = [p for p in pairs if p[2].get("kw_correct") is not None]; t2 = [p[0] for p in sub]
            rep["keyword_comparison"] = dict(n=len(sub), judge=binary_stats(t2, [int(p[1] >= thr[n]) for p in sub]), keyword=binary_stats(t2, [int(bool(p[2]["kw_correct"])) for p in sub]))
    if all(n in names for n in ("faithfulness", "answer_relevancy", "correctness")):
        jl = {it["id"]: judge_label(it, {n: sc(it, n) for n in names}, thr) for it in items}; ok = [it for it in items if jl[it["id"]] is not None]
        h = [it["label"] for it in ok]; j = [jl[it["id"]] for it in ok]; agree = sum(x == y for x, y in zip(h, j))
        rep["five_class"] = dict(n=len(ok), n_undecidable=len(items) - len(ok), agreement=agreement(h, j), agreement_ci=wilson(agree, len(ok)), kappa=cohens_kappa(h, j), kappa_ci=kappa_ci(h, j),
                                 labels=list(LABELS), confusion=confusion(h, j, LABELS), judge_labels=dict(Counter(j)))
        rep["disagreements"] = [dict(id=it["id"], human=it["label"], judge=jl[it["id"]], answer=it["answer"][:160], scores={n: (None if sc(it, n) is None else round(sc(it, n), 2)) for n in names},
                                     reasons={n: (main[it["id"]][n]["reason"] or "")[:160] for n in names if main[it["id"]][n]["judged"]}) for it in ok if jl[it["id"]] != it["label"]]
    if len(runs) > 1 and "correctness" in names:                     # how much does the headline agreement depend on the seed? and does a median over runs help?
        pairs = [(target_correct(it["label"]), [r[it["id"]]["correctness"]["score"] for r in runs]) for it in items]
        pairs = [(t, sc_) for t, sc_ in pairs if t is not None and all(x is not None for x in sc_)]; truth = [t for t, _ in pairs]; t_c = thr["correctness"]
        med = lambda xs: sorted(xs)[len(xs) // 2]
        rep["correctness_by_run"] = dict(n=len(pairs), per_run=[dict(agreement=agreement(truth, [int(sc_[k] >= t_c) for _, sc_ in pairs]), kappa=cohens_kappa(truth, [int(sc_[k] >= t_c) for _, sc_ in pairs])) for k in range(len(runs))],
                                         median=dict(agreement=agreement(truth, [int(med(sc_) >= t_c) for _, sc_ in pairs]), kappa=cohens_kappa(truth, [int(med(sc_) >= t_c) for _, sc_ in pairs])))
    if len(runs) > 1: rep["variance"] = variance_report(runs, [it["id"] for it in items], names, thr)
    return rep

def _f(x, d=2): return "n/a" if x is None else f"{x:.{d}f}"
def _ci(c): return "n/a" if not c or c[0] is None else f"{c[0]:.2f} to {c[1]:.2f}"

def render_md(rep, meta):
    L = ["# Judge validation (S4)", "", f"Judge `{meta['judge_model']}` (digest {meta['judge_digest']}), generator `{meta['generator_model']}`, DeepEval {meta['deepeval']}, {meta['date']}. "
         f"Human labels: n={rep['n_labeled']} ({rep['label_counts']}). Thresholds fixed in advance: {rep['thresholds']}. Judge settings: {meta['judge_settings']}.",
         "", "Agreement intervals are Wilson 95%; kappa intervals are percentile bootstrap (2000 resamples). With n about 30 intervals are wide: treat the point values as rough.", "",
         "## Agreement with the human labels (binary targets, thresholds fixed in advance)", "", "| judge metric vs human target | n | human + / - | agreement [95% CI] | kappa [95% CI] | band | note |", "|---|---|---|---|---|---|---|"]
    for n, b in rep["binary"].items():
        s = b["stats"]; L.append(f"| {n} | {s['n']} | {s['human_pos']} / {s['human_neg']} | {_f(s.get('agreement'))} [{_ci(s.get('agreement_ci'))}] | {_f(s.get('kappa'))} [{_ci(s.get('kappa_ci'))}] | {kappa_band(s.get('kappa'))} | {s.get('note', '')} (excluded {b['n_excluded']}) |")
    k = rep.get("keyword_comparison")
    if k:
        L += ["", f"### Judge correctness vs the old keyword metric, same {k['n']} answerable items", "", "| metric | agreement [CI] | kappa [CI] | FP / FN |", "|---|---|---|---|"]
        for nm in ("judge", "keyword"):
            s = k[nm]; L.append(f"| {nm} | {_f(s.get('agreement'))} [{_ci(s.get('agreement_ci'))}] | {_f(s.get('kappa'))} [{_ci(s.get('kappa_ci'))}] | {s.get('fp', 'n/a')} / {s.get('fn', 'n/a')} |")
    f5 = rep.get("five_class")
    if f5:
        L += ["", f"## Five-class agreement (fixed rule over judge scores + abstain check): n={f5['n']} ({f5['n_undecidable']} undecidable because of judge errors)", "",
              f"Agreement {_f(f5['agreement'])} [{_ci(f5['agreement_ci'])}], kappa {_f(f5['kappa'])} [{_ci(f5['kappa_ci'])}] ({kappa_band(f5['kappa'])}). Rows = human, columns = judge.", "",
              "| human \\ judge | " + " | ".join(f5["labels"]) + " |", "|---|" + "---|" * len(f5["labels"])]
        L += [f"| {l} | " + " | ".join(str(x) for x in row) + " |" for l, row in zip(f5["labels"], f5["confusion"])]
    L += ["", "## Threshold sweep (diagnostic, IN-SAMPLE: picking the best row from the same 30 labels would be optimistic)", "", "| metric | " + " | ".join(f"{s['threshold']}" for s in next(iter(rep["binary"].values()))["sweep"]) + " |", "|---|---|---|---|---|"] if rep["binary"] else []
    for n, b in rep["binary"].items(): L.append(f"| {n} (agreement / kappa) | " + " | ".join(f"{_f(s['agreement'])} / {_f(s['kappa'])}" for s in b["sweep"]) + " |")
    cr = rep.get("correctness_by_run")
    if cr: L += ["", f"## Correctness vs human by run (n={cr['n']}, items scored in every run)", "", "| run | agreement | kappa |", "|---|---|---|"] + [f"| {k + 1} | {_f(x['agreement'])} | {_f(x['kappa'])} |" for k, x in enumerate(cr["per_run"])] + [f"| median of runs | {_f(cr['median']['agreement'])} | {_f(cr['median']['kappa'])} |"]
    v = rep.get("variance")
    if v:
        L += ["", "## Run-to-run variance (judged items only; seeds differ per run)", "", "| metric | n | runs | mean range | max range | items whose score changed | run means | verdict flips | pairwise kappa |", "|---|---|---|---|---|---|---|---|---|"]
        for n, x in v.items(): L.append(f"| {n} | {x['n']} | {x['runs']} | {_f(x['mean_range'], 3)} | {_f(x['max_range'], 2)} | {_f(x['share_changed'])} | {x['run_means']} | {x['verdict_flips']} | {[_f(k) for k in x['pairwise_kappa']]} |")
        L += ["", "At temperature 0 the seed has no effect, so near-zero variance is a determinism check; it says nothing about robustness to prompt wording or to a different judge model."]
    d = rep.get("disagreements")
    if d is not None:
        L += ["", f"## Human vs judge disagreements ({len(d)})", "", "| id | human | judge | scores | answer | judge reasons |", "|---|---|---|---|---|---|"]
        L += [f"| {x['id']} | {x['human']} | {x['judge']} | {x['scores']} | {x['answer'].replace('|', '/')} | {'; '.join(f'{a}: {b}' for a, b in x['reasons'].items()).replace('|', '/')} |" for x in d]
    L += ["", f"Judge errors (items without a score, excluded above): {rep['judge_errors']}"]
    return "\n".join(L) + "\n"

def console_summary(rep, meta):
    L = [f"\nJUDGE VALIDATION  judge={meta['judge_model']}  generator={meta['generator_model']}  n_labeled={rep['n_labeled']} {rep['label_counts']}"]
    for n, b in rep["binary"].items():
        s = b["stats"]; L.append(f"  {n:18s} n={s['n']} (human +{s['human_pos']}/-{s['human_neg']})  agreement {_f(s.get('agreement'))} [{_ci(s.get('agreement_ci'))}]  kappa {_f(s.get('kappa'))} [{_ci(s.get('kappa_ci'))}] {kappa_band(s.get('kappa'))}" + (f"  NOTE: {s['note']}" if s.get("note") else ""))
    k = rep.get("keyword_comparison")
    if k: L.append(f"  keyword metric on the same {k['n']} items: agreement {_f(k['keyword'].get('agreement'))}, kappa {_f(k['keyword'].get('kappa'))}   (judge: {_f(k['judge'].get('agreement'))}, {_f(k['judge'].get('kappa'))})")
    f5 = rep.get("five_class")
    if f5: L.append(f"  five-class: n={f5['n']} agreement {_f(f5['agreement'])} [{_ci(f5['agreement_ci'])}] kappa {_f(f5['kappa'])} [{_ci(f5['kappa_ci'])}] {kappa_band(f5['kappa'])}")
    cr = rep.get("correctness_by_run")
    if cr: L.append(f"  correctness by run (n={cr['n']}): agreement " + ", ".join(_f(x["agreement"]) for x in cr["per_run"]) + f"  kappa " + ", ".join(_f(x["kappa"]) for x in cr["per_run"]) + f"  | median of runs: agreement {_f(cr['median']['agreement'])} kappa {_f(cr['median']['kappa'])}")
    for n, x in (rep.get("variance") or {}).items(): L.append(f"  variance {n:18s} n={x['n']} runs={x['runs']} mean range {_f(x['mean_range'], 3)} max {_f(x['max_range'])} verdict flips {x['verdict_flips']} run means {x['run_means']}")
    L.append(f"  judge errors: {rep['judge_errors']}"); return "\n".join(L)


# ---------------------------------------------------------------- commands
def cmd_make_labels(cfg, jc, a, log=print):
    from eval import judge as J
    path = a.labels or jc["labels"]["file"]
    if os.path.exists(path) and has_labels(path) and not a.force: raise SystemExit(f"{path} already contains human labels; refusing to overwrite (use --force to replace it).")
    items = J.collect_answers(cfg, None, log); gen = J._gen_model(cfg)
    if gen: J.unload(gen)
    chosen = select_for_labeling(items, jc["labels"]["seed"]); write_template(path, chosen, force=True)
    log(f"\nwrote {len(chosen)} answers to {path}  (strata A/B/C/D/E: {dict(Counter(stratum(it) for it in chosen))})")
    log("Next: open it in Excel/Sheets, fill the `human_label` column (correct | incorrect | hallucinated | properly_abstained | evasive) following data/labels/LABELING_GUIDE.md,")
    log("save as CSV (keep UTF-8), then run: python cli.py judge --validate"); return 0

def cmd_validate(cfg, jc, a, log=print):
    from eval import judge as J
    from rag.pipeline import load_golden
    from rag.metrics import correctness as kw_correctness
    path = a.labels or jc["labels"]["file"]; rows, unl = read_labels(path); G = {g["id"]: g for g in load_golden()}
    if not rows: raise SystemExit(f"{path} has no labeled rows: fill the human_label column first.")
    if len(rows) < 25: log(f"warning: only {len(rows)} labeled rows ({unl} unlabeled); the brief asks for about 30 and intervals will be very wide")
    names = a.metrics.split(",") if a.metrics else jc["metrics"]; thr = jc["thresholds"]; gen = J._gen_model(cfg)
    for w in J.check_distinct(jc["judge"]["model"], gen): log("warning: " + w)
    items = []
    for r in rows:
        g = G[r["id"]]; items.append(dict(id=r["id"], type=g["type"], role=g["role"], question=r["question"], reference=g.get("reference", "") if g["answerable"] else "", answerable=bool(g["answerable"]),
                                            contexts=r["contexts_list"], answer=r["answer"], label=r["label"], kw_correct=float(kw_correctness(r["answer"], g)) if g["answerable"] else None))
    runs, n_runs = [], a.runs or jc["runs"]; t0 = time.time(); stats = []
    for k in range(n_runs):
        judge = J.make_judge(jc, seed=jc["judge"]["seed"] + k, temperature=a.temperature, cache=not a.no_cache); fns = J.deepeval_metric_fns(judge, thr, names)
        log(f"--- run {k + 1}/{n_runs}: seed {judge.seed}, temperature {judge.temperature}, {len(items)} items")
        runs.append(J.score_all(items, fns, log)); stats.append(judge.stats)
    J.unload(jc["judge"]["model"])
    rep = build_report(items, runs, thr, names); meta = J.run_meta(jc, gen); meta["temperature_used"] = a.temperature if a.temperature is not None else jc["judge"]["temperature"]
    os.makedirs("results", exist_ok=True); out = a.out or "results/judge_validation.json"; md = a.md or "results/judge_validation.md"
    with open(out, "w", encoding="utf-8") as f: json.dump(dict(meta=meta, report=rep, judge_stats=stats, wall_min=round((time.time() - t0) / 60, 1), runs=runs), f, indent=1, ensure_ascii=False)
    with open(md, "w", encoding="utf-8") as f: f.write(render_md(rep, meta))
    log(console_summary(rep, meta)); log(f"saved -> {out} and {md}   ({(time.time() - t0) / 60:.1f} min)"); return 0


KEYS = {"1": "correct", "2": "incorrect", "3": "hallucinated", "4": "properly_abstained", "5": "evasive"}

def _save_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f: w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)

def cmd_label(cfg, jc, a, log=print, ask=input):
    """Terminal labeler: one answer at a time, keys 1-5, s = skip, q = quit. Saves after every label, resumes on the next run (only unlabeled rows are shown)."""
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    path = a.labels or jc["labels"]["file"]
    try:
        with open(path, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
    except FileNotFoundError: raise SystemExit(f"{path} not found. Create it first: python cli.py judge --make-labels")
    todo = [r for r in rows if not norm_label(r["human_label"])]
    log(f"{len(rows) - len(todo)} labeled, {len(todo)} to go. Definitions: data/labels/LABELING_GUIDE.md. Judge by the reference and the contexts, not real-world truth.")
    for n, r in enumerate(todo, 1):
        log(f"\n===== {n}/{len(todo)}  {r['id']}  ({r['type']}) =====\nQUESTION : {r['question']}\nREFERENCE: {r['reference']}")
        for i, c in enumerate([c for c in r["contexts"].split(CTX_SEP) if c.strip()], 1): log(f"CONTEXT {i}: {c.strip()}")
        log(f"ANSWER   : {r['answer']}\n1 correct | 2 incorrect | 3 hallucinated | 4 properly_abstained | 5 evasive | s skip | q quit")
        while True:
            k = ask("label> ").strip().lower()
            if k in KEYS: r["human_label"] = KEYS[k]; _save_rows(path, rows); break
            if k == "s": break
            if k == "q": _save_rows(path, rows); log("saved. Run the same command to continue."); return 0
            log("press 1-5, s or q")
    left = sum(1 for r in rows if not norm_label(r["human_label"]))
    log(f"\ndone: {len(rows) - left}/{len(rows)} labeled" + (f" ({left} skipped: run again to label them)" if left else ". Next: python cli.py judge --validate")); return 0
