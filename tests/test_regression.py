"""Regression gate, fully offline (hash embedder + MockLLM, no Ollama). These tests prove the COMPARATOR and the plumbing, not a model."""
import copy, json, os, subprocess, sys, tempfile, unittest
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
from rag.pipeline import load_cfg
from eval import regression as R

TOL = R.load_tolerances(os.path.join(ROOT, "configs/tolerances.yaml"))

def synth(offline=False):
    """A small synthetic snapshot: 69 answerable + 13 unanswerable questions, 5 safety cases."""
    q = {}
    for i in range(69): q[f"a{i:02d}"] = dict(type="factual", question=f"qa{i}", answer="ok", hit=1.0, mrr=1.0, doc_recall=1.0, correct=1.0, grounded=0.7)
    for i in range(13): q[f"u{i:02d}"] = dict(type="unanswerable", question=f"qu{i}", answer="I don't know", abstained=1.0)
    cases = {f"s{i}": dict(category=c, passed=True, question=f"sq{i}", answer="fine") for i, c in enumerate(("injection", "pii", "jailbreak", "abstention", "injection"))}
    m = dict(n_answerable=69, n_unanswerable=13, hit_at_k=1.0, mrr=1.0, doc_recall=1.0, correctness=1.0, groundedness=0.7, abstention_accuracy=1.0, p50_ms=400.0, p95_ms=800.0,
             tokens_in_mean=270.0, tokens_out_mean=25.0, tokens_per_query=295.0, est_cost_usd=0.005)
    return dict(version=R.VERSION, meta=dict(created="t", label="x", offline=offline, config={"retriever": {"k": 4}}, config_hash="c1", data_hash="d1", models={"m": "sha256:aaa"}, hardware="hw", wall_s=1),
                metrics=m, questions=q, safety=dict(acl_violations=0, categories={c: dict(n=10, passed=10) for c in R.SAFETY_CATS}), safety_cases=cases)

def lose(snap, ids, key, metric, n_total):
    """Mark questions as failed on `key` and recompute the aggregate `metric`."""
    for i in ids: snap["questions"][i][key] = 0.0
    snap["metrics"][metric] = sum(v[key] for v in snap["questions"].values() if key in v) / n_total
    return snap

class Stats(unittest.TestCase):
    def test_sign_test_exact(self):
        self.assertAlmostEqual(R.sign_test_p(0, 6), 2 / 64); self.assertAlmostEqual(R.sign_test_p(0, 5), 2 / 32); self.assertEqual(R.sign_test_p(3, 3), 1.0); self.assertEqual(R.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(R.sign_test_p(1, 8), 2 * (1 + 9) / 512)
    def test_min_detectable_change_is_six_questions(self):
        self.assertEqual(R.min_detectable(), 6)   # matches the S1 note: won 6 / lost 0 is the smallest significant change
    def test_hashes_ignore_line_endings_and_key_order(self):
        self.assertEqual(R.config_hash({"a": 1, "b": {"c": 2, "d": 3}}), R.config_hash({"b": {"d": 3, "c": 2}, "a": 1})); self.assertNotEqual(R.config_hash({"a": 1}), R.config_hash({"a": 2}))
        with tempfile.TemporaryDirectory() as d:
            for sub in ("corpus", "golden", "safety"): os.makedirs(f"{d}/data/{sub}")
            def write(nl):
                for p in ("corpus/x.json", "golden/golden.jsonl", "safety/s.json"):
                    with open(f"{d}/data/{p}", "wb") as f: f.write(("line1" + nl + "line2" + nl).encode())
                return R.data_hash(d)
            self.assertEqual(write("\n"), write("\r\n"))
    def test_overrides_and_config_diff(self):
        c = R.apply_overrides({"retriever": {"enforce_acl": True, "k": 4}, "llm": {"model": "a"}}, ["retriever.enforce_acl=false", "llm.model=llama3.2:1b", "chunking.size=25"])
        self.assertIs(c["retriever"]["enforce_acl"], False); self.assertEqual(c["llm"]["model"], "llama3.2:1b"); self.assertEqual(c["chunking"]["size"], 25); self.assertEqual(c["retriever"]["k"], 4)
        self.assertEqual([d["key"] for d in R.config_diff({"a": {"b": 1}, "c": 2}, {"a": {"b": 2}, "c": 2, "e": 1})], ["a.b", "e"])

class Comparator(unittest.TestCase):
    def cmp(self, new, base=None): return R.compare(base or synth(), new, TOL)
    def failed(self, new, base=None): return self.cmp(new, base)["failed_gates"]
    def test_identical_run_passes_and_has_no_flips(self):
        c = self.cmp(synth()); self.assertEqual((c["status"], c["exit_code"], c["failed_gates"]), ("PASS", 0, [])); self.assertEqual(c["flips"], {}); self.assertEqual(c["safety_flips"], {"pass_to_fail": [], "fail_to_pass": []})
    def test_retrieval_tolerance_boundary_one_question_ok_two_fail(self):
        self.assertEqual(self.failed(lose(synth(), ["a00"], "hit", "hit_at_k", 69)), [])           # -0.0145 >= -0.02
        self.assertEqual(self.failed(lose(synth(), ["a00", "a01"], "hit", "hit_at_k", 69)), ["retrieval"])   # -0.029
    def test_generation_tolerance_boundary_three_ok_four_fail(self):
        self.assertEqual(self.failed(lose(synth(), ["a00", "a01", "a02"], "correct", "correctness", 69)), [])   # -0.043
        c = self.cmp(lose(synth(), ["a00", "a01", "a02", "a03"], "correct", "correctness", 69)); self.assertEqual(c["failed_gates"], ["generation"])   # -0.058
        row = next(x for x in c["checks"] if x["metric"] == "correctness"); self.assertIn("0 won / 4 lost", row["note"]); self.assertEqual(len(c["flips"]["generation"]["pass_to_fail"]), 4)
    def test_improvement_never_fails_and_is_listed_as_fail_to_pass(self):
        b = lose(synth(), ["a00", "a01"], "correct", "correctness", 69); c = self.cmp(synth(), b)
        self.assertEqual(c["status"], "PASS"); self.assertEqual(len(c["flips"]["generation"]["fail_to_pass"]), 2)
    def test_floor_stops_slow_drift_after_refreeze(self):
        b = lose(synth(), [f"a{i:02d}" for i in range(20)], "correct", "correctness", 69)   # 0.71 baseline: already below the 0.80 floor
        self.assertEqual(self.failed(copy.deepcopy(b), b), ["generation"])      # same as baseline, still fails: the floor is absolute
        off = copy.deepcopy(b); off["meta"]["offline"] = True; o2 = copy.deepcopy(b); o2["meta"]["offline"] = True
        self.assertEqual(self.failed(o2, off), [])                                # stand-in runs: floors are not applied
    def test_acl_violation_is_a_hard_fail_even_if_nothing_else_changed(self):
        n = synth(); n["safety"]["acl_violations"] = 1; c = self.cmp(n); self.assertEqual(c["failed_gates"], ["safety"])
        self.assertIn("acl_violations (hard gate)", [x["metric"] for x in c["checks"] if x["status"] == "FAIL"])
    def test_safety_swap_one_fixed_one_broken_still_fails(self):
        b = synth(); b["safety_cases"]["s0"]["passed"] = False; b["safety"]["categories"]["injection"]["passed"] = 9   # baseline: injection s0 failed
        n = synth(); n["safety_cases"]["s4"]["passed"] = False; n["safety"]["categories"]["injection"]["passed"] = 9     # now s0 is fixed but s4 broke: same count (1 failing)
        c = self.cmp(n, b); self.assertEqual(c["failed_gates"], ["safety"]); self.assertEqual([x["id"] for x in c["safety_flips"]["pass_to_fail"]], ["s4"]); self.assertEqual([x["id"] for x in c["safety_flips"]["fail_to_pass"]], ["s0"])
        self.assertEqual([x["metric"] for x in c["checks"] if x["status"] == "FAIL"], ["new failing cases"])   # the category count alone would have let it through

    def test_safety_category_failures_have_zero_tolerance_and_a_ceiling(self):
        n = synth(); n["safety"]["categories"]["jailbreak"]["passed"] = 9; self.assertEqual(self.failed(n), ["safety"])
        b = synth(); b["safety"]["categories"]["injection"]["passed"] = 7; n = copy.deepcopy(b); self.assertEqual(self.failed(n, b), ["safety"])   # 3 failing > ceiling 1 even though equal to the baseline
    def test_latency_boundaries(self):
        n = synth(); n["metrics"]["p95_ms"] = 1000.0; self.assertEqual(self.failed(n), [])               # exactly +25%: allowed
        n["metrics"]["p95_ms"] = 1001.0; self.assertEqual(self.failed(n), ["latency"])
        b = synth(); b["metrics"]["p95_ms"] = 100.0; n = synth(); n["metrics"]["p95_ms"] = 140.0
        self.assertEqual(self.failed(n, b), [])                                                           # +40% but only +40 ms (< 50 ms slack)
        n["metrics"]["p95_ms"] = 2100.0; self.assertEqual(self.failed(n), ["latency"]); n2 = synth(); n2["metrics"]["p95_ms"] = 2100.0; b2 = synth(); b2["metrics"]["p95_ms"] = 2100.0; self.assertEqual(self.failed(n2, b2), ["latency"])   # ceiling
    def test_token_growth_fails(self):
        n = synth(); n["metrics"]["tokens_per_query"] = 295 * 1.20; self.assertEqual(self.failed(n), ["tokens"]); n["metrics"]["tokens_per_query"] = 295 * 1.10; self.assertEqual(self.failed(n), [])
    def test_several_failures_are_all_named(self):
        n = lose(synth(), ["a00", "a01"], "hit", "hit_at_k", 69); n["safety"]["acl_violations"] = 3; n["metrics"]["p95_ms"] = 5000.0
        self.assertEqual(self.failed(n), ["retrieval", "safety", "latency"])
    def test_invalid_comparisons_exit_2_and_judge_nothing(self):
        for mut, text in ((lambda s: s["meta"].update(data_hash="d2"), "data changed"), (lambda s: s["meta"].update(offline=True), "offline"), (lambda s: s.update(safety=None, safety_cases={}), "safety")):
            n = synth(); mut(n); c = self.cmp(n); self.assertEqual((c["status"], c["exit_code"], c["checks"]), ("INVALID", 2, [])); self.assertIn(text, " ".join(c["invalid"]))
    def test_model_digest_change_is_a_warning_with_the_names(self):
        n = synth(); n["meta"]["models"] = {"m": "sha256:bbb"}; c = self.cmp(n); self.assertEqual(c["status"], "PASS"); self.assertTrue(any("digest" in w and "sha256:aaa" in w for w in c["warnings"]))
    def test_baseline_cannot_be_frozen_if_it_breaks_the_hard_gate_or_floors(self):
        self.assertEqual(R.check_baseline_ok(synth(), TOL), []); s = synth(); s["safety"]["acl_violations"] = 2; self.assertTrue(R.check_baseline_ok(s, TOL))
        s = lose(synth(), [f"a{i:02d}" for i in range(20)], "correct", "correctness", 69); self.assertTrue(any("correctness" in p for p in R.check_baseline_ok(s, TOL)))
        s["meta"]["offline"] = True; self.assertEqual(R.check_baseline_ok(s, TOL), [])
    def test_noise_study_reports_range_and_unstable_questions(self):
        a, b = synth(), lose(synth(), ["a00"], "correct", "correctness", 69); ns = R.noise_study([a, a, b], TOL)
        self.assertEqual(next(r["range"] for r in ns["metrics"] if r["metric"] == "hit_at_k"), 0); self.assertGreater(next(r["range"] for r in ns["metrics"] if r["metric"] == "correctness"), 0)
        self.assertEqual([(u["id"], u["metric"]) for u in ns["unstable"]], [("a00", "correct")]); self.assertIn("range", R.noise_markdown(ns, TOL))

class EndToEndOffline(unittest.TestCase):
    """Real pipeline with stand-ins: a no-op passes and each deliberate break fails with the right gate named."""
    @classmethod
    def setUpClass(cls):
        cls.cfg = R.apply_overrides(load_cfg("configs/baseline.yaml"), [])
        cls.base = R.take_snapshot(cls.cfg, offline=True)
    def run_with(self, *sets): return R.compare(self.base, R.take_snapshot(R.apply_overrides(self.cfg, sets), offline=True), TOL)
    def test_snapshot_has_what_the_spec_requires(self):
        s = self.base; self.assertEqual(len(s["questions"]), 82); self.assertTrue({"config_hash", "data_hash", "models", "hardware", "config"} <= set(s["meta"]))
        self.assertTrue({"p50_ms", "p95_ms", "tokens_per_query", "hit_at_k", "correctness"} <= set(s["metrics"])); self.assertEqual(s["safety"]["acl_violations"], 0); self.assertGreater(len(s["safety_cases"]), 60)
        json.dumps(s)
    def test_noop_passes_with_zero_flips(self):
        c = self.run_with(); self.assertEqual((c["status"], c["exit_code"], c["flips"]), ("PASS", 0, {})); self.assertEqual(c["config_diff"], [])
    def test_acl_off_fails_the_safety_gate(self):
        c = self.run_with("retriever.enforce_acl=false"); self.assertEqual(c["exit_code"], 1); self.assertIn("safety", c["failed_gates"])
        self.assertTrue(any(x["metric"].startswith("acl_violations") and x["status"] == "FAIL" and x["new"] > 0 for x in c["checks"])); self.assertEqual([d["key"] for d in c["config_diff"]], ["retriever.enforce_acl"])
    def test_tiny_chunks_fail_the_retrieval_gate(self):
        c = self.run_with("chunking.strategy=fixed", "chunking.size=25", "chunking.overlap=0"); self.assertEqual(c["exit_code"], 1); self.assertIn("retrieval", c["failed_gates"]); self.assertGreater(len(c["flips"]["retrieval"]["pass_to_fail"]), 10)
    def test_vulnerable_model_swap_fails_generation_and_safety(self):
        c = self.run_with("llm.naive=true"); self.assertEqual(c["exit_code"], 1); self.assertTrue({"generation", "safety"} <= set(c["failed_gates"]))
    def test_report_names_the_failed_gate(self):
        c = self.run_with("retriever.enforce_acl=false"); new = R.take_snapshot(R.apply_overrides(self.cfg, ["retriever.enforce_acl=false"]), offline=True)
        md = R.to_markdown(c, self.base, new, "baselines/x.json"); self.assertIn("failed gate(s):", md); self.assertIn("OFFLINE STAND-INS", md); self.assertIn("## Flipped questions", md); self.assertIn("retriever.enforce_acl", md)

def read(path):
    with open(path, encoding="utf-8") as f: return f.read()

class CliExitCodes(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "cli.py", "regress", "--offline", *args], cwd=ROOT, capture_output=True, text=True)
    def test_save_compare_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            b = f"{d}/b.json"; r = self.run_cli("--save-baseline", b); self.assertEqual(r.returncode, 0, r.stderr + r.stdout); self.assertTrue(os.path.exists(b))
            ok = self.run_cli("--compare", b, "--out", f"{d}/n.json", "--md", f"{d}/n.md"); self.assertEqual(ok.returncode, 0, ok.stdout[-800:]); self.assertIn("PASS", read(f"{d}/n.md"))
            bad = self.run_cli("--compare", b, "--set", "retriever.enforce_acl=false", "--out", f"{d}/m.json", "--md", f"{d}/m.md"); self.assertEqual(bad.returncode, 1); self.assertIn("failed gate(s):", read(f"{d}/m.md"))
            inv = self.run_cli("--compare", b, "--no-safety", "--out", f"{d}/i.json", "--md", f"{d}/i.md"); self.assertEqual(inv.returncode, 2)
    def test_compare_inherits_hardware_note_unless_given(self):
        with tempfile.TemporaryDirectory() as d:
            b = f"{d}/b.json"; self.assertEqual(self.run_cli("--save-baseline", b, "--hardware", "my laptop").returncode, 0)
            self.assertEqual(self.run_cli("--compare", b, "--out", f"{d}/n.json", "--md", f"{d}/n.md").returncode, 0); self.assertNotIn("hardware note differs", read(f"{d}/n.md"))
            self.assertEqual(self.run_cli("--compare", b, "--hardware", "other pc", "--out", f"{d}/o.json", "--md", f"{d}/o.md").returncode, 0); self.assertIn("hardware note differs", read(f"{d}/o.md"))
    def test_refuses_to_freeze_a_baseline_that_leaks(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("--save-baseline", f"{d}/b.json", "--set", "retriever.enforce_acl=false", "--out", f"{d}/rej.json"); self.assertEqual(r.returncode, 1); self.assertFalse(os.path.exists(f"{d}/b.json")); self.assertIn("REFUSING", r.stdout)

if __name__ == "__main__": unittest.main()
