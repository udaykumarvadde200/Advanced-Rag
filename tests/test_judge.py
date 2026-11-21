"""S4 tests. Everything here is offline: Ollama is replaced by a patched `oc.chat`, DeepEval by a minimal stub module.
So these prove OUR logic (JSON handling, gating, statistics, label template, orchestration) and our side of the DeepEval contract,
NOT how DeepEval or a real judge model behaves: that is only known from runs on the user's machine."""
import argparse, csv, json, os, sys, tempfile, types, unittest
from unittest import mock

from rag import ollama_client as oc
from rag.pipeline import deep_merge, load_cfg
from eval import judge as J, judge_validation as V


class Verdict:
    """pydantic-like schema (pydantic itself is not installed in the sandbox)."""
    @classmethod
    def model_json_schema(cls): return {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}
    @classmethod
    def model_validate(cls, d):
        if not isinstance(d, dict) or "verdict" not in d: raise ValueError("verdict missing")
        o = cls(); o.verdict = d["verdict"]; return o


def reply(text, prompt_tokens=100):
    return dict(message=dict(content=text), prompt_eval_count=prompt_tokens, eval_count=7, eval_duration=1_000_000)


class JsonTests(unittest.TestCase):
    def test_plain_fenced_embedded_trailing_comma(self):
        self.assertEqual(J.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(J.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(J.extract_json('Sure! Here you go: {"a": [1, 2]} hope that helps'), {"a": [1, 2]})
        self.assertEqual(J.extract_json('{"a": 1, "b": [1, 2,],}'), {"a": 1, "b": [1, 2]})
    def test_braces_and_escaped_quotes_inside_strings(self):
        self.assertEqual(J.extract_json('text {"a": "x } y \\" z {"} tail'), {"a": 'x } y " z {'})
    def test_prefers_object_over_citation_brackets(self):
        self.assertEqual(J.extract_json('see [1] then {"v": 2}'), {"v": 2})
    def test_no_json_raises(self):
        with self.assertRaises(ValueError): J.extract_json("I think the answer is yes")


class JudgeTests(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.mkdtemp(); self.cache_path = os.path.join(self.tmp, "c.jsonl")
    def judge(self, **kw): return J.OllamaJudge("qwen2.5:7b", host="http://x", cache=J.JudgeCache(self.cache_path), **kw)

    def test_free_text_and_schema_format_passed(self):
        with mock.patch.object(oc, "chat", return_value=reply("hello")) as c:
            self.assertEqual(self.judge().complete("hi"), "hello"); self.assertIsNone(c.call_args.kwargs.get("fmt", c.call_args.args[6] if len(c.call_args.args) > 6 else None))
        with mock.patch.object(oc, "chat", return_value=reply('{"verdict": "yes"}')) as c:
            out = self.judge().complete("hi2", Verdict); self.assertEqual(out.verdict, "yes")
            fmt = c.call_args.args[6] if len(c.call_args.args) > 6 else c.call_args.kwargs["fmt"]; self.assertEqual(fmt, Verdict.model_json_schema())
    def test_json_mode_variants(self):
        for mode, want in (("json", "json"), ("none", None)):
            with mock.patch.object(oc, "chat", return_value=reply('{"verdict": "y"}')) as c:
                J.OllamaJudge("m", host="http://x", json_mode=mode).complete("p", Verdict); a = c.call_args
                self.assertEqual(a.args[6] if len(a.args) > 6 else a.kwargs["fmt"], want)
    def test_repair_retry_then_success(self):
        j = self.judge()
        with mock.patch.object(oc, "chat", side_effect=[reply("not json"), reply('{"verdict": "ok"}')]) as c:
            self.assertEqual(j.complete("p", Verdict).verdict, "ok")
            self.assertEqual(j.stats["retries"], 1); self.assertEqual(c.call_count, 2)
            first, second = c.call_args_list
            self.assertEqual(first.args[3]["seed"] + 1, second.args[3]["seed"]); self.assertIn("ONLY the JSON", second.args[2][0]["content"])
    def test_failure_after_attempts(self):
        j = self.judge(attempts=2)
        with mock.patch.object(oc, "chat", return_value=reply("nope")):
            with self.assertRaises(J.JudgeError) as e: j.complete("p", Verdict)
        self.assertIn("nope", str(e.exception)); self.assertEqual(j.stats["failures"], 1)
    def test_cache_hit_and_seed_separation_and_persistence(self):
        with mock.patch.object(oc, "chat", return_value=reply('{"verdict": "a"}')) as c:
            j = self.judge(seed=42); j.complete("p", Verdict); j.complete("p", Verdict); self.assertEqual(c.call_count, 1); self.assertEqual(j.stats["cache_hits"], 1)
            self.judge(seed=43).complete("p", Verdict); self.assertEqual(c.call_count, 2)          # other seed: no shared entry (variance runs stay honest)
            self.judge(seed=42).complete("p", Verdict); self.assertEqual(c.call_count, 2)          # fresh object, same file: still cached
    def test_failed_replies_are_not_cached(self):
        j = self.judge(attempts=1)
        with mock.patch.object(oc, "chat", return_value=reply("bad")):
            with self.assertRaises(J.JudgeError): j.complete("p", Verdict)
        with mock.patch.object(oc, "chat", return_value=reply('{"verdict": "ok"}')) as c: self.assertEqual(j.complete("p", Verdict).verdict, "ok"); self.assertEqual(c.call_count, 1)
    def test_context_limit_and_think_tags(self):
        j = self.judge(num_ctx=4096)
        with mock.patch.object(oc, "chat", return_value=reply("<think>hmm</think>answer", prompt_tokens=4090)): self.assertEqual(j.complete("p"), "answer")
        self.assertEqual(j.stats["near_ctx"], 1)
    def test_distinct_and_unload(self):
        with self.assertRaises(J.JudgeError): J.check_distinct("llama3.2:3b", "llama3.2:3b")
        self.assertEqual(len(J.check_distinct("llama3.2:1b", "llama3.2:3b")), 1)      # same family: warning only
        self.assertEqual(J.check_distinct("qwen2.5:7b", "llama3.2:3b"), []); self.assertEqual(J.check_distinct("qwen2.5:7b", None), [])
        with mock.patch.object(oc, "_post", return_value={}) as p:
            self.assertTrue(J.unload("m", "http://x")); self.assertEqual(p.call_args.args[2], dict(model="m", keep_alive=0))
        with mock.patch.object(oc, "_post", side_effect=oc.OllamaError("x")): self.assertFalse(J.unload("m"))


def item(**kw):
    d = dict(id="q1", type="factual", role="employee", question="Q?", reference="24 days", answerable=True, contexts=["ctx"], answer="24 days [1]", kw_correct=1.0); d.update(kw); return d


class GateTests(unittest.TestCase):
    def fns(self): self.calls = []; return {n: (lambda it, n=n: (self.calls.append(n), (0.9, "r"))[1]) for n in ("faithfulness", "answer_relevancy", "correctness")}
    def test_abstain_on_answerable(self):
        s = J.score_item(item(answer="I don't know"), self.fns())
        self.assertEqual(self.calls, []); self.assertEqual(s["correctness"]["score"], 0.0); self.assertIsNone(s["faithfulness"]["score"]); self.assertFalse(s["correctness"]["judged"])
    def test_unanswerable(self):
        s = J.score_item(item(answerable=False, answer="I don't know", reference=""), self.fns()); self.assertIsNone(s["correctness"]["score"]); self.assertEqual(self.calls, [])
        s = J.score_item(item(answerable=False, answer="It is 5 days.", reference=""), self.fns())
        self.assertEqual(s["correctness"]["score"], 0.0); self.assertEqual(sorted(self.calls), ["answer_relevancy", "faithfulness"])
    def test_normal_item_judged_once_per_metric(self):
        s = J.score_item(item(), self.fns()); self.assertEqual(sorted(self.calls), ["answer_relevancy", "correctness", "faithfulness"]); self.assertTrue(all(v["judged"] and v["score"] == 0.9 for v in s.values()))
    def test_no_context_skips_faithfulness(self): self.assertIsNone(J.score_item(item(contexts=[]), self.fns())["faithfulness"]["score"])
    def test_injection_gate(self): self.assertEqual(J.gate("injection_obedience", item(answer="I don't know"))[1], 1.0); self.assertTrue(J.gate("injection_obedience", item())[0])
    def test_error_recorded_not_fatal(self):
        def boom(it): raise ValueError("bad json")
        s = J.score_item(item(), dict(faithfulness=boom, correctness=lambda it: (1.0, "ok")), dict(n=0, max=6))
        self.assertIn("ValueError", s["faithfulness"]["error"]); self.assertIsNone(s["faithfulness"]["score"]); self.assertEqual(s["correctness"]["score"], 1.0)
    def test_ollama_down_is_fatal(self):
        def down(it): raise oc.OllamaError("Cannot reach Ollama")
        with self.assertRaises(oc.OllamaError): J.score_item(item(), dict(faithfulness=down))
    def test_guard_aborts_on_consecutive_failures(self):
        def boom(it): raise ValueError("x")
        with self.assertRaises(J.JudgeError): J.score_all([item(id=f"q{i}") for i in range(10)], dict(faithfulness=boom), log=lambda m: None, max_consecutive=3)
    def test_aggregate(self):
        items = [item(id="a", type="factual"), item(id="b", type="multi_hop"), item(id="c", type="factual")]
        sc = {"a": dict(correctness=dict(score=1.0, judged=True, error=None)), "b": dict(correctness=dict(score=0.0, judged=True, error=None)), "c": dict(correctness=dict(score=None, judged=False, error="x"))}
        g = J.aggregate(items, sc, ["correctness"])["correctness"]
        self.assertEqual((g["mean"], g["n_scored"], g["n_judged"], g["n_errors"]), (0.5, 2, 2, 1)); self.assertEqual(g["by_type"], {"factual": 1.0, "multi_hop": 0.0})


class StatsTests(unittest.TestCase):
    def test_kappa_textbook(self):
        a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15; b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15       # po .70, pe .50 -> kappa .40
        self.assertAlmostEqual(V.agreement(a, b), 0.70); self.assertAlmostEqual(V.cohens_kappa(a, b), 0.40)
    def test_kappa_edges(self):
        self.assertAlmostEqual(V.cohens_kappa(list("abcab"), list("abcab")), 1.0)
        self.assertIsNone(V.cohens_kappa([1, 1, 1], [1, 1, 1])); self.assertIsNone(V.cohens_kappa([], []))
        self.assertLess(V.cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]), 0)
        with self.assertRaises(ValueError): V.agreement([1], [1, 0])
    def test_wilson_and_kappa_ci(self):
        lo, hi = V.wilson(27, 30); self.assertTrue(lo < 0.9 < hi); self.assertEqual(V.wilson(0, 0), [None, None])
        a = [1] * 20 + [0] * 10; b = a[:]; b[0] = 0; b[25] = 1; ci = V.kappa_ci(a, b); k = V.cohens_kappa(a, b)
        self.assertTrue(ci[0] <= k <= ci[1]); self.assertEqual(ci, V.kappa_ci(a, b))                    # deterministic
    def test_binary_stats_warns_on_rare_class(self):
        s = V.binary_stats([1] * 28 + [0] * 2, [1] * 28 + [0, 1]); self.assertIn("rarer class", s["note"]); self.assertEqual((s["tp"], s["fp"], s["fn"], s["tn"]), (28, 1, 0, 1))
    def test_confusion_orientation(self):
        self.assertEqual(V.confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"]), [[1, 1], [0, 1]])
    def test_bands(self): self.assertEqual(V.kappa_band(0.5), "moderate"); self.assertEqual(V.kappa_band(None), "undefined"); self.assertEqual(V.kappa_band(0.9), "almost perfect")


def pool():
    out = []; types = ["factual", "paraphrase", "multi_hop"]
    for i in range(60): out.append(item(id=f"a{i}", type=types[i % 3], kw_correct=1.0))                       # A
    for i in range(12): out.append(item(id=f"b{i}", type=types[i % 3], kw_correct=0.0, answer="It is 3 days."))  # B
    for i in range(4): out.append(item(id=f"c{i}", kw_correct=0.0, answer="I don't know"))                      # C
    for i in range(13): out.append(item(id=f"d{i}", answerable=False, answer="I don't know", kw_correct=None, type="unanswerable"))  # D
    for i in range(3): out.append(item(id=f"e{i}", answerable=False, answer="It is 9.", kw_correct=None, type="unanswerable"))       # E
    return out


class SelectionTests(unittest.TestCase):
    def test_quotas_and_determinism(self):
        ch = V.select_for_labeling(pool(), seed=7); ids = [x["id"] for x in ch]
        self.assertEqual(len(ch), 30); self.assertEqual(len(set(ids)), 30)
        from collections import Counter
        self.assertEqual(Counter(V.stratum(x) for x in ch), Counter(V.QUOTAS)); self.assertEqual(ids, [x["id"] for x in V.select_for_labeling(pool(), seed=7)])
        self.assertNotEqual(ids, [x["id"] for x in V.select_for_labeling(pool(), seed=8)])
    def test_round_robin_covers_types(self): self.assertEqual({x["type"] for x in V.select_for_labeling(pool()) if V.stratum(x) == "A"}, {"factual", "paraphrase", "multi_hop"})
    def test_shortfall_is_filled(self):
        small = [x for x in pool() if not x["id"].startswith("e")]; self.assertEqual(len(V.select_for_labeling(small)), 30)       # stratum E empty
    def test_not_a_function_of_judge_output(self):
        p = pool()
        for x in p: x["scores"] = {"correctness": {"score": 0.123}}                                                             # selection must ignore any judge data
        self.assertEqual([x["id"] for x in V.select_for_labeling(p)], [x["id"] for x in V.select_for_labeling(pool())])


class CsvTests(unittest.TestCase):
    def setUp(self): self.path = os.path.join(tempfile.mkdtemp(), "labels", "h.csv")
    def test_roundtrip_unicode_commas_newlines_and_label_normalising(self):
        its = [item(id="q1", question='What, exactly, is "24"?', answer="24 days \u2013 yes\nline2", contexts=["a, b", "c \u00e9"]), item(id="q2", answerable=False, answer="I don't know", reference="")]
        V.write_template(self.path, its)
        with open(self.path, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
        rows[0]["human_label"] = " Properly Abstained "; rows[1]["human_label"] = ""
        with open(self.path, "w", newline="", encoding="utf-8-sig") as f: w = csv.DictWriter(f, V.COLS); w.writeheader(); w.writerows(rows)
        ok, unl = V.read_labels(self.path)
        self.assertEqual((len(ok), unl), (1, 1)); r = ok[0]
        self.assertEqual((r["label"], r["answer"], r["contexts_list"], r["question"]), ("properly_abstained", "24 days \u2013 yes\nline2", ["a, b", "c \u00e9"], 'What, exactly, is "24"?'))
    def test_invalid_label_lists_offenders(self):
        V.write_template(self.path, [item()])
        with open(self.path, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
        rows[0]["human_label"] = "wrong"
        with open(self.path, "w", newline="", encoding="utf-8-sig") as f: w = csv.DictWriter(f, V.COLS); w.writeheader(); w.writerows(rows)
        with self.assertRaises(ValueError) as e: V.read_labels(self.path)
        self.assertIn("q1", str(e.exception)); self.assertIn("Allowed", str(e.exception))
    def test_never_overwrites_labels(self):
        V.write_template(self.path, [item()]); V.write_template(self.path, [item(id="q9")])                                      # unlabeled: fine
        with open(self.path, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
        rows[0]["human_label"] = "correct"
        with open(self.path, "w", newline="", encoding="utf-8-sig") as f: w = csv.DictWriter(f, V.COLS); w.writeheader(); w.writerows(rows)
        with self.assertRaises(FileExistsError): V.write_template(self.path, [item(id="q10")])
        V.write_template(self.path, [item(id="q10")], force=True)
    def test_missing_file_message(self):
        with self.assertRaises(FileNotFoundError) as e: V.read_labels(self.path + ".nope")
        self.assertIn("--make-labels", str(e.exception))


THR = dict(faithfulness=0.8, answer_relevancy=0.7, correctness=0.5)

class JudgeLabelTests(unittest.TestCase):
    def jl(self, c, r, f, **kw): return V.judge_label(item(**kw), dict(correctness=c, answer_relevancy=r, faithfulness=f), THR)
    def test_rule_table(self):
        self.assertEqual(self.jl(0.9, 0.9, 0.9), "correct"); self.assertEqual(self.jl(0.2, 0.9, 0.9), "incorrect"); self.assertEqual(self.jl(0.2, 0.9, 0.3), "hallucinated")
        self.assertEqual(self.jl(0.2, 0.2, 0.9), "evasive"); self.assertEqual(self.jl(None, 0.9, 0.9), None)
        self.assertEqual(self.jl(None, None, None, answer="I don't know"), "evasive"); self.assertEqual(self.jl(None, None, None, answer="I don't know", answerable=False), "properly_abstained")
        self.assertEqual(self.jl(0.0, 0.9, 0.9, answerable=False, answer="It is 9"), "hallucinated"); self.assertIsNone(self.jl(0.2, None, 0.9))
    def test_targets(self):
        self.assertEqual([V.target_correct(l) for l in V.LABELS], [1, 0, 0, None, 0]); self.assertEqual([V.target_faithful(l) for l in V.LABELS], [1, None, 0, None, None])
        self.assertEqual([V.target_relevant(l) for l in V.LABELS], [1, 1, 1, None, 0])


def rec(score, judged=True, error=None, reason="r"): return dict(score=score, judged=judged, error=error, reason=reason, s=0.0)

class ReportTests(unittest.TestCase):
    def setUp(self):
        # (id, human label, answerable, answer, kw_correct, correctness, relevancy, faithfulness)
        spec = [("q1", "correct", True, "24 days", 1.0, 0.9, 0.9, 0.9), ("q2", "correct", True, "10 days", 1.0, 0.8, 0.9, 1.0), ("q3", "incorrect", True, "5 days", 0.0, 0.1, 0.9, 0.9),
                ("q4", "hallucinated", True, "42 days", 0.0, 0.0, 0.9, 0.2), ("q5", "evasive", True, "It depends.", 0.0, 0.0, 0.1, 0.9), ("q6", "properly_abstained", False, "I don't know", None, None, None, None),
                ("q7", "correct", True, "7 days", 0.0, 0.7, 0.9, 0.9)]                                   # q7: the keyword metric is wrong, the judge is right
        self.items, self.run = [], {}
        for i, lab, ans, a, kw, c, r, f in spec:
            self.items.append(item(id=i, label=lab, answerable=ans, answer=a, kw_correct=kw))
            self.run[i] = dict(correctness=rec(c, judged=c is not None), answer_relevancy=rec(r, judged=r is not None), faithfulness=rec(f, judged=f is not None))
        self.names = ["faithfulness", "answer_relevancy", "correctness"]
    def test_five_class_and_binary_numbers(self):
        rep = V.build_report(self.items, [self.run], THR, self.names); f5 = rep["five_class"]
        self.assertEqual((f5["n"], f5["n_undecidable"], f5["agreement"]), (7, 0, 1.0)); self.assertEqual(rep["disagreements"], [])
        c = rep["binary"]["correctness"]["stats"]; self.assertEqual((c["n"], c["human_pos"], c["human_neg"], c["agreement"]), (6, 3, 3, 1.0))       # q6 excluded
        k = rep["keyword_comparison"]; self.assertEqual(k["n"], 6); self.assertAlmostEqual(k["keyword"]["agreement"], 5 / 6); self.assertEqual(k["judge"]["agreement"], 1.0)          # q7 flips the keyword metric
        self.assertEqual(rep["binary"]["faithfulness"]["stats"]["n"], 4)                                                                                                    # correct x3 + hallucinated
    def test_disagreement_listed(self):
        self.run["q3"]["correctness"] = rec(0.8); rep = V.build_report(self.items, [self.run], THR, self.names)
        self.assertEqual([(d["id"], d["human"], d["judge"]) for d in rep["disagreements"]], [("q3", "incorrect", "correct")]); self.assertLess(rep["five_class"]["agreement"], 1.0)
    def test_judge_error_is_excluded_and_counted(self):
        self.run["q1"]["correctness"] = rec(None, error="ValueError: x"); rep = V.build_report(self.items, [self.run], THR, self.names)
        self.assertEqual(rep["judge_errors"]["correctness"], 1); self.assertEqual(rep["five_class"]["n_undecidable"], 1); self.assertEqual(rep["binary"]["correctness"]["stats"]["n"], 5)
    def test_variance(self):
        import copy
        r2 = copy.deepcopy(self.run); r2["q1"]["correctness"]["score"] = 0.3; r3 = copy.deepcopy(self.run)                                    # q1 flips across the 0.5 threshold
        v = V.build_report(self.items, [self.run, r2, r3], THR, self.names)["variance"]["correctness"]
        self.assertEqual(v["verdict_flips"], 1); self.assertAlmostEqual(v["max_range"], 0.6); self.assertEqual(v["n"], 6)                   # only q6 (gated, not judged) is left out
        same = V.build_report(self.items, [self.run, copy.deepcopy(self.run)], THR, self.names)["variance"]["correctness"]; self.assertEqual((same["verdict_flips"], same["mean_range"], same["share_changed"]), (0, 0.0, 0.0))
    def test_correctness_by_run_and_median(self):
        import copy
        r2 = copy.deepcopy(self.run); r2["q1"]["correctness"]["score"] = 0.1; r3 = copy.deepcopy(self.run)          # run 2 wrongly fails q1; runs 1 and 3 are right
        c = V.build_report(self.items, [self.run, r2, r3], THR, self.names)["correctness_by_run"]
        self.assertEqual(c["n"], 6); self.assertEqual([x["agreement"] for x in c["per_run"]], [1.0, 5 / 6, 1.0]); self.assertEqual(c["median"]["agreement"], 1.0)
        self.assertNotIn("correctness_by_run", V.build_report(self.items, [self.run], THR, self.names))
    def test_variance_ignores_gated_items(self):
        v = V.variance_report([self.run, self.run], [i["id"] for i in self.items], ["correctness"], THR)["correctness"]; self.assertEqual(v["n"], 6)  # q6 has no score
    def test_render(self):
        rep = V.build_report(self.items, [self.run, self.run], THR, self.names); meta = dict(judge_model="j", judge_digest="d", generator_model="g", deepeval="x", date="d", judge_settings={})
        md = V.render_md(rep, meta); self.assertIn("Five-class", md); self.assertIn("keyword", md); self.assertIn("kappa", V.console_summary(rep, meta))


def fake_deepeval():
    """Minimal stand-in for the parts of DeepEval we use; it records constructor arguments and uses the model the way DeepEval does."""
    calls = dict(cases=[], metrics=[]); m = types.ModuleType("deepeval"); models = types.ModuleType("deepeval.models"); metrics = types.ModuleType("deepeval.metrics"); tc = types.ModuleType("deepeval.test_case")
    class Base:
        def __init__(self, model_name=None, *a, **k): self.model_name = model_name; self.model = self.load_model()
    models.DeepEvalBaseLLM = Base
    class Case:
        def __init__(self, **kw): self.__dict__.update(kw); calls["cases"].append(kw)
    class P: INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT, RETRIEVAL_CONTEXT = "input", "actual_output", "expected_output", "retrieval_context"
    tc.LLMTestCase, tc.LLMTestCaseParams = Case, P
    def metric(name):
        class M:
            def __init__(self, **kw): self.kw = kw; self.model = kw["model"]; calls["metrics"].append((name, kw))
            def measure(self, case, _show_indicator=True):
                out = self.model.generate("rate it", schema=Verdict); self.score = {"yes": 0.75}.get(out.verdict, 0.0); self.reason = "because"; return self.score
        M.__name__ = name; return M
    metrics.FaithfulnessMetric, metrics.AnswerRelevancyMetric, metrics.ToxicityMetric, metrics.GEval = metric("Faithfulness"), metric("AnswerRelevancy"), metric("Toxicity"), metric("GEval")
    return dict(deepeval=m, **{"deepeval.models": models, "deepeval.metrics": metrics, "deepeval.test_case": tc}), calls, Base


class DeepEvalWiringTests(unittest.TestCase):
    def test_metric_construction_and_adapter(self):
        mods, calls, _ = fake_deepeval()
        with mock.patch.dict(sys.modules, mods), mock.patch.object(oc, "chat", return_value=reply('{"verdict": "yes"}')):
            judge = J.OllamaJudge("qwen2.5:7b", host="http://x"); fns = J.deepeval_metric_fns(judge, J.DEFAULT_JUDGE_CFG["thresholds"], list(J.METRICS))
            self.assertEqual(set(fns), set(J.METRICS)); score, why = fns["correctness"](item()); self.assertEqual((score, why), (0.75, "because"))
        c = calls["cases"][0]; self.assertEqual((c["input"], c["actual_output"], c["expected_output"], c["retrieval_context"]), ("Q?", "24 days [1]", "24 days", ["ctx"]))
        by = {kw.get("name", n): kw for n, kw in calls["metrics"]}
        self.assertTrue(all(kw["async_mode"] is False for kw in by.values())); self.assertEqual(by["Correctness"]["evaluation_params"], ["input", "actual_output", "expected_output"])
        self.assertEqual(by["InjectionObedience"]["evaluation_params"], ["input", "actual_output", "retrieval_context"]); self.assertEqual(by["Faithfulness"]["threshold"], 0.8)
        self.assertEqual(by["Faithfulness"]["model"].get_model_name(), "ollama/qwen2.5:7b"); self.assertTrue(by["Faithfulness"]["include_reason"])
    def test_old_style_base_class_without_init_args(self):
        mods, calls, _ = fake_deepeval()
        class Old:                                                                                                    # older DeepEval: no-arg __init__, load_model not called by the base
            def __init__(self): pass
        mods["deepeval.models"].DeepEvalBaseLLM = Old
        with mock.patch.dict(sys.modules, mods), mock.patch.object(oc, "chat", return_value=reply("plain text")):
            mdl = J.deepeval_model(J.OllamaJudge("m", host="http://x")); self.assertEqual(mdl.model_name, "m"); self.assertEqual(mdl.generate("p"), "plain text")
            import asyncio; self.assertEqual(asyncio.run(mdl.a_generate("p")), "plain text")
    def test_unknown_metric(self):
        mods, _, _ = fake_deepeval()
        with mock.patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError): J.deepeval_metric_fns(J.OllamaJudge("m", host="http://x"), J.DEFAULT_JUDGE_CFG["thresholds"], ["nope"])


class LabelerTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "l.csv"); V.write_template(self.path, [item(id="q1"), item(id="q2", answerable=False, answer="I don't know"), item(id="q3")])
        self.a = argparse.Namespace(labels=self.path)
    def run_keys(self, keys):
        it = iter(keys); return V.cmd_label({}, dict(labels=dict(file=self.path)), self.a, lambda m: None, ask=lambda p: next(it))
    def labels(self):
        with open(self.path, newline="", encoding="utf-8-sig") as f: return [r["human_label"] for r in csv.DictReader(f)]
    def test_label_quit_resume_skip(self):
        self.run_keys(["1", "x", "q"]); self.assertEqual(self.labels(), ["correct", "", ""])               # saved after each label; junk key re-asks
        self.run_keys(["4", "s"]); self.assertEqual(self.labels(), ["correct", "properly_abstained", ""])    # resumes at q2; q3 skipped
        self.run_keys(["5"]); self.assertEqual(self.labels(), ["correct", "properly_abstained", "evasive"])
        ok, unl = V.read_labels(self.path); self.assertEqual((len(ok), unl), (3, 0))                       # file stays valid for --validate
    def test_missing_file(self):
        self.a.labels = self.path + ".nope"
        with self.assertRaises(SystemExit): self.run_keys([])


class OrchestrationTests(unittest.TestCase):
    """Whole make-labels -> label -> validate flow offline: hash embedder + MockLLM generate the answers; the judge metrics are fakes."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.cfg = deep_merge(load_cfg("configs/ollama.yaml"), dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False)))
        self.jc = J.load_judge_cfg("configs/judge.yaml"); self.jc["labels"]["file"] = os.path.join(self.tmp, "labels.csv"); self.jc["cache"] = None
        def ns(**kw):
            d = dict(labels=None, force=False, metrics=None, runs=None, temperature=None, no_cache=True, out=os.path.join(self.tmp, "o.json"), md=os.path.join(self.tmp, "o.md"), limit=None); d.update(kw)
            return argparse.Namespace(**d)
        self.ns = ns
    def read_out(self):
        with open(os.path.join(self.tmp, "o.json"), encoding="utf-8") as f: res = json.load(f)
        with open(os.path.join(self.tmp, "o.md"), encoding="utf-8") as f: return res, f.read()
    def test_collect_answers_fields(self):
        items = J.collect_answers(self.cfg, limit=8, log=lambda m: None)
        self.assertEqual(len(items), 8); it = items[0]
        for k in ("id", "type", "role", "question", "reference", "answerable", "contexts", "answer", "kw_correct", "grounded_proxy"): self.assertIn(k, it)
        self.assertTrue(all(isinstance(c, str) and "\n" not in c for x in items for c in x["contexts"]))
    def test_make_labels_then_validate(self):
        with mock.patch.object(J, "unload", return_value=True):
            self.assertEqual(V.cmd_make_labels(self.cfg, self.jc, self.ns(), lambda m: None), 0)
        with open(self.jc["labels"]["file"], newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 30); self.assertEqual(list(rows[0]), V.COLS); self.assertTrue(all(r["human_label"] == "" for r in rows))
        for r in rows: r["human_label"] = "properly_abstained" if r["answer"].startswith("I don't know") else "correct"                 # a pretend human
        with open(self.jc["labels"]["file"], "w", newline="", encoding="utf-8-sig") as f: w = csv.DictWriter(f, V.COLS); w.writeheader(); w.writerows(rows)
        with self.assertRaises(SystemExit): V.cmd_make_labels(self.cfg, self.jc, self.ns(), lambda m: None)                             # labels are never clobbered
        fake = lambda judge, thr, names: {n: (lambda it: (1.0, "fake")) for n in names}
        with mock.patch.object(J, "deepeval_metric_fns", side_effect=fake), mock.patch.object(J, "unload", return_value=True):
            self.assertEqual(V.cmd_validate(self.cfg, self.jc, self.ns(runs=2), lambda m: None), 0)
        res, md = self.read_out(); self.assertEqual(res["report"]["n_labeled"], 30); self.assertEqual(len(res["runs"]), 2)
        self.assertIn("variance", res["report"]); self.assertIn("Five-class", md)
    def test_score_command_plumbing(self):
        fake = lambda judge, thr, names: {n: (lambda it: (0.9, "fake")) for n in names}
        with mock.patch.object(J, "deepeval_metric_fns", side_effect=fake), mock.patch.object(J, "unload", return_value=True), mock.patch.object(J, "run_meta", return_value=dict(judge_model="j", judge_digest="d", generator_model=None, deepeval="x", thresholds={}, judge_settings={}, date="d")):
            self.assertEqual(J.cmd_score(self.cfg, self.jc, self.ns(limit=10), lambda m: None), 0)
        res, md = self.read_out(); self.assertEqual(res["n_items"], 10); self.assertIn("keyword_vs_judge", res); self.assertIn("Judge scores", md)


if __name__ == "__main__": unittest.main()
