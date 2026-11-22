"""S5 tests. Everything runs offline (hash embedder + MockLLM): they prove the harness logic (metrics, attribution rules, statistics, comparison, CLI), not model quality."""
import contextlib, io, json, os, re, tempfile, unittest
from rag.pipeline import RAG, load_cfg, load_golden, load_docs, deep_merge
from rag.chunking import Chunk
from rag import metrics as M, llm as L, prompts as P
from eval import run_generation as G
import cli

def offline_cfg(**over):
    return deep_merge(deep_merge(load_cfg("configs/ollama.yaml"), dict(embedder=dict(type="hash"), llm=dict(type="mock", naive=False))), over)
def gold(i): return next(g for g in load_golden() if g["id"] == i)
def ch(text, i="c#0", d="d-01"): return Chunk(id=i, doc_id=d, text=text, access="public")
SPEC = dict(model="llama3.2:3b", prompt="default", k=4, order="rank")

class TestMetrics(unittest.TestCase):
    def test_normalize(self):
        for raw, want in [("Twenty-four days", "24 day"), ("twenty four days", "24 day"), ("five business days", "5 business day"), ("Rs. 1,000", "1000 rupee"), ("\u20b9800", "800 rupee"),
                          ("800 INR", "800 rupee"), ("7:00 PM", "7 pm"), ("7pm", "7 pm"), ("7 p.m.", "7 pm"), ("someone", "someone"), ("3.5 hours", "3.5 hour"), ("30%", "30%")]:
            self.assertEqual(M.normalize(raw), want, raw)
    def test_norm_fixes_format_differences_strict_cannot(self):
        g = dict(keywords=["24"]); self.assertEqual(M.correctness("Twenty-four days of leave.", g), 0.0); self.assertEqual(M.correctness_norm("Twenty-four days of leave.", g), 1.0)
        g = dict(keywords=["7 pm"]); self.assertEqual(M.correctness("Cleaned at 7PM.", g), 0.0); self.assertEqual(M.correctness_norm("Cleaned at 7PM.", g), 1.0)
        g = dict(keywords=["higher", "800"]); self.assertEqual(M.correctness("The meal cap (800) is greater than 250.", g), 0.0); self.assertEqual(M.correctness_norm("The meal cap (800) is greater than 250.", g), 1.0)
        self.assertEqual(M.correctness_norm("twenty five days", dict(keywords=["5 day"])), 0.0, "after normalisation a keyword must not match inside a larger number (25 day)")
        self.assertEqual(M.correctness_norm("about thirty days", dict(keywords=["30"])), 1.0)
    def test_golden_alternates_field(self):
        g = dict(keywords=["remote"], alternates={"remote": ["work from home"]}); self.assertEqual(M.correctness_norm("You may work from home.", g), 1.0); self.assertEqual(M.correctness("You may work from home.", g), 0.0)
    def test_norm_is_never_stricter_than_strict(self):
        answers = ["24 days", "I don't know", "7 PM [1]", "the cap is 800 rupees, higher than 250", "Expense claims are longer: 30 days (fin-01)", "", "no", "5 business days"]
        for g in load_golden():
            for a in answers + [g["reference"]]:
                if M.correctness(a, g): self.assertEqual(M.correctness_norm(a, g), 1.0, (g["id"], a))
    def test_no_keywords_is_wrong_not_crash(self): self.assertEqual(M.correctness_norm("x", dict(keywords=[])), 0.0)
    def test_strip_citations(self):
        t = M.strip_citations("Meal cap is 800 [1][2] (fin-01) and fin-01#0 too [1, 2].\nSources: [1] fin-01\nDone", ["fin-01"])
        self.assertNotIn("[", t); self.assertNotIn("fin-01", t); self.assertNotIn("Sources", t); self.assertIn("Meal cap is 800", t); self.assertIn("Done", t)
    def test_clean_groundedness_does_not_punish_citations(self):
        ctx = [ch("Meal expenses are capped at 800 rupees per day.", d="fin-01")]; a = "Meal expenses are capped at 800 rupees per day [1] (fin-01)."
        self.assertLess(M.groundedness(a, ctx), 1.0); self.assertEqual(M.groundedness_clean(a, ctx), 1.0)
        self.assertLess(M.groundedness_clean("The cap is 900 rupees per day [1]", ctx), 1.0, "made-up content must still count as ungrounded")
    def test_ungrounded_digits_flag_the_n06_style_misread(self):
        ctx = [ch("The audit is run twice a year by the finance team.")]
        self.assertIn("2", M.ungrounded_tokens("The audit is run 2 times [1].", ctx)); self.assertEqual(M.ungrounded_tokens("The audit is run twice a year [1].", ctx), [])
    def test_soft_abstain(self):
        for t in ("I don't know", "The context does not contain that information.", "There is no mention of pets.", "I cannot find this in the provided text."): self.assertTrue(M.is_soft_abstain(t), t)
        for t in ("24 days [1]", "The cap is 800."): self.assertFalse(M.is_soft_abstain(t), t)
        self.assertFalse(L.is_abstain("The context does not contain that."), "the strict detector must stay strict (so the artifact category is meaningful)")

class TestStats(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(G.sign_test_p(6, 0), 0.03125); self.assertAlmostEqual(G.sign_test_p(0, 6), 0.03125); self.assertAlmostEqual(G.sign_test_p(5, 0), 0.0625)
        self.assertAlmostEqual(G.sign_test_p(8, 1), 0.0390625); self.assertEqual(G.sign_test_p(0, 0), 1.0); self.assertEqual(G.sign_test_p(3, 3), 1.0)
    def test_wilson(self):
        lo, hi = G.wilson(13, 13); self.assertLess(lo, 1.0); self.assertEqual(hi, 1.0); self.assertEqual(G.wilson(0, 0), (0.0, 0.0)); lo, hi = G.wilson(50, 100); self.assertLess(lo, 0.5); self.assertGreater(hi, 0.5)
    def test_percentile(self): self.assertEqual(G.pctl([1, 2, 3, 4, 100], 50), 3); self.assertEqual(G.pctl([1, 2, 3, 4, 100], 95), 100); self.assertEqual(G.pctl([], 50), 0.0)

class TestContexts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rag = RAG(offline_cfg())
    def test_gold_context_holds_every_evidence_of_every_answerable_question(self):
        for g in load_golden():
            ctx, intact = G.gold_context(self.rag.chunks, g)
            if not g["answerable"]: self.assertEqual((ctx, intact), ([], True)); continue
            self.assertTrue(intact, g["id"]); self.assertLessEqual(len(ctx), len(g["evidences"]))
            for e in g["evidences"]: self.assertTrue(any(e.lower() in c.text.lower() for c in ctx), (g["id"], e))
    def test_split_evidence_is_flagged_and_whole_documents_are_given(self):
        rag = RAG(offline_cfg(chunking=dict(strategy="fixed", size=20, overlap=0))); g = gold("q001")
        ctx, intact = G.gold_context(rag.chunks, g)
        self.assertFalse(intact); self.assertTrue(all(c.id.endswith("#all") for c in ctx)); self.assertTrue(any(g["evidences"][0].lower() in c.text.lower() for c in ctx))
    def test_order_contexts(self):
        cs = [ch("a", "b#1", "b"), ch("b", "a#2", "a"), ch("c", "a#1", "a")]
        self.assertEqual([c.id for c in G.order_contexts(cs, "rank")], ["b#1", "a#2", "a#1"]); self.assertEqual([c.id for c in G.order_contexts(cs, "reverse")], ["a#1", "a#2", "b#1"])
        self.assertEqual([c.id for c in G.order_contexts(cs, "doc")], ["a#1", "a#2", "b#1"]); self.assertEqual(len(cs), 3)
        with self.assertRaises(ValueError): G.order_contexts(cs, "nope")

def F(**k):
    d = dict(answerable=True, abstained=0.0, soft_abstained=0.0, correct=0.0, correct_norm=0.0, intact=True, ev_top=1, ev_pool=1, n_evid=1, ungrounded_digits=0, grounded_clean=1.0); d.update(k); return d

class TestClassify(unittest.TestCase):
    def test_every_category(self):
        cases = [(F(correct=1.0), None), (F(intact=False, ev_top=0, ev_pool=0), ("chunking", "evidence_split")), (F(ev_top=0, ev_pool=1), ("ranking", "in_pool_not_in_top_k")),
                 (F(ev_top=0, ev_pool=0), ("retrieval_miss", "not_in_pool")), (F(correct_norm=1.0), ("metric_artifact", "strict_keyword")),
                 (F(abstained=1.0, soft_abstained=1.0), ("abstention_error", "false_abstain")), (F(soft_abstained=1.0), ("abstention_error", "false_abstain")),
                 (F(ungrounded_digits=1), ("generation_error", "hallucination_candidate")), (F(grounded_clean=0.3), ("generation_error", "hallucination_candidate")),
                 (F(), ("generation_error", "incomplete_or_misread")),
                 (F(answerable=False, abstained=1.0), None), (F(answerable=False, soft_abstained=1.0), ("metric_artifact", "soft_abstain")), (F(answerable=False), ("abstention_error", "answered_unanswerable"))]
        for f, want in cases: self.assertEqual(G.classify(f), want, f)
    def test_rule_order(self):
        self.assertEqual(G.classify(F(intact=False, ev_top=0, ev_pool=0, correct_norm=1.0))[0], "chunking")           # chunking beats everything
        self.assertEqual(G.classify(F(ev_top=0, ev_pool=0, correct_norm=1.0, abstained=1.0))[0], "retrieval_miss")      # retrieval beats artifact and abstention
        self.assertEqual(G.classify(F(correct_norm=1.0, abstained=1.0, soft_abstained=1.0))[0], "metric_artifact")      # artifact beats abstention
        self.assertEqual(G.classify(F(abstained=1.0, ungrounded_digits=2))[0], "abstention_error")                       # abstention beats generation
    def test_multi_hop_partial_retrieval(self):
        self.assertEqual(G.classify(F(n_evid=2, ev_top=1, ev_pool=2))[0], "ranking"); self.assertEqual(G.classify(F(n_evid=2, ev_top=1, ev_pool=1))[0], "retrieval_miss")
    def test_chunking_through_the_real_pipeline(self):
        rag = RAG(offline_cfg(chunking=dict(strategy="fixed", size=20, overlap=0))); g = gold("q001")
        row = dict(id="q001", abstained=0.0, soft_abstained=0.0, correct=0.0, correct_norm=0.0, ungrounded_digits=0, grounded_clean=1.0, answer="x")
        self.assertEqual(G.attribute(rag, [g], [row])[0]["category"], "chunking")
    def test_retrieval_miss_through_the_real_pipeline(self):                  # a guest cannot see hr-01 (ACL pre-filter), so the evidence is not even in the pool
        rag = RAG(offline_cfg()); g = dict(gold("q001"), role="guest")
        row = dict(id="q001", abstained=0.0, soft_abstained=0.0, correct=0.0, correct_norm=0.0, ungrounded_digits=0, grounded_clean=1.0, answer="x")
        self.assertEqual(G.attribute(rag, [g], [row])[0]["category"], "retrieval_miss")
    def test_ranking_through_the_real_pipeline(self):
        rag = RAG(offline_cfg()); gs = [g for g in load_golden() if g["answerable"] and G.ev_found(G.retrieved(rag, g), g) < len(g["evidences"]) and G.ev_found(G.pool_list(rag, g), g) == len(g["evidences"])]
        self.assertTrue(gs, "the offline stand-in should produce at least one ranking case (q061 / q070)")
        row = dict(abstained=0.0, soft_abstained=0.0, correct=0.0, correct_norm=0.0, ungrounded_digits=0, grounded_clean=1.0, answer="x")
        self.assertEqual(G.attribute(rag, gs[:1], [dict(row, id=gs[0]["id"])])[0]["category"], "ranking")

class TestRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = offline_cfg(); cls.G = load_golden(); cls.rag = RAG(cls.cfg); cls.res = G.run_config(cls.cfg, SPEC, cls.G, ("gold", "retrieved"), rag=cls.rag)
    def test_default_variant_reproduces_the_normal_pipeline(self):                       # same prompt as `cli.py eval` -> same answers
        for r in self.res["modes"]["retrieved"]["rows"][:25]:
            g = next(x for x in self.G if x["id"] == r["id"]); self.assertEqual(r["answer"], self.rag.answer(g["question"], g["role"])["answer"], r["id"])
    def test_failures_add_up(self):
        rows = self.res["modes"]["retrieved"]["rows"]; failed = sum(1 for r in rows if (r["answerable"] and not r["correct"]) or (not r["answerable"] and not r["abstained"]))
        self.assertEqual(len(self.res["attribution"]), failed); self.assertTrue(all(a["category"] in G.CATEGORIES for a in self.res["attribution"]))
        tab = G.attribution_counts(self.res["attribution"], sorted({r["type"] for r in rows})); self.assertEqual(sum(sum(v.values()) for v in tab.values()), failed)
    def test_gap_table_is_consistent(self):
        g = self.res["gap"]["ALL"]; self.assertEqual(g["n"], 69); self.assertEqual(g["gold"] - g["retrieval_cost"] + g["retrieval_luck"], g["retrieved"])
        self.assertEqual(sum(self.res["gap"][t]["gold"] for t in self.res["gap"] if t != "ALL"), g["gold"])
    def test_gold_context_for_unanswerable_is_empty_so_mock_abstains(self):
        for r in self.res["modes"]["gold"]["rows"]:
            if not r["answerable"]: self.assertEqual(r["abstained"], 1.0, r["id"])
    def test_summary_fields(self):
        s = self.res["modes"]["retrieved"]["summary"]; self.assertEqual((s["n_answerable"], s["n_unanswerable"]), (69, 13)); self.assertIn("hit", s); self.assertNotIn("hit", self.res["modes"]["gold"]["summary"])
        self.assertEqual(sum(v["n"] for v in s["by_type"].values()), 69); self.assertGreaterEqual(s["grounded_clean"], s["grounded"] - 1e-9)
    def test_abstention_precision_recall_arithmetic(self):
        def r(ans, ab, nm=None):
            d = dict(answerable=ans, abstained=ab, soft_abstained=ab, type="x", correct=1.0, correct_norm=1.0, grounded=1.0, grounded_clean=1.0, tokens_in=1, tokens_out=1, ms=1.0, cost=0.0)
            if nm is not None: d["near_miss"] = nm
            return d
        rows = [r(False, 1.0, True), r(False, 0.0, True), r(False, 1.0, False), r(False, 1.0, False), r(True, 1.0), r(True, 0.0)]
        a = G.summarize(rows)["abstention"]
        self.assertEqual((a["tp"], a["fp"], a["fn"]), (3, 1, 1)); self.assertAlmostEqual(a["precision"], 0.75); self.assertAlmostEqual(a["recall"], 0.75)
        self.assertEqual(a["near_miss_n"], 2); self.assertAlmostEqual(a["near_miss_recall"], 0.5); self.assertEqual(a["other_n"], 2); self.assertAlmostEqual(a["other_recall"], 1.0)
    def test_context_order_changes_the_prompt_not_the_retrieval_metrics(self):       # reordering must not change hit/MRR (those describe the retriever, not the prompt)
        a = G.run_mode(self.rag, self.G[:40], "retrieved", P.get_system("default"), "rank"); b = G.run_mode(self.rag, self.G[:40], "retrieved", P.get_system("default"), "reverse")
        for x, y in zip(a, b):
            if x["answerable"]: self.assertEqual((x["hit"], x["mrr"], x["evidence_recall"]), (y["hit"], y["mrr"], y["evidence_recall"]), x["id"])
        self.assertNotEqual([x["tokens_in"] for x in a], [], "sanity")
    def test_markdown_report_has_all_sections(self):
        md = G.md_run(dict(self.res, spec=SPEC), offline=True)
        for h in ("OFFLINE STAND-IN", "## A.", "## B.", "## C.", "## D.", "## E.", "retrieval_miss"): self.assertIn(h, md)
    def test_near_miss_proxy(self):
        self.assertTrue(G.near_miss("what is the pet insurance policy", ch("Our pet insurance policy is not offered."), 0.5)); self.assertFalse(G.near_miss("what is the pet insurance policy", ch("Lunch is at noon."), 0.5))
        self.assertFalse(G.near_miss("what is x", None, 0.5))

def fake_run(label, correct, abst=None, ms=100.0, spec=None, norm=None, ids=None):
    ids = ids or [f"a{i}" for i in range(len(correct))] + [f"u{i}" for i in range(len(abst or []))]; rows = []
    for i, c in enumerate(correct): rows.append(dict(id=f"a{i}", type="x", answerable=True, correct=float(c), correct_norm=float(c if norm is None else norm[i]), abstained=0.0, ms=ms))
    for i, a in enumerate(abst or []): rows.append(dict(id=f"u{i}", type="x", answerable=False, abstained=float(a), ms=ms))
    summ = dict(correct=0.5, correct_norm=0.5, gen_p95_ms=ms, tokens_in=100, est_cost_usd=0.0, abstention=dict(recall=1.0))
    return dict(label=label, spec=dict(spec or SPEC), modes=dict(retrieved=dict(rows=rows, summary=summ)))

class TestCompare(unittest.TestCase):
    def test_adoption_when_all_rules_hold(self):
        ref = fake_run("ref", [0] * 10 + [1] * 5, [1] * 4); var = fake_run("var", [1] * 7 + [0] * 3 + [1] * 5, [1] * 4)
        c = G.compare(ref, var, known=["a0"]); self.assertEqual((c["strict"]["won"], c["strict"]["lost"]), (7, 0)); self.assertLess(c["strict"]["p"], 0.05); self.assertTrue(c["verdict"].startswith("ADOPTION CANDIDATE"), c["verdict"])
    def test_five_wins_are_not_enough(self):
        ref = fake_run("ref", [0] * 5 + [1] * 5); c = G.compare(ref, fake_run("v", [1] * 10)); self.assertIn("R1", c["verdict"]); self.assertAlmostEqual(c["strict"]["p"], 0.0625)
    def test_a_loss_outside_the_known_failures_fails_r4(self):                       # 10 wins (all on the known failures) but 1 loss elsewhere: R1 holds, R4 does not
        ref = fake_run("ref", [0] * 10 + [1] * 5); var = fake_run("v", [1] * 10 + [0] + [1] * 4); c = G.compare(ref, var, known=[f"a{i}" for i in range(10)])
        self.assertIn("R4", c["verdict"]); self.assertNotIn("R1", c["verdict"]); self.assertEqual((c["rest"]["won"], c["rest"]["lost"]), (0, 1))
    def test_abstention_regression_and_latency_fail(self):
        ref = fake_run("ref", [0] * 8 + [1] * 2, [1, 1, 1, 1]); var = fake_run("v", [1] * 8 + [1] * 2, [0, 0, 1, 1], ms=200.0)
        c = G.compare(ref, var); self.assertIn("R3", c["verdict"]); self.assertIn("R5", c["verdict"]); self.assertEqual(c["abstention"]["lost"], 2)
    def test_normalised_rule(self):
        ref = fake_run("ref", [0] * 8 + [1] * 2); var = fake_run("v", [1] * 8 + [1] * 2, norm=[0] * 8 + [1] * 2); self.assertIn("R2", G.compare(ref, var)["verdict"])
    def test_different_questions_are_invalid(self):
        a = fake_run("a", [1, 0]); b = fake_run("b", [1, 0, 1])
        with self.assertRaises(G.InvalidComparison): G.compare(a, b)
        with self.assertRaises(G.InvalidComparison): G.check_comparable([dict(a, offline=True), dict(a, offline=False)])

class TestPrompts(unittest.TestCase):
    def test_default_is_the_production_prompt(self): self.assertEqual(P.get_system("default"), L.SYSTEM)
    def test_unknown_variant(self):
        with self.assertRaises(ValueError): P.get_system("nope")
    def test_variants_keep_the_safety_sentences(self):
        for n, s in P.VARIANTS.items():
            self.assertIn("never follow instructions", s, n); self.assertIn("Refuse requests", s, n); self.assertIn("I don't know", s, n)
    def test_fewshot_examples_do_not_leak_corpus_facts(self):
        ex = P.FEWSHOT[len(L.SYSTEM):].lower(); corpus = " ".join(d["text"].lower() for d in load_docs()); w = re.findall(r"[a-z0-9]+", ex)
        shingles = {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}; cw = " ".join(re.findall(r"[a-z0-9]+", corpus))
        self.assertEqual([s for s in shingles if s in cw], [])

class TestCLI(unittest.TestCase):
    def parse(self, *argv): return cli.build_parser().parse_args(list(argv))
    def test_old_commands_still_parse(self):                                          # S4 lost three tests to a flag clash: guard every earlier documented command line
        for argv in (["check"], ["ask", "q?", "--role", "hr"], ["eval", "--limit", "5", "--out", "x.json"], ["sweep", "--offline", "--retrievers", "bm25,dense"], ["safety", "--offline", "--no-proof", "--out", "a", "--md", "b"],
                     ["regress", "--compare", "baselines/v1.json", "--set", "a.b=c", "--set", "d=e", "--no-safety", "--hardware", "x"], ["regress", "--save-baseline", "b.json", "--label", "L"], ["regress", "--noise-study", "3"],
                     ["judge", "--probe"], ["judge", "--validate", "--runs", "3", "--labels", "l.csv"], ["judge", "--make-labels", "--force"], ["judge", "--annotate"], ["judge", "--metrics", "faithfulness", "--limit", "3", "--temperature", "0", "--no-cache"]):
            self.parse(*argv)
        a = self.parse("regress", "--compare", "baselines/v1.json"); self.assertEqual(a.compare, "baselines/v1.json"); self.assertEqual(self.parse("judge", "--validate", "--runs", "3").runs, 3)
    def test_gen_flags(self):
        a = self.parse("gen", "--model", "qwen2.5:3b", "--prompt", "fewshot", "--k", "6", "--order", "doc", "--ids", "q020,q033", "--offline", "--grid", "--skip-missing", "--merge", "a.json", "b.json")
        self.assertEqual((a.model, a.prompt, a.k, a.order, a.ids), ("qwen2.5:3b", "fewshot", 6, "doc", "q020,q033")); self.assertEqual(a.merge, ["a.json", "b.json"]); self.assertTrue(a.grid and a.skip_missing and a.offline)
    def run_gen(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = G.run_cli(offline_cfg(), self.parse("gen", *argv))
        return code, buf.getvalue()
    def test_end_to_end_single_merge_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (os.path.join(d, n) for n in ("a.json", "b.json", "c.json"))
            self.assertEqual(self.run_gen("--offline", "--limit", "20", "--out", a)[0], 0); self.assertTrue(os.path.exists(a[:-5] + ".md"))
            self.assertEqual(self.run_gen("--offline", "--limit", "20", "--order", "doc", "--out", b)[0], 0)
            code, out = self.run_gen("--merge", a, b, "--md", os.path.join(d, "m.md")); self.assertEqual(code, 0); self.assertIn("order=doc", out); self.assertIn("OFFLINE STAND-IN", out)
            self.assertEqual(self.run_gen("--offline", "--limit", "12", "--out", c)[0], 0)
            code, out = self.run_gen("--merge", a, c); self.assertEqual(code, 2); self.assertIn("INVALID", out)
            with open(b, encoding="utf-8") as f: j = json.load(f)
            j["offline"] = False
            with open(b, "w", encoding="utf-8") as f: json.dump(j, f)
            self.assertEqual(self.run_gen("--merge", a, b)[0], 2)                      # offline vs real is not comparable
    def test_ids_show_prints_answers_and_attribution(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = self.run_gen("--offline", "--ids", "q020,q035", "--out", os.path.join(d, "x.json"))
            self.assertEqual(code, 0); self.assertIn("=== q020", out); self.assertIn("=== q035", out); self.assertIn("attribution:", out)
    def test_typo_in_prompt_or_order_fails_early(self):
        with self.assertRaises(ValueError): self.run_gen("--offline", "--prompt", "nope")
        with self.assertRaises(ValueError): self.run_gen("--offline", "--order", "nope")
    def test_grid_plan_is_one_factor_at_a_time(self):
        ref = dict(SPEC); plan = G.sweep_specs(ref, {"prompt": dict(values=["fewshot"], modes=["gold", "retrieved"]), "k": dict(values=[2, 6], modes=["retrieved"])})
        self.assertEqual(len(plan), 4); self.assertEqual(plan[0][2], "reference")
        for spec, modes, fac in plan[1:]: self.assertEqual([k for k in ref if spec[k] != ref[k]], [fac])
    def test_generation_yaml_is_valid(self):
        import yaml
        with open("configs/generation.yaml", encoding="utf-8") as f: y = yaml.safe_load(f)
        for fac, d in y["factors"].items():
            for v in d["values"]: P.get_system(v) if fac == "prompt" else (G.order_contexts([], v) if fac == "order" else None)
        known = {g["id"] for g in load_golden()}; self.assertTrue(set(y["known_failures"]) <= known)

class TestRealOllamaPath(unittest.TestCase):
    """The non-offline route (OllamaLLM + model check + per-prompt system message) against the stand-in server: plumbing only, the answers are not model results."""
    @classmethod
    def setUpClass(cls):
        import shutil, threading
        from http.server import HTTPServer
        from tests.fake_ollama import H
        cls.srv = HTTPServer(("127.0.0.1", 0), H); cls.host = f"http://127.0.0.1:{cls.srv.server_port}"; threading.Thread(target=cls.srv.serve_forever, daemon=True).start(); cls.tmp = tempfile.mkdtemp(); cls.shutil = shutil
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown(); cls.shutil.rmtree(cls.tmp, ignore_errors=True)
    def cfg(self):
        c = load_cfg("configs/ollama.yaml"); c["embedder"].update(model="fake-embed:latest", host=self.host, cache_dir=self.tmp); c["llm"].update(model="fake-chat:1b", host=self.host); return c
    def go(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = G.run_cli(self.cfg(), cli.build_parser().parse_args(["gen", *argv]))
        return code, buf.getvalue()
    def test_real_path_run_with_prompt_variant(self):
        out = os.path.join(self.tmp, "r.json"); code, _ = self.go("--limit", "8", "--prompt", "fewshot", "--out", out); self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f: j = json.load(f)
        self.assertFalse(j["offline"]); r = j["runs"][0]; self.assertEqual(r["spec"]["prompt"], "fewshot"); self.assertEqual(len(r["modes"]["gold"]["rows"]), 8)
        self.assertGreater(r["modes"]["retrieved"]["summary"]["tokens_in"], 0)
    def test_missing_model_gives_a_pull_hint_and_exit_1(self):
        code, out = self.go("--limit", "3", "--model", "not-pulled:7b", "--out", os.path.join(self.tmp, "x.json")); self.assertEqual(code, 1); self.assertIn("ollama pull not-pulled:7b", out)
    def test_grid_skips_unpulled_models_but_not_the_reference(self):
        import yaml
        gc = os.path.join(self.tmp, "gen.yaml")
        with open(gc, "w", encoding="utf-8") as f: yaml.safe_dump(dict(reference=dict(model="fake-chat:1b", prompt="default", k=4, order="rank"), factors=dict(model=dict(values=["ghost:7b"], modes=["retrieved"]), order=dict(values=["doc"], modes=["retrieved"]))), f)
        out = os.path.join(self.tmp, "g.json"); code, text = self.go("--grid", "--limit", "6", "--skip-missing", "--gen-config", gc, "--out", out); self.assertEqual(code, 0); self.assertIn("SKIPPED", text); self.assertIn("ollama pull ghost:7b", text)
        with open(out, encoding="utf-8") as f: labels = [r["label"] for r in json.load(f)["runs"]]
        self.assertEqual(labels, ["fake-chat-1b_default_k4_rank", "fake-chat-1b_default_k4_doc"])
        code, text = self.go("--grid", "--limit", "6", "--skip-missing", "--gen-config", gc, "--model", "x", "--out", out)             # (--model is ignored by --grid: the reference comes from the gen config)
        self.assertEqual(code, 0)
        with open(gc, "w", encoding="utf-8") as f: yaml.safe_dump(dict(reference=dict(model="ghost:7b", prompt="default", k=4, order="rank"), factors={}), f)
        code, text = self.go("--grid", "--limit", "6", "--skip-missing", "--gen-config", gc, "--out", out); self.assertEqual(code, 1); self.assertIn("ollama pull ghost:7b", text)   # a missing REFERENCE is never skipped

if __name__ == "__main__": unittest.main()
