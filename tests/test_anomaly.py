"""Anomaly rules A1-A12 (P0.3) and the toy mechanisms behind them.

  python3 -m unittest tests.test_anomaly                 # ~1 min
  UARCHRE_SLOW=1 python3 -m unittest tests.test_anomaly  # + every toy_cases.yaml case (~2 min)

Each fixture in tests/fixtures/anomaly_cases.py must fire exactly its expected rule ids.
"""

import os
import unittest

from tests.fixtures import anomaly_cases as AC
from uarchre.analysis.anomaly import (
    Curve,
    RunSet,
    evaluate,
    fired,
    apply,
    transitions,
    load_thresholds,
    probe_params,
    RULE_IDS,
    SEVERITY,
    RULES,
)
from uarchre.analysis.knee import Knee, find_knee, attribute
from uarchre.backends.platform.toy import run_toy, ToyConfig
from uarchre.kernels import Kernel, Instr, Op
from uarchre.kernels.window import window_probe

BIG = dict(int_prf=512, fp_prf=512, lq=256, sq=256)


def knee_of(filler, ns, cfg, depth=1, delay="miss"):
    mj, st = probe_params(depth, delay)
    return find_knee(
        ns,
        [run_toy(window_probe(filler, n, depth=depth, delay=delay), cfg, iters=30, warm=10) for n in ns],
        mj,
        st,
    )


class TestRegistry(unittest.TestCase):
    def test_every_rule_has_an_id_and_severity(self):
        self.assertEqual(set(SEVERITY) - set(RULES), {"offset.missing"})  # emitted by the A11 rule
        self.assertEqual(set(RULE_IDS.values()), set(SEVERITY))
        thr = load_thresholds()
        self.assertFalse(thr["frozen"])  # to be frozen before the first M1 run
        for code in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A12"):
            self.assertIn(code, thr)


class TestFixtures(unittest.TestCase):
    """Synthetic, cross-curve, published and gem5 fixtures: exactly the expected rules."""

    def check(self, cases):
        self.assertTrue(cases)
        for name, rs, expect in cases:
            with self.subTest(fixture=name):
                self.assertEqual(fired(evaluate(rs)), expect)

    def test_synthetic_curves(self):
        self.check(AC.synth_cases())

    def test_cross_curve(self):
        self.check(AC.cross_cases())

    @unittest.skipUnless(AC.gem5_rob_cases(), "no results/gem5/p11-rob*")
    def test_gem5_rob_fires_nothing(self):
        self.check(AC.gem5_rob_cases())

    def test_gem5_campaign_only_uncalibrated(self):
        cases = AC.gem5_campaign_cases()
        if not cases:
            self.skipTest("no backend campaign results")
        self.check(cases)


class TestFirestorm(unittest.TestCase):
    def setUp(self):
        _, self.rs, self.expect = AC.firestorm()
        self.findings = evaluate(self.rs)

    def test_fires_a4_and_a5(self):
        self.assertEqual(fired(self.findings), ["delay.disagreement", "ratio.retire_group"])
        a4 = [f for f in self.findings if f.rule == "ratio.retire_group"][0]
        self.assertEqual(a4.evidence["group"], 7)
        self.assertAlmostEqual(a4.evidence["ratio"], 2264 / 330, places=2)
        self.assertEqual({f.structure for f in self.findings if f.rule == "delay.disagreement"}, {"lq", "sq"})

    def test_no_lq_or_sq_reported_as_measured(self):
        """Our probe is the miss delay: attribute() would report LQ 127, SQ 100."""
        knees = {
            f: Knee(c.knee.knee, c.knee.ci, 2.0, 0.99)
            for (f, d, dl), c in self.rs.curves.items()
            if dl == "miss"
        }
        est = attribute(knees, {"emitter": "aarch64", "platform": "m1", "depth": 1, "delay": "miss"})
        apply(est, self.findings)
        for s in ("lq", "sq"):
            self.assertNotEqual(est[s]["status"], "measured", s)
            self.assertIn("delay.disagreement", est[s]["anomalies"])
            self.assertEqual(est[s]["bound"]["upper"], 330)


class TestTransitions(unittest.TestCase):
    def test_staircase_is_one_transition(self):
        x = list(range(50, 75))
        y = [600 if n <= 60 else 616 if n == 61 else 902 if n == 62 else 1202 for n in x]
        self.assertEqual(len(transitions(x, y)), 1)

    def test_two_plateaus_are_two(self):
        x = list(range(2, 90, 2))
        y = [200 if n <= 8 else 300 if n <= 64 else 600 for n in x]
        tr = transitions(x, y)
        self.assertEqual([(a, b) for a, b, _ in tr], [(8, 10), (64, 66)])


class TestToyDefaults(unittest.TestCase):
    """With every mechanism off, run_toy reproduces the original model bit for bit."""

    GOLDEN = [
        (("nop", 61, 1, "miss"), dict(rob=64, int_prf=88, fp_prf=92, lq=40, sq=36), 216.0),
        (("nop", 62, 1, "miss"), dict(rob=64, int_prf=88, fp_prf=92, lq=40, sq=36), 216.4915254237288),
        (("int", 45, 2, "miss"), dict(rob=64, int_prf=88, fp_prf=92, lq=40, sq=36), 410.0),
        (("fp", 62, 3, "miss"), dict(rob=64, int_prf=88, fp_prf=92, lq=40, sq=36), 1202.0),
        (("store", 45, 3, "miss"), dict(rob=64, int_prf=88, fp_prf=92, lq=40, sq=36), 1206.0),
        (("nop", 50, 8, "int"), dict(rob=64, lq=40, sq=36), 204.0),
        (("nop", 60, 1, "miss"), dict(rob=64, miss_latency_jitter=0.2, seed=3), 218.42372881355934),
        (("load", 63, 2, "miss"), dict(rob=64, miss_latency_jitter=0.1, seed=5, width=2), 827.0338983050848),
    ]

    def test_defaults_bit_identical(self):
        for (f, n, d, dl), cfg, want in self.GOLDEN:
            with self.subTest(f=f, n=n, d=d, delay=dl):
                self.assertEqual(run_toy(window_probe(f, n, depth=d, delay=dl), ToyConfig(**cfg)), want)


class TestToyMechanisms(unittest.TestCase):
    def test_retire_groups_ratio_is_seven(self):
        cfg = ToyConfig(rob=48, retire_group=7, failable_per_group=1, width=8, miss_latency=300, **BIG)
        nop = knee_of("nop", range(318, 336), cfg).knee
        for f in ("load", "store", "branch"):
            k = knee_of(f, range(40, 52), cfg).knee
            self.assertAlmostEqual(nop / k, 7, delta=0.5, msg=f)

    def test_classic_ratio_is_one(self):
        cfg = ToyConfig(rob=64, width=8, miss_latency=300, **BIG)
        nop = knee_of("nop", range(56, 68), cfg).knee
        for f in ("load", "store", "branch"):
            self.assertAlmostEqual(nop / knee_of(f, range(56, 68), cfg).knee, 1, delta=0.05)

    def test_issue_time_sq_depends_on_the_delay(self):
        base = dict(rob=128, int_prf=512, fp_prf=512, lq=32, sq=24)
        issue = ToyConfig(
            **base, lsq_alloc="issue", ls_sched=16, lq_release_early=True, sq_release_early=True
        )
        miss = knee_of("store", range(30, 48), issue).knee
        alu = knee_of("store", range(110, 130), issue, depth=8, delay="fp").knee
        self.assertEqual(miss, 39)  # SQ 24 + scheduler 16 - 1
        self.assertGreater(alu - miss, 60)  # the ALU reading reaches the ROB
        disp = ToyConfig(**base)
        self.assertEqual(
            knee_of("store", range(16, 32), disp).knee,
            knee_of("store", range(16, 32), disp, depth=8, delay="fp").knee,
        )

    def test_two_level_sq_steps_twice(self):
        cfg = ToyConfig(rob=256, sq_levels=(16, 48), int_prf=512, fp_prf=512, lq=256)
        ns = list(range(2, 90, 2))
        ys = [run_toy(window_probe("store", n), cfg, iters=30, warm=10) for n in ns]
        self.assertEqual(len(transitions(ns, ys)), 2)
        one = ToyConfig(rob=256, sq=64, int_prf=512, fp_prf=512, lq=256)
        ys1 = [run_toy(window_probe("store", n), one, iters=30, warm=10) for n in ns]
        self.assertEqual(len(transitions(ns, ys1)), 1)

    def test_predictors_speed_up_only_predictable_chains(self):
        def per_load(pattern, **kw):
            k = Kernel("chase", [Instr(Op.MISS_LOAD, 0)] * 8, n_chains=1, params={"chain": pattern})
            return run_toy(k, ToyConfig(**kw), iters=20, warm=8) / 8

        self.assertEqual({per_load(p) for p in ("random", "stride", "constant")}, {200.0})
        lap = {p: per_load(p, load_addr_predict=True) for p in ("random", "stride", "constant")}
        lvp = {p: per_load(p, load_value_predict=True) for p in ("random", "stride", "constant")}
        self.assertEqual((lap["random"], lvp["random"], lvp["stride"]), (200.0, 200.0, 200.0))
        for v in (lap["stride"], lap["constant"], lvp["constant"]):
            self.assertLess(v, 200 / 4)

    def test_dispatch_throughput_limit(self):
        """filler_throughput limits entry into the window: 1 store/cycle at an
        80-cycle miss leaves no knee at depth 1."""
        cfg = ToyConfig(
            rob=512, miss_latency=80, sq=96, int_prf=512, fp_prf=512, lq=256, filler_throughput={"store": 1}
        )
        self.assertIsNone(knee_of("store", range(30, 110, 4), cfg).knee)


class TestApply(unittest.TestCase):
    def test_tie_sets_lower_bound(self):
        rs = RunSet("synth", "t", ctx={"emitter": "abstract", "platform": "toy"}, offsets=AC.offsets_table())
        rs.add(Curve.from_knee("nop", 94))
        rs.add(Curve.from_knee("int", 92))
        est = {"int_prf_headroom": {"value": 94.0, "status": "measured", "anomalies": []}}
        apply(est, evaluate(rs))
        e = est["int_prf_headroom"]
        self.assertEqual((e["anomalies"], e["bound"]), (["tie.rob"], {"lower": 94.0}))
        self.assertTrue(e["status"].startswith("ambiguous"))


class TestIntegration(unittest.TestCase):
    """The path sweep.py takes: attribute(), then the registry replaces knee.py's flags."""

    def test_attribute_then_apply_carries_exactly_the_registry_ids(self):
        ctx = {"emitter": "abstract", "platform": "toy", "depth": 1, "delay": "miss"}
        for name, rs, expect in AC.synth_cases():
            with self.subTest(fixture=name):
                c = list(rs.curves.values())[0]
                knees = {c.filler: c.knee}
                fs = evaluate(rs)
                est = attribute(knees, {**ctx, "depth": c.depth}, AC.offsets_table())
                apply(est, fs, replace=True)
                got = sorted({a for e in est.values() for a in e["anomalies"]})
                self.assertEqual(got, expect)

    def test_load_results_round_trip(self):
        if not os.path.exists(os.path.join(AC.GEM5, "proto-rob64-d2", "sweep_raw.csv")):
            self.skipTest("no results/gem5/proto-rob64-d2")
        rs = AC.gem5_runset("proto", ["proto-rob64-d2"])
        c = rs.curves[("nop", 2, "miss")]
        self.assertEqual((c.knee.knee, c.knee.ci_method, c.knee.n_reps), (61.0, "bootstrap", 3))
        self.assertTrue(c.order and rs.clock)


@unittest.skipUnless(os.environ.get("UARCHRE_SLOW"), "set UARCHRE_SLOW=1 to run the toy configurations")
class TestToyCases(unittest.TestCase):
    def test_toy_configurations(self):
        curves = AC.run_toy_curves(jobs=int(os.environ.get("UARCHRE_JOBS", "6")))
        for name, rs, expect in AC.toy_runsets(curves):
            with self.subTest(case=name):
                self.assertEqual(fired(evaluate(rs)), expect)


if __name__ == "__main__":
    unittest.main()
