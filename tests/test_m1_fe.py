"""Frontend runner for Apple silicon (targets/m1/m1_fe.py) on a modelled M1: no Mac, no gem5.

python3 -m unittest tests.test_m1_fe
"""

import contextlib
import io
import json
import math
import os
import random
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "targets", "m1")))

import m1_fe  # noqa: E402
from uarchre.kernels.frontend import FEKernel  # noqa: E402

GHZ = 3.2
RATIO = 1024 * 24e6 / (GHZ * 1e9)  # 24 MHz ticks x 1024 per core cycle
MS = GHZ * 1e6  # core cycles per millisecond


def lru_misses(n, cap, ways):
    """Misses per lap of a cyclic walk over n items in a set-associative LRU structure."""
    sets = cap // ways
    over = min(max(0, n - cap), sets)
    return (ways + 1) * over if n <= cap + sets else n


def model(name, laps):
    """Core cycles of one rep: 8-wide fetch, 1536 x 128 B L1i (12-way), L0 BTB 1024 (4-way) with no
    bubble, 2 bubbles from the L1i, 13-cycle mispredicts."""
    if m := re.match(r"fe_width(?:_mix)?_L(\d+)$", name):
        return laps * max(1, math.ceil(int(m.group(1)) / 8))
    if m := re.match(r"fe_l1i3_(\d+)$", name):
        n = int(m.group(1)) // 128
        return laps * (4 * n + 30 + 6 * lru_misses(n, 1536, 12))
    if m := re.match(r"fe_btb2_n(\d+)_s\d+$", name):
        n = int(m.group(1))
        return laps * (n + 2 + 2 * lru_misses(n, 1024, 4) + 17 * lru_misses(n, 49152, 12 * 32))
    raise ValueError(name)


class FakeRunner:
    """native.Runner's interface over `model`: quantized 24 MHz ticks, rep noise, a slow cold rep 0."""

    def __init__(self, smoke=False, noise=2e-4, reject_every=0):
        self.smoke, self.noise, self.reject_every = smoke, noise, reject_every
        self.counter = "pmccntr" if smoke else "cntvct"
        self.attempts, self.rng, self.saved = [], random.Random(0), {}

    def compile(self, src, name):
        self.last_src = src
        return name

    def launch(self, binpath, argv=(), timeout=1800, label=""):
        laps, reps, seed, mode = (int(a) for a in argv)
        k = re.match(r"fe_mispredict_w(\d+)$", binpath)
        if k:
            cpb = 0.55 + (6.5 if mode == 1 else 0.0)
            cyc = laps * 64 * int(k.group(1)) * cpb
        else:
            cyc = model(binpath, laps)
        bad = self.reject_every and len(self.attempts) % self.reject_every == self.reject_every - 1
        self.attempts.append({"label": label, "accepted": not bad})
        lines = ["counter_hz 24576000.0", "qos_class 33"]
        for i in range(reps):
            c = cyc * (1.05 if i == 0 else 1.0) * (1.6 if bad else 1.0) * (1 + self.rng.gauss(0, self.noise))
            if self.rng.random() < 0.05:
                c *= 1.01  # an interrupt
            lines.append(f"rep {i} cycles {round(c * RATIO / 1024) * 1024}")
        return "\n".join(lines) + "\n", RATIO * (1 + self.rng.gauss(0, self.noise))

    def save(self, family, poc, raw, summary):
        self.saved[poc] = json.loads(json.dumps({"raw": raw, "summary": summary}, default=float))
        return os.path.join(m1_fe.native.RESULTS, "test", family, poc)


def tiny_chain(n, seed=0, isa=None):
    return FEKernel(f"fe_btb2_n{n}_s{seed}", "nop", n, "taken branch", {"n": n, "seed": seed}, isa="aarch64")


def rows(summary):
    return {r["structure"]: r for r in summary}


def run(fn, r=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(r or FakeRunner(), None)


class TestPlan(unittest.TestCase):
    """M1 sweeps bracket the published values with 30 % to spare, and every timed rep is >= 1 ms."""

    P = m1_fe.M1

    def brackets(self, grid, value, below=3):
        self.assertGreaterEqual(sum(g <= 0.7 * value for g in grid), below)
        self.assertGreaterEqual(max(grid), 1.3 * value)

    def test_published_values(self):
        self.assertEqual(m1_fe.published("frontend.fetch_width")[0], 8)
        self.assertEqual(m1_fe.published("frontend.l1i", "size_kib")[0], 192)
        self.assertEqual(m1_fe.published("frontend.btb_levels[0]", "entries")[0], 1024)
        self.assertEqual(m1_fe.published("frontend.btb_levels[1]", "taken_bubble_cycles")[0], 2)
        v, src = m1_fe.published("frontend.mispredict_penalty")
        self.assertEqual(v, 13)
        self.assertIn("7cpu-m1", src)

    def test_ranges(self):
        w = m1_fe.published("frontend.fetch_width")[0]
        self.assertLessEqual(min(self.P["width"]["Ls"]), 2)
        self.assertGreater(max(self.P["width"]["Ls"]), 1.3 * w)
        l1i_lines = m1_fe.published("frontend.l1i", "size_kib")[0] * 1024 // m1_fe.LINE
        self.brackets(self.P["l1i"]["grid"], l1i_lines)
        l0 = m1_fe.published("frontend.btb_levels[0]", "entries")[0]
        self.brackets(self.P["btb"]["grid"], l0)
        self.assertGreaterEqual(
            self.P["btb"]["top"], 1.3 * l1i_lines * m1_fe.LINE // 4
        )  # jumps filling the L1i
        self.assertLessEqual(self.P["jump"]["n"], 0.7 * l0)

    def test_timed_region(self):
        P = self.P
        self.assertGreaterEqual(P["width"]["laps"], MS)
        c = P["l1i"]
        for n in c["grid"]:
            fp = m1_fe.LINE * n
            self.assertGreaterEqual(max(c["min_laps"], c["work"] * 32 // fp) * fp / 32, MS)
        b = P["btb"]
        for n in b["grid"] + [b["top"]]:
            self.assertGreaterEqual(max(b["min_laps"], b["work"] // n) * n, MS)
        self.assertGreaterEqual(P["jump"]["laps"] * P["jump"]["n"], MS)
        m = P["mispredict"]
        self.assertGreaterEqual(64 * m["words"] * 0.5 * m1_fe.published("frontend.mispredict_penalty")[0], MS)
        self.assertGreaterEqual(m["pattern_laps"] * 64 * m["words"] * 0.5, MS)  # one taken branch per cycle

    def test_smoke_is_small(self):
        S = m1_fe.SMOKE
        self.assertLessEqual(len(S["l1i"]["grid"]) + len(S["btb"]["grid"]), 20)
        self.assertEqual(S["outer"], 1)


class TestConvert(unittest.TestCase):
    def test_mach_output(self):
        out = "counter_hz 24576000.0\nqos_class 33\n"
        out += "rep 0 cycles 90112\nrep 1 cycles 81920\nrep 2 cycles 83968\n"
        d = m1_fe.convert(out, 8.0, 1000, 3)
        self.assertEqual(d["ticks"], [90112, 81920, 83968])
        self.assertEqual(d["cpu"], [81920 / 8000, 83968 / 8000])  # rep 0 dropped

    def test_gem5_output(self):
        out = "rep 0 cycles 500 instret 300\nrep 1 cycles 400 instret 300\n"
        self.assertEqual(m1_fe.convert(out, 1.0, 100, 2)["cpu"], [4.0])
        self.assertEqual(m1_fe.convert("rep 0 cycles 7\n", 1.0, 7, 1)["cpu"], [1.0])

    def test_missing_reps(self):
        with self.assertRaises(RuntimeError):
            m1_fe.convert("rep 0 cycles 5\n", 1.0, 1, 3, "k")

    def test_point_uses_counter_and_args(self):
        r = FakeRunner()
        k = m1_fe.width_loop(8, isa=m1_fe.A64)
        launches = m1_fe.point(r, k, 1 << 20, 4, outer=2, seeds=(5,), mode=2)
        self.assertIn("cntvct_el0", r.last_src)
        self.assertEqual([x["seed"] for x in launches], [5, 5])
        self.assertEqual(len(launches[0]["cpu"]), 3)
        self.assertAlmostEqual(m1_fe.value(launches), 1.0, delta=0.01)

    def test_boot_resamples_launches(self):
        launches = [{"cpu": [1.0, 1.0], "accepted": True}, {"cpu": [2.0, 2.0], "accepted": True}]
        b = m1_fe.boot(launches)
        self.assertTrue(set(b) <= {1.0, 1.5, 2.0})
        self.assertGreater(b.std(), 0.2)
        launches[1]["accepted"] = False
        self.assertEqual(set(m1_fe.boot(launches)), {1.0})


class TestOnModel(unittest.TestCase):
    """Each PoC end to end on the modelled M1 with the M1 sweep plan."""

    def test_width(self):
        raw, summary = run(m1_fe.run_width)
        for f in m1_fe.FILLERS:
            row = rows(summary)[f"frontend.fetch_width[{f}]"]
            self.assertEqual((row["value"], row["ci"], row["published"]), (8, [8, 8], 8))
            self.assertEqual(raw["estimates"][f]["shape"], "staircase")
            self.assertEqual(row["flags"], [])

    def test_l1i(self):
        r = FakeRunner()
        raw, summary = run(m1_fe.run_l1i, r)
        cap = rows(summary)["frontend.l1i.size_kib"]
        self.assertAlmostEqual(cap["value"], 192, delta=2 * m1_fe.LINE / 1024)
        self.assertLessEqual(cap["ci"][0], cap["value"])
        self.assertGreaterEqual(cap["ci"][1], cap["value"])
        self.assertLess(cap["ci"][1] - cap["ci"][0], 4)
        self.assertEqual(cap["published"], 192)
        jump = rows(summary)["frontend.taken_jump_cycles"]
        self.assertAlmostEqual(jump["value"], 1.0, delta=0.02)
        self.assertEqual(jump["published"], 1.0)
        self.assertEqual(raw["search"]["bracket"][1] - raw["search"]["bracket"][0], 1)

    def test_btb(self):
        with mock.patch.object(m1_fe, "jump_chain", tiny_chain):
            raw, summary = run(m1_fe.run_btb)
        rs = rows(summary)
        l0 = rs["frontend.btb_levels[0].entries"]
        self.assertTrue(l0["ci"][0] <= 1024 <= l0["ci"][1], l0)
        self.assertLessEqual(abs(l0["value"] - 1024), 2)
        self.assertAlmostEqual(rs["frontend.btb_levels[0].taken_bubble_cycles"]["value"], 0, delta=0.05)
        self.assertAlmostEqual(rs["frontend.btb_levels[1].taken_bubble_cycles"]["value"], 2, delta=0.05)
        l1 = rs["frontend.btb_levels[1].entries"]
        self.assertLessEqual(abs(l1["value"] - 49152), 32)
        self.assertIsNone(l1["published"])
        self.assertTrue(all(s["consistent"] for lv in raw["levels"] for s in lv["seeds"].values()))

    def test_mispredict(self):
        r = FakeRunner()
        raw, summary = run(m1_fe.run_mispredict, r)
        pen = rows(summary)["frontend.mispredict_penalty"]
        self.assertAlmostEqual(pen["value"], 13, delta=0.05)
        self.assertEqual(len(raw["penalties"]), len(m1_fe.M1["mispredict"]["seeds"]))
        k = m1_fe.random_branches(m1_fe.M1["mispredict"]["words"], isa=m1_fe.A64)
        self.assertNotIn(f"size_t words  = {k.data_words};", r.last_src)  # enough fresh data for 9 reps
        self.assertIn(f"size_t words  = {9 * k.params['words'] + 1};", r.last_src)

    def test_rejected_launches_are_dropped(self):
        r = FakeRunner(reject_every=4)
        raw, summary = run(m1_fe.run_mispredict, r)
        pen = rows(summary)["frontend.mispredict_penalty"]
        self.assertAlmostEqual(pen["value"], 13, delta=0.05)
        self.assertTrue(any(f.startswith("unaccepted") for f in pen["flags"]))

    def test_smoke_plan_runs(self):
        r = FakeRunner(smoke=True)
        for fn in m1_fe.POCS.values():
            with mock.patch.object(m1_fe, "jump_chain", tiny_chain):
                raw, summary = run(fn, r)
            for row in summary:
                self.assertEqual(
                    set(row), {"structure", "value", "ci", "unit", "published", "source", "flags", "note"}
                )
            json.dumps(raw, default=float)


class TestCLI(unittest.TestCase):
    def test_help_and_unknown(self):
        for argv, code in ((["-h"], 0), (["P9.9"], 2)):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    m1_fe.main(argv)
            self.assertEqual(e.exception.code, code)

    def test_main_runs_the_named_pocs(self):
        fake = FakeRunner()
        with mock.patch.object(m1_fe.native, "Runner", lambda a: fake):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                m1_fe.main(["P3.7", "P3.1", "--tag", "t", "--force"])
        self.assertEqual(set(fake.saved), {"P3.7", "P3.1"})
        self.assertIn("frontend.mispredict_penalty", out.getvalue())
        self.assertEqual(set(m1_fe.POCS), set(m1_fe.EST_MIN))
        self.assertEqual(m1_fe.FAMILY, "fe")


if __name__ == "__main__":
    unittest.main()
