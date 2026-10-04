"""Frontend kernels and estimators (uarchre/kernels/frontend.py).

python3 -m unittest tests.test_frontend
"""

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uarchre.kernels import frontend as fe  # noqa: E402


class TestKernels(unittest.TestCase):
    def test_width_loop_fits_one_block(self):
        k = fe.width_loop(16)
        lines = [ln for ln in k.asm.splitlines() if ln and not ln.endswith(":") and not ln.startswith(".")]
        self.assertEqual(len(lines), 16)
        self.assertIn(".p2align 6", k.asm)
        with self.assertRaises(ValueError):
            fe.width_loop(17)
        with self.assertRaises(ValueError):
            fe.width_loop(1)
        m = fe.width_loop(10, filler="mix")
        self.assertEqual(m.asm.count("fadd.d"), 4)
        self.assertEqual(m.asm.count("nop"), 4)

    def test_straight_nops_footprint(self):
        k = fe.straight_nops(32768, filler="nop")
        self.assertEqual(k.params["insts"] * 4, 32768)
        self.assertIn(".rept 8189", k.asm)
        m = fe.straight_nops(32768)  # default mix: 8189 = 6 * 1364 + 5
        self.assertIn(".rept 1364", m.asm)
        self.assertEqual(m.asm.count("\nnop") - 4, 5)  # 4 NOPs inside the .rept block, 5 after
        self.assertIn("beqz a0, 9b", k.asm)  # exit test is a short backward branch

    def test_chain_order_is_one_cycle_without_fallthrough(self):
        for n in (3, 17, 1000):
            o = fe.chain_order(n, seed=3)
            self.assertEqual(sorted(o), list(range(n)))
            self.assertEqual(o[0], 0)
            self.assertTrue(all(o[i + 1] != o[i] + 1 for i in range(n - 1)))
        with self.assertRaises(ValueError):
            fe.chain_order(2, seed=0)

    def test_jump_chain_layout(self):
        n = 50
        k = fe.jump_chain(n, seed=1)
        jumps = re.findall(r"^j (\S+)$", k.asm, re.M)[2:]  # skip the entry/exit trampoline
        self.assertEqual(len(jumps), n)
        # exactly one jump returns to the lap head; the rest form a path
        self.assertEqual(jumps.count("5b"), 1)
        self.assertIn("beqz a0, 9b", k.asm)  # exit test is a short backward branch
        labels = re.findall(r"^(fej\d+):$", k.asm, re.M)
        self.assertEqual(len(labels), n)
        nxt = dict(zip(labels, jumps))
        seen, cur = set(), "fej0"
        while cur != "5b":
            self.assertNotIn(cur, seen)
            seen.add(cur)
            cur = nxt[cur]
        self.assertEqual(len(seen), n)

    def test_random_branches(self):
        k = fe.random_branches(words=2)
        self.assertEqual(k.units_per_lap, 128)
        self.assertEqual(k.asm.count("beqz t0"), 64)  # 64 static branches
        self.assertEqual(k.data_words, 9)  # 4 reps' worth of fresh data
        self.assertIn("data + (size_t)r * 2u", fe.harness_c(k))

    def test_harness_has_kernel_and_counters(self):
        src = fe.harness_c(fe.width_loop(4))
        self.assertIn("kern(laps, data);", src)  # no per-rep offset unless asked
        self.assertIn("kern:", src)
        self.assertIn("rdcycle", src)
        self.assertIn("rdinstret", src)
        self.assertIn(".option norvc", src)

    def test_parse_reps(self):
        out = "rep 0 cycles 100 instret 50\nrep 1 cycles 90 instret 50\ngem5 exit\n"
        self.assertEqual(fe.parse_reps(out), [(100, 50), (90, 50)])


class TestEstimators(unittest.TestCase):
    Ls = list(range(2, 17))

    def test_width_staircase(self):
        for w in (2, 3, 4, 6, 8):
            cpi = [math.ceil(L / w) for L in self.Ls]
            r = fe.width_estimate(self.Ls, cpi)
            self.assertEqual(r["width"], w)
            if w <= 8:
                self.assertEqual(r["shape"], "staircase")

    def test_width_ramp(self):
        for w in (2, 3, 4, 6):
            cpi = [max(math.ceil(L / 8), L / w) for L in self.Ls]
            r = fe.width_estimate(self.Ls, cpi)
            self.assertEqual(r["width"], w)
            self.assertEqual(r["shape"], "ramp")

    def test_onset_set_associative_ramp(self):
        # C = 256 entries, 4 ways -> 64 sets; overflow sets miss (ways+1) per lap
        C, A, P, base = 256, 4, 5.0, 3.0
        ns = list(range(128, 400, 8)) + [C, C + 1]
        cpu = []
        for n in ns:
            over = min(max(0, n - C), C // A)
            miss = (A + 1) * over if n <= C + C // A else n
            cpu.append(base + 2.0 / n + P * miss / n)
        exc = fe.lap_excess(ns, cpu, [n for n in ns if n < 200])
        lo, hi = fe.onset(ns, exc, tau=P / 2)
        self.assertEqual((lo, hi), (C, C + 1))

    def test_onset_none(self):
        self.assertIsNone(fe.onset([1, 2, 3], [0, 0.1, 0.2], tau=1))


class TestCapacitySearch(unittest.TestCase):
    """validation/gem5/fe_common.capacity_search on synthetic set-associative curves."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation", "gem5"))
        import fe_common

        cls.fc = fe_common

    @staticmethod
    def model(C, A, P=3.0, base=1.0, overhead=30.0, harness=0.0):
        sets = C // A

        def m(n):
            over = min(max(0, n - C), sets)
            miss = (A + 1) * over if n <= C + sets else n
            return base + (overhead + harness * (n >= C)) / n + P * miss / n

        return m

    def test_exact_onset(self):
        grid = self.fc.geom_grid(32, 12288, 4)
        for C, A in [(256, 1), (1024, 1), (1024, 4), (4096, 1), (4096, 8)]:
            r = self.fc.capacity_search(self.model(C, A), grid, log=None)
            self.assertEqual(tuple(r["bracket"]), (C, C + 1), (C, A))

    def test_never_uses_truth_in_grid(self):
        # the result does not depend on whether C is a grid point
        grid = self.fc.geom_grid(32, 12288, 4)
        r = self.fc.capacity_search(self.model(1000, 1), grid, log=None)
        self.assertEqual(tuple(r["bracket"]), (1000, 1001))

    def test_no_plateau(self):
        r = self.fc.capacity_search(self.model(16, 1), self.fc.geom_grid(64, 1024, 4), log=None)
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()
