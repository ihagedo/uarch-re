"""x86 host kernels and hygiene helpers (P0.4, P1.6, P1.8, P4.5).

python3 -m unittest tests.test_x86
"""

import os
import platform
import shutil
import sys
import unittest

import numpy as np

from uarchre.backends.isa_base import emit_c
from uarchre.kernels import Op
from uarchre.kernels.window import window_probe
from uarchre.kernels import x86_micro as xm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "targets", "x86"))
import hygiene  # noqa: E402

X86 = platform.machine() == "x86_64" and shutil.which("gcc")


class TestVariants(unittest.TestCase):
    def test_variant_replaces_int_filler_only(self):
        em = xm.VariantX86("mov")
        self.assertEqual(em.filler(Op.INT, 0), "movq %%r8, %%r10")
        self.assertEqual(em.filler(Op.NOP, 0), "nop")
        src = emit_c(window_probe("int", 4), em)
        self.assertEqual(src.count("movq %%r8, %%r1"), 8)  # 2 x N movs, rotated over r10-r15
        self.assertNotIn("leaq (%%r8,%%r9)", src)

    def test_unit_variants_have_the_documented_size(self):
        for name, (fn, entries, _) in xm.VARIANTS.items():
            asm = fn("%%r10")
            instrs = [ln for ln in asm.split("\\n\\t") if ln and not ln.endswith(":")]
            self.assertEqual(len(instrs), entries, name)

    def test_split_branch_cannot_fuse(self):
        lines = xm.VARIANTS["br_split6"][0]("%%r10").split("\\n\\t")
        i = next(k for k, ln in enumerate(lines) if ln.startswith("cmpq"))
        self.assertFalse(lines[i + 1].startswith("jne"))
        lines = xm.VARIANTS["br_nop7"][0]("%%r10").split("\\n\\t")
        i = next(k for k, ln in enumerate(lines) if ln.startswith("cmpq"))
        self.assertTrue(lines[i + 1].startswith("jne"))


class TestMicro(unittest.TestCase):
    def test_tests_are_unique_and_rotate(self):
        self.assertEqual(len(xm.MICRO), len(xm.P16) + len(xm.P18) + len(xm.P18X) - 2)
        body = xm.MICRO["add_tput"].body()
        self.assertGreater(len({ln.split(", ")[1] for ln in body}), 8)  # independent chains
        self.assertEqual(len({ln for ln in xm.MICRO["add_lat"].body()}), 1)  # one dependent chain

    def test_generated_c_has_every_test(self):
        src = xm.micro_c(xm.P16[:3], iters=10)
        for t in xm.P16[:3]:
            self.assertIn(f"t_{t.name}(", src)

    @unittest.skipUnless(X86, "needs an x86-64 host with gcc")
    def test_runs_and_add_chain_is_one_cycle(self):
        res = xm.run_micro([xm.MICRO["add_lat"], xm.MICRO["imul_lat"]], cpu=None, reps=3, iters=2000)
        add = np.median(xm.cycles(res["add_lat"]))
        imul = np.median(xm.cycles(res["imul_lat"]))
        self.assertAlmostEqual(add, 1.0, delta=0.25)  # the probe is itself a 1-cycle chain
        self.assertAlmostEqual(imul / add, 3.0, delta=0.6)


class TestHygiene(unittest.TestCase):
    def rows(self, seed=0, reps=5):
        rng = np.random.default_rng(seed)
        out, i = [], 0
        for rep in range(reps):
            for n in rng.permutation(np.arange(16, 640, 16)):
                y = 250.0 if n <= 500 else 500.0
                out.append(
                    {
                        "key": "nop",
                        "n": int(n),
                        "outer_rep": rep,
                        "order_index": i,
                        "value": y * (1 + 0.01 * rng.standard_normal()),
                    }
                )
                i += 1
        return out

    def test_a8_quiet_on_clean_data(self):
        self.assertFalse(hygiene.drift_rules(self.rows())["fires"])

    def test_a8_fires_on_injected_step_and_ramp(self):
        r = self.rows()
        self.assertTrue(hygiene.drift_rules(hygiene.inject(r, "step", 0.05))["b"]["fires"])
        self.assertTrue(hygiene.drift_rules(hygiene.inject(r, "ramp", 0.08))["c"]["fires"])

    def test_a8_ratio_rule(self):
        out = hygiene.drift_rules(self.rows(), ratio_pairs=[(0.60, 0.66)] * 10)
        self.assertTrue(out["a"]["fires"])

    def test_siblings(self):
        if os.path.exists("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list"):
            self.assertNotIn(0, hygiene.siblings(0))


if __name__ == "__main__":
    unittest.main()
