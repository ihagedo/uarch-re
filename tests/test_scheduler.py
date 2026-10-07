"""P1.5 / P1.7 kernels (uarchre/kernels/scheduler.py) and their gem5 helpers.

python3 -m unittest tests.test_scheduler
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from uarchre.kernels.scheduler import (
    sched_probe,
    retire_drain,
    nop_stream,
    dispatch_probe,
    from_window,
    multi_source,
    parse_multi,
    chain_nodes_needed,
    c_source,
)
from uarchre.kernels.window import window_probe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "validation", "gem5"))


class TestSchedProbe(unittest.TestCase):
    def test_dependent_fillers(self):
        k = sched_probe("add", 5)
        asm = k.params["asm"]
        self.assertEqual(asm[0], "ld %0, 0(%0)")
        self.assertEqual(asm[6], "ld %1, 0(%1)")
        self.assertTrue(all(ln.startswith("add ") and ln.endswith("%0, t0") for ln in asm[1:6]))
        self.assertTrue(all(ln.endswith("%1, t0") for ln in asm[7:12]))
        self.assertEqual(len(asm), 12)

    def test_memory_fillers_get_a_dependent_scratch_address(self):
        asm = sched_probe("load", 3).params["asm"]
        self.assertEqual(asm[:3], ["ld %0, 0(%0)", "andi a5, %0, 0", "add a5, a5, %3"])
        self.assertTrue(all("(a5)" in ln for ln in asm[3:6]))
        self.assertTrue(all("(a6)" in ln for ln in asm[9:12]))
        self.assertTrue(all(ln.startswith("sd t0") for ln in sched_probe("store", 2).params["asm"][3:5]))

    def test_preload_comes_first_and_keeps_overhead_constant(self):
        asm = sched_probe("add", 2, pre="mul", m=3).params["asm"]
        self.assertEqual([ln.split()[0] for ln in asm[:6]], ["ld", "mul", "mul", "mul", "add", "add"])
        # helpers are emitted at m = 0 too, so the baseline of a sharing test has the same overhead
        self.assertEqual(sched_probe("add", 2, pre="load", m=0).params["asm"][1], "andi a5, %0, 0")

    def test_fp_filler_is_an_fp_destination_fed_by_the_miss(self):
        self.assertEqual(sched_probe("fp", 1).params["asm"][1], "fmv.d.x ft0, %0")


class TestRetireKernels(unittest.TestCase):
    def test_drain_layout(self):
        k = retire_drain(10, links=4, exhaust=9)
        asm = k.params["asm"]
        self.assertEqual(asm.count("fsqrt.d ft8, ft8"), 4)
        self.assertEqual(asm.count("nop"), 10)
        self.assertEqual(sum(ln.startswith("add ") for ln in asm), 9)
        self.assertEqual(k.params["depth"], 0)  # no pointer chase
        self.assertEqual(nop_stream(7).params["asm"], ["nop"] * 7)

    def test_dispatch_probe(self):
        asm = dispatch_probe(3, 2).params["asm"]
        self.assertEqual([ln.split()[0] for ln in asm], ["add", "add", "add", "fadd.d", "fadd.d"])

    def test_from_window_keeps_the_body(self):
        k = from_window(window_probe("nop", 2))
        self.assertEqual(k.params["asm"], ["ld %0, 0(%0)", "nop", "nop", "ld %1, 0(%1)", "nop", "nop"])
        self.assertEqual(k.params["depth"], 1)


class TestMultiPoint(unittest.TestCase):
    def test_parse(self):
        out = (
            "point 0 rep 0 cycles_per_iter 10.5\npoint 0 rep 1 cycles_per_iter 9.0\n"
            "point 1 rep 0 cycles_per_iter 20.0\ncycles_per_iter done\n"
        )
        self.assertEqual(parse_multi(out, 2), {0: [10.5, 9.0], 1: [20.0]})

    def test_chain_budget(self):
        ks = [sched_probe("add", 1), retire_drain(1)]
        self.assertEqual(chain_nodes_needed(ks, 10, 3), 30)

    def test_source_has_one_function_per_point(self):
        src = multi_source("t", [sched_probe("add", 1), nop_stream(2)])
        self.assertIn("point1(void **pa", src)
        self.assertIn("{ point0, point1 }", src)

    @unittest.skipUnless(shutil.which("riscv64-linux-gnu-gcc"), "needs the RISC-V cross compiler")
    def test_cross_compiles(self):
        for src in (
            multi_source("t", [sched_probe("load", 2), retire_drain(3), dispatch_probe(2, 2)]),
            c_source(sched_probe("store", 2, pre="mul", m=1)),
        ):
            with tempfile.TemporaryDirectory() as d:
                c = os.path.join(d, "k.c")
                with open(c, "w") as fh:
                    fh.write(src)
                r = subprocess.run(
                    [
                        "riscv64-linux-gnu-gcc",
                        "-O1",
                        "-march=rv64gc",
                        "-mabi=lp64d",
                        "-static",
                        c,
                        "-o",
                        os.path.join(d, "k"),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr)


class TestGem5Helpers(unittest.TestCase):
    def test_iq_set(self):
        from sched_common import iq_set

        s = iq_set([(24, [("alu", 4)])])
        self.assertTrue(
            s.startswith("cpu.instQueues=[IQUnit(numEntries=24, fuPool=FUPool(FUList=[IntALU(count=4)]))")
        )
        self.assertIn("System_Unit", s)  # catch-all queue so no op class is left without a queue

    def test_reference_model(self):
        from sched_common import expected_knee

        queues = [
            (24, {"IntAlu"}),
            (20, {"IntAlu", "IntMult"}),
            (20, {"MemRead"}),
            (16, {"MemRead", "MemWrite"}),
        ]
        self.assertEqual(expected_knee(queues, "add"), 44)
        self.assertEqual(expected_knee(queues, "mul", pre="add", m=30), 14)  # 6 adds spill into q1
        self.assertEqual(expected_knee(queues, "load"), 34)  # 2 misses resident
        self.assertEqual(expected_knee(queues, "store", pre="load", m=20), 14)  # 18 fit in q2, 2 spill
        uni = [(48, {"IntAlu", "IntMult", "MemRead", "MemWrite", "FloatCvt"})]
        self.assertEqual(expected_knee(uni, "add"), 46)
        self.assertEqual(expected_knee(uni, "add", pre="load", m=4), 40)  # misses, helpers, preload

    def test_hockey_fit(self):
        from sched_common import hockey

        ms = [0, 8, 16, 20, 24, 28, 32]
        b, c, slope = hockey(ms, [20, 20, 20, 20, 20, 16, 12])
        self.assertEqual((b, c), (24, 20))
        self.assertAlmostEqual(slope, -1.0)


if __name__ == "__main__":
    unittest.main()
