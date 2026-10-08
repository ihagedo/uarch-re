"""AArch64 emitter (P0.6): lowering, the M1 register rules, cross-compilation.

python3 -m unittest tests.test_aarch64
"""

import re
import shutil
import unittest

from uarchre.backends.isa.aarch64 import AArch64, COUNTERS
from uarchre.backends.isa_base import emit_c, run_args
from uarchre.kernels import FILLERS, DELAYS, MIXES, Op
from uarchre.kernels.window import window_probe

REG = re.compile(r"\b([xwdsvq])(\d+)\b")
ALL_FILLERS = list(FILLERS) + list(MIXES)
ALL_DELAYS = ["miss"] + list(DELAYS)


def regs(line):
    """{('x'|'v', n)} named in one asm line (w/x share a number, d/s/q/v too)."""
    return {("x" if c in "xw" else "v", int(n)) for c, n in REG.findall(line)}


class TestAArch64Emitter(unittest.TestCase):
    def setUp(self):
        self.em = AArch64()

    def test_every_op_lowers(self):
        for op in Op:
            if op == Op.MISS_LOAD:
                self.assertEqual(self.em.miss(1), "ldr %1, [%1]")
            elif op in (Op.DELAY_FP, Op.DELAY_INT):
                self.assertTrue(self.em.delay(op, 0))
            else:
                self.assertTrue(self.em.filler(op, 3), op)

    def test_every_window_variant_lowers(self):
        for f in ALL_FILLERS:
            for d in ALL_DELAYS:
                for depth in (1, 2):
                    src = emit_c(window_probe(f, 9, depth=depth, delay=d), self.em)
                    self.assertIn("cycles_per_iter", src)

    def test_no_adr_and_no_reserved_registers(self):
        """adr/adrp are eliminated at rename from A15 on; x18 is Apple's platform
        register, x29/x30 the frame and link registers."""
        for f in ALL_FILLERS:
            for d in ALL_DELAYS:
                body = self.em.lower(window_probe(f, 16, depth=2, delay=d))
                text = "\n".join(body + list(self.em.loop_head()))
                self.assertNotRegex(text, r"\badrp?\b")
                for r in ("x18", "x29", "x30", "w18", "sp"):
                    self.assertNotRegex(text, rf"\b{r}\b")
        for c in ("x18", "x29", "x30"):
            self.assertNotIn(c, self.em.clobbers)

    def test_every_register_is_claimed_before_the_loop(self):
        """Unclaimed registers run slower on M1 (Handley), so the loop head writes all of them."""
        head = list(self.em.loop_head())
        written = set()
        for line in head:
            op = line.split()[0] if line.split() else ""
            if op in ("mov", "movz", "movk", "scvtf", "fmov"):
                written |= regs(line.split(",")[0])
        self.assertIn("cmp x6, x7", head)  # NZCV
        for f in ALL_FILLERS:
            for d in ALL_DELAYS:
                k = window_probe(f, 16, depth=2, delay=d)
                body = self.em.lower(k)[len(head) :]
                used = set().union(*(regs(ln) for ln in body)) - {("x", 31)}
                self.assertLessEqual(used, written, f"{f}/{d}: {sorted(used - written)}")

    def test_clobbers_cover_every_named_register(self):
        named = set()
        for f in ALL_FILLERS:
            for d in ALL_DELAYS:
                for ln in self.em.lower(window_probe(f, 16, depth=2, delay=d)):
                    named |= regs(ln)
        clob = set().union(*(regs(c) for c in self.em.clobbers))
        self.assertLessEqual(named, clob)

    def test_filler_values_and_addresses_vary(self):
        offs = {self.em.filler(Op.LOAD, i).split("#")[1] for i in range(8)}
        self.assertEqual(len(offs), 8)
        dests = {self.em.filler(Op.INT, i).split(",")[0] for i in range(8)}
        self.assertEqual(len(dests), 8)
        head = "\n".join(self.em.loop_head())
        self.assertNotRegex(head, r"mov x[67], #[01]\b")  # no 0/1 filler sources

    def test_m1_chain_rules(self):
        self.assertEqual(self.em.line, 128)
        self.assertGreaterEqual(self.em.default_bytes, 256 << 20)
        k = window_probe("nop", 8, depth=2)
        self.assertEqual(
            run_args(self.em, k, chain_bytes=4 << 20, iters=40, reps=2)[:2], [str(4 << 20), "40"]
        )

    def test_probe_is_a_reg_reg_chain(self):
        self.assertEqual(self.em.probe_add(), "add %0, %0, %1")
        self.assertNotIn("#", self.em.probe_add())

    def test_counters(self):
        self.assertIn("pmccntr_el0", AArch64().cycle_counter_c)
        self.assertIn("cntvct_el0", AArch64("cntvct").cycle_counter_c)
        self.assertIn("mach_absolute_time", AArch64("mach").cycle_counter_c)
        self.assertEqual(AArch64().default_probe_adds, 1 << 12)
        self.assertEqual(AArch64("mach").default_probe_adds, 1 << 22)
        self.assertEqual(set(COUNTERS), {"pmccntr", "cntvct", "mach"})
        with self.assertRaises(ValueError):
            AArch64("rdtsc")

    def test_registered_for_gem5(self):
        from uarchre.backends.platform.native import EMITTERS

        self.assertIsInstance(EMITTERS["aarch64"], AArch64)

    def test_arm_truth(self):
        """ARM ground truth: 42 int / 44 vector / 6 CC architectural registers
        mapped at reset in gem5 v25.1; FP fillers use the vector pool; a cmp
        writes 3 CC registers."""
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation", "gem5"))
        from arm_sweep import arm_truth

        t = arm_truth({"numROBEntries": 64})
        self.assertEqual((t["rob"], t["int_prf_headroom"], t["fp_prf_headroom"]), (64, 256 - 42, 256 - 44))
        self.assertEqual(t["flags_prf"], 5 * 256)  # ArmO3CPU default
        t = arm_truth({"numPhysCCRegs": 96, "numPhysIntRegs": 128})
        self.assertEqual((t["flags_prf_headroom"], t["flags_ops"], t["int_prf_headroom"]), (90, 30, 86))

    @unittest.skipUnless(
        shutil.which("aarch64-linux-gnu-gcc"), "needs the AArch64 cross compiler (. tools/env.sh)"
    )
    def test_cross_compiles(self):
        from uarchre.backends.platform.gem5 import TOOLCHAIN
        from uarchre.backends.platform.native import compile_source

        cc, cflags = TOOLCHAIN["aarch64"]
        for counter in ("pmccntr", "cntvct"):
            em = AArch64(counter)
            for f in ALL_FILLERS:
                for d in ALL_DELAYS:
                    k = window_probe(f, 12, depth=2, delay=d)
                    compile_source(
                        emit_c(k, em), k.name + "_" + counter, "aarch64", cc=cc, cflags=cflags, static=True
                    )


if __name__ == "__main__":
    unittest.main()
