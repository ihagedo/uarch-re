"""Generated C harness and toy-model noise.

python3 -m unittest tests.test_backends
"""

import platform
import shutil
import subprocess
import unittest

from uarchre.backends import RunResult
from uarchre.backends.isa_base import emit_c, parse_output, run_args
from uarchre.backends.isa.riscv64 import RISCV64
from uarchre.backends.isa.x86_64 import X86_64
from uarchre.backends.platform.toy import run_toy, ToyConfig
from uarchre.kernels.window import window_probe

OUT = """clock_ratio_pre 0.5766
rep 0 cycles_per_iter 248.127
rep 1 cycles_per_iter 246.595
rep 2 cycles_per_iter 243.851
clock_ratio_post 0.5180
cycles_per_iter 243.851
"""


class TestHarness(unittest.TestCase):
    def test_parse(self):
        r = parse_output(OUT)
        self.assertEqual(r, RunResult([248.127, 246.595, 243.851], 0.5766, 0.518))
        self.assertEqual((r.best, r.median), (243.851, 246.595))

    def test_parse_old_output(self):
        self.assertEqual(parse_output("cycles_per_iter 12.5\n"), RunResult([12.5]))

    def test_wrapping_chain_is_refused(self):
        k = window_probe("nop", 8, depth=2)
        with self.assertRaises(ValueError):
            run_args(X86_64(), k, chain_bytes=1 << 20, iters=20000, reps=5)
        self.assertEqual(
            run_args(RISCV64(), k, chain_bytes=4 << 20, iters=40, reps=2, seed=3),
            [str(4 << 20), "40", "2", str(1 << 12), "3"],
        )

    def test_probe_is_not_an_add_immediate(self):
        """Golden Cove-class cores fold add-immediate chains at rename, so the
        probe adds reg-reg."""
        self.assertNotIn("$", X86_64().probe_add())
        src = emit_c(window_probe("nop", 4), X86_64())
        self.assertIn("rep %d cycles_per_iter", src)
        self.assertIn("clock_ratio_pre", src)

    @unittest.skipUnless(platform.machine() == "x86_64" and shutil.which("gcc"), "needs x86-64 gcc")
    def test_native_run(self):
        from uarchre.backends.platform.native import run_native_reps

        r = run_native_reps(window_probe("nop", 16), chain_bytes=8 << 20, iters=500, reps=3)
        self.assertEqual(len(r.cycles), 3)
        self.assertGreater(r.clock_ratio_pre, 0.2)  # TSC ticks per core cycle; ~0.1 means adds were folded

    @unittest.skipUnless(
        shutil.which("riscv64-linux-gnu-gcc"), "needs the RISC-V cross compiler (. tools/env.sh)"
    )
    def test_riscv_compiles(self):
        from uarchre.backends.platform.native import compile_kernel
        from uarchre.backends.platform.gem5 import CROSS, CFLAGS

        b = compile_kernel(window_probe("store", 8, depth=2), "riscv64", cc=CROSS, cflags=CFLAGS, static=True)
        self.assertIn(b"ELF", subprocess.run(["head", "-c", "4", b], capture_output=True).stdout)


class TestToyNoise(unittest.TestCase):
    K = window_probe("nop", 40)

    def test_no_jitter_ignores_seed(self):
        self.assertEqual(
            run_toy(self.K, ToyConfig(rob=64, seed=1)), run_toy(self.K, ToyConfig(rob=64, seed=2))
        )

    def test_jitter_is_seeded(self):
        a = run_toy(self.K, ToyConfig(rob=64, miss_latency_jitter=0.2, seed=1))
        self.assertEqual(a, run_toy(self.K, ToyConfig(rob=64, miss_latency_jitter=0.2, seed=1)))
        self.assertNotEqual(a, run_toy(self.K, ToyConfig(rob=64, miss_latency_jitter=0.2, seed=2)))


if __name__ == "__main__":
    unittest.main()
