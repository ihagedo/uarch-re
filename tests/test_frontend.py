"""Frontend kernels and estimators (uarchre/kernels/frontend.py).

python3 -m unittest tests.test_frontend
"""

import ast
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
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

    def test_page_loop_layout(self):
        for isa in (fe.RV, fe.A64):
            k = fe.page_loop(20, 16380, isa=isa)
            body = k.asm.split("Lpg_loop:\n", 1)[1].splitlines()
            self.assertEqual(len(body), 20)
            self.assertEqual(body[-2:], [isa["dec"], isa["loop"].format(label="Lpg_loop")])
            self.assertIn(".p2align 14\nLpg_base:\n.org Lpg_base + 16380\n", k.asm)
            self.assertEqual((k.name, k.isa), ("fe_page_L20_o16380", isa["isa"]))
        self.assertEqual(fe.page_loop(10, 2044, filler="mix", isa=fe.A64).asm.count("fadd d"), 4)
        for bad in (
            dict(L=1, offset=0),
            dict(L=4, offset=2),
            dict(L=4, offset=16384),
            dict(L=4, offset=0, page=3),
        ):
            with self.assertRaises(ValueError):
                fe.page_loop(**bad)

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
        self.assertEqual(fe.parse_reps("counter_hz 1\nrep 0 cycles 7\n"), [(7, None)])

    def test_riscv_sources_unchanged(self):
        ks = []
        for f in ("nop", "mix"):
            ks += [fe.width_loop(L, filler=f) for L in range(2, 17)]
            ks += [fe.straight_nops(64 * n, filler=f) for n in (64, 257, 1000, 4096)]
        ks += [fe.jump_chain(n, s) for n in (32, 513, 4097, 12288) for s in (0, 1, 2)]
        ks += [fe.random_branches(w) for w in (2, 512)]
        src = "".join(fe.harness_c(k) for k in ks)
        self.assertEqual(hashlib.md5(src.encode()).hexdigest(), "ba2ba593b4775fb0decc59d54006f2c0")


def a64_kernels():
    A = fe.A64
    return [
        fe.width_loop(16, filler="mix", isa=A),
        fe.width_loop(5, isa=A),
        fe.straight_nops(64 * 40, isa=A),
        fe.straight_nops(4096, filler="nop", isa=A),
        fe.jump_chain(300, 1, isa=A),
        fe.random_branches(4, isa=A),
        fe.page_loop(64, 16380, isa=A),
        fe.page_loop(9, 2044, filler="mix", isa=A),
    ]


class TestAArch64(unittest.TestCase):
    REG = re.compile(r"\b([xwdsvq])(\d+)\b")

    def regs(self, text):
        return {("x" if c in "xw" else "v", int(n)) for c, n in self.REG.findall(text)}

    def test_layouts(self):
        A = fe.A64
        k = fe.width_loop(16, filler="mix", isa=A)
        body = [ln for ln in k.asm.splitlines() if not ln.endswith(":") and not ln.startswith(".")]
        self.assertEqual(len(body), 16)
        self.assertEqual(k.asm.count("fadd d"), 7)
        self.assertEqual(body[-2:], ["sub x0, x0, #1", "cbnz x0, 1b"])
        k = fe.straight_nops(32768, isa=A)
        self.assertEqual(k.params["insts"] * 4, 32768)
        self.assertIn("cbz x0, 9b", k.asm)
        self.assertEqual(k.asm.splitlines()[-2], "b 1b")  # large laps close with b
        k = fe.jump_chain(50, seed=1, isa=A)
        self.assertEqual(len(re.findall(r"^b (Lfej\d+|5b)$", k.asm, re.M)), 50)
        self.assertEqual(len(re.findall(r"^Lfej\d+:$", k.asm, re.M)), 50)
        k = fe.random_branches(words=2, isa=A)
        self.assertEqual(k.asm.count("cbz x6, 1f"), 64)
        self.assertEqual((k.units_per_lap, k.data_words, k.isa), (128, 9, "aarch64"))

    def test_m1_register_rules(self):
        """The claim block writes every register a kernel names (x0, x1 are the arguments) and
        NZCV; no adr/adrp, x18, x29, x30 or callee-saved registers."""
        claim = fe._a64_claim()
        self.assertEqual(claim[-1], "cmp x6, x7")
        written = set().union(*(self.regs(ln.split(",")[0]) for ln in claim))
        for k in a64_kernels():
            used = self.regs(k.asm)
            self.assertLessEqual(used - {("x", 0), ("x", 1)}, written, k.name)
            text = k.asm + "\n".join(claim)
            self.assertNotRegex(text, r"\badrp?\b|\b[xw](18|29|30)\b|\bsp\b")
            self.assertFalse({r for r in self.regs(text) if 19 <= r[1] <= 28 and r[0] == "x"}, k.name)
            self.assertFalse({r for r in self.regs(text) if 8 <= r[1] <= 15 and r[0] == "v"}, k.name)

    def test_harness_counters(self):
        k = fe.random_branches(4, isa=fe.A64)
        src = fe.harness_c(k)
        self.assertIn("pmccntr_el0", src)
        self.assertIn("pmevcntr0_el0", src)
        self.assertIn(" instret %llu", src)
        self.assertIn("kern(laps, data + (size_t)r * 4u);", src)
        self.assertIn('#define KERN "_kern"', src)  # Mach-O symbol
        src = fe.harness_c(k, counter="mach")
        self.assertIn("mach_absolute_time", src)
        self.assertNotIn("pmevcntr0_el0", src)
        self.assertNotIn("instret", src)

    @unittest.skipUnless(
        shutil.which("aarch64-linux-gnu-gcc"), "needs the AArch64 cross compiler (. tools/env.sh)"
    )
    def test_cross_compiles(self):
        from uarchre.backends.platform.gem5 import TOOLCHAIN

        cc, cflags = TOOLCHAIN["aarch64"]
        with tempfile.TemporaryDirectory() as d:
            for k in a64_kernels():
                for counter in ("pmccntr", "cntvct"):
                    c = os.path.join(d, f"{k.name}_{counter}.c")
                    with open(c, "w") as fh:
                        fh.write(fe.harness_c(k, counter))
                    subprocess.run([cc, *cflags, "-static", c, "-o", c[:-2]], check=True)

    @unittest.skipUnless(
        shutil.which("aarch64-linux-gnu-cpp") and (shutil.which("llvm-mc") or shutil.which("llvm-mc-19")),
        "needs the AArch64 cross preprocessor and llvm-mc",
    )
    def test_macho_assembles(self):
        """The kernel asm as clang on macOS sees it (__APPLE__) assembles for arm64-apple-macos."""
        mc = shutil.which("llvm-mc") or shutil.which("llvm-mc-19")
        with tempfile.TemporaryDirectory() as d:
            for k in a64_kernels():
                pp = subprocess.run(
                    ["aarch64-linux-gnu-cpp", "-D__APPLE__", "-P"],
                    input=fe.harness_c(k, "cntvct"),
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                block = re.search(r"__asm__\((.*?)\);", pp, re.S).group(1)
                asm = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', block))
                self.assertIn("_kern:", asm)
                s = os.path.join(d, "k.s")
                with open(s, "w") as fh:
                    fh.write(asm)
                r = subprocess.run(
                    [mc, "-triple=arm64-apple-macos11", "-filetype=obj", s, "-o", s + ".o"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, f"{k.name}: {r.stderr[:500]}")


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

    def page_curves(self, fetch, Ls=range(2, 65)):
        """(crossing, inside) cycles per iteration: 8-wide rename (sub + cbnz fused) under `fetch`."""
        ramp = [max(1, (L - 1) / 8) for L in Ls]
        cross = [max(fetch(L), r) for L, r in zip(Ls, ramp)]
        return list(Ls), cross, ramp

    def test_step_period(self):
        cases = {
            16: lambda L: 1 + math.ceil((L - 1) / 16),  # one 16-instruction block a cycle, split at a page
            8: lambda L: 1 + math.ceil((L - 1) / 8),
            None: lambda L: 1,  # no split
        }
        for period, fetch in cases.items():
            for Ls in (range(2, 65), [2, 4, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 26]):
                r = fe.step_period(*self.page_curves(fetch, Ls))
                self.assertEqual(r["period"], period, (period, list(Ls)))
                if period:
                    self.assertAlmostEqual(r["heights"][period], 1, places=6)
        r = fe.step_period(*self.page_curves(cases[16], range(2, 65)))
        self.assertEqual(r["heights"][8], 0)
        self.assertLess(r["fetch_bound"][-1], 26)  # the rename ramp takes over

    def test_step_period_two_staircases(self):
        """8 a cycle plus a refill per 64 B line (gem5's fetch buffer): the smaller period wins."""
        Ls = list(range(2, 40))
        y = [3 + math.ceil((L - 1) / 8) + 2 * math.ceil((L - 1) / 16) for L in Ls]
        r = fe.step_period(Ls, y, y)
        self.assertEqual(r["period"], 8)
        self.assertAlmostEqual(r["heights"][16], 2, places=6)
        self.assertIsNone(fe.step_period([2, 3, 4], [1, 1, 1], [1, 1, 1])["period"])  # too few points

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
