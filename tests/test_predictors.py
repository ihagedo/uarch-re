"""Frontend predictor kernels (P3.4, P3.5, P3.6, P3.8, P3.9).

python3 -m unittest tests.test_predictors
"""

import hashlib
import re
import shutil
import subprocess
import unittest

from uarchre.kernels import predictors as kp

A64 = "aarch64"


def a64_sources(counter="pmccntr"):
    return {
        "ras": kp.ras_source(dmax=16, seed=1, isa=A64, counter=counter),
        "fq": kp.fetchq_source([8, 12, 16], isa=A64, counter=counter),
        "fd": kp.fdip_source([3, 0], n_lines=256, isa=A64, counter=counter),
        "hist": kp.history_source([0, 5], isa=A64, counter=counter),
        "ind": kp.indirect_source(32, [8, 32], targets=2, isa=A64, counter=counter),
    }


def toplevel_asm(src):
    body = re.search(r"^__asm__\(\n(.*?)\n\);\n", src, re.S | re.M).group(1)
    return [ln.strip()[1:-3] for ln in body.splitlines()]


class TestParse(unittest.TestCase):
    def test_parse_points(self):
        out = "pt 3 rep 0 cyc 10.5\npt 3 rep 1 cyc 11\npt corr:8 rep 0 cyc 2e1\nnoise\n"
        self.assertEqual(kp.parse_points(out), {"3": [10.5, 11.0], "corr:8": [20.0]})


class TestGenerators(unittest.TestCase):
    def test_ras_depth_table(self):
        src = kp.ras_source(dmax=10, depths=[1, 5, 10])
        self.assertIn("ras_f9, ras_f5, ras_f0", src)  # entry for depth D is f_{dmax-D}
        self.assertEqual(src.count("jal ra, ras_f"), 9)  # direct calls only
        self.assertIn(".option norvc", src)

    def test_ras_placement_seed_changes_layout_only(self):
        a, b = kp.ras_source(dmax=8, seed=0), kp.ras_source(dmax=8, seed=3)
        self.assertNotEqual(a, b)
        strip = lambda s: s.replace('    "nop\\n"\n', "")
        self.assertEqual(strip(a), strip(b))

    def test_fetchq_block(self):
        src = kp.fetchq_source(20, copies=16, chain=12)
        self.assertEqual(kp.fetchq_source([4, 8], copies=16).count('"nop\\n"'), 16 * 12)
        self.assertEqual(src.count('"nop\\n"'), 16 * 20)
        self.assertEqual(src.count("csrr t2, fflags"), 16)
        with self.assertRaises(AssertionError):
            kp.fetchq_source(1000)

    def test_fdip_order_is_a_page_local_permutation(self):
        order = kp.fdip_order(256, seed=1)
        self.assertEqual(sorted(order), list(range(256)))
        for i in range(0, 256, 64):
            self.assertEqual(sorted(order[i : i + 64]), list(range(i, i + 64)))
        self.assertNotEqual(order[:64], list(range(64)))

    def test_fdip_barriers(self):
        src = kp.fdip_source([4, 8], n_lines=128)
        self.assertEqual(src.count("lla ra,"), 128 // 4 + 128 // 8)
        self.assertEqual(kp.fdip_source(0, n_lines=128).count("lla ra,"), 0)

    def test_history_variants(self):
        src = kp.history_source([5, 7])
        for v in ("corr", "rand", "const"):
            self.assertIn(f"hist_{v}_5:", src)
        self.assertEqual(src.count('"j 3f\\n"'), 3 * (5 + 7))

    def test_indirect_sites(self):
        src = kp.indirect_source(64, [8, 64], targets=2)
        self.assertEqual(src.count('"jalr ra, 0(s2)\\n"'), 64)
        self.assertIn("ind_leaf1", src)

    def test_riscv_sources_unchanged(self):
        src = "".join(
            [
                kp.ras_source(dmax=128, iters=32, reps=3, warm=4, seed=2),
                kp.ras_source(dmax=10, depths=[1, 5, 10]),
                kp.fetchq_source([8, 12, 16], chain=10),
                kp.fdip_source([0], seed=2),
                kp.fdip_source([4, 8], n_lines=128),
                kp.history_source([0, 16, 130], seed=1),
                kp.indirect_source(64, [8, 64], targets=2),
            ]
        )
        self.assertEqual(hashlib.md5(src.encode()).hexdigest(), "a8eb27b4883d3e803f5afdc7809999fc")


class TestAArch64(unittest.TestCase):
    def test_structure(self):
        src = a64_sources()
        ras, fq, hist, ind = (toplevel_asm(src[k]) for k in ("ras", "fq", "hist", "ind"))
        self.assertEqual(sum(ln.startswith("bl ras_f") for ln in ras), 15)
        self.assertEqual(ras.count("str x30, [sp, #-16]!"), 15)  # nested calls save the link register
        self.assertIn('__asm__("ras_f0")', src["ras"])  # Mach-O would add an underscore
        self.assertEqual(fq.count("mrs x9, fpsr"), 3 * 16)
        self.assertEqual(fq.count("nop"), 16 * (8 + 12 + 16))
        self.assertEqual(hist.count("b 3f"), 3 * 5)
        self.assertEqual(hist.count("cbz xzr, 4f"), 2)
        self.assertEqual((ind.count("blr x9"), ind.count("br x0")), (32, 1))

    def test_m1_rules(self):
        """No adr/adrp/lla, no x18/x29, no .option; every counter backend works."""
        for counter, read in (
            ("pmccntr", "pmccntr_el0"),
            ("cntvct", "cntvct_el0"),
            ("mach", "mach_absolute_time"),
        ):
            for name, src in a64_sources(counter).items():
                self.assertIn(read, src)
                asm = "\n".join(toplevel_asm(src))
                self.assertNotRegex(asm, r"\b(adrp?|lla|la)\b|\b[xw](18|29)\b|\.option", name)
        with self.assertRaises(ValueError):
            kp.ras_source(dmax=4, isa="x86_64")

    def test_fdip_barrier_table(self):
        """The C table holds the targets RISC-V loads with lla, in traversal order."""
        n, p = 256, 3
        src = kp.fdip_source([p], n_lines=n, isa=A64)
        table = [int(x) for x in re.search(r"fd0_bt\[\] = \{ ([\d, ]+) \}", src).group(1).split(", ")]
        order = kp.fdip_order(n)
        self.assertEqual(table, [order[i + 1] if i + 1 < n else n for i in range(p - 1, n, p)])
        lla = re.findall(r"lla ra, fd0_(l\d+|tail)", kp.fdip_source([p], n_lines=n))
        self.assertEqual(sorted(lla), sorted(f"l{t}" if t < n else "tail" for t in table))
        self.assertEqual(toplevel_asm(src).count("ldr x30, [x10], #8"), len(table))


@unittest.skipUnless(
    shutil.which("riscv64-linux-gnu-gcc"), "needs the RISC-V cross compiler (. tools/env.sh)"
)
class TestCompile(unittest.TestCase):
    def _syms(self, src, name):
        from uarchre.backends.platform.gem5 import compile_c

        b = compile_c(src, name)
        out = subprocess.run(["riscv64-linux-gnu-nm", b], capture_output=True, text=True, check=True).stdout
        return {ln.split()[2]: int(ln.split()[0], 16) for ln in out.splitlines() if len(ln.split()) == 3}

    def test_fetchq_landing_lines_share_cache_sets_not_btb_slots(self):
        syms = self._syms(kp.fetchq_source([8, 12, 16]), "t_fq")
        l2sets = set()
        for v in range(3):
            land = [syms[f"fq_land{v}_{k}"] for k in range(16)]
            self.assertEqual({(a >> 6) % 64 for a in land}, {63})  # one L1i set (32 KiB, 8-way)
            self.assertEqual(len({(a >> 6) % 512 for a in land}), 1)  # one L2 set (256 KiB, 8-way) ...
            l2sets |= {(land[0] >> 6) % 512}
            self.assertEqual(len({a & 0xFFF for a in land}), 16)  # distinct BTB slots
            blocks = [syms[f"fq_blk{v}_{k}"] for k in range(16)]
            self.assertTrue(all((a >> 6) % 64 != 63 for a in blocks))
        self.assertEqual(len(l2sets), 3)  # ... a different one per variant

    def test_all_compile(self):
        for src, name in [
            (kp.ras_source(dmax=16), "t_ras"),
            (kp.history_source(4), "t_hist"),
            (kp.indirect_source(32, [8, 32]), "t_ind"),
            (kp.fdip_source(4, n_lines=128), "t_fd"),
        ]:
            self.assertTrue(self._syms(src, name))


@unittest.skipUnless(
    shutil.which("aarch64-linux-gnu-gcc"), "needs the AArch64 cross compiler (. tools/env.sh)"
)
class TestCompileAArch64(unittest.TestCase):
    def _bin(self, src, name):
        from uarchre.backends.platform.gem5 import compile_c

        return compile_c(src, name, isa=A64)

    def test_no_adrp_between_counter_reads(self):
        """Timed code reaches kernels by bl/blr from registers loaded before the first read."""
        for name, src in a64_sources().items():
            b = self._bin(src, f"t_a64_{name}")
            dis = subprocess.run(
                ["aarch64-linux-gnu-objdump", "-d", "--no-show-raw-insn", b], capture_output=True, text=True
            ).stdout
            main = dis.split("<main>:\n", 1)[1].split("\n\n", 1)[0].splitlines()
            reads = [i for i, ln in enumerate(main) if "pmccntr_el0" in ln]
            self.assertGreaterEqual(len(reads), 2, name)
            for a, z in zip(reads[0::2], reads[1::2]):
                self.assertFalse([ln for ln in main[a:z] if re.search(r"\badrp?\b", ln)], name)

    def test_fetchq_layout_matches_riscv(self):
        b = self._bin(kp.fetchq_source([8, 12, 16], isa=A64), "t_a64_fq")
        out = subprocess.run(["aarch64-linux-gnu-nm", b], capture_output=True, text=True, check=True).stdout
        syms = {ln.split()[2]: int(ln.split()[0], 16) for ln in out.splitlines() if len(ln.split()) == 3}
        for v in range(3):
            land = [syms[f"fq_land{v}_{k}"] for k in range(16)]
            self.assertEqual({(a >> 6) % 64 for a in land}, {63})
            self.assertEqual(len({(a >> 6) % 512 for a in land}), 1)
            self.assertEqual(len({a & 0xFFF for a in land}), 16)


if __name__ == "__main__":
    unittest.main()
