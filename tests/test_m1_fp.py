"""Predictor and decoupling runner for Apple silicon (targets/m1/m1_fp.py) on a modelled M1: no Mac, no gem5.

python3 -m unittest tests.test_m1_fp
"""

import ast
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "targets", "m1")))

import m1_fp  # noqa: E402
from uarchre.kernels import predictors as kp  # noqa: E402

GHZ = 3.2
RATIO = 1024 * 24e6 / (GHZ * 1e9)  # 24 MHz ticks x 1024 per core cycle
MS = GHZ * 1e6  # core cycles per millisecond


def table(src, name):
    return [int(x) for x in re.search(rf"{name}\[\] = \{{ ([\d, ]+) \}}", src).group(1).split(", ")]


def model(name, src, hist=100, ras=48, ind=1000, overlap=1.0, barrier=13.0, fq=200):
    """{key: core cycles per unit} and units per timed region for one harness of m1_fp."""
    if name.startswith("m1fp_hist"):
        units = int(re.search(r"\(t1 - t0\) / (\d+)\);", src).group(1))
        out = {}
        for v, n in re.findall(r'printf\("pt (\w+):(\d+) rep', src):
            n = int(n)
            out[f"{v}:{n}"] = 12 + n + (6.5 if v == "rand" or (v == "corr" and n + 1 > hist) else 0)
        return out, units
    if name.startswith("m1fp_ras"):
        units = int(re.search(r"\(t1 - t0\) / (\d+)\);", src).group(1))
        return {str(d): 8 + 2 * d + 13 * max(0, d - ras) for d in table(src, "depth")}, units
    if name.startswith("m1fp_ind"):
        rounds, t = (int(x) for x in re.search(r"/ \((\d+) \* (\d+)\)\);", src).groups())
        return {str(s): 30 + 2 * s + (15 * s if s * t > ind else 0) for s in table(src, "sizes")}, rounds * t
    if name.startswith("m1fp_fd"):
        iters, n = (int(x) for x in re.search(r"/ \((\d+) \* (\d+)\)\);", src).groups())
        miss = 25 if n * m1_fp.FD_LINE > 192 << 10 else 0
        cost = lambda p: 4 + (barrier + miss if p == 1 else miss / overlap)
        return {p: cost(int(p)) for p in re.findall(r'printf\("pt (\d+) rep', src)}, iters * n
    if name.startswith("m1fp_fq"):
        laps, copies = (int(x) for x in re.search(r"/ \((\d+) \* (\d+)\)\);", src).groups())
        bs = [int(b) for b in re.findall(r'printf\("pt (\d+) rep', src)]
        chain = src.count('"fsqrt d1, d1\\n"') // (copies * len(bs))
        return {str(b): 13 * chain + b / 8 + (90 if b > fq else 0) for b in bs}, laps * copies
    raise ValueError(name)


class FakeRunner:
    """native.Runner's interface over `model`: quantized 24 MHz ticks (cycles under smoke), a little noise."""

    def __init__(self, smoke=False, noise=1e-4, **kw):
        self.smoke, self.noise, self.kw = smoke, noise, kw
        self.counter = "pmccntr" if smoke else "cntvct"
        self.rng, self.short, self.labels, self.srcs = random.Random(0), [], [], []

    def compile(self, src, name):
        self.srcs.append(src)
        return name, src

    def launch(self, binpath, argv=(), timeout=1800, label=""):
        cyc, units = model(*binpath, **self.kw)
        self.labels.append(label)
        ratio, q = (1.0, 1) if self.smoke else (RATIO, 1024)
        lines = [] if self.smoke else ["counter_hz 24576000.0"]
        for key, c in cyc.items():
            if units * c < MS:
                self.short.append((label, key, units * c))
            for i in range(m1_fp.REPS):
                ticks = round(units * c * (1 + self.rng.gauss(0, self.noise)) * ratio / q) * q
                lines.append(f"pt {key} rep {i} cyc {ticks / units:.4f}")
        return "\n".join(lines) + "\n", ratio


def run(poc, r=None, args=None):
    r = r or FakeRunner()
    with contextlib.redirect_stdout(io.StringIO()):
        raw, summary = m1_fp.POCS[poc](r, args)
    json.dumps({"raw": raw, "summary": summary}, default=float)  # what native.Runner.save writes
    return r, raw, {s["structure"]: s for s in summary}


class TestPlan(unittest.TestCase):
    """Sweeps bracket the published values with 30 % to spare; unknown ones are swept wide."""

    def test_history_grid(self):
        h = m1_fp.native.published("frontend.direction_predictor.history_bits") or m1_fp.HIST_PUBLISHED
        self.assertEqual(m1_fp.HIST_PUBLISHED, 100)
        self.assertGreaterEqual(sum(n + 1 <= 0.7 * h for n in m1_fp.HIST_GRID), 5)
        self.assertGreaterEqual(max(m1_fp.HIST_GRID) + 1, 2 * h)

    def test_unknown_sizes_swept_wide(self):
        ras = m1_fp.native.published("frontend.ras_depth.value") or 0
        self.assertGreaterEqual(m1_fp.RAS_DMAX, max(128, 1.3 * ras))
        self.assertLessEqual(min(m1_fp.IND_SIZES), 8)
        self.assertGreaterEqual(max(m1_fp.IND_SIZES), 16384)
        self.assertGreaterEqual(max(m1_fp.FQ_GRID), 512)

    def test_fdip_footprints(self):
        l1i = m1_fp.native.published("frontend.l1i.size_kib") << 10
        self.assertEqual(m1_fp.FD_LINE, m1_fp.native.published("memory.l2.line_bytes"))
        ref, *big = (n * m1_fp.FD_LINE for n in m1_fp.FD_LINES)
        self.assertLessEqual(ref, 0.7 * l1i)
        self.assertTrue(all(1.3 * l1i <= b <= 0.1 * (12 << 20) for b in big))
        self.assertFalse(m1_fp.native.published("decoupling.fdip_signature.present"))

    def test_fetchq_layout_fits(self):
        L = m1_fp.FQ_LAYOUT
        self.assertEqual(L["slot"], 16 << 10)  # macOS page: the virtual layout fixes the L1i set
        self.assertGreater(L["copies"], 2 * 12)  # also thrashes a 6-way L1i indexed by bit 14
        src = kp.fetchq_source(
            m1_fp.FQ_GRID[-m1_fp.FQ_PACK :],
            chain=max(m1_fp.FQ_CHAINS),
            stride=m1_fp.FQ_PACK * L["slot"],
            isa="aarch64",
            counter="mach",
            **L,
        )
        self.assertEqual(src.count('"mrs x9, fpsr\\n"'), m1_fp.FQ_PACK * L["copies"])
        self.assertEqual(src.count('"b fq_land0_0_1\\n"'), 1)

    def test_iters_for(self):
        args = types.SimpleNamespace(min_ms=2.0)
        self.assertEqual(m1_fp.iters_for(args, 12), -(-2 * MS // 12))
        self.assertEqual(m1_fp.iters_for(None, 1e9), 1)
        self.assertEqual(
            [g for g, _ in m1_fp.octaves(range(1, 9), lambda d: 2 * d + 8)], [[1, 2, 3], [4, 5, 6, 7, 8]]
        )


class TestConvert(unittest.TestCase):
    def test_ticks_to_cycles(self):
        out = "counter_hz 24576000.0\npt corr:8 rep 0 cyc 153.6000\npt corr:8 rep 1 cyc 76.8\n"
        self.assertEqual(m1_fp.to_cycles(kp.parse_points(out), 7.68), {"corr:8": [20.0, 10.0]})

    def test_measure_rejects_empty_output(self):
        r = types.SimpleNamespace(compile=lambda s, n: n, launch=lambda b, label="": ("counter_hz 1\n", 1.0))
        with self.assertRaises(RuntimeError):
            m1_fp.measure(r, "", "x", "P3.4")


class TestModelledM1(unittest.TestCase):
    """Each PoC on canned output from a model with a known answer; every timed point >= 1 ms."""

    def check(self, r):
        self.assertEqual(r.short, [], "timed points under 1 ms")

    def test_history(self):
        r, raw, rows = run("P3.4")
        row = rows["frontend.direction_predictor.history"]
        self.assertEqual(row["value"], 100)
        self.assertEqual((row["published"], row["unit"]), (100, "taken branches"))
        self.assertIn("fine", {x["phase"] for x in raw})
        self.check(r)

    def test_ras(self):
        r, raw, rows = run("P3.6")
        row = rows["frontend.ras_depth"]
        self.assertEqual(row["value"], 48)
        self.assertLessEqual(row["ci"][0], 48)
        self.assertEqual(len(raw), len(m1_fp.RAS_SEEDS))
        self.assertEqual(sorted(map(int, raw[0]["T"])), list(range(1, m1_fp.RAS_DMAX + 1)))
        self.check(r)

    def test_indirect(self):
        r, raw, rows = run("P3.5")
        for t in m1_fp.IND_TARGETS:
            row = rows[f"frontend.indirect_predictor.capacity (T={t})"]
            self.assertLessEqual(abs(row["value"] - 1000), 16 * t, row)
            self.assertIn("not corrected", row["note"])
        self.assertEqual(
            {len(re.findall(r"blr x9", s)) for s in r.srcs}, {max(m1_fp.IND_SIZES)}
        )  # one layout
        self.check(r)

    def test_fdip_absent_present_and_predicted_barrier(self):
        r, _, rows = run("P3.8")
        row = rows["decoupling.fdip_signature"]
        self.assertAlmostEqual(row["value"], 1.0, delta=0.05)
        self.assertIn("fdip.absent", row["flags"])
        self.assertEqual(row["published"], "absent")
        self.assertAlmostEqual(rows["decoupling.l1i_miss_cycles"]["value"], 25, delta=0.5)
        self.assertTrue(all(".org fd0_lines + 128\\n" in s for s in r.srcs))
        self.check(r)
        _, _, rows = run("P3.8", FakeRunner(overlap=4.0))
        self.assertAlmostEqual(rows["decoupling.fdip_signature"]["value"], 4.0, delta=0.1)
        self.assertIn("fdip.present", rows["decoupling.fdip_signature"]["flags"])
        _, _, rows = run("P3.8", FakeRunner(barrier=0.0))
        self.assertIsNone(rows["decoupling.fdip_signature"]["value"])
        self.assertIn("barrier.predicted", rows["decoupling.fdip_signature"]["flags"])

    def test_fetchq(self):
        r, raw, rows = run("P3.9")
        row = rows["frontend.fetch_to_rename_buffer"]
        self.assertEqual(row["value"], 200)
        self.assertIn("'mrs x9, fpsr'", row["note"])
        self.check(r)
        r, raw, _ = run("P3.9", FakeRunner(), types.SimpleNamespace(serializer="isb"))
        self.assertEqual({x["serializer"] for x in raw}, {"isb"})
        self.assertTrue(all('"isb\\n"' in s and "mrs x9, fpsr" not in s for s in r.srcs))

    def test_smoke_sizes(self):
        """--gem5-smoke runs a few points per PoC through the same code."""
        n = 0
        for poc in m1_fp.POCS:
            r, raw, rows = run(poc, FakeRunner(smoke=True))
            self.assertTrue(rows and raw, poc)
            n += len(r.labels)
        self.assertLessEqual(n, 40)


class TestCli(unittest.TestCase):
    def test_parse(self):
        a = m1_fp.parser().parse_args(
            ["P3.6", "P3.9", "--serializer", "isb", "--min-ms", "2", "--gem5-smoke"]
        )
        self.assertEqual((a.pocs, a.serializer, a.min_ms, a.gem5_smoke), (["P3.6", "P3.9"], "isb", 2.0, True))
        self.assertEqual(m1_fp.parser().parse_args([]).pocs, [])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m1_fp.main(["P9.9"])

    def test_contract(self):
        self.assertEqual(m1_fp.FAMILY, "fp")
        self.assertEqual(set(m1_fp.POCS), set(m1_fp.EST_MIN))


class TestDefaultsUnchanged(unittest.TestCase):
    def test_aarch64_gem5_sources(self):
        """The gem5-validated AArch64 layouts do not move with the M1 parameters."""
        src = "".join(
            [
                kp.fetchq_source([16, 32, 48, 64], chain=12, isa="aarch64"),
                kp.fdip_source([0], seed=2, isa="aarch64"),
                kp.fdip_source([3, 0], n_lines=256, isa="aarch64", counter="mach"),
            ]
        )
        self.assertEqual(hashlib.md5(src.encode()).hexdigest(), "7f8edd73ac99fad4afd911e0525bf182")


def m1_sources(counter):
    """One source per PoC as the M1 run builds them (smaller grids)."""
    L = m1_fp.FQ_LAYOUT
    return {
        "hist": kp.history_source([0, 99, 256], iters=10, isa="aarch64", counter=counter),
        "ind": kp.indirect_source(
            max(m1_fp.IND_SIZES), [8, 16384], targets=2, isa="aarch64", counter=counter
        ),
        "ras": kp.ras_source(m1_fp.RAS_DMAX, [1, 256], seed=2, isa="aarch64", counter=counter),
        "fd": kp.fdip_source([1, 0], n_lines=1024, isa="aarch64", counter=counter, line_bytes=m1_fp.FD_LINE),
        "fq": kp.fetchq_source(
            [0, 512],
            chain=20,
            stride=2 * L["slot"],
            isa="aarch64",
            counter=counter,
            serializer="isb; dsb sy",
            **L,
        ),
    }


@unittest.skipUnless(
    shutil.which("aarch64-linux-gnu-gcc"), "needs the AArch64 cross compiler (. tools/env.sh)"
)
class TestCrossCompile(unittest.TestCase):
    def test_linux_static(self):
        """The cntvct (Linux / Asahi) build of every M1 source links; the fetch-queue landing lines share
        one 16 KiB-period L1i set that no block uses."""
        from uarchre.backends.platform.gem5 import compile_c

        bins = {
            name: compile_c(src, f"t_m1fp_{name}", isa="aarch64")
            for name, src in m1_sources("cntvct").items()
        }
        nm = subprocess.run(
            ["aarch64-linux-gnu-nm", bins["fq"]], capture_output=True, text=True, check=True
        ).stdout
        syms = {ln.split()[2]: int(ln.split()[0], 16) for ln in nm.splitlines() if len(ln.split()) == 3}
        L = m1_fp.FQ_LAYOUT
        for v, nops in enumerate([0, 512]):
            land = {s: a for s, a in syms.items() if s.startswith(f"fq_land{v}_")}
            self.assertEqual(len(land), L["copies"] * L["hops"])
            sets = {(a % L["slot"]) // L["line_bytes"] for a in land.values()}
            self.assertEqual(len(sets), L["hops"])
            for k in range(L["copies"]):
                a = syms[f"fq_blk{v}_{k}"] % L["slot"]
                end = a + 4 * (nops + 20 + 2 + 1) - 1  # chain 20, two serializing instructions, the jump
                used = set(range(a // L["line_bytes"], end // L["line_bytes"] + 1))
                self.assertFalse(used & sets)


@unittest.skipUnless(
    shutil.which("aarch64-linux-gnu-cpp") and (shutil.which("llvm-mc") or shutil.which("llvm-mc-19")),
    "needs the AArch64 cross preprocessor and llvm-mc",
)
class TestMachO(unittest.TestCase):
    def test_assembles(self):
        """Every M1 kernel as clang on macOS sees it (__APPLE__) assembles for arm64-apple-macos."""
        mc = shutil.which("llvm-mc") or shutil.which("llvm-mc-19")
        with tempfile.TemporaryDirectory() as d:
            for name, src in m1_sources("cntvct").items():
                pp = subprocess.run(
                    ["aarch64-linux-gnu-cpp", "-D__APPLE__", "-P"],
                    input=src,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                block = re.search(r"^__asm__\(\n(.*?)\n\);", pp, re.S | re.M).group(1)
                asm = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', block))
                s = os.path.join(d, f"{name}.s")
                with open(s, "w") as fh:
                    fh.write(asm)
                r = subprocess.run(
                    [mc, "-triple=arm64-apple-macos11", "-filetype=obj", s, "-o", s + ".o"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[:500]}")


if __name__ == "__main__":
    unittest.main()
