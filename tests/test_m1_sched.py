"""targets/m1/m1_sched.py: sweep design against firestorm.yaml, conversion, analysis, CLI, builds.

python3 -m unittest tests.test_m1_sched
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "targets", "m1"))
import m1_sched as ms  # noqa: E402
import native  # noqa: E402
from uarchre.backends.isa.aarch64 import SRC_A  # noqa: E402
from uarchre.kernels.scheduler import (  # noqa: E402
    A64AsmEmitter,
    dispatch_probe,
    multi_source,
    nop_stream,
    retire_drain,
    sched_probe,
)

A64 = "aarch64"
CLOCK = 3.2e9  # Firestorm, cycles per second
FSQRT_CYCLES = 13  # Firestorm fsqrt d latency


def pub(path):
    return native.published(path)["value"]


def class_capacity(cls):
    """([lo, hi] published entries the class can use, dispatch buffer) or (None, buffer)."""
    p, _, _ = ms.queue_published(cls)
    db = ms.QUEUES[cls][2]
    p = p if p is None or isinstance(p, list) else [p, p]
    return p, pub(f"decoupling.dispatch_buffers.{db}") if db else 0


def ports(word):
    return sum(word in p["ops"].split() for p in native.published("backend.ports"))


class TestSweepDesign(unittest.TestCase):
    def test_size_grids_bracket_published(self):
        for cls, (lo, hi, _) in ms.SIZES.items():
            p, db = class_capacity(cls)
            if p is None:
                continue
            self.assertLessEqual(lo, 0.7 * p[0], cls)
            self.assertGreaterEqual(hi, 1.3 * (p[1] + db), cls)

    def test_add_sweeps_widest(self):
        lo, hi, _ = ms.SIZES["add"]
        self.assertLessEqual(lo, min(g[0] for g in ms.SIZES.values()))
        self.assertGreaterEqual(hi, max(g[1] for g in ms.SIZES.values()))
        self.assertLessEqual(lo, 0.7 * class_capacity("add")[1])

    def test_published_split(self):
        self.assertEqual(ms.queue_published("mul")[0], [26, 56])  # 2 x 13 paired, or 28 + 28 per port
        self.assertEqual(ms.queue_published("fp")[0], [72, 144])
        self.assertEqual(ms.queue_published("load")[0], 48)
        self.assertEqual(ms.queue_published("add")[0], 134)  # 24 + 26 + 16 + 12 + 28 + 28
        self.assertIn("backend.ports (dougallj-firestorm, jiegec-firestorm)", ms.queue_published("add")[2])
        self.assertIn("dispatch buffer 12", ms.queue_published("mul")[1])

    def test_preloads_stay_below_their_own_queue(self):
        for x, y, mvals in ms.PAIRS:
            self.assertEqual(mvals[0], 0)
            p, _ = class_capacity(x)
            cap = p[0] if p is not None else ms.SIZES[x][1] / 2
            self.assertLess(max(mvals), 0.85 * cap, (x, y))
            self.assertIn(y, ms.SIZES)

    def test_dispatch_blocks_saturate(self):
        rename = pub("backend.rename_width")
        rate = {"add": ports("alu"), "mul": ports("mul")}
        self.assertEqual(rate, {"add": 6, "mul": 2})
        for x, blocks in ms.DISPATCH.items():
            q = 0.75 * ms.SIZES[x][1]  # largest buffer the block must still fill
            self.assertGreaterEqual(min(blocks) * (1 - rate[x] / rename), q, x)
            self.assertGreaterEqual(max(ms.DISPATCH_M), 1.3 * q * rename / rate[x], x)
            db = pub(f"decoupling.dispatch_buffers.{ms.DISPATCH_DB[x]}")
            self.assertLess(db * rename / rate[x], max(ms.DISPATCH_M) / 2)

    def test_drain_fits_the_window(self):
        rows = native.published("backend.retire.retire_groups")["range"][0]
        prf = native.published("backend.int_prf")["range"][1]
        rename = pub("backend.rename_width")
        links, exhaust, kmax = ms.DRAIN["links"], ms.DRAIN["exhaust"], max(ms.DRAIN_K)
        self.assertGreater(exhaust, 1.3 * prf)  # rename stalls on int registers at every K
        self.assertLessEqual(links + kmax / pub("backend.retire.group_max_uops") + prf / 4, 0.85 * rows)
        self.assertLessEqual((links + kmax + prf) / rename, 0.7 * links * FSQRT_CYCLES)
        self.assertGreater(kmax / (pub("backend.retire.retire_rate") * 7), 10)  # cycles of slope signal

    def test_timed_regions_last_a_millisecond(self):
        dram = pub("memory.dram_latency")
        t = {
            "sched": dram,
            "dispatch": min(min(b) / r for b, r in ((ms.DISPATCH["add"], 6), (ms.DISPATCH["mul"], 2))),
            "drain": ms.DRAIN["links"] * FSQRT_CYCLES,
            "stream": min(ms.STREAM_K) / pub("frontend.fetch_width"),
        }
        for k, cyc in t.items():
            self.assertGreaterEqual(ms.ITERS[k] * cyc / CLOCK, 1e-3, k)

    def test_one_launch_never_wraps_a_chain(self):
        nodes = (ms.DEFAULTS["chain_mb"] << 20) // ms.LINE
        per = int(0.9 * nodes / (ms.ITERS["sched"] * ms.REPS))
        self.assertGreaterEqual(per, len(range(*ms.SIZES["add"][:2], ms.SIZES["add"][2])) + 1)


def _regs(line):
    return {("x" if c in "xw" else "v", int(n)) for c, n in re.findall(r"\b([xwdsvq])(\d+)\b", line)}


def m1_kernels():
    return [
        sched_probe("fadd", 9, isa=A64),
        sched_probe("add", 3, pre="fadd", m=2, isa=A64),
        sched_probe("load", 5, pre="store", m=3, isa=A64),
        sched_probe("mul", 4, pre="add", m=8, isa=A64),
        sched_probe("fp", 3, isa=A64),
        dispatch_probe(16, 5, x="add", y="nop", isa=A64),
        dispatch_probe(8, 3, x="mul", y="nop", isa=A64),
        retire_drain(64, 24, 528, A64),
        nop_stream(64, A64),
    ]


class TestKernels(unittest.TestCase):
    def test_fadd_filler(self):
        asm = sched_probe("fadd", 2, isa=A64).params["asm"]
        self.assertEqual(asm[:5], ["ldr %0, [%0]", "add x19, x6, %0, lsr #63", "fmov d24, x19"] + [
            f"fadd d{i}, d24, d30" for i in (16, 17)
        ])  # fmt: skip
        self.assertEqual(asm[5:8], ["ldr %1, [%1]", "add x20, x6, %1, lsr #63", "fmov d25, x20"])
        exp = (struct.unpack("<Q", struct.pack("<Q", SRC_A))[0] >> 52) & 0x7FF
        self.assertTrue(0 < exp < 0x7FF)  # the fadds read a normal double

    def test_fp_filler_unchanged(self):
        self.assertEqual(sched_probe("fp", 1, isa=A64).params["asm"], ["ldr %0, [%0]", "fmov d16, %0"] + [
            "ldr %1, [%1]", "fmov d16, %1"
        ])  # fmt: skip

    def test_nop_trailer(self):
        asm = dispatch_probe(2, 3, x="mul", y="nop", isa=A64).params["asm"]
        self.assertEqual(asm, ["mul x9, x6, x7", "mul x10, x6, x7", "nop", "nop", "nop"])

    def test_m1_register_rules(self):
        em = A64AsmEmitter("cntvct")
        head = em.loop_head()
        written = set()
        for ln in head:
            if ln.split()[0] in ("mov", "movz", "movk", "scvtf", "fmov"):
                written |= _regs(ln.split(",")[0])
        clob = set().union(*(_regs(c) for c in em.clobbers))
        for k in m1_kernels():
            body = em.lower(k)[len(head) :]
            used = set().union(*(_regs(ln) for ln in body)) - {("x", 31)}
            self.assertLessEqual(used, written, k.name)
            self.assertLessEqual(used, clob, k.name)
            text = "\n".join(em.lower(k))
            self.assertNotRegex(text, r"\badrp?\b")
            for r in ("x18", "w18", "x29", "x30", "sp"):
                self.assertNotRegex(text, rf"\b{r}\b")


# ------------------------------------------------------------------ fake hardware


CAPS = {"add": 100, "mul": 30, "fp": 70, "fadd": 70, "load": 50, "store": 40}
RATIO = 7.68  # 24 MHz x 1024 ticks per 3.2 GHz cycle


def model(p):
    """Cycles per iteration of a kernel on a toy core with the CAPS queues (mul shares add's)."""
    if "sat" in p:
        rate, q = (6, 100) if p["x"] == "add" else (2, 40)
        return max(p["sat"] / rate, (p["sat"] - q) / rate + p["m"] / 8) + 3
    if "links" in p:
        return p["links"] * 13 + p["k"] / 56
    if "filler" not in p:
        return p["k"] / 8 + 1
    cap = CAPS[p["filler"]] - (p["m"] if p.get("pre") == "mul" and p["filler"] == "add" else 0)
    return 400.0 if p["n"] <= cap else 800.0


class FakeRunner:
    smoke, counter = False, "cntvct"

    def __init__(self):
        self.attempts = []

    def compile(self, src, name):
        return src

    def launch(self, binpath, argv, label=""):
        reps = int(argv[2])
        out = ["counter_hz 24576000000.0"]
        for i, p in enumerate(json.loads(binpath)):
            for r in range(reps):
                c = model(p) * (1.2 if r == 0 else 1.0) * (1 + 1e-4 * (i % 3))
                out.append(f"point {i} rep {r} cycles_per_iter {c * RATIO:.3f}")
        self.attempts.append({"label": label, "accepted": True, "ghz": 3.2})
        return "\n".join(out + ["cycles_per_iter done"]), RATIO


def fake_source(name, kernels, em=None):
    return json.dumps([k.params | {"asm": None} for k in kernels])


def args(*argv):
    return ms.build_parser().parse_args(["--outer-reps", "2", *argv])


@mock.patch.object(ms, "multi_source", fake_source)
@mock.patch.object(ms, "log", lambda *a: None)
class TestAnalysis(unittest.TestCase):
    def test_conversion_drops_rep0_and_divides_by_the_ratio(self):
        r = FakeRunner()
        cfg = ms.config(r, args())
        vals, rec = ms.launch(r, cfg, "t", [sched_probe("add", 10, isa=A64)], 100, 1 << 20, 0)
        self.assertAlmostEqual(vals[0], 400.0, places=2)
        self.assertEqual(len(rec["ticks"][0]), ms.REPS)
        self.assertTrue(rec["accepted"])

    def test_sizes(self):
        raw, rows = ms.run_sched(FakeRunner(), args("--parts", "sizes", "--fp-filler", "both"))
        got = {row["structure"]: row for row in rows}
        for cls, cap in CAPS.items():
            row = got[f"scheduler.{cls}"]
            self.assertEqual(row["value"], cap, cls)
            self.assertLessEqual(row["ci"][0], cap)
            self.assertGreaterEqual(row["ci"][1], cap)
            self.assertIn("offset 0 applied", row["note"])
        self.assertEqual(got["scheduler.mul"]["published"], [26, 56])
        self.assertEqual(got["scheduler.add"]["published"], 134)
        self.assertEqual(raw["sizes"]["add"]["offset"], 0)
        json.dumps(raw, default=float)

    def test_pairing(self):
        pairs = [("mul", "add", [0, 8, 16]), ("fp", "add", [0, 16, 32])]
        with mock.patch.object(ms, "PAIRS", pairs):
            _, rows = ms.run_sched(FakeRunner(), args("--parts", "pairing"))
        shared, separate = rows
        self.assertEqual(shared["value"], 0)
        self.assertIn("shared from the first op", shared["note"])
        self.assertIn("separate queues", separate["note"])
        self.assertIn("pairing.no_break", separate["flags"])

    def test_dispatch(self):
        _, rows = ms.run_sched(FakeRunner(), args("--parts", "dispatch"))
        got = {row["structure"]: row for row in rows}
        self.assertAlmostEqual(got["dispatch.add"]["value"], 100, delta=3)
        self.assertAlmostEqual(got["dispatch.mul"]["value"], 40, delta=3)
        self.assertEqual(got["dispatch.add"]["published"], 24)
        self.assertIn("t0 = 0 applied", got["dispatch.add"]["note"])

    def test_retire(self):
        _, rows = ms.run_retire(FakeRunner(), args())
        got = {row["structure"]: row for row in rows}
        self.assertAlmostEqual(got["retire.drain"]["value"], 56, delta=0.5)
        self.assertAlmostEqual(got["retire.rate"]["value"], 8, delta=0.1)
        self.assertEqual(got["retire.rate"]["published"], 8)
        self.assertEqual(got["retire.drain"]["published"], 56)
        self.assertAlmostEqual(got["nop_stream"]["value"], 8, delta=0.1)
        self.assertEqual(got["nop_stream"]["flags"], [])


class TestCLI(unittest.TestCase):
    def test_parse(self):
        a = ms.build_parser().parse_args(["P1.7", "--gem5-smoke", "--fp-filler", "fadd", "--parts", "sizes"])
        self.assertEqual((a.pocs, a.fp_filler, a.parts, a.gem5_smoke), (["P1.7"], "fadd", "sizes", True))
        self.assertEqual(set(ms.POCS), set(ms.EST_MIN))
        self.assertEqual(ms.FAMILY, "sched")

    def test_unknown_poc(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            ms.main(["P9.9", "--gem5-smoke"])

    def test_common_options_only(self):
        """m1_kernels.py hands over a namespace without this module's own options."""
        a = native.add_args(argparse.ArgumentParser()).parse_args([])
        cfg = ms.config(FakeRunner(), a)
        self.assertEqual((cfg["fp"], cfg["outer_reps"], cfg["chain"]), (["fp"], 3, 256 << 20))


def _sources():
    ks = [
        sched_probe("add", ms.SIZES["add"][1], isa=A64),
        sched_probe("fadd", 40, isa=A64),
        sched_probe("store", 20, pre="load", m=40, isa=A64),
        dispatch_probe(max(ms.DISPATCH["add"]), max(ms.DISPATCH_M), x="add", y="nop", isa=A64),
        retire_drain(max(ms.DRAIN_K), ms.DRAIN["links"], ms.DRAIN["exhaust"], A64),
        nop_stream(max(ms.STREAM_K), A64),
    ]
    return ks


class TestBuilds(unittest.TestCase):
    @unittest.skipUnless(shutil.which("aarch64-linux-gnu-gcc"), "needs the AArch64 cross compiler")
    def test_cross_compiles(self):
        src = multi_source("t", _sources(), em=A64AsmEmitter("cntvct"))
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "k.c")
            with open(c, "w") as fh:
                fh.write(src)
            r = subprocess.run(
                ["aarch64-linux-gnu-gcc", "-O1", "-static", c, "-o", os.path.join(d, "k")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr[:2000])

    @unittest.skipUnless(shutil.which("llvm-mc") or shutil.which("llvm-mc-19"), "needs llvm-mc")
    def test_macho_assembles(self):
        """The timed loops (operands as registers) assemble for arm64-apple-macos."""
        mc = shutil.which("llvm-mc") or shutil.which("llvm-mc-19")
        em = A64AsmEmitter("mach")
        with tempfile.TemporaryDirectory() as d:
            for k in _sources():
                text = "\n".join(em.lower(k))
                text = re.sub(r"%([0-3])", r"x\1", text)
                s = os.path.join(d, "k.s")
                with open(s, "w") as fh:
                    fh.write(".text\n_kern:\n" + text.replace("\\n\\t", "\n") + "\nret\n")
                r = subprocess.run(
                    [mc, "-triple=arm64-apple-macos11", "-filetype=obj", s, "-o", s + ".o"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, f"{k.name}: {r.stderr[:500]}")


if __name__ == "__main__":
    unittest.main()
