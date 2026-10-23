"""targets/m1/m1_mem.py: sweep ranges against firestorm.yaml, output conversion, and every PoC end to end
on canned output from a synthetic M1 (no gem5, no Mac); the AArch64 sources cross-compile and their
inline asm assembles for arm64-apple-macos.

    python3 -m unittest tests.test_m1_mem
"""

import contextlib
import io
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "targets", "m1"))

import m1_mem as T  # noqa: E402
import native  # noqa: E402
from uarchre.kernels import memory as M  # noqa: E402

KB, MB = 1 << 10, 1 << 20
RATIO = 7.5  # counter ticks per core cycle in the canned output
TRUTH = dict(l1d=128 * KB, l2=8 * MB, l1_line=64, l2_line=128, ways=8, dtlb=160, stlb=3072, itlb=192)
TRUTH.update(mlp=20, pairs=200, fwd=5.0, dram=320.0)
TRUTH.update(tlb_ways=4, tlb_victim=32, fwd2=6.0, bw=60.0, bw2=63.0, slc="lru")
HZ = 24576000  # counter_hz in the canned output


def ramp(x, c, frac):
    """LRU miss fraction: 0 up to capacity c, 1 from c(1 + frac)."""
    return min(1.0, max(0.0, (x / c - 1) / frac))


def lat(f):
    m1, m2 = ramp(f, TRUTH["l1d"], 1 / 8), ramp(f, TRUTH["l2"], 1 / 12)
    tlb = 6 * ramp(f, 160 * 16 * KB, 0.1)
    return 3 + m1 * (15 + tlb) + m2 * (TRUTH["dram"] - 18 - tlb)


def slc_lat(pos, kind):
    """Reverse-probe cycles per load pos MiB in: L2 hits to 12 MiB, SLC hits after; DRAM past an 18 MiB
    edge (sd 1.5 MiB) for LRU, a DRAM fraction 1 - exp(-(pos - 12) / 4) for random replacement."""
    if kind == "lru":
        d = 0.5 * math.erfc(-(pos - 18) / (1.5 * math.sqrt(2)))
    else:
        d = 1 - math.exp(-max(0.0, pos - 12) / 4)
    return (18 if pos < 12 else 90) + d * (300 - 90)


def model(argv, t=None):
    """Canned stdout of the memory probe for argv (values in ticks); t overrides TRUTH."""
    mode, a, t = argv[0], [str(x) for x in argv[1:]], {**TRUTH, **(t or {})}
    out = [f"counter_hz {HZ}", "clock_ratio 7.5000", "page_bytes 16384"]

    def emit(tag, key, cyc, reps):
        n = int(reps)
        out.extend(f"R {tag} {key} {r} {cyc * RATIO * (1 + 0.002 * (r - n // 2)):.4f}" for r in range(n))

    if mode == "lat":
        for f in map(int, a[6:]):
            if a[0] == "seq":
                s = int(a[1])
                cyc = {64: 4.0, 128: 8.0, 256: 20.0}.get(s, TRUTH["dram"])
            else:
                cyc = lat(f) if int(a[1]) == 64 else TRUTH["dram"]
            emit("lat", f, cyc, a[3])
    elif mode == "pair":
        f = int(a[3])
        for d in map(int, a[4:]):
            if f <= 8 * MB:  # X an L2 hit; X + D an L1 hit inside X's L1 line
                emit("pair", d, 18 + (3 if d < TRUTH["l1_line"] else 18), a[2])
            else:  # X from DRAM; X + D in its L1 line, its L2 line, or neither
                second = 3 if d < TRUTH["l1_line"] else 18 if d < TRUTH["l2_line"] else TRUTH["dram"]
                emit("pair", d, TRUTH["dram"] + second, a[2])
    elif mode == "conflict":
        for k in map(int, a[4:]):
            emit("conflict", k, 3 if k <= TRUTH["ways"] else 18, a[2])
    elif mode == "page":
        for p in map(int, a[4:]):
            base = 3 if p <= 1024 else 18
            tlb = 0 if a[3] == "1" else 6 * ramp(p, TRUTH["dtlb"], 0.05) + 20 * ramp(p, TRUTH["stlb"], 0.05)
            emit("dense" if a[3] == "1" else "page", p, base + tlb, a[1])
    elif mode == "code":
        dense = "d=1" in a
        for p in (int(x) for x in a[2:] if not x.startswith("d=")):
            emit("code", p, 2 + 5 * (p > 1536) + (0 if dense else 10 * ramp(p, TRUTH["itlb"], 0.05)), a[1])
    elif mode == "mlp":
        for k in (int(x) for x in a[4:] if "=" not in x):
            emit("mlp", k, TRUTH["dram"] * max(1, k / TRUTH["mlp"]), a[2])
    elif mode == "indep":
        emit("indep", int(a[4]), 1.5, a[2])
    elif mode == "wchain":
        emit("wchainA", int(a[0]), TRUTH["dram"] * (1.01 if int(a[0]) >= 256 * MB else 0.8), a[2])
        emit("wchainAB", int(a[0]), TRUTH["dram"] * 1.05, a[2])
    elif mode == "memdep":
        for n in (int(x) for x in a[2:] if x.isdigit()):
            emit("memdep", n, 2 + 0.5 * max(0, n - TRUTH["pairs"]) if argv[-1] != "noalias" else 2, a[1])
    elif mode == "fwd":
        emit("fwd", 0, TRUTH["fwd"], a[1])
    elif mode == "fwd2":
        emit("fwd2", 0, t["fwd2"], a[1])
    elif mode == "tlbway":  # 32 sets on the low VPN bits; every K-th page uses 32 / gcd(K, 32) of them
        k, dense = int(a[2]), a[3] == "1"
        cap = t["tlb_ways"] * 32 // math.gcd(k, 32) + t["tlb_victim"]
        for n in (int(x) for x in a[4:] if x.isdigit()):
            emit(
                "tlbdense" if dense else "tlbway", n, 3 + (0 if dense else 3 * min(2, max(0, n - cap))), a[1]
            )
    elif mode == "slc":
        node, per = int(a[0]), int(a[1])
        for f in map(int, a[4:]):
            for b in range(f // node // 16 * 16 // per):
                emit(f"rev{f}", b, slc_lat((b + 0.5) * per * node / MB, t["slc"]), a[2])
                emit(f"cold{f}", b, 300.0, a[2])
    elif mode == "bw":
        nbytes = int(a[0]) // 512 * 512
        for s in map(int, a[2:]):
            emit("bw", s, nbytes / (t["bw" if s == 1 else "bw2"] * 1e9) * HZ / RATIO, a[1])
    return "\n".join(out) + "\n"


class FakeRunner:
    """native.Runner's interface; launches return canned output, the binary name picks the control."""

    counter = "cntvct"

    def __init__(self, **truth):
        self.attempts, self.argvs, self.truth = [], [], truth

    def compile(self, src, name):
        return name

    def launch(self, binpath, argv=(), timeout=1800, label=""):
        argv = list(argv) + (["noalias"] if binpath == "memdep_n" and argv[0] == "memdep" else [])
        self.argvs.append(argv)
        return model(argv, self.truth), RATIO


def args(**kw):
    return SimpleNamespace(gem5_smoke=False, launches=2, **kw)


def by(rows):
    return {r["structure"]: r for r in rows}


class TestPlan(unittest.TestCase):
    """Every M1 sweep brackets the published value with 30 % margin and keeps a timed region >= ~1 ms."""

    def setUp(self):
        self.p = T.plan()

    def brackets(self, grid, value):
        self.assertLessEqual(min(grid), 0.7 * value)
        self.assertGreaterEqual(max(grid), 1.3 * value)

    def test_caches(self):
        p = self.p["P2.1"]
        fs = sorted(f for g, _ in p["coarse"] for f in g)
        self.brackets(fs, native.published("memory.l1d.size_kib") * KB)
        self.brackets(fs, native.published("memory.l2.size_mib") * MB)
        self.assertGreaterEqual(max(fs), 16 * (12 + 8) * MB)  # well past L2 + SLC: the DRAM level
        for k, (lo, hi) in p["lat"].items():
            self.assertTrue([f for f in fs if lo <= f <= hi], k)
        lines = [native.published(f"memory.{c}.line_bytes") for c in ("l1d", "l2")]
        self.assertLessEqual(min(p["pair_d"]), min(lines) / 2)
        self.assertGreaterEqual(max(p["pair_d"]), 2 * max(lines))
        self.assertGreater(p["slot"], max(p["pair_d"]) + 8)
        self.assertGreaterEqual(max(p["conflict_k"]), 1.3 * native.published("memory.l1d.ways"))
        way = native.published("memory.l1d.size_kib") * KB // native.published("memory.l1d.ways")
        self.assertTrue(all(s % way == 0 for s in p["conflict_s"]))
        small = p["coarse"][0][1]
        self.assertGreaterEqual(small * 3 / 3.2e9, 1.5e-3)  # L1 hits at 3 cycles, 3.2 GHz

    def test_tlbs(self):
        p = self.p["P2.2"]
        self.brackets(p["page"], native.published("memory.dtlb_l1.entries"))
        self.brackets(p["page"], native.published("memory.dtlb_l2.entries"))
        # the packed control stays inside the L1 dTLB at the largest P
        self.assertLessEqual(
            max(p["page"]) * 128, 0.85 * native.published("memory.dtlb_l1.entries") * 16 * KB
        )
        self.assertGreaterEqual(max(p["code"]), 2 * native.published("frontend.l1i.size_kib") * KB / 128)
        self.assertLessEqual(max(p["code"]) * 16 * KB, 120 * MB)  # code_main's branch reach
        self.assertGreaterEqual(p["nloads"] * 3 / 3.2e9, 1.5e-3)

    def test_mlp(self):
        p = self.p["P2.3"]
        self.assertLess((p["reps"] + 1) * p["iters"], p["F"] // 128 // max(p["K"]))  # chains never meet
        self.assertLessEqual(max(p["K"]), M.MLP_A64_KMAX)
        self.assertGreaterEqual(p["F"], 16 * (12 + 8) * MB)
        self.assertGreaterEqual(p["iters"] * 300 / 3.2e9, 1e-3)

    def test_prefetch_and_memdep(self):
        p = self.p["P2.4"]
        self.assertIn(256 * MB, p["wchain"])  # m1_sweep.py's default --chain-mb
        for s in p["seq"]:  # lines touched by a stride chain outgrow L2 + SLC
            self.assertGreaterEqual(p["F"] // s * 128, 1.3 * (12 + 8) * MB)
        p = self.p["P2.5"]
        self.assertLessEqual(max(p["N"] + p["control"]), T.MD_NMAX)
        self.assertLessEqual(8192 + 16 * T.MD_NMAX, 16384)
        self.assertGreaterEqual(p["blocks"] * 1.5 / 3.2e9, 1e-3)
        self.assertEqual(set(T.EST_MIN), set(T.POCS))

    def test_tlb_ways(self):
        p, ent = self.p["P2.2-ways"], native.published("memory.dtlb_l1.entries")
        self.assertEqual(sorted(p["K"]), [1, 16, 32])
        self.assertLessEqual(min(p["N"]), 0.7 * 5)  # 5 ways x 32 sets
        self.assertGreaterEqual(max(p["N"]), 1.3 * (2 * 4 + 32))  # 4 x 32 + 32 victim at K = 16
        self.assertEqual(p["N"], list(range(min(p["N"]), max(p["N"]) + 1)))  # resolution 1
        self.assertLessEqual(
            max(p["N"]), 16 * KB // 128
        )  # a line of its own per node: no L1D or L2 set shared
        self.assertLess(max(p["N"]), ent)  # K = 1 stays flat
        self.assertGreaterEqual(p["nloads"] * 3 / 3.2e9, 1.5e-3)

    def test_bw(self):
        p = self.p["P2.3-bw"]
        self.assertGreaterEqual(p["bytes"], 1 << 30)
        self.assertGreaterEqual(p["bytes"] / 100e9, 1e-3)  # a pass outlasts 1 ms even at 100 GB/s
        self.assertGreaterEqual(p["launches"], 5)
        self.assertEqual(p["streams"][:2], [1, 2])

    def test_slc(self):
        p = self.p["P2.1-slc"]
        l2, slc = (native.published(f"memory.{c}.size_mib") * MB for c in ("l2", "slc"))
        self.assertLessEqual(min(p["F"]), 0.7 * 7.5 * MB)  # below a core's effective L2
        self.assertGreaterEqual(max(p["F"]), 1.3 * (l2 + slc))
        self.assertLessEqual(max(p["F"]) // (16 * KB), native.published("memory.dtlb_l2.entries"))
        self.assertGreaterEqual(p["x"], 3 * (l2 + slc))  # the eviction read flushes L2 and SLC
        self.assertEqual((p["node"], p["per"] % 16, p["hit"], p["l2_mib"]), (128, 0, 90, 12))
        self.assertTrue(all(f % (16 * p["node"]) == 0 for f in p["F"]))  # whole chase blocks
        s = T.plan(smoke=True)["P2.1-slc"]
        self.assertGreater(max(s["F"]), 256 * KB)  # past gem5's L2
        self.assertGreaterEqual(s["x"], 4 * 256 * KB)

    def test_smoke_plan_is_small(self):
        s = T.plan(smoke=True)
        self.assertEqual(set(s), set(T.POCS))
        self.assertLessEqual(max(f for g, _ in s["P2.1"]["coarse"] for f in g), 1 * MB)
        self.assertLess(
            (s["P2.3"]["reps"] + 1) * s["P2.3"]["iters"], s["P2.3"]["F"] // 128 // max(s["P2.3"]["K"])
        )


class TestConvert(unittest.TestCase):
    def test_ticks_to_cycles(self):
        text = "counter_hz 24576000\nclock_ratio 7.6\npage_bytes 16384\n"
        text += "R lat 4096 0 22.5000\nR lat 4096 1 30.0000\n"
        lr = T.convert(text, 7.5)
        self.assertEqual(lr["runs"]["lat"]["4096"], [3.0, 4.0])
        self.assertEqual(lr["scalars"], {"counter_hz": 24576000.0, "clock_ratio": 7.6, "page_bytes": 16384})
        self.assertEqual(T.pooled([lr, lr], "lat"), ([4096], [3.5]))

    def test_merge_and_region(self):
        a = T.convert("R lat 1 0 1\nR lat 2 0 2\n", 1.0)
        b = T.convert("R lat 3 0 3\n", 1.0)
        m = T.merge([[a, a], [b, b]])
        self.assertEqual(len(m), 2)
        self.assertEqual(sorted(m[0]["runs"]["lat"]), ["1", "2", "3"])
        self.assertEqual(T.region(m, "lat", 2, 3), (2.5, [2.5, 2.5]))

    def test_steps(self):
        xs = list(range(10, 200, 10))
        ys = [0.1] * 5 + [6.0] * 7 + [26.0] * 7
        self.assertEqual(T.steps(xs, ys), [(50, 60, 0.1, 6.0), (120, 130, 6.0, 26.0)])
        self.assertEqual(T.steps(xs, [0.2 * (i % 2) for i in range(len(xs))]), [])

    def test_geom_and_thin(self):
        g = T.geom(16, 16384, 8)
        self.assertEqual((g[0], g[-1]), (16, 16384))
        self.assertEqual(len(g), 81)
        self.assertEqual(T.thin(list(range(100)), 10)[-1], 99)
        self.assertLessEqual(len(T.thin(list(range(100)), 10)), 11)

    def test_published(self):
        self.assertEqual(T.pub("memory.l1d.size_kib")[0], 128)
        self.assertEqual(T.pub("memory.dram_latency")[0], [300, 350])
        self.assertIn("7cpu-m1", T.pub("memory.dtlb_l2.entries")[1])
        self.assertIsNone(T.pub("memory.mlp")[0])
        self.assertEqual(T.pub(None), (None, ""))

    def test_cli(self):
        a = T.parser().parse_args(["P2.1", "P2.3", "--gem5-smoke", "--tag", "x", "--launches", "5"])
        self.assertEqual((a.pocs, a.gem5_smoke, a.tag, a.launches), (["P2.1", "P2.3"], True, "x", 5))
        self.assertEqual(T.parser().parse_args([]).pocs, [])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            T.main(["P9.9", "--gem5-smoke"])
        self.assertEqual(
            T.launches(SimpleNamespace(gem5_smoke=False)), 3
        )  # m1_kernels.py passes no --launches


class TestEndToEnd(unittest.TestCase):
    """Each PoC's run function on the synthetic M1: the readings land on its truth."""

    def run_poc(self, poc, **truth):
        self.r = FakeRunner(**truth)
        raw, rows = T.POCS[poc](self.r, args())
        for x in rows:
            self.assertEqual(
                set(x), {"structure", "value", "ci", "unit", "published", "source", "flags", "note"}
            )
        return raw, by(rows)

    def test_caches(self):
        raw, rows = self.run_poc("P2.1")
        self.assertAlmostEqual(
            rows["L1D capacity"]["value"], TRUTH["l1d"] / KB, delta=0.02 * TRUTH["l1d"] / KB
        )
        self.assertAlmostEqual(rows["L2 capacity"]["value"], TRUTH["l2"] / MB, delta=0.02 * TRUTH["l2"] / MB)
        self.assertEqual(rows["L1D capacity"]["published"], 128)
        self.assertAlmostEqual(rows["L1D load-to-use"]["value"], 3.0, delta=0.05)
        self.assertAlmostEqual(rows["L2 load-to-use"]["value"], 18.0, delta=0.1)
        self.assertAlmostEqual(rows["DRAM load-to-use"]["value"], TRUTH["dram"], delta=1)
        self.assertEqual(rows["L1D line"]["value"], TRUTH["l1_line"])
        self.assertEqual(rows["L2 line"]["value"], TRUTH["l2_line"])
        self.assertEqual(rows["L1D associativity"]["value"], TRUTH["ways"])
        self.assertEqual(rows["L1D associativity"]["flags"], [])
        lo, hi = rows["L1D capacity"]["ci"]
        self.assertLessEqual(lo, rows["L1D capacity"]["value"])
        self.assertGreaterEqual(hi, rows["L1D capacity"]["value"])

    def test_tlbs(self):
        raw, rows = self.run_poc("P2.2")
        for name, t in (("L1 dTLB", "dtlb"), ("L2 TLB", "stlb"), ("L1 iTLB", "itlb")):
            v = rows[f"{name} entries"]["value"]
            self.assertAlmostEqual(v, TRUTH[t], delta=max(2, 0.03 * TRUTH[t]), msg=name)
        self.assertAlmostEqual(rows["L1 dTLB miss penalty"]["value"], 6, delta=0.5)
        self.assertAlmostEqual(rows["L2 TLB miss penalty"]["value"], 20, delta=0.5)
        self.assertIn("16384 B pages", rows["L1 dTLB entries"]["note"])
        self.assertTrue(any(a[0] == "code" and "d=1" in a for a in self.r.argvs))

    def test_mlp(self):
        raw, rows = self.run_poc("P2.3")
        x = rows["misses in flight"]
        self.assertEqual(x["value"], TRUTH["mlp"])
        self.assertEqual(x["flags"], [])
        mlp = [a for a in self.r.argvs if a[0] == "mlp"]
        self.assertEqual(len(mlp), 2 * len(T.MLP_K))  # one K per launch, a seed per outer launch
        self.assertEqual({a[4] for a in mlp}, {0, 1})
        self.assertEqual(len(raw["launches"]), 2)

    def test_prefetch(self):
        raw, rows = self.run_poc("P2.4")
        self.assertAlmostEqual(rows["DRAM latency"]["value"], TRUTH["dram"], delta=1)
        self.assertEqual(rows["window chains, 256 MiB: latency / DRAM"]["flags"], [])
        self.assertEqual(rows["window chains, 64 MiB: latency / DRAM"]["flags"], ["not a DRAM miss"])
        self.assertEqual(rows["dependent 128 B-stride chain / random"]["flags"], ["prefetched"])
        self.assertEqual(rows["dependent 512 B-stride chain / random"]["flags"], [])
        self.assertEqual(rows["stride prefetcher: largest dependent stride caught"]["value"], 256)

    def test_memdep(self):
        raw, rows = self.run_poc("P2.5")
        x = rows["store-load pairs tracked"]
        self.assertAlmostEqual(x["value"], TRUTH["pairs"], delta=1)
        self.assertEqual(x["flags"], [])
        self.assertAlmostEqual(rows["store-to-load forwarding round trip"]["value"], TRUTH["fwd"], delta=0.05)
        x = rows["store-to-load forwarding, forced-sequential pair"]
        self.assertAlmostEqual(x["value"], TRUTH["fwd2"], delta=0.05)
        self.assertIn("str x1, [x1]; ldr x2, [x2]", x["note"])

    def test_tlb_ways_victim(self):
        raw, rows = self.run_poc("P2.2-ways")
        self.assertEqual(rows["L1 dTLB, 32-page stride: pages before the rise"]["value"], 36)
        self.assertEqual(rows["L1 dTLB, 16-page stride: pages before the rise"]["value"], 40)
        self.assertEqual(rows["L1 dTLB, consecutive pages: pages before the rise"]["flags"], ["no rise"])
        self.assertEqual(rows["L1 dTLB, consecutive pages: pages before the rise"]["published"], 160)
        self.assertEqual(
            (rows["L1 dTLB ways per set"]["value"], rows["L1 dTLB victim entries"]["value"]), (4, 32)
        )
        self.assertIn("+6.0 cycles", rows["L1 dTLB, 32-page stride: pages before the rise"]["note"])

    def test_tlb_ways_set_associative(self):
        _, rows = self.run_poc("P2.2-ways", tlb_ways=5, tlb_victim=0)
        self.assertEqual(rows["L1 dTLB, 32-page stride: pages before the rise"]["value"], 5)
        self.assertEqual(rows["L1 dTLB, 16-page stride: pages before the rise"]["value"], 10)
        self.assertEqual(
            (rows["L1 dTLB ways per set"]["value"], rows["L1 dTLB victim entries"]["value"]), (5, 0)
        )
        self.assertEqual(rows["L1 dTLB ways per set"]["flags"], [])

    def test_tlb_ways_fully_associative(self):
        _, rows = self.run_poc("P2.2-ways", tlb_ways=0, tlb_victim=160)
        self.assertIsNone(rows["L1 dTLB, 32-page stride: pages before the rise"]["value"])
        self.assertIsNone(rows["L1 dTLB ways per set"]["value"])
        self.assertEqual(rows["L1 dTLB ways per set"]["flags"], ["no set conflict"])

    def test_slc(self):
        raw, rows = self.run_poc("P2.1-slc")
        w = rows["SLC replacement, CPU-placed lines: edge width"]
        self.assertAlmostEqual(w["value"], (1.036 + 0.674) * 1.5 / 18, delta=0.03)
        self.assertEqual(w["flags"], ["step"])
        self.assertAlmostEqual(rows["SLC: reach of the reverse probe"]["value"], 18, delta=0.2)
        self.assertIn("22 MiB", w["note"])  # the first footprint whose probe reaches DRAM
        prof = raw["profiles"][str(max(raw["plan"]["F"]))]
        self.assertAlmostEqual(prof["hit_rate"][0], 1.0)
        self.assertAlmostEqual(prof["hit_rate"][-1], 0.0)
        self.assertAlmostEqual(prof["dram"], 300.0, delta=1)
        self.assertEqual(len(prof["pos_mib"]), len(prof["hit_rate"]))

    def test_slc_random(self):
        _, rows = self.run_poc("P2.1-slc", slc="random")
        w = rows["SLC replacement, CPU-placed lines: edge width"]
        k25, k50, k85 = (12 + 4 * math.log(1 / (1 - q)) for q in (0.25, 0.5, 0.85))
        self.assertAlmostEqual(w["value"], (k85 - k25) / k50, delta=0.03)
        self.assertEqual(w["flags"], ["gradual"])
        self.assertAlmostEqual(rows["SLC: reach of the reverse probe"]["value"], k50, delta=0.2)

    def test_slc_reach_within_l2(self):
        raw = {"plan": {**T.plan()["P2.1-slc"], "F": [8 * MB, 16 * MB]}}
        lines = [
            f"R rev{f} {b} 0 {RATIO * (18 if b < 32 else 300)}"
            for f in raw["plan"]["F"]
            for b in range(f // (256 * KB))
        ]
        lines += [f"R cold{f} {b} 0 {RATIO * 300}" for f in raw["plan"]["F"] for b in range(f // (256 * KB))]
        raw["launches"] = [T.convert("\n".join(lines) + "\n", RATIO)]
        rows = by(T.analyze_slc(raw))
        self.assertEqual(
            rows["SLC replacement, CPU-placed lines: edge width"]["flags"], ["step", "reach within L2"]
        )
        self.assertAlmostEqual(rows["SLC: reach of the reverse probe"]["value"], 8, delta=0.2)

    def test_bw(self):
        raw, rows = self.run_poc("P2.3-bw")
        one = rows["DRAM read bandwidth, one core"]
        self.assertAlmostEqual(one["value"], TRUTH["bw"], delta=0.3)
        self.assertEqual(one["flags"], [])
        self.assertIn("1 GiB per pass", one["note"])
        two = rows["DRAM read bandwidth, one core, 2 interleaved streams"]
        self.assertAlmostEqual(two["value"], TRUTH["bw2"], delta=0.3)
        self.assertIn("1.05 x one stream", two["note"])
        self.assertEqual(len(raw["launches"]), 5)  # several launches even when --launches is 2
        lr = {**raw["launches"][0], "scalars": {}, "ratio": 1.0}  # gem5: core cycles at GEM5_HZ
        lr["runs"] = {"bw": {"1": [T.GEM5_HZ * 1e-3]}}
        self.assertAlmostEqual(T.bw_rates(lr, 1, 10**6)[0], 1.0)

    def test_memdep_without_a_knee(self):
        rows = by(T.analyze_memdep(raw_memdep_flat()))
        self.assertIsNone(rows["store-load pairs tracked"]["value"])
        self.assertIn("no rise", rows["store-load pairs tracked"]["flags"])


def probe_levels(n, levels, uniform, seed=1):
    """Level each line of a reverse probe comes from (len(levels) = DRAM) after one fill over n lines in
    random order. levels: (sets, ways, "lru" | "random"), caches start full of other lines and each
    lower level holds the victims of the one above (exclusive); sets: line mod sets, or random."""
    rng = random.Random(seed)
    tags = [[[] for _ in range(s)] for s, _, _ in levels]
    smap = [{} for _ in levels]

    def ways(i, x):
        return tags[i][x % levels[i][0] if uniform else smap[i].setdefault(x, rng.randrange(levels[i][0]))]

    def insert(i, x):
        if i < len(levels):
            st = ways(i, x)
            if len(st) == levels[i][1]:
                insert(i + 1, st.pop(0 if levels[i][2] == "lru" else rng.randrange(len(st))))
            st.append(x)

    def access(x):
        i = next((i for i in range(len(levels)) if x in ways(i, x)), len(levels))
        if i < len(levels):
            ways(i, x).remove(x)
        if i == 0:
            ways(0, x).append(x)
        else:
            insert(0, x)
        return i

    for j in range(2 * sum(s * w for s, w, _ in levels)):
        access(-1 - j)
    order = list(range(n))
    rng.shuffle(order)
    for x in order:
        access(x)
    return [access(x) for x in reversed(order)]


class TestSLCStatistic(unittest.TestCase):
    """The edge width on simulated caches: LRU reads a step, random replacement a gradual edge."""

    def verdict(self, levels, lat, node, uniform):
        cap = sum(s * w for s, w, _ in levels) * node
        p = dict(
            F=T.geom(cap, 2.4 * cap, 4, 16 * node), node=node, per=64, reps=1, x=0, hit=lat[-2], l2_mib=0
        )
        out = []
        for f in p["F"]:
            lv = probe_levels(f // node // 16 * 16, levels, uniform)
            for b in range(len(lv) // p["per"]):
                y = sum(lat[x] for x in lv[b * p["per"] : (b + 1) * p["per"]]) / p["per"]
                out += [f"R rev{f} {b} 0 {y:.4f}", f"R cold{f} {b} 0 {lat[-1]:.4f}"]
        rows = by(T.analyze_slc({"plan": p, "launches": [T.convert("\n".join(out) + "\n", 1.0)]}))
        return rows["SLC replacement, CPU-placed lines: edge width"]["flags"][0]

    def test_l2_and_exclusive_slc(self):
        """12-way L2 + 16-way SLC holding its victims, sets on random physical pages, M1 latencies."""
        for slc, want in (("lru", "step"), ("random", "gradual")):
            got = self.verdict([(256, 12, "lru"), (128, 16, slc)], [18, 90, 300], 128, uniform=False)
            self.assertEqual(got, want, slc)

    def test_gem5_l2(self):
        """gem5's L2 alone: 8 ways, a physically contiguous buffer (set = line mod sets)."""
        for pol, want in (("lru", "step"), ("random", "gradual")):
            self.assertEqual(self.verdict([(256, 8, pol)], [17, 158], 64, uniform=True), want, pol)


def raw_memdep_flat():
    lr = T.convert("".join(f"R memdep {n} 0 {2 * RATIO}\n" for n in (4, 8, 16, 32, 64)), RATIO)
    fw = T.convert("R fwd 0 0 37.5\n", RATIO)
    return {"plan": T.plan()["P2.5"], "coarse": [lr], "control": [lr], "fwd": [fw]}


def apple_stubs(d):
    """Apple SDK prototypes the __APPLE__ branches use, so a Linux cross compiler can type-check them."""
    os.makedirs(os.path.join(d, "mach"))
    os.makedirs(os.path.join(d, "libkern"))
    with open(os.path.join(d, "mach", "mach_time.h"), "w") as fh:
        fh.write(
            "#include <stdint.h>\ntypedef int kern_return_t;\n"
            "struct mach_timebase_info { uint32_t numer; uint32_t denom; };\n"
            "typedef struct mach_timebase_info *mach_timebase_info_t;\n"
            "typedef struct mach_timebase_info mach_timebase_info_data_t;\n"
            "kern_return_t mach_timebase_info(mach_timebase_info_t info);\n"
            "uint64_t mach_absolute_time(void);\n"
        )
    with open(os.path.join(d, "libkern", "OSCacheControl.h"), "w") as fh:
        fh.write("#include <stddef.h>\nvoid sys_icache_invalidate(void *start, size_t len);\n")
    with open(os.path.join(d, "apple.h"), "w") as fh:
        fh.write(
            "typedef unsigned int qos_class_t;\n#define QOS_CLASS_USER_INTERACTIVE ((qos_class_t)0x21)\n"
            "int pthread_set_qos_class_self_np(qos_class_t qos_class, int relative_priority);\n"
            "qos_class_t qos_class_self(void);\nvoid pthread_jit_write_protect_np(int enabled);\n"
            "#define MAP_JIT 0x0800\n"
        )


CROSS = shutil.which("aarch64-linux-gnu-gcc")
MC = shutil.which("llvm-mc") or shutil.which("llvm-mc-19")


class TestMacSources(unittest.TestCase):
    def sources(self, counter):
        p = T.plan()["P2.5"]
        return {
            "chase": M.chase_source("aarch64", counter),
            "memdep_a": M.memdep_source(p["N"], True, "aarch64", counter),
            "memdep_n": M.memdep_source(p["control"], False, "aarch64", counter),
        }

    def test_kernel_pieces(self):
        src = M.chase_source("aarch64", "mach")
        self.assertIn('printf("page_bytes %zu\\n", PAGE_BYTES)', src)
        self.assertIn('if (!strncmp(argv[a], "d=", 2)) { code_dense = atoi(argv[a] + 2); continue; }', src)
        self.assertIn("pthread_jit_write_protect_np(0);", src)
        self.assertIn("sys_icache_invalidate(code, bytes);", src)
        for mode in ("tlbway", "slc", "bw"):
            self.assertIn(f'!strcmp(mode, "{mode}")) {{\n        return {mode}_main(argc, argv);', src)
        self.assertIn('printf("R fwd2 0 %d %.4f\\n"', M.memdep_source([4], True, "aarch64", "mach"))

    @unittest.skipUnless(CROSS, "needs the AArch64 cross compiler (. tools/env.sh)")
    def test_cross_compiles(self):
        with tempfile.TemporaryDirectory() as d:
            for name, src in self.sources("cntvct").items():
                c = os.path.join(d, name + ".c")
                with open(c, "w") as fh:
                    fh.write(src)
                r = subprocess.run(
                    [CROSS, "-O1", "-march=armv8-a", "-static", "-Wall", "-Werror", c, "-o", c[:-2]],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-2000:]}")

    @unittest.skipUnless(CROSS and MC, "needs the AArch64 cross compiler and llvm-mc")
    def test_macos_branches(self):
        """__APPLE__ C type-checks against the SDK prototypes; every inline asm block assembles for Mach-O."""
        with tempfile.TemporaryDirectory() as d:
            apple_stubs(os.path.join(d, "inc"))
            for name, src in self.sources("mach").items():
                c = os.path.join(d, name + ".c")
                with open(c, "w") as fh:
                    fh.write(src)
                inc = os.path.join(d, "inc")
                cmd = [CROSS, "-O1", "-D__APPLE__", "-I", inc, "-include", os.path.join(inc, "apple.h")]
                r = subprocess.run(
                    [*cmd, "-Wall", "-Werror", "-S", c, "-o", c + ".s"], capture_output=True, text=True
                )
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-2000:]}")
                with open(c + ".s") as fh:
                    blocks = re.findall(r"^#APP\n(.*?)^#NO_APP", fh.read(), re.S | re.M)
                self.assertGreater(len(blocks), 3)
                s = os.path.join(d, name + "_asm.s")
                with open(s, "w") as fh:
                    fh.write("".join(".p2align 2\n" + b for b in blocks))
                r = subprocess.run(
                    [MC, "-triple=arm64-apple-macos11", "-filetype=obj", s, "-o", s + ".o"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[:1500]}")


if __name__ == "__main__":
    unittest.main()
