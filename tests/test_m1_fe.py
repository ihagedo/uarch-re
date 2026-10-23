"""Frontend runner for Apple silicon (targets/m1/m1_fe.py) on a modelled M1: no Mac, no gem5.

python3 -m unittest tests.test_m1_fe
"""

import contextlib
import io
import json
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
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "targets", "m1")))

import m1_fe  # noqa: E402
import m1_kernels  # noqa: E402
from tests.test_m1_mem import apple_stubs  # noqa: E402
from uarchre.kernels.frontend import FEKernel, stride_chain_source  # noqa: E402

GHZ = 3.2
RATIO = 1024 * 24e6 / (GHZ * 1e9)  # 24 MHz ticks x 1024 per core cycle
MS = GHZ * 1e6  # core cycles per millisecond
LINE = m1_fe.native.l1i_line(None)  # firestorm.yaml
PAGE = 16 << 10
M1_L1I = {"size": 192 << 10, "ways": 12, "index": "virtual"}
GEM5_L1I = {"size": 32 << 10, "ways": 8, "index": "virtual"}


def lru_misses(n, cap, ways):
    """Misses per lap of a cyclic walk over n items in a set-associative LRU structure."""
    sets = cap // ways
    over = min(max(0, n - cap), sets)
    return (ways + 1) * over if n <= cap + sets else n


def page_cycles(L, cross, fetch):
    """Cycles per iteration of page_loop(L): `fetch` 16 (one 16-instruction block a cycle, split at a page),
    8 (8 a cycle, split at a page) or none (no split); 8-wide rename, the loop's sub + cbnz fused."""
    rename = max(1, (L - 1) / 8)
    if fetch == "none":
        return rename
    w = int(fetch)
    return max(1 + math.ceil((L - 1) / w), rename) if cross else max(math.ceil(L / w), rename)


def set_misses(n, stride, l1i, frames=None):
    """Misses per lap of a cyclic walk over n lines `stride` apart in an LRU L1i; `frames` maps each
    16 KiB page to a physical frame (physical index), None indexes by virtual address."""
    sets = {}
    for i in range(n):
        a = i * stride
        if frames is not None:
            a = frames[a // PAGE] * PAGE + a % PAGE
        k = a // LINE % (l1i["size"] // l1i["ways"] // LINE)
        sets[k] = sets.get(k, 0) + 1
    return sum(m for m in sets.values() if m > l1i["ways"])


def model(name, laps, fetch="8"):
    """Core cycles of one rep: 8-wide fetch, 3072 x 64 B L1i (12-way), L0 BTB 1024 (4-way) with no
    bubble, 2 bubbles from the L1i, 13-cycle mispredicts."""
    if m := re.match(r"fe_width(?:_mix)?_L(\d+)$", name):
        return laps * max(1, math.ceil(int(m.group(1)) / 8))
    if m := re.match(r"fe_page_L(\d+)_o(\d+)$", name):
        return laps * page_cycles(int(m.group(1)), int(m.group(2)) % PAGE == PAGE - 4, fetch)
    if m := re.match(r"fe_l1i3_(\d+)$", name):
        n = int(m.group(1)) // LINE
        return laps * (4 * n + 30 + 6 * lru_misses(n, (192 << 10) // LINE, 12))
    if m := re.match(r"fe_btb2_n(\d+)_s\d+$", name):
        n = int(m.group(1))
        return laps * (n + 2 + 2 * lru_misses(n, 1024, 4) + 17 * lru_misses(n, 49152, 12 * 32))
    raise ValueError(name)


class FakeRunner:
    """native.Runner's interface over `model`: quantized 24 MHz ticks, rep noise, a slow cold rep 0."""

    def __init__(self, smoke=False, noise=2e-4, reject_every=0, fetch="8", l1i=M1_L1I):
        self.smoke, self.noise, self.reject_every = smoke, noise, reject_every
        self.fetch, self.l1i = fetch, l1i
        self.counter = "pmccntr" if smoke else "cntvct"
        self.attempts, self.rng, self.saved = [], random.Random(0), {}

    def compile(self, src, name):
        self.last_src = src
        return name

    def ways(self, argv):
        """stride_chain_source's output: 2 cycles of lap control, 1 per jump, 18 per L1i miss."""
        work, reps, seed, stride, line, *ns = (int(a) for a in argv)
        assert line == LINE
        pages = max(ns) * stride // PAGE + 1
        frames = self.rng.sample(range(1 << 16), pages) if self.l1i["index"] == "physical" else None
        lines = ["counter_hz 24576000.0"]
        for n in ns:
            cpj = 1 + (2 + 18 * set_misses(n, stride, self.l1i, frames)) / n
            for r in range(reps):
                lines.append(f"pt {n} rep {r} cyc {cpj * RATIO * (1 + self.rng.gauss(0, self.noise)):.4f}")
        return "\n".join(lines) + "\n"

    def launch(self, binpath, argv=(), timeout=1800, label=""):
        bad = self.reject_every and len(self.attempts) % self.reject_every == self.reject_every - 1
        if binpath == "fe_ways":
            self.attempts.append({"label": label, "accepted": not bad})
            return self.ways(argv), RATIO
        laps, reps, seed, mode = (int(a) for a in argv)
        k = re.match(r"fe_mispredict_w(\d+)$", binpath)
        if k:
            cpb = 0.55 + (6.5 if mode == 1 else 0.0)
            cyc = laps * 64 * int(k.group(1)) * cpb
        else:
            cyc = model(binpath, laps, self.fetch)
        self.attempts.append({"label": label, "accepted": not bad})
        lines = ["counter_hz 24576000.0", "qos_class 33"]
        for i in range(reps):
            c = cyc * (1.05 if i == 0 else 1.0) * (1.6 if bad else 1.0) * (1 + self.rng.gauss(0, self.noise))
            if self.rng.random() < 0.05:
                c *= 1.01  # an interrupt
            lines.append(f"rep {i} cycles {round(c * RATIO / 1024) * 1024}")
        return "\n".join(lines) + "\n", RATIO * (1 + self.rng.gauss(0, self.noise))

    def save(self, family, poc, raw, summary):
        self.saved[poc] = json.loads(json.dumps({"raw": raw, "summary": summary}, default=float))
        return os.path.join(m1_fe.native.RESULTS, "test", family, poc)


def tiny_chain(n, seed=0, isa=None):
    return FEKernel(f"fe_btb2_n{n}_s{seed}", "nop", n, "taken branch", {"n": n, "seed": seed}, isa="aarch64")


def rows(summary):
    return {r["structure"]: r for r in summary}


def run(fn, r=None, args=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(r or FakeRunner(), args)


class TestPlan(unittest.TestCase):
    """M1 sweeps bracket the published values with 30 % to spare, and every timed rep is >= 1 ms."""

    P = m1_fe.M1

    def brackets(self, grid, value, below=3):
        self.assertGreaterEqual(sum(g <= 0.7 * value for g in grid), below)
        self.assertGreaterEqual(max(grid), 1.3 * value)

    def test_published_values(self):
        self.assertEqual(m1_fe.published("frontend.fetch_width")[0], 8)
        self.assertEqual(m1_fe.published("frontend.l1i", "size_kib")[0], 192)
        self.assertEqual(m1_fe.published("frontend.btb_levels[0]", "entries")[0], 1024)
        self.assertEqual(m1_fe.published("frontend.btb_levels[1]", "taken_bubble_cycles")[0], 2)
        v, src = m1_fe.published("frontend.mispredict_penalty")
        self.assertEqual(v, 13)
        self.assertIn("7cpu-m1", src)

    def test_ranges(self):
        w = m1_fe.published("frontend.fetch_width")[0]
        self.assertLessEqual(min(self.P["width"]["Ls"]), 2)
        self.assertGreater(max(self.P["width"]["Ls"]), 1.3 * w)
        l1i_bytes = m1_fe.published("frontend.l1i", "size_kib")[0] * 1024
        self.brackets(self.P["l1i"]["grid"], l1i_bytes)
        self.assertEqual(LINE, 64)
        self.assertFalse(any(b % 128 for b in self.P["l1i"]["grid"]))  # whole lines at 64 or 128 B
        l0 = m1_fe.published("frontend.btb_levels[0]", "entries")[0]
        self.brackets(self.P["btb"]["grid"], l0)
        self.assertGreaterEqual(self.P["btb"]["top"], 1.3 * l1i_bytes // 4)  # jumps filling the L1i
        self.assertLessEqual(self.P["jump"]["n"], 0.7 * l0)

    def test_page_and_ways_ranges(self):
        g = self.P["page"]
        self.assertEqual(g["Ls"], list(range(2, 65)))  # every L: a step every 8 or 16
        self.assertGreaterEqual(max(g["Ls"]), 3 * 16)
        self.assertLessEqual(g["extra_L"], m1_fe.published("backend.rename_width")[0])
        self.assertEqual((g["page"], g["page"] - 4 * g["split"]), (PAGE, PAGE - 4))
        w = self.P["ways"]
        self.assertEqual(w["ns"], list(range(2, 25)))
        for ways in (6, 12):  # the readings; 6 ways at 16 KiB and 12 at 32 KiB both knee inside
            self.brackets(w["ns"], ways)
            self.assertIn((192 << 10) // ways, w["strides"])
        self.assertGreater(max(w["strides"]), 32 << 10)  # control
        self.assertGreaterEqual(w["launches"], 8)

    def test_timed_region(self):
        P = self.P
        self.assertGreaterEqual(P["width"]["laps"], MS)
        self.assertGreaterEqual(P["page"]["laps"], MS)  # >= 1 cycle per iteration
        self.assertGreaterEqual(P["ways"]["work"], MS)  # >= 1 cycle per jump
        c = P["l1i"]
        for fp in c["grid"]:
            self.assertGreaterEqual(max(c["min_laps"], c["work"] * 32 // fp) * fp / 32, MS)
        b = P["btb"]
        for n in b["grid"] + [b["top"]]:
            self.assertGreaterEqual(max(b["min_laps"], b["work"] // n) * n, MS)
        self.assertGreaterEqual(P["jump"]["laps"] * P["jump"]["n"], MS)
        m = P["mispredict"]
        self.assertGreaterEqual(64 * m["words"] * 0.5 * m1_fe.published("frontend.mispredict_penalty")[0], MS)
        self.assertGreaterEqual(m["pattern_laps"] * 64 * m["words"] * 0.5, MS)  # one taken branch per cycle

    def test_smoke_is_small(self):
        S = m1_fe.SMOKE
        self.assertLessEqual(len(S["l1i"]["grid"]) + len(S["btb"]["grid"]), 20)
        self.assertEqual(S["outer"], 1)
        self.assertLessEqual(2 * len(S["page"]["Ls"]), 30)
        self.assertLessEqual(len(S["ways"]["strides"]) * S["ways"]["launches"], 3)
        way = GEM5_L1I["size"] // GEM5_L1I["ways"]
        self.assertEqual(S["ways"]["strides"], [way // 2, way, 2 * way])  # knees 16, 8, 8
        self.brackets(S["ways"]["ns"], 2 * GEM5_L1I["ways"])


class TestConvert(unittest.TestCase):
    def test_mach_output(self):
        out = "counter_hz 24576000.0\nqos_class 33\n"
        out += "rep 0 cycles 90112\nrep 1 cycles 81920\nrep 2 cycles 83968\n"
        d = m1_fe.convert(out, 8.0, 1000, 3)
        self.assertEqual(d["ticks"], [90112, 81920, 83968])
        self.assertEqual(d["cpu"], [81920 / 8000, 83968 / 8000])  # rep 0 dropped

    def test_gem5_output(self):
        out = "rep 0 cycles 500 instret 300\nrep 1 cycles 400 instret 300\n"
        self.assertEqual(m1_fe.convert(out, 1.0, 100, 2)["cpu"], [4.0])
        self.assertEqual(m1_fe.convert("rep 0 cycles 7\n", 1.0, 7, 1)["cpu"], [1.0])

    def test_missing_reps(self):
        with self.assertRaises(RuntimeError):
            m1_fe.convert("rep 0 cycles 5\n", 1.0, 1, 3, "k")

    def test_point_uses_counter_and_args(self):
        r = FakeRunner()
        k = m1_fe.width_loop(8, isa=m1_fe.A64)
        launches = m1_fe.point(r, k, 1 << 20, 4, outer=2, seeds=(5,), mode=2)
        self.assertIn("cntvct_el0", r.last_src)
        self.assertEqual([x["seed"] for x in launches], [5, 5])
        self.assertEqual(len(launches[0]["cpu"]), 3)
        self.assertAlmostEqual(m1_fe.value(launches), 1.0, delta=0.01)

    def test_boot_resamples_launches(self):
        launches = [{"cpu": [1.0, 1.0], "accepted": True}, {"cpu": [2.0, 2.0], "accepted": True}]
        b = m1_fe.boot(launches)
        self.assertTrue(set(b) <= {1.0, 1.5, 2.0})
        self.assertGreater(b.std(), 0.2)
        launches[1]["accepted"] = False
        self.assertEqual(set(m1_fe.boot(launches)), {1.0})


class TestOnModel(unittest.TestCase):
    """Each PoC end to end on the modelled M1 with the M1 sweep plan."""

    def test_width(self):
        raw, summary = run(m1_fe.run_width)
        for f in m1_fe.FILLERS:
            row = rows(summary)[f"frontend.fetch_width[{f}]"]
            self.assertEqual((row["value"], row["ci"], row["published"]), (8, [8, 8], 8))
            self.assertEqual(raw["estimates"][f]["shape"], "staircase")
            self.assertEqual(row["flags"], [])

    def test_l1i(self):
        for line in (None, 128):  # firestorm.yaml's 64 B, or --l1i-line 128
            raw, summary = run(m1_fe.run_l1i, FakeRunner(), SimpleNamespace(l1i_line=line))
            line = line or LINE
            self.assertEqual(raw["plan"]["line"], line)
            cap = rows(summary)["frontend.l1i.size_kib"]
            self.assertAlmostEqual(cap["value"], 192, delta=2 * line / 1024)
            self.assertLessEqual(cap["ci"][0], cap["value"])
            self.assertGreaterEqual(cap["ci"][1], cap["value"])
            self.assertLess(cap["ci"][1] - cap["ci"][0], 4)
            self.assertEqual(cap["published"], 192)
            self.assertIn(f"lines of {line} B", cap["note"])
            jump = rows(summary)["frontend.taken_jump_cycles"]
            self.assertAlmostEqual(jump["value"], 1.0, delta=0.02)
            self.assertEqual(jump["published"], 1.0)
            self.assertEqual(raw["search"]["bracket"][1] - raw["search"]["bracket"][0], 1)

    def test_page(self):
        """Each fetch reading on its own modelled M1: step period and crossing cost."""
        for fetch, period, extra in (("16", 16, 1), ("8", 8, 1), ("none", None, 0)):
            raw, summary = run(m1_fe.run_page, FakeRunner(fetch=fetch))
            rs = rows(summary)
            row = rs["frontend.fetch_width[page]"]
            self.assertEqual(row["value"], period, fetch)
            self.assertEqual(row["ci"], [period, period] if period else None)
            self.assertEqual((row["published"], row["unit"], row["flags"]), (8, "instr/fetch", []))
            self.assertAlmostEqual(rs["frontend.page_cross_cycles"]["value"], extra, delta=0.02)
            self.assertEqual(len(raw["per_launch"]), m1_fe.M1["outer"])
            self.assertEqual(raw["line"], LINE)
        offsets = {p["where"]: p["offset"] for p in raw["points"]}
        self.assertEqual(offsets, {"cross": PAGE - 4, "inside": 2048 - 4})

    def test_ways(self):
        """12 ways; 6 ways indexed by virtual address; 6 ways indexed by physical address (random frames)."""
        for ways, index, knees in ((12, "virtual", (12, 12, 12)), (6, "virtual", (12, 6, 6))):
            r = FakeRunner(l1i={"size": 192 << 10, "ways": ways, "index": index})
            raw, summary = run(m1_fe.run_ways, r)
            rs = rows(summary)
            for s, k in zip(m1_fe.M1["ways"]["strides"], knees):
                row = rs[f"frontend.l1i.ways[{s >> 10}KiB]"]
                self.assertEqual((row["value"], row["ci"], row["unit"]), (k, [k, k], "lines"), (ways, s))
                self.assertIsNone(row["published"])
                self.assertEqual(row["flags"], [])
            self.assertEqual(raw["line"], LINE)
            self.assertEqual(len(raw["launches"]["32768"]), m1_fe.M1["ways"]["launches"])
        r = FakeRunner(l1i={"size": 192 << 10, "ways": 6, "index": "physical"})
        rs = rows(run(m1_fe.run_ways, r)[1])
        for s in (16, 32, 64):
            lo, hi = rs[f"frontend.l1i.ways[{s}KiB]"]["ci"]
            self.assertTrue(6 <= lo < hi <= 12, (s, lo, hi))  # the knee moves between launches

    def test_ways_reports_missing_points(self):
        r = FakeRunner()
        r.ways = lambda argv: "pt 2 rep 0 cyc 7.7\n"
        with self.assertRaises(RuntimeError):
            run(m1_fe.run_ways, r)

    def test_btb(self):
        with mock.patch.object(m1_fe, "jump_chain", tiny_chain):
            raw, summary = run(m1_fe.run_btb)
        rs = rows(summary)
        l0 = rs["frontend.btb_levels[0].entries"]
        self.assertTrue(l0["ci"][0] <= 1024 <= l0["ci"][1], l0)
        self.assertLessEqual(abs(l0["value"] - 1024), 2)
        self.assertAlmostEqual(rs["frontend.btb_levels[0].taken_bubble_cycles"]["value"], 0, delta=0.05)
        self.assertAlmostEqual(rs["frontend.btb_levels[1].taken_bubble_cycles"]["value"], 2, delta=0.05)
        l1 = rs["frontend.btb_levels[1].entries"]
        self.assertLessEqual(abs(l1["value"] - 49152), 32)
        self.assertIsNone(l1["published"])
        self.assertTrue(all(s["consistent"] for lv in raw["levels"] for s in lv["seeds"].values()))

    def test_mispredict(self):
        r = FakeRunner()
        raw, summary = run(m1_fe.run_mispredict, r)
        pen = rows(summary)["frontend.mispredict_penalty"]
        self.assertAlmostEqual(pen["value"], 13, delta=0.05)
        self.assertEqual(len(raw["penalties"]), len(m1_fe.M1["mispredict"]["seeds"]))
        k = m1_fe.random_branches(m1_fe.M1["mispredict"]["words"], isa=m1_fe.A64)
        self.assertNotIn(f"size_t words  = {k.data_words};", r.last_src)  # enough fresh data for 9 reps
        self.assertIn(f"size_t words  = {9 * k.params['words'] + 1};", r.last_src)

    def test_rejected_launches_are_dropped(self):
        r = FakeRunner(reject_every=4)
        raw, summary = run(m1_fe.run_mispredict, r)
        pen = rows(summary)["frontend.mispredict_penalty"]
        self.assertAlmostEqual(pen["value"], 13, delta=0.05)
        self.assertTrue(any(f.startswith("unaccepted") for f in pen["flags"]))

    def test_smoke_plan_runs(self):
        r = FakeRunner(smoke=True, l1i=GEM5_L1I)
        for poc, fn in m1_fe.POCS.items():
            with mock.patch.object(m1_fe, "jump_chain", tiny_chain):
                raw, summary = run(fn, r)
            for row in summary:
                self.assertEqual(
                    set(row), {"structure", "value", "ci", "unit", "published", "source", "flags", "note"}
                )
            json.dumps(raw, default=float)
            if poc == "P3.2-ways":  # gem5's 8 ways of 4 KiB
                self.assertEqual([x["value"] for x in summary], [16, 8, 8])
            if poc == "P3.1-page":
                self.assertEqual(summary[0]["value"], 8)


class TestCLI(unittest.TestCase):
    def test_help_and_unknown(self):
        for argv, code in ((["-h"], 0), (["P9.9"], 2)):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    m1_fe.main(argv)
            self.assertEqual(e.exception.code, code)

    def test_main_runs_the_named_pocs(self):
        fake = FakeRunner()
        with mock.patch.object(m1_fe.native, "Runner", lambda a: fake):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                m1_fe.main(["P3.7", "P3.1", "--tag", "t", "--force"])
        self.assertEqual(set(fake.saved), {"P3.7", "P3.1"})
        self.assertIn("frontend.mispredict_penalty", out.getvalue())
        self.assertEqual(set(m1_fe.POCS), set(m1_fe.EST_MIN))
        self.assertEqual(m1_fe.FAMILY, "fe")

    def test_kernels_cli_runs_a_new_part_alone(self):
        fake = FakeRunner(smoke=True, l1i=GEM5_L1I)
        with mock.patch.object(m1_kernels.native, "Runner", lambda a: fake):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                m1_kernels.main(["run", "fe", "P3.2-ways", "--gem5-smoke", "--l1i-line", "64"])
        self.assertEqual(set(fake.saved), {"P3.2-ways"})
        self.assertIn("frontend.l1i.ways[4KiB]: 8 lines", out.getvalue())


CROSS = shutil.which("aarch64-linux-gnu-gcc")
MC = shutil.which("llvm-mc") or shutil.which("llvm-mc-19")


class TestWaysSource(unittest.TestCase):
    """stride_chain_source for every counter; its macOS branch against the SDK prototypes."""

    def test_pieces(self):
        src = stride_chain_source("mach")
        for piece in ("flags |= MAP_JIT;", "pthread_jit_write_protect_np(0);", "sys_icache_invalidate(code"):
            self.assertIn(piece, src)
        self.assertIn('printf("pt %zu rep %d cyc %.4f\\n"', src)
        self.assertIn('"dsb ish', stride_chain_source("pmccntr"))  # gem5: no EL0 cache maintenance
        self.assertIn("__builtin___clear_cache", stride_chain_source("cntvct"))

    @unittest.skipUnless(CROSS, "needs the AArch64 cross compiler (. tools/env.sh)")
    def test_cross_compiles(self):
        with tempfile.TemporaryDirectory() as d:
            for counter in ("cntvct", "pmccntr"):
                c = os.path.join(d, f"ways_{counter}.c")
                with open(c, "w") as fh:
                    fh.write(stride_chain_source(counter))
                r = subprocess.run(
                    [CROSS, "-O1", "-static", "-Wall", "-Werror", c, "-o", c[:-2]],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    @unittest.skipUnless(CROSS and MC, "needs the AArch64 cross compiler and llvm-mc")
    def test_macos_branch(self):
        with tempfile.TemporaryDirectory() as d:
            inc = os.path.join(d, "inc")
            apple_stubs(inc)
            c = os.path.join(d, "ways.c")
            with open(c, "w") as fh:
                fh.write(stride_chain_source("mach"))
            cmd = [CROSS, "-O1", "-D__APPLE__", "-I", inc, "-include", os.path.join(inc, "apple.h"), "-Wall"]
            r = subprocess.run([*cmd, "-Werror", "-S", c, "-o", c + ".s"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            with open(c + ".s") as fh:
                blocks = re.findall(r"^#APP\n(.*?)^#NO_APP", fh.read(), re.S | re.M)
            self.assertGreaterEqual(len(blocks), 2)
            s = os.path.join(d, "ways_asm.s")
            with open(s, "w") as fh:
                fh.write("".join(".p2align 2\n" + b for b in blocks))
            r = subprocess.run(
                [MC, "-triple=arm64-apple-macos11", "-filetype=obj", s, "-o", s + ".o"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr[:1500])


if __name__ == "__main__":
    unittest.main()
