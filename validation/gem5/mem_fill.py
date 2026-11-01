#!/usr/bin/env python3
"""P2.3-fill on gem5 ARM O3 with known L1D MSHRs: mem_mlp's far cache with an 8 MiB L2, so both copies
fit. 16 and 8 MSHRs merge misses to one line; tgts_per_mshr=1 does not. Prints the plain, same-line and
far readings and the line / far ratio (2: one entry per line).

  validation/gem5/mem_fill.py
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "validation", "gem5")]
import mem_mlp  # noqa: E402
from uarchre.backends.platform.gem5 import compile_c, run_gem5_binary  # noqa: E402
from uarchre.kernels import memory as M  # noqa: E402

F, ITERS, REPS = 2 << 20, 200, 2
KS = list(range(1, 13)) + [14, 16, 18, 20, 24, 28, 32, 36, 40, 44]
SETS = ["l2.size=8192kB", "l2.tag_latency=100", "l2.data_latency=100", "l2.mshrs=128"]
PARAMS = {**mem_mlp.PARAMS, "numPhysVecRegs": 512}
TRUTH = {"m16": ["cpu.dcache.mshrs=16"], "m8": ["cpu.dcache.mshrs=8"],
         "m16-t1": ["cpu.dcache.mshrs=16", "cpu.dcache.tgts_per_mshr=1"]}  # fmt: skip
b = compile_c(M.chase_source("aarch64", "pmccntr"), "memprobe", isa="aarch64")


def one(job):
    name, where = job
    o = {"plain": [], "line": ["o=8"], "far": [f"o={F}"]}[where]
    argv = ["mlp", F, ITERS, REPS, 0, f"x={16 << 20}", "n=64", *o, *KS]
    out = run_gem5_binary(b, [str(a) for a in argv], params=PARAMS, sets=SETS + TRUTH[name], isa="aarch64")[0]
    ts = {}
    for k, _, v in re.findall(r"^R mlp (\d+) (\d+) ([\d.]+)", out, re.M):
        ts.setdefault(int(k), []).append(float(v))
    ks = sorted(ts)
    return job, mem_mlp.reading(ks, [min(ts[k]) for k in ks]), [round(min(ts[k])) for k in ks]


jobs = [(n, w) for n in TRUTH for w in ("plain", "line", "far")]
with ThreadPoolExecutor(9) as ex:
    res = {j: (r, t) for j, r, t in ex.map(one, jobs)}
for n in TRUTH:
    got = {w: res[(n, w)][0]["reading"] for w in ("plain", "line", "far")}
    print(f"{n:7s} plain {got['plain']:4.0f}  line {got['line']:4.0f}  far {got['far']:4.0f}  line/far "
          f"{got['line'] / got['far']:.2f}  {[res[(n, w)][0]['why'] for w in ('line', 'far')]}")  # fmt: skip
    for w in ("line", "far"):
        print(f"    {w} T(K):", res[(n, w)][1])
