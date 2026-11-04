#!/usr/bin/env python3
"""P2.1-store on gem5 ARM O3: the store mode with the default L1D (allocates on a store miss) and with a
WriteAllocator (whole-line streams skip allocation). 8 KiB of 64 B lines, a 128 KiB eviction read
(4 x the L1D, half the L2), 4 reps; prints the chase's cycles per load and each store row's fraction.

  validation/gem5/mem/mem_store.py
"""

import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [ROOT]
from uarchre.backends.platform.gem5 import compile_c, run_gem5_binary  # noqa: E402
from uarchre.kernels import memory as M  # noqa: E402

b = compile_c(M.chase_source("aarch64", "pmccntr"), "memprobe", isa="aarch64")
TRUTH = {"allocate": [], "write-around": ["cpu.dcache.write_allocator=WriteAllocator(no_allocate_limit=2)"]}


def one(name):
    out = run_gem5_binary(b, ["store", "64", "8192", "131072", "4", "0", "1", "2", "3"], sets=TRUTH[name],
                          isa="aarch64")[0]  # fmt: skip
    t = {}
    for v, _, c in re.findall(r"^R store(\d) \d+ (\d+) ([\d.]+)", out, re.M):
        t.setdefault(int(v), []).append(float(c))
    return name, {v: statistics.median(c[1:]) for v, c in sorted(t.items())}


with ThreadPoolExecutor(2) as ex:
    for name, c in ex.map(one, TRUTH):
        a = {v: (c[0] - c[v]) / (c[0] - c[1]) for v in (2, 3)}
        print(f"{name:13s} none {c[0]:.2f} load {c[1]:.2f} store {c[2]:.2f} whole {c[3]:.2f} | "
              f"allocated: store {a[2]:.2f} whole {a[3]:.2f}")  # fmt: skip
