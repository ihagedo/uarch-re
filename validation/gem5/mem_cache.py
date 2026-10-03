#!/usr/bin/env python3
"""P2.1 on gem5 O3 (RISC-V): cache capacity, latency, line size and associativity.

Phases (results/gem5/mem-cache/<phase>.json): coarse (random chase, 4 KiB..8 MiB, five
cache configurations), fine (a finer grid inside each transition coarse found, never
placed from the configured size), latency (L1D and L2 hit latency), line (pair chase:
the cost per pair doubles once D reaches the line size), assoc (K nodes a set span apart).

    python3 validation/gem5/mem_cache.py coarse fine latency line assoc
"""

import os
import sys

from mem_common import run_jobs, load, med, RESULTS, log

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from uarchre.kernels.memory import plateau_onset

OUT = os.path.join(RESULTS, "mem-cache")
KB, MB = 1 << 10, 1 << 20
NL, REPS = 1000, 3

CAP = {  # name: (L1D size, L1D assoc, L2 size)
    "l1d16k-l2128k": (16 * KB, 8, 128 * KB),
    "l1d32k-l2256k": (32 * KB, 8, 256 * KB),
    "l1d48k-l2512k": (48 * KB, 12, 512 * KB),
    "l1d64k-l21m": (64 * KB, 8, 1 * MB),
    "l1d128k-l22m": (128 * KB, 8, 2 * MB),
}


def cap_sets(name):
    l1, a1, l2 = CAP[name]
    return [f"cpu.dcache.size={l1 // KB}kB", f"cpu.dcache.assoc={a1}", f"l2.size={l2 // KB}kB"]


def coarse_grid():
    return [int(4 * KB * 2 ** (k / 2)) // 64 * 64 for k in range(0, 2 * 11 + 1)]  # 4 KiB .. 8 MiB


def coarse_jobs():
    fs = coarse_grid()
    # two full warm passes for everything up to 2 MiB (the largest L2), the
    # default cap (20000 loads) above, where every load misses anyway
    small = [f for f in fs if f <= 2 * MB]
    big = [f for f in fs if f > 2 * MB]
    argv = ["lat", "rand", 64, NL, REPS, 0, 1, f"w={2 * (2 * MB) // 64}", *small, "w=20000", *big]
    return [{"key": n, "cfg": n, "argv": argv, "params": {}, "sets": cap_sets(n)} for n in CAP]


def levels(xs, ys):
    """Transitions on a coarse curve: plateaus are runs of points within 15 %
    of each other; a transition is the gap between two plateaus. Returns
    [(last x of lower plateau, first x of upper plateau, lo level, hi level)]."""
    pts = sorted(zip(xs, ys))
    groups = [[pts[0]]]
    for x, y in pts[1:]:
        if y <= groups[-1][0][1] * 1.15 and y >= groups[-1][-1][1] / 1.15:
            groups[-1].append((x, y))
        else:
            groups.append([(x, y)])
    plats = [g for g in groups if len(g) >= 2]
    out = []
    for lo, hi in zip(plats, plats[1:]):
        out.append((lo[-1][0], hi[0][0], med([y for _, y in lo]), med([y for _, y in hi])))
    return out


def fine_grid(lo, hi):
    step = 1 << max(6, (lo // 64).bit_length() - 1)  # power of two, lo/128 < step <= lo/64
    start = lo // step * step - 2 * step
    return list(range(start, hi + step, step))


def coarse_curve(rec):
    d = rec["runs"]["lat"]
    xs = sorted(int(k) for k in d)
    return xs, [med(d[str(x)]) for x in xs]


def fine_jobs():
    co = load(os.path.join(OUT, "coarse.json"))
    jobs = []
    for n in CAP:
        xs, ys = coarse_curve(co[n])
        for i, (lo, hi, ylo, yhi) in enumerate(levels(xs, ys)[:2]):
            fs = fine_grid(lo, hi)
            w = 2 * max(fs) // 64
            jobs.append(
                {
                    "key": f"{n}/t{i}",
                    "cfg": n,
                    "transition": i,
                    "bracket": [lo, hi],
                    "levels": [ylo, yhi],
                    "argv": ["lat", "rand", 64, NL, REPS, 0, 1, f"w={w}", *fs],
                    "params": {},
                    "sets": cap_sets(n),
                }
            )
    return jobs


def detect_capacity(fine_rec, coarse_levels):
    """Last footprint still at the lower plateau (onset of the rise)."""
    d = fine_rec["runs"]["lat"]
    xs = sorted(int(k) for k in d)
    ys = [med(d[str(x)]) for x in xs]
    lo_level, hi_level = coarse_levels
    return plateau_onset(xs, ys, lo_level, hi_level, frac=0.1), (xs, ys)


L1_LAT = [1, 2, 3, 4, 6]
L2_LAT = [5, 10, 15, 20, 30]


def latency_jobs():
    argv = ["lat", "rand", 64, NL, REPS, 0, 1, 8 * KB, 128 * KB]
    jobs = []
    for x in L1_LAT:
        jobs.append(
            {
                "key": f"l1d{x}",
                "level": "l1d",
                "hit_latency": x,
                "argv": argv,
                "params": {},
                "sets": [f"cpu.dcache.tag_latency={x}", f"cpu.dcache.data_latency={x}"],
            }
        )
    for x in L2_LAT:
        jobs.append(
            {
                "key": f"l2{x}",
                "level": "l2",
                "hit_latency": x,
                "argv": argv,
                "params": {},
                "sets": [f"l2.tag_latency={x}", f"l2.data_latency={x}"],
            }
        )
    return jobs


LINES = [32, 64, 128, 256]
PAIR_D = [8, 16, 32, 64, 128, 256, 512]


def line_jobs():
    jobs = []
    for b in LINES:
        sets = [f"cache_line_size={b}"]
        if b < 64:
            sets.append(f"cpu.fetchBufferSize={b}")  # O3 needs fetch buffer <= line
        jobs.append(
            {
                "key": f"line{b}",
                "line": b,
                "argv": ["pair", 1024, NL, REPS, 8 * MB, *PAIR_D],
                "params": {},
                "sets": sets,
            }
        )
    return jobs


L1_ASSOC = [2, 4, 8, 16]
L2_ASSOC = [4, 8, 16]


def assoc_jobs():
    jobs = []
    ks = list(range(1, 25))
    for a in L1_ASSOC:
        jobs.append(
            {
                "key": f"l1a{a}",
                "level": "l1d",
                "assoc": a,
                "S": 32 * KB,
                "argv": ["conflict", 32 * KB, NL, REPS, 0, *ks],
                "params": {},
                "sets": [f"cpu.dcache.assoc={a}"],
            }
        )
    for a in L2_ASSOC:
        jobs.append(
            {
                "key": f"l2a{a}",
                "level": "l2",
                "assoc": a,
                "l1_assoc": 4,
                "S": 256 * KB,
                "argv": ["conflict", 256 * KB, NL, REPS, 0, *ks],
                "params": {},
                "sets": ["cpu.dcache.assoc=4", f"l2.assoc={a}"],
            }
        )
    return jobs


PHASES = {
    "coarse": coarse_jobs,
    "fine": fine_jobs,
    "latency": latency_jobs,
    "line": line_jobs,
    "assoc": assoc_jobs,
}

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    want = sys.argv[1:] or ["coarse", "latency", "line", "assoc", "fine"]
    # independent phases run side by side; `fine` needs `coarse` first
    first = [p for p in want if p != "fine"]
    with ThreadPoolExecutor(len(first) or 1) as ex:
        list(ex.map(lambda p: run_jobs(f"mem_cache_{p}", PHASES[p](), os.path.join(OUT, f"{p}.json")), first))
    if "fine" in want:
        run_jobs("mem_cache_fine", fine_jobs(), os.path.join(OUT, "fine.json"))
    log("mem_cache", "all phases done")
