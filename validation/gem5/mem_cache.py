#!/usr/bin/env python3
"""P2.1 on gem5 O3 (RISC-V or ARM): cache capacity, latency, line size and associativity.

Phases (results/gem5/mem-cache/<phase>.json): coarse (random chase, 4 KiB..8 MiB, five cache
configurations), fine (a finer grid inside each transition coarse found, never placed from the
configured size), latency (L1D and L2 hit latency), line (pair chase: the cost per pair doubles once
D reaches the line size), assoc (K nodes a set span apart). ARM: ARM_PHASES into arm-memory-cache/.

    python3 validation/gem5/mem_cache.py [--isa aarch64] coarse fine latency line assoc
"""

import os
import sys

from mem_common import run_jobs, load, med, RESULTS, log, isa_args, for_isa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from uarchre.kernels.memory import plateau_onset

OUT = os.path.join(RESULTS, "mem-cache")
ARM_OUT = os.path.join(RESULTS, "arm-memory-cache")
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


def coarse_jobs(names=tuple(CAP)):
    fs = coarse_grid()
    # two full warm passes for everything up to 2 MiB (the largest L2), the
    # default cap (20000 loads) above, where every load misses anyway
    small = [f for f in fs if f <= 2 * MB]
    big = [f for f in fs if f > 2 * MB]
    argv = ["lat", "rand", 64, NL, REPS, 0, 1, f"w={2 * (2 * MB) // 64}", *small, "w=20000", *big]
    return [{"key": n, "cfg": n, "argv": argv, "params": {}, "sets": cap_sets(n)} for n in names]


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


def fine_jobs(names=tuple(CAP), out=OUT):
    co = load(os.path.join(out, "coarse.json"))
    jobs = []
    for n in names:
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


def line_size(xs, ys):
    """First D where the pair cost passes half the step from the smallest D to the largest."""
    pts = sorted(zip(xs, ys))
    lo, hi = pts[0][1], pts[-1][1]
    return next((int(x) for x, y in pts if y > lo + 0.5 * (hi - lo)), None)


def assoc_onset(xs, ys):
    """Last K of a one-set chain still at the K = 1 level (+10% of the step to the highest level)."""
    pts = sorted(zip(xs, ys))
    last, _ = plateau_onset([x for x, _ in pts], [y for _, y in pts], pts[0][1], max(ys), frac=0.1)
    return None if last is None else int(last)


L1_LAT = [1, 2, 3, 4, 6]
L2_LAT = [5, 10, 15, 20, 30]


def latency_jobs(l1_lat=L1_LAT, l2_lat=L2_LAT):
    argv = ["lat", "rand", 64, NL, REPS, 0, 1, 8 * KB, 128 * KB]
    jobs = []
    for x in l1_lat:
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
    for x in l2_lat:
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


def line_jobs(lines=LINES):
    jobs = []
    for b in lines:
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


def assoc_jobs(l1_assoc=L1_ASSOC, l2_assoc=L2_ASSOC):
    jobs = []
    ks = list(range(1, 25))
    for a in l1_assoc:
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
    for a in l2_assoc:
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

# aarch64: three configurations and two or three values per parameter
ARM_CAP = ["l1d16k-l2128k", "l1d48k-l2512k", "l1d128k-l22m"]
ARM_PHASES = {
    "coarse": lambda: coarse_jobs(ARM_CAP),
    "fine": lambda: fine_jobs(ARM_CAP, ARM_OUT),
    "latency": lambda: latency_jobs([2, 4], [10, 20]),
    "line": lambda: line_jobs([32, 64, 128]),
    "assoc": lambda: assoc_jobs([2, 4, 8], []),
}

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    isa, want = isa_args(sys.argv[1:])
    want = want or ["coarse", "latency", "line", "assoc", "fine"]
    phases, out, name = PHASES, OUT, "mem_cache"
    if isa == "aarch64":
        phases, out, name = ARM_PHASES, ARM_OUT, "arm_mem_cache"
    # independent phases run side by side; `fine` needs `coarse` first
    first = [p for p in want if p != "fine"]
    with ThreadPoolExecutor(len(first) or 1) as ex:
        list(
            ex.map(
                lambda p: run_jobs(f"{name}_{p}", for_isa(phases[p](), isa), os.path.join(out, f"{p}.json")),
                first,
            )
        )
    if "fine" in want:
        run_jobs(f"{name}_fine", for_isa(phases["fine"](), isa), os.path.join(out, "fine.json"))
    log(name, "all phases done")
