#!/usr/bin/env python3
"""P2.4 on gem5 O3 (RISC-V): prefetcher detection and reliable misses.

Chains over 4 MiB (16x the L2) with no prefetcher, a StridePrefetcher on the L1D, or
one on the L2: independent 64 B strides, dependent 64 B and 256 B strides, a random
chain (plus L1 and L2 hit references) and the window probe's own two chains, whose L2
miss fraction is read from gem5 counters dumped around the timed chase.

    python3 validation/gem5/mem_prefetch.py [--isa aarch64]
"""

import os

from mem_common import run_jobs, RESULTS, isa_args, for_isa

OUT = os.path.join(RESULTS, "mem-prefetch", "results.json")
F = 4 << 20
NL, REPS = 2000, 3

PREFETCH = {
    "off": [],
    "l1d_stride": ["cpu.dcache.prefetcher=StridePrefetcher(degree=4)"],
    "l2_stride": ["l2.prefetcher=StridePrefetcher(degree=4)"],
}
CHAINS = {
    "indep64": ["indep", 64, NL, REPS, 0, F],
    "seq64": ["lat", "seq", 64, NL, REPS, 0, 1, F],
    "seq256": ["lat", "seq", 256, NL, REPS, 0, 1, F],
    "rand": ["lat", "rand", 64, NL, REPS, 0, 1, 8 << 10, 128 << 10, F],
    "wchain": ["wchain", F, NL, REPS, 1],
}


def jobs():
    return [
        {"key": f"{pf}/{ch}", "pf": pf, "chain": ch, "argv": argv, "params": {}, "sets": sets}
        for pf, sets in PREFETCH.items()
        for ch, argv in CHAINS.items()
    ]


if __name__ == "__main__":
    import sys

    isa, _ = isa_args(sys.argv[1:])
    if isa == "aarch64":
        run_jobs(
            "arm_mem_prefetch",
            for_isa(jobs(), isa),
            os.path.join(RESULTS, "arm-memory-prefetch", "results.json"),
        )
    else:
        run_jobs("mem_prefetch", jobs(), OUT)
