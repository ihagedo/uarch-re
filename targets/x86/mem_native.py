#!/usr/bin/env python3
"""P2 memory probes on the x86 host, pinned to P-core CPU 0.

    python3 targets/x86/mem_native.py footprint   # P2.4 / P2.1: random-chain latency vs footprint
    python3 targets/x86/mem_native.py chains      # P2.4: chain types at a DRAM footprint
    python3 targets/x86/mem_native.py wchain      # P2.4: the window probe's own chains
    python3 targets/x86/mem_native.py tlb         # P2.2: page-stride vs dense chains
    python3 targets/x86/mem_native.py capfine conflict   # P2.1: capacity near L1/L2/L3; L1D ways
Appends to results/native/mem-<cmd>/results.json; no argument runs footprint, chains and wchain.
"""

import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.kernels import memory as M  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hygiene  # noqa: E402

CPU = 0
WATCH = [0, 1]  # CPU 0 and its SMT sibling
LAUNCHES = 5
RESULTS = os.path.join(ROOT, "results", "native")
LOGS = os.path.join(
    os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER', 'u')}/uarch-re"), "logs", "memory"
)
KB, MB = 1 << 10, 1 << 20


def log(msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "mem_native.log"), "a") as fh:
        fh.write(line + "\n")


def save(name, recs):
    d = os.path.join(RESULTS, f"mem-{name}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "results.json")
    old = []
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
    with open(path, "w") as fh:
        json.dump(old + recs, fh, indent=1)
    log(f"{name}: +{len(recs)} records -> {path}")


def launch(tag, argv_head, keys, shuffle=True, **meta):
    """LAUNCHES launches of argv_head + keys (keys shuffled per launch)."""
    recs = []
    for li in range(LAUNCHES):
        ks = list(keys)
        if shuffle:  # so slow drift does not look like a footprint effect
            random.Random(1000 * li + len(ks)).shuffle(ks)
        waited = hygiene.wait_quiet(WATCH, 0.2)
        load0 = os.getloadavg()
        cmd = ["taskset", "-c", str(CPU), M.binary("x86_64"), *[str(a) for a in argv_head + ks]]
        out, info = hygiene.run_monitored(cmd, watch=WATCH)
        runs, sc = M.parse_runs(out)
        recs.append(
            {
                "tag": tag,
                "launch": li,
                "argv": [*argv_head],
                "keys": ks,
                "cpu": CPU,
                "runs": {t: {str(k): v for k, v in d.items()} for t, d in runs.items()},
                "scalars": sc,
                "loadavg_before": load0,
                "loadavg_after": os.getloadavg(),
                "waited_s": round(waited, 2),
                "monitor": info,
                **meta,
            }
        )
        log(
            f"{tag} launch {li}: {info['wall_s']:.1f} s, clock_ratio {sc.get('clock_ratio')}, "
            f"load {load0[0]:.1f}, busy {info['busy']}, nivcsw {info['nivcsw']}"
        )
    return recs


def grid(lo, hi, per_octave):
    out, x = [], lo
    k = 0
    while True:
        x = int(round(lo * 2 ** (k / per_octave) / 64)) * 64
        if x > hi:
            break
        out.append(x)
        k += 1
    return sorted(set(out))


def footprint():
    fs = grid(4 * KB, 1024 * MB, 4)
    recs = []
    for pages in (0, 1, 2):
        recs += launch(f"rand-pages{pages}", ["lat", "rand", 64, 100000, 3, pages, 1], fs, pages=pages)
    save("footprint", recs)


def chains():
    F = 256 * MB
    recs = []
    for name, head in {
        "rand64": ["lat", "rand", 64],
        "seq64": ["lat", "seq", 64],
        "seq256": ["lat", "seq", 256],
        "seq4096": ["lat", "seq", 4096],
    }.items():
        recs += launch(name, [*head, 100000, 3, 0, 1], [F], chain=name)
    recs += launch("indep64", ["indep", 64, 100000, 3, 0], [F], chain="indep64")
    save("chains", recs)


def wchain():
    recs = []
    # 2.5 MiB = 40k nodes (old runs), 5 MiB, 64 MiB (sweep.py --chain-mb default), 256 MiB for reference
    for cb in (2560 * KB, 5 * MB, 64 * MB, 256 * MB):
        recs += launch(f"wchain-{cb}", ["wchain", cb, 20000, 3], [], shuffle=False, chain_bytes=cb)
    save("wchain", recs)


def tlb():
    ps = sorted({int(round(8 * 2 ** (k / 8))) for k in range(0, 8 * 13 + 1)})  # 8 .. 65536 pages
    recs = []
    recs += launch("page-4k", ["page", 200000, 3, 1, 0], ps, pages=1, dense=0)
    recs += launch("dense-4k", ["page", 200000, 3, 1, 1], ps, pages=1, dense=1)
    recs += launch("page-2m", ["page", 200000, 3, 2, 0], ps, pages=2, dense=0)
    save("tlb", recs)


def capfine():
    regions = {
        "l1": range(16 * KB, 129 * KB, 4 * KB),
        "l2": range(512 * KB, 4 * MB + 1, 128 * KB),
        "l3": range(8 * MB, 49 * MB, 2 * MB),
    }
    recs = []
    for name, fs in regions.items():
        recs += launch(f"cap-{name}", ["lat", "rand", 64, 100000, 3, 2, 1], list(fs), pages=2, region=name)
    save("capfine", recs)


def conflict():
    # one L1D set: S = 64 KiB is a multiple of any L1 set span; 2 MiB pages so DTLB sets do not conflict first
    ks = list(range(1, 33))
    recs = launch("conflict-64k", ["conflict", 64 * KB, 100000, 3, 2], ks, S=64 * KB, pages=2)
    save("conflict", recs)


if __name__ == "__main__":
    cmds = {
        "footprint": footprint,
        "chains": chains,
        "wchain": wchain,
        "tlb": tlb,
        "capfine": capfine,
        "conflict": conflict,
    }
    for c in sys.argv[1:] or ["footprint", "chains", "wchain"]:
        cmds[c]()
