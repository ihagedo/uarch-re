#!/usr/bin/env python3
"""P1.5 on gem5 O3 (RISC-V or ARM): scheduler queues, queue pairing, dispatch-buffer probe.

gem5 dispatches in order to the first capable queue with room; memory ops keep their
entry until they complete. Experiments (results/gem5/[arm-]sched-p1_5/<name>.json): unified
(one queue, Q = 16..96), queues (paired layouts in LAYOUTS), sharing (m class-X ops
preloaded before the Y fillers) and dispatch (Handley's probe, S adds then M fadds).

    validation/gem5/sched_p1_5.py unified queues sharing dispatch --jobs 12 [--isa aarch64]
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sched_common import ROOT, run_multi, config_truth, iq_set, warm
from uarchre.kernels.scheduler import sched_probe, dispatch_probe, CLASSES, ASM_EMITTERS

OUT = os.path.join(ROOT, "results", "gem5", "sched-p1_5")
ISA = "riscv64"
BASE = {
    "numROBEntries": 512,
    "numPhysIntRegs": 512,
    "numPhysFloatRegs": 512,
    "LQEntries": 128,
    "SQEntries": 128,
}
UNIFIED = [16, 24, 32, 48, 64, 96]
LAYOUTS = {  # name: (a0, a1, f, l, s)
    "P1": (16, 12, 24, 12, 10),
    "P2": (24, 20, 36, 20, 16),
    "P3": (12, 28, 16, 28, 8),
}
SEEDS = (0, 1)
ITERS, REPS = 60, 3
GATE = None  # bounds concurrent gem5 launches from this process


def LOG(*a):
    print(*a, flush=True)


def paired(a0, a1, f, ld, s):
    return iq_set(
        [
            (a0, [("alu", 2)]),
            (a1, [("alu", 2), ("muldiv", 2)]),
            (f, [("fp", 2)]),
            (ld, [("rd", 2)]),
            (s, [("rdwr", 1)]),
        ]
    )


def capacity(layout, cls):
    a0, a1, f, ld, s = LAYOUTS[layout]
    return {"add": a0 + a1, "mul": a1, "fp": f, "load": ld + s, "store": s}[cls]


def pmap(f, xs):
    xs = list(xs)
    if not xs:
        return []
    with ThreadPoolExecutor(len(xs)) as ex:
        return list(ex.map(f, xs))


def chain_for(npoints, chase=1):
    need = npoints * ITERS * REPS * chase * 2  # 2x margin: visited nodes stay far from the chain size
    b = 1 << 20
    while b // ASM_EMITTERS[ISA].line < need:
        b <<= 1
    return b


def curve(name, filler, ns, params, sets, seed, pre=None, m=0):
    """One gem5 launch: the scheduler probe at every N in `ns`."""
    ks = [sched_probe(filler, n, pre, m, ISA) for n in ns]
    with GATE:
        res, outdir = run_multi(f"p15_{name}", ks, params, sets, ITERS, REPS, chain_for(len(ns)), seed)
    vals = {n: warm(res[i]) for i, n in enumerate(ns)}
    return {
        "ns": list(ns),
        "values": [vals[n] for n in ns],
        "raw": [res[i] for i in range(len(ns))],
        "seed": seed,
        "truth": config_truth(outdir),
    }


def exp_unified():
    def one(args):
        q, seed = args
        c = curve(f"uni{q}_add_s{seed}", "add", range(1, q + 17), {**BASE, "numIQEntries": q}, [], seed)
        LOG(f"unified Q={q} seed {seed} done")
        return {"q": q, **c}

    return {"base": BASE, "sizes": UNIFIED, "runs": pmap(one, [(q, s) for q in UNIFIED for s in SEEDS])}


def exp_queues():
    def one(args):
        lay, cls, seed = args
        c = curve(
            f"{lay}_{cls}_s{seed}",
            cls,
            range(1, capacity(lay, cls) + 13),
            BASE,
            [paired(*LAYOUTS[lay])],
            seed,
        )
        LOG(f"queues {lay} {cls} seed {seed} done")
        return {"layout": lay, "filler": cls, **c}

    jobs = [(lay, c, s) for lay in LAYOUTS for c in CLASSES for s in SEEDS]
    return {"base": BASE, "layouts": LAYOUTS, "runs": pmap(one, jobs)}


SHARING = [  # (layout, preload X, probe Y, m values)
    ("P2", "mul", "add", [0, 4, 8, 12, 16]),
    ("P2", "add", "mul", [0, 8, 16, 20, 24, 28, 32, 36]),
    ("P2", "fp", "add", [0, 8, 16, 24]),
    ("P2", "store", "load", [0, 4, 8, 12]),
    ("P2", "load", "store", [0, 8, 12, 16, 18, 20, 24, 28]),
    ("U48", "load", "add", [0, 4, 8, 12, 16]),
]


def exp_sharing():
    def one(args):
        lay, x, y, m = args
        if lay == "U48":
            params, sets, cap = {**BASE, "numIQEntries": 48}, [], 48
        else:
            params, sets, cap = BASE, [paired(*LAYOUTS[lay])], capacity(lay, y)
        c = curve(f"{lay}_{x}{m}_{y}", y, range(1, cap + 7), params, sets, 0, pre=x, m=m)
        LOG(f"sharing {lay} {x}->{y} m={m} done")
        return {"layout": lay, "x": x, "y": y, "m": m, **c}

    jobs = [(lay, x, y, m) for lay, x, y, ms in SHARING for m in ms]
    return {"base": BASE, "layouts": LAYOUTS, "pairs": SHARING, "runs": pmap(one, jobs)}


# RISC-V nop is addi x0 and takes an IntAlu entry, so the trailing work is fadds
DISPATCH_Q = [16, 24, 32, 40, 48]
DISPATCH_S = [160, 152, 168]  # three add-block lengths: the free run wobbles with the block's phase
DISPATCH_M = list(range(0, 321, 16))


def split_layout(q_int):
    return iq_set([(q_int, [("alu", 2)]), (16, [("muldiv", 2)]), (128, [("fp", 4)]), (32, [("rdwr", 2)])])


def exp_dispatch():
    def one(args):
        q, sat = args
        ks = [dispatch_probe(sat, m, isa=ISA) for m in DISPATCH_M]
        name = f"p15_disp_q{q}" + ("" if sat == 160 else f"_s{sat}")
        with GATE:
            res, outdir = run_multi(name, ks, BASE, [split_layout(q)], 100, REPS, 256 << 10, 0)
        LOG(
            f"dispatch q={q} S={sat}: "
            + " ".join(f"{m}:{warm(res[i]):.0f}" for i, m in enumerate(DISPATCH_M))
        )
        return {
            "q_int": q,
            "S": sat,
            "ms": DISPATCH_M,
            "values": [warm(res[i]) for i in range(len(ks))],
            "raw": [res[i] for i in range(len(ks))],
            "truth": config_truth(outdir),
        }

    return {
        "base": BASE,
        "S": DISPATCH_S,
        "runs": pmap(one, [(q, s) for q in DISPATCH_Q for s in DISPATCH_S]),
    }


EXPS = {"unified": exp_unified, "queues": exp_queues, "sharing": exp_sharing, "dispatch": exp_dispatch}

# --isa aarch64 validates the AArch64 kernels: three sizes, layout P2, the add->mul hinge, one seed
ARM_UNIFIED = [16, 32, 64]
ARM_LAYOUTS = {"P2": LAYOUTS["P2"]}
ARM_SHARING = [("P2", "add", "mul", [0, 8, 16, 20, 24, 28, 32])]
ARM_DISPATCH_Q = [16, 32, 48]


def main():
    global GATE, ISA, OUT, BASE, UNIFIED, LAYOUTS, SHARING, DISPATCH_Q, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", choices=list(EXPS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    args = ap.parse_args()
    GATE = threading.BoundedSemaphore(args.jobs)
    if args.isa == "aarch64":
        ISA, OUT = "aarch64", os.path.join(ROOT, "results", "gem5", "arm-sched-p1_5")
        BASE = {**BASE, "numPhysVecRegs": 512}  # ARM scalar FP renames into the vector pool
        UNIFIED, LAYOUTS, SHARING, DISPATCH_Q, SEEDS = (
            ARM_UNIFIED,
            ARM_LAYOUTS,
            ARM_SHARING,
            ARM_DISPATCH_Q,
            (0,),
        )
    os.makedirs(OUT, exist_ok=True)

    def one(name):
        t0 = time.time()
        res = EXPS[name]()
        res["elapsed_s"] = round(time.time() - t0)
        with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=0, default=str)
        LOG(f"== {name} done in {res['elapsed_s']} s")

    pmap(one, args.exps)


if __name__ == "__main__":
    main()
