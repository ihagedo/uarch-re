#!/usr/bin/env python3
"""P1.7 retire bandwidth on gem5 O3 (RISC-V or ARM): three probes, commitWidth 1..8.

  drain   Handley's retire drain: cycles/iter = a + K / retire width
  window  P1.1 window probe, 64-entry ROB: slope past the knee = 2 / min(retire, rename)
  stream  K NOPs alone: 1 / slope is the narrowest stage (6 int ALUs on RISC-V; ARM nops skip the IQ)
CONTROLS run at commitWidth 8 with another stage changed; a retire method reads 8 on all.

  validation/gem5/retire_p1_7.py --jobs 12 [--isa aarch64]  # results/gem5/[arm-]retire-p1_7/data.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sched_common import ROOT, run_multi, config_truth, iq_set
from uarchre.kernels.scheduler import retire_drain, nop_stream, from_window
from uarchre.kernels.window import window_probe

OUT = os.path.join(ROOT, "results", "gem5", "retire-p1_7")
ISA = "riscv64"
WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8]
WINDOW_WIDTHS = [2, 4, 6, 8]
CONTROLS = {  # name: (top-level params, --set overrides)
    "F4C8": ({"fetchWidth": 4, "decodeWidth": 4, "renameWidth": 4, "dispatchWidth": 4, "commitWidth": 8}, []),
    "R4C8": ({"renameWidth": 4, "commitWidth": 8}, []),
    # 8 integer ALUs instead of the default 6: retire is then the narrowest stage at width 8
    "C8A8": ({"commitWidth": 8}, [iq_set([(256, [("alu", 8), ("muldiv", 2), ("fp", 4), ("rdwr", 4)])])]),
}
DRAIN_K = list(range(0, 257, 16))
WINDOW_N = list(range(32, 161, 4))
STREAM_K = [64, 128, 192, 256]
DRAIN_BASE = {"numROBEntries": 512, "numPhysIntRegs": 80}  # int headroom 48 (ARM 38) < 64 adds
WINDOW_BASE = {"numROBEntries": 64}
ARM_WIDTHS = [2, 4, 6, 8]  # --isa aarch64 validates the AArch64 kernels: these widths, control F4C8


def warm(cycles):
    """Median over inner reps after the first (rep 0 pays cold I-cache misses)."""
    c = cycles[1:] if len(cycles) > 1 else cycles
    return float(statistics.median(c))


def launches():
    cfgs = [(f"C{c}", {"commitWidth": c}, []) for c in WIDTHS] + [(n, p, s) for n, (p, s) in CONTROLS.items()]
    for name, p, sets in cfgs:
        pts = [("drain", k, retire_drain(k, isa=ISA)) for k in DRAIN_K] + [
            ("stream", k, nop_stream(k, ISA)) for k in STREAM_K
        ]
        yield (
            name,
            p,
            pts,
            dict(params={**DRAIN_BASE, **p}, sets=sets, iters=100, reps=3, chain_bytes=256 << 10),
        )
        if name in CONTROLS or p["commitWidth"] in WINDOW_WIDTHS:
            pts = [("window", n, from_window(window_probe("nop", n), ISA)) for n in WINDOW_N]
            yield (
                name,
                p,
                pts,
                dict(params={**WINDOW_BASE, **p}, sets=sets, iters=150, reps=3, chain_bytes=2 << 20),
            )


def main():
    global ISA, OUT, WIDTHS, CONTROLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    args = ap.parse_args()
    if args.isa == "aarch64":
        ISA, OUT = "aarch64", os.path.join(ROOT, "results", "gem5", "arm-retire-p1_7")
        WIDTHS, CONTROLS = ARM_WIDTHS, {"F4C8": CONTROLS["F4C8"]}
    os.makedirs(OUT, exist_ok=True)
    todo = list(launches())
    print(f"{len(todo)} gem5 launches, {sum(len(t[2]) for t in todo)} points", flush=True)
    t0 = time.time()

    def one(job):
        name, p, pts, kw = job
        res, outdir = run_multi(f"p17_{name}_{pts[0][0]}", [k for _, _, k in pts], **kw)
        t = config_truth(outdir)
        truth = {
            k: t[k]
            for k in (
                "commitWidth",
                "renameWidth",
                "fetchWidth",
                "decodeWidth",
                "dispatchWidth",
                "numROBEntries",
                "numPhysIntRegs",
            )
        }
        truth["int_alus"] = sum(
            f["count"] for q in t["instQueues"] for f in q["fus"] if f["ops"] == ["IntAlu"]
        )
        rows = [
            {
                "method": m,
                "config": name,
                "params": p,
                "x": x,
                "cycles": res[i],
                "value": warm(res[i]),
                "truth": truth,
            }
            for i, (m, x, _) in enumerate(pts)
        ]
        print(
            f"[{time.time() - t0:5.0f}s] {name} {pts[0][0]}: "
            + " ".join(f"{r['x']}:{r['value']:.1f}" for r in rows),
            flush=True,
        )
        return rows

    with ThreadPoolExecutor(args.jobs) as ex:
        rows = [r for rs in ex.map(one, todo) for r in rs]
    with open(os.path.join(OUT, "data.json"), "w") as fh:
        json.dump({"rows": rows, "elapsed_s": round(time.time() - t0)}, fh, indent=0)
    print(f"wrote {OUT}/data.json")


if __name__ == "__main__":
    main()
