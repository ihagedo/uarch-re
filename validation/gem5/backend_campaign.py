#!/usr/bin/env python3
"""Backend window-structure campaign on gem5 O3 (RISC-V): P1.1-P1.4.

One ./sweep.py run per entry into results/gem5/<tag>/; a tag with knees.json is skipped, so
the script resumes. Structures not under test are set large (BIG). ALU-delay runs use 12
dependent fsqrt.d (~290 cycles per group) and a 1 MB chain, since nothing is chased.

  validation/gem5/backend_campaign.py [--only p12] [--list]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG = os.path.join(
    os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER')}/uarch-re"), "logs", "backend"
)
BIG = dict(numROBEntries=512, numPhysIntRegs=512, numPhysFloatRegs=512, LQEntries=256, SQEntries=256)


def rng(v, step, below=None, above=24):
    lo = max(2, v // 2 if below is None else v - below)
    return f"{lo}:{v + above}:{step}"


def sweep(tag, gem5, fillers, n, depth=1, delay="miss", reps=1):
    return dict(tag=tag, gem5=gem5, fillers=fillers, n=n, depth=depth, delay=delay, reps=reps)


def campaign():
    c = []
    # P1.1 reorder window: ALU (FP) delay; the miss-delay grid is p11_rob.sh
    for r in [32, 48, 64, 96, 128, 192, 256]:
        c.append(sweep(f"p11-rob{r}-fp12", dict(numROBEntries=r), "nop", rng(r, 8), 12, "fp"))
    # P1.1 seeds: three chain layouts per point (P0.2 on gem5)
    for r, d in [(32, 1), (128, 2), (256, 1)]:
        c.append(sweep(f"p11-rob{r}-d{d}-seeds", dict(numROBEntries=r), "nop", rng(r, 8), d, reps=3))
    # P1.2 register headroom (truth = physical - 32 architectural)
    for h in [32, 64, 96, 128, 192]:
        c.append(sweep(f"p12-int{h}", {**BIG, "numPhysIntRegs": 32 + h}, "int", rng(h, 8)))
        c.append(sweep(f"p12-fp{h}", {**BIG, "numPhysFloatRegs": 32 + h}, "fp", rng(h, 8)))
    # P1.3 load and store queues, miss delay and FP delay
    for q in [16, 24, 32, 48, 64]:
        for delay, depth in [("miss", 1), ("fp", 12)]:
            tag = "" if delay == "miss" else "-fp12"
            c.append(sweep(f"p13-lq{q}{tag}", {**BIG, "LQEntries": q}, "load", rng(q, 4), depth, delay))
            c.append(sweep(f"p13-sq{q}{tag}", {**BIG, "SQEntries": q}, "store", rng(q, 4), depth, delay))
    # P1.4 near-ties and shared pools (ROB 128)
    c.append(
        sweep("p14-tie", {**BIG, "numROBEntries": 128, "numPhysIntRegs": 32 + 124}, "nop,int", rng(128, 8))
    )
    c.append(
        sweep("p14-sep", {**BIG, "numROBEntries": 128, "numPhysIntRegs": 32 + 80}, "nop,int", rng(128, 8, 96))
    )
    c.append(
        sweep(
            "p14-mix",
            {**BIG, "numROBEntries": 128, "numPhysIntRegs": 32 + 96, "numPhysFloatRegs": 32 + 96},
            "nop,int,fp,mix",
            rng(128, 8, 96),
        )
    )
    return c


def run(s, jobs):
    out = os.path.join(ROOT, "results", "gem5", s["tag"])
    if os.path.exists(os.path.join(out, "knees.json")):
        return s["tag"], "skip"
    gem5 = ",".join(f"{k}={v}" for k, v in s["gem5"].items())
    cmd = [
        sys.executable,
        os.path.join(ROOT, "sweep.py"),
        "gem5",
        "--tag",
        s["tag"],
        "--gem5",
        gem5,
        "--fillers",
        s["fillers"],
        "--n",
        s["n"],
        "--depth",
        str(s["depth"]),
        "--delay",
        s["delay"],
        "--chain-mb",
        "4" if s["delay"] == "miss" else "1",
        "--iters",
        "40",
        "--outer-reps",
        str(s["reps"]),
        "--jobs",
        str(jobs),
        "--no-plot",
    ]
    with open(os.path.join(LOG, s["tag"] + ".log"), "w") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return s["tag"], f"exit {r.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parallel", type=int, default=10, help="sweeps at once")
    ap.add_argument("--jobs", type=int, default=4, help="gem5 launches per sweep at once")
    args = ap.parse_args()
    todo = [s for s in campaign() if s["tag"].startswith(args.only)]
    if args.list:
        for s in todo:
            print(s)
        return
    os.makedirs(LOG, exist_ok=True)
    with ThreadPoolExecutor(args.parallel) as ex:
        for tag, status in ex.map(lambda s: run(s, args.jobs), todo):
            print(f"{tag:24s} {status}", flush=True)


if __name__ == "__main__":
    main()
