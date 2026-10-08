#!/usr/bin/env python3
"""Do the window probe's chains miss to DRAM? (P0.4)

Times the window kernel with no fillers (two independent pointer chases) at 1-512 MiB
per chain on each CPU, in TSC ticks and ns per iteration. Latency should plateau at
DRAM well past the 24 MiB L3; the campaigns use 256 MiB per chain.

  targets/x86/footprint.py --cpus 2,14 --tag x86-footprint   # results/native/<tag>/footprint.{csv,json}
"""

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)

import numpy as np  # noqa: E402
from uarchre.kernels.window import window_probe  # noqa: E402
from uarchre.backends.isa_base import run_args, parse_output  # noqa: E402
from uarchre.backends.platform.native import compile_kernel, compile_source, EMITTERS  # noqa: E402
import hygiene  # noqa: E402

TSC_C = r"""
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
int main(void) {
    struct timespec a, b; unsigned aux;
    clock_gettime(CLOCK_MONOTONIC, &a); uint64_t t0 = __rdtscp(&aux);
    do clock_gettime(CLOCK_MONOTONIC, &b);
    while ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec) < 5e8);
    uint64_t t1 = __rdtscp(&aux);
    printf("tsc_ghz %.6f\n", (t1 - t0) / ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)));
    return 0;
}
"""


def tsc_ghz(cpu):
    b = compile_source(TSC_C, "tsc_freq")
    out = subprocess.run(["taskset", "-c", str(cpu), b], capture_output=True, text=True, check=True).stdout
    return float(out.split()[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x86-footprint")
    ap.add_argument("--cpus", default="2,14")
    ap.add_argument("--mb", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    try:
        os.sched_setaffinity(0, set(range(4, 20)) - {14})
    except OSError:
        pass
    outdir = os.path.join("results", "native", args.tag)
    os.makedirs(outdir, exist_ok=True)
    k = window_probe("nop", 0, depth=1)
    b = compile_kernel(k, "x86_64")
    ghz = tsc_ghz(int(args.cpus.split(",")[0]))
    rows = []
    for cpu in [int(c) for c in args.cpus.split(",")]:
        sib = hygiene.siblings(cpu)
        for mb in [int(m) for m in args.mb.split(",")]:
            nodes = (mb << 20) // 64
            iters = min(args.iters, nodes // args.reps - 1)
            for L in range(args.launches):
                hygiene.wait_quiet(sib + [cpu], 0.1, timeout=900 if sib else 0.5)
                cmd = [
                    "taskset",
                    "-c",
                    str(cpu),
                    b,
                    *run_args(EMITTERS["x86_64"], k, mb << 20, iters, args.reps, None, L),
                ]
                out, info = hygiene.run_monitored(cmd, sib + [cpu])
                r = parse_output(out)
                rows.append(
                    {
                        "cpu": cpu,
                        "mb_per_chain": mb,
                        "launch": L,
                        "iters": iters,
                        "ticks_median": r.median,
                        "ns_median": r.median / ghz,
                        "clock_ratio_pre": r.clock_ratio_pre,
                        "clock_ratio_post": r.clock_ratio_post,
                        "load1": os.getloadavg()[0],
                        "share": info["share"],
                    }
                )
                print(rows[-1], flush=True)
    with open(os.path.join(outdir, "footprint.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for cpu in sorted({r["cpu"] for r in rows}):
        summ[cpu] = {
            mb: float(
                np.median([r["ns_median"] for r in rows if r["cpu"] == cpu and r["mb_per_chain"] == mb])
            )
            for mb in sorted({r["mb_per_chain"] for r in rows})
        }
    json.dump(
        {
            "tsc_ghz": ghz,
            "ns_per_iter": summ,
            "note": "two independent chains per iteration: "
            "one iteration costs about one load latency when both miss",
        },
        open(os.path.join(outdir, "footprint.json"), "w"),
        indent=1,
    )
    print(json.dumps(summ, indent=1), "tsc_ghz", ghz)


if __name__ == "__main__":
    main()
