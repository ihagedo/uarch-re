#!/usr/bin/env python3
"""P0.6 smoke test: every window-probe variant cross-compiles and runs on gem5 ARM.

Every filler under every delay at depth 2, ROB 64, at N = 16 (fits) and N = 96 (past the
ROB and LSQ). A variant passes if N = 96 costs >= 1.3x N = 16 and its clock probe reads
~1 cycle per dependent add. Writes results/gem5/<tag>/smoke.csv.

  validation/gem5/arm_smoke.py [--jobs 12] [--tag arm-smoke-variants]
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import uarchre.backends.platform.gem5 as g5
from uarchre.kernels import FILLERS, MIXES
from uarchre.kernels.window import window_probe
from arm_sweep import ARM_CONFIG

DELAYS = ["miss", "fp", "int"]
NS = (16, 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tag", default="arm-smoke-variants")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    g5.CONFIG = ARM_CONFIG
    fillers = list(FILLERS) + list(MIXES)
    points = [(f, d, n) for f in fillers for d in DELAYS for n in NS]

    def one(p):
        f, d, n = p
        k = window_probe(f, n, depth=args.depth, delay=d)
        try:
            r = g5.run_gem5_reps(
                k, {"numROBEntries": 64}, chain_bytes=4 << 20, iters=40, reps=2, isa="aarch64"
            )
            return p, r, ""
        except Exception as e:  # report every failure, keep going
            return p, None, str(e).splitlines()[0][:200]

    with ThreadPoolExecutor(args.jobs) as ex:
        res = {p: (r, err) for p, r, err in ex.map(one, points)}

    out = os.path.join(ROOT, "results", "gem5", args.tag)
    os.makedirs(out, exist_ok=True)
    rows, n_ok = [], 0
    print(
        f"{'filler':7s} {'delay':5s} {'c/it N=16':>10s} {'c/it N=96':>10s} {'ratio':>6s} "
        f"{'probe':>6s}  result"
    )
    for f in fillers:
        for d in DELAYS:
            (lo, e1), (hi, e2) = res[(f, d, NS[0])], res[(f, d, NS[1])]
            if lo is None or hi is None:
                print(f"{f:7s} {d:5s} {'':>10s} {'':>10s} {'':>6s} {'':>6s}  FAIL {e1 or e2}")
                rows.append(dict(filler=f, delay=d, ok=False, error=e1 or e2))
                continue
            ratio = hi.best / lo.best
            probe = hi.clock_ratio_post
            ok = ratio > 1.3 and probe is not None and 0.95 <= probe <= 1.1
            n_ok += ok
            print(
                f"{f:7s} {d:5s} {lo.best:10.1f} {hi.best:10.1f} {ratio:6.2f} {probe:6.3f}  "
                f"{'ok' if ok else 'FAIL'}"
            )
            rows.append(
                dict(
                    filler=f,
                    delay=d,
                    ok=ok,
                    cpi_n16=lo.best,
                    cpi_n96=hi.best,
                    ratio=round(ratio, 3),
                    clock_ratio_pre=hi.clock_ratio_pre,
                    clock_ratio_post=probe,
                    error="",
                )
            )
    with open(os.path.join(out, "smoke.csv"), "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "filler",
                "delay",
                "ok",
                "cpi_n16",
                "cpi_n96",
                "ratio",
                "clock_ratio_pre",
                "clock_ratio_post",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n{n_ok}/{len(rows)} variants ok; wrote {os.path.relpath(out, ROOT)}/smoke.csv")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
