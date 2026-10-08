#!/usr/bin/env python3
"""P3.1 fetch and decode width on gem5 O3.

One of fetchWidth / decodeWidth is w in {2, 3, 4, 6, 8}, every other width 8. width_loop(L):
L = 2..16 instructions in one fetch block ending in a taken branch; filler "mix" is NOPs and
fadd.d ("nop" alone caps at gem5's 6 integer ALUs). Width = round(max L / cycles per iteration);
a staircase ceil(L/w) marks fetch (stops at a taken branch), a ramp L/w a queue-fed stage.

    . tools/env.sh && python3 validation/gem5/fe_width.py [--isa aarch64]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_common import run_fe, pmap, config_value, write_rows, write_json, med, pop_isa, out_dir
from uarchre.kernels.frontend import width_loop, width_estimate, SPELLING

WIDTHS = [2, 3, 4, 6, 8]
ARM_WIDTHS = {"fetch": [3, 4, 8], "decode": [3, 4, 6]}  # --isa aarch64: a reduced sweep
LS = list(range(2, 17))
LAPS = 2000
FILLERS = ["mix", "nop"]


def configs(isa="riscv64"):
    widths = {"fetch": WIDTHS, "decode": WIDTHS} if isa == "riscv64" else ARM_WIDTHS
    cfg = [("fetch", w, {"fetchWidth": w}) for w in widths["fetch"]]
    # decodeWidth=8 is the all-8 config, already run as fetch 8; reuse it
    cfg += [("decode", w, {"decodeWidth": w}) for w in widths["decode"] if w != 8]
    return cfg


def main(argv=()):
    isa, _ = pop_isa(argv)
    out = out_dir("fe-width", isa)
    jobs = [(f, stage, w, p, L) for f in FILLERS for stage, w, p in configs(isa) for L in LS]

    def one(j):
        f, stage, w, p, L = j
        r = run_fe(width_loop(L, filler=f, isa=SPELLING[isa]), laps=LAPS, reps=3, params=p)
        key = "fetchWidth" if stage == "fetch" else "decodeWidth"
        took = config_value(r["outdir"], "system.cpu", key)
        assert took == str(w), (stage, w, took)
        return [
            {"filler": f, "stage": stage, "width": w, "L": L, "rep": i, "cycles_per_iter": c, "ipc": ipc}
            for i, (c, ipc) in enumerate(zip(r["cpu"], r["ipc"]))
        ]

    done = [0]

    def log(i, res):
        done[0] += 1
        if done[0] % 15 == 0:
            print(f"  {done[0]}/{len(jobs)}", flush=True)

    rows = [r for rs in pmap(one, jobs, log=log) for r in rs]
    write_rows(os.path.join(out, "raw.csv"), rows)

    summary = []
    for f, (stage, w, _) in [(f, c) for f in FILLERS for c in configs(isa)]:
        cpi = [
            med(
                [
                    r["cycles_per_iter"]
                    for r in rows
                    if r["filler"] == f and r["stage"] == stage and r["width"] == w and r["L"] == L
                ]
            )
            for L in LS
        ]
        est = width_estimate(LS, cpi)
        expect = "staircase" if (stage == "fetch" or w == 8) else "ramp"
        summary.append(
            {
                "filler": f,
                "stage": stage,
                "truth": w,
                **est,
                "cpi": cpi,
                "Ls": LS,
                "error": est["width"] - w,
                "shape_ok": est["shape"] == expect,
                "expected_shape": expect,
            }
        )
        print(
            f"{f:3s} {stage:6s} truth {w}  detected {est['width']}  peak ipc {est['peak_ipc']:.3f}  "
            f"shape {est['shape']} (expect {expect})",
            flush=True,
        )
    write_json(os.path.join(out, "summary.json"), summary)


if __name__ == "__main__":
    main(sys.argv[1:])
