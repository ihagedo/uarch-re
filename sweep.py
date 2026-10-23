#!/usr/bin/env python3
"""Run window-probe sweeps on a platform and analyze them.

  ./sweep.py toy    --tag rob64 --toy rob=64,int_prf=72,lq=24,sq=16 --fillers nop,int,load,store
  ./sweep.py native --tag host  --fillers nop,int,fp,load,store --n 8:400:8
  ./sweep.py gem5   --tag rob128 --gem5 numROBEntries=128 --fillers nop

Writes results/<platform>/<tag>/{sweep_raw.csv,sweep.csv,knees.json,spec.yaml,sweep.png}.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uarchre.kernels.window import window_probe
from uarchre.analysis.knee import STRUCT, attribute
from uarchre.analysis.bootstrap import bootstrap_knee
from uarchre.analysis.protocol import run_protocol
from uarchre.analysis import anomaly
from uarchre.spec import spec_from_estimates, write_spec

OUTER_REPS = {"toy": 1, "gem5": 3, "native": 5}
EMITTER = {"toy": "abstract", "gem5": "riscv64", "native": "x86_64"}


def parse_range(s):
    a, b, step = (int(v) for v in s.split(":"))
    return list(range(a, b + 1, step))


def parse_kv(s):
    d = {}
    for kv in (s or "").split(","):
        if kv:
            k, v = kv.split("=")
            d[k] = float(v) if "." in v else int(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["toy", "native", "gem5"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fillers", default="nop,int,load,store")
    ap.add_argument("--n", default="4:96:4")
    ap.add_argument("--toy", default="", help="ToyConfig overrides k=v,...")
    ap.add_argument("--gem5", default="", help="gem5 O3 param overrides k=v,...")
    ap.add_argument("--set", action="append", default=[], help="gem5 dotted-path override, repeatable")
    ap.add_argument("--isa", default=None, help="gem5 ISA: riscv64 (default) or aarch64")
    ap.add_argument("--chain-mb", type=int, default=64)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--depth", type=int, default=1, help="dependent delay ops per chain group")
    ap.add_argument(
        "--delay",
        default="miss",
        choices=["miss", "fp", "int"],
        help="miss = pointer-chase misses; fp / int = dependent ALU delay chain",
    )
    ap.add_argument(
        "--outer-reps",
        type=int,
        default=None,
        help="process launches per point (default: toy 1, gem5 3, native 5)",
    )
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed; chain seed of outer rep r is seed + r")
    ap.add_argument("--jobs", type=int, default=1, help="concurrent launches (toy/gem5 only)")
    ap.add_argument("--pin-cpu", type=int, default=None, help="native: taskset to this CPU")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ns = parse_range(args.n)
    fillers = args.fillers.split(",")
    outer_reps = args.outer_reps or OUTER_REPS[args.platform]
    if args.platform == "native" and args.jobs > 1:
        sys.exit("--jobs > 1 on native hardware makes the runs perturb each other")
    outdir = os.path.join("results", args.platform, args.tag)
    os.makedirs(outdir, exist_ok=True)

    truth = None
    skipped = []
    if args.platform == "toy":
        from dataclasses import replace
        from uarchre.backends.platform.toy import run_toy, ToyConfig

        cfg = ToyConfig(**parse_kv(args.toy))
        truth = cfg.headroom()
        run = lambda k, rep: run_toy(k, replace(cfg, seed=cfg.seed + args.seed + rep))
    elif args.platform == "native":
        from uarchre.backends.platform.native import run_native_reps

        run = lambda k, rep: run_native_reps(
            k,
            chain_bytes=args.chain_mb << 20,
            iters=args.iters or 20000,
            reps=5,
            seed=args.seed + rep,
            pin_cpu=args.pin_cpu,
        )
    else:
        from uarchre.backends.platform.gem5 import run_gem5_reps, gem5_truth

        params = parse_kv(args.gem5)
        truth = gem5_truth(params, args.isa or "riscv64")
        isa = args.isa or "riscv64"
        EMITTER["gem5"] = isa

        def run(k, rep):
            # gem5 v25.1 deterministically asserts in timebuf.hh at some exact timings:
            # retry once with 3 more iterations, then skip the point and record it
            for extra in (0, 3):
                try:
                    return run_gem5_reps(
                        k,
                        params,
                        chain_bytes=args.chain_mb << 20,
                        iters=(args.iters or 300) + extra,
                        seed=args.seed + rep,
                        sets=args.set,
                        isa=isa,
                    )
                except RuntimeError as err:
                    print(
                        f"  gem5 failed on {k.name} rep {rep} (iters +{extra}): "
                        f"{str(err).splitlines()[-1][:120]}",
                        flush=True,
                    )
            skipped.append({"kernel": k.name, "rep": rep})
            return None

    # an ALU delay group retires link by link, so its step is a ramp: read its onset
    min_jump = 1 + 0.4 / args.depth if args.delay == "miss" else 1.3
    steps = 1 if args.depth == 1 else 2
    t0 = time.time()
    records, reps = run_protocol(
        lambda f, n, rep: run(window_probe(f, n, depth=args.depth, delay=args.delay), rep),
        fillers,
        ns,
        outer_reps=outer_reps,
        seed=args.seed,
        min_jump=min_jump,
        expected_steps=steps,
        jobs=args.jobs,
        log=lambda m: print(m, flush=True),
    )
    knees = {
        f: bootstrap_knee(reps[f], min_jump=min_jump, seed=args.seed, expected_steps=steps) for f in fillers
    }
    for f, k in knees.items():
        print(
            f"{f}: knee={k.knee} ci={k.ci} ({k.ci_method}, {k.n_reps} reps) jump={k.jump_ratio:.2f} "
            f"q={k.quality:.2f} {k.anomaly or ''} {' '.join(k.flags)}"
        )

    with open(os.path.join(outdir, "sweep_raw.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "filler",
                "depth",
                "delay",
                "n",
                "outer_rep",
                "inner_rep",
                "cycles_per_iter",
                "clock_ratio_pre",
                "clock_ratio_post",
                "chain_seed",
                "order_index",
                "t_ns",
            ]
        )
        for r in records:
            for i, c in enumerate(r.result.cycles):
                w.writerow(
                    [
                        r.key,
                        args.depth,
                        args.delay,
                        r.n,
                        r.outer_rep,
                        i,
                        c,
                        r.result.clock_ratio_pre,
                        r.result.clock_ratio_post,
                        args.seed + r.outer_rep,
                        r.order_index,
                        r.t_ns,
                    ]
                )
    rows = [
        {"filler": f, "n": n, "cycles_per_iter": statistics.median(v)}
        for f in fillers
        for n, v in sorted(reps[f].items())
    ]
    with open(os.path.join(outdir, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filler", "n", "cycles_per_iter"])
        w.writeheader()
        w.writerows(rows)

    ctx = {
        "emitter": EMITTER[args.platform],
        "platform": args.platform,
        "depth": args.depth,
        "delay": args.delay,
    }
    est = attribute({f: k for f, k in knees.items() if f in STRUCT}, ctx)  # movimm: knee only
    # anomaly rules A1-A12 on this run's curves; the registry alone sets the flags
    widths = (
        {"default": cfg.width}
        if args.platform == "toy"
        else {"default": 8}
        if args.platform == "gem5"
        else None
    )
    rs = anomaly.runset_from_knees(
        knees,
        reps,
        depth=args.depth,
        delay=args.delay,
        platform=args.platform,
        core=args.tag,
        ctx=ctx,
        widths=widths,
        order={f: {(r.n, r.outer_rep): r.order_index for r in records if r.key == f} for f in fillers},
        clock=[
            {"rep": r.outer_rep, "pre": r.result.clock_ratio_pre, "post": r.result.clock_ratio_post}
            for r in records
        ],
    )
    findings = anomaly.evaluate(rs)
    anomaly.apply(est, findings, replace=True)
    report = {
        "platform": args.platform,
        "tag": args.tag,
        "config": args.toy or args.gem5,
        "sets": args.set,
        "context": ctx,
        "protocol": {"outer_reps": outer_reps, "seed": args.seed, "coarse": args.n, "points": len(records)},
        "truth": truth,
        "knees": {f: k.to_dict() for f, k in knees.items()},
        "estimates": est,
        "findings": [f.to_dict() for f in findings],
        "skipped": skipped,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(outdir, "knees.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    spec = spec_from_estimates(
        f"{args.platform}/{args.tag}",
        {"native": "x86_64", "toy": "riscv64"}.get(args.platform, EMITTER["gem5"]),
        est,
        os.path.join(outdir, "sweep.csv"),
        documented=truth,
    )
    write_spec(os.path.join(outdir, "spec.yaml"), spec)

    print("\nestimates:")
    for s, e in est.items():
        t = f"  (truth {truth[s]})" if truth and s in truth else ""
        off = e["offset"]["value"]
        print(
            f"  {s:18s} {str(e['value']):>8s} ci {e['ci']}  offset {'none' if off is None else f'{off:g}'} "
            f"[{e['offset']['calibration']}]  {e['status']} {' '.join(e['anomalies'])}{t}"
        )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for f in fillers:
            pts = [(r["n"], r["cycles_per_iter"]) for r in rows if r["filler"] == f]
            (line,) = ax.plot(*zip(*pts), marker=".", label=f)
            if outer_reps > 1:
                xs = [n for n, v in reps[f].items() for _ in v]
                ys = [y for v in reps[f].values() for y in v]
                ax.scatter(xs, ys, s=4, color=line.get_color(), alpha=0.3, lw=0)
            if knees[f].knee is not None:
                ax.axvspan(knees[f].ci[0], knees[f].ci[1] + 1, color=line.get_color(), alpha=0.15, lw=0)
                ax.axvline(knees[f].knee, ls=":", color="gray")
        ax.set_xlabel("filler count N")
        ax.set_ylabel("cycles / iteration")
        ax.set_title(f"window probe: {args.platform} / {args.tag}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sweep.png"), dpi=120)
    print(f"\nwrote {outdir}/")


if __name__ == "__main__":
    main()
