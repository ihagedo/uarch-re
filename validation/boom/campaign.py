#!/usr/bin/env python3
"""Window-probe sweeps on BOOM RTL (P0.5, P1.1 at L2).

sweep.py's protocol and analysis with the BOOM driver as the platform, one Verilator
launch per (filler, N, outer rep); outer reps vary the chain seed (the simulator is
deterministic). Writes results/boom/<tag>/ in sweep.py's formats plus sims.csv.

  taskset -c 4-19 validation/boom/campaign.py --config SmallBoomV3Config --tag small-d1 --n 4:48:4
  taskset -c 4-19 validation/boom/campaign.py --all --only small-d2,medium-d2 --jobs 4
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.kernels.window import window_probe
from uarchre.analysis.knee import attribute
from uarchre.analysis.bootstrap import bootstrap_knee
from uarchre.analysis.protocol import run_protocol
from uarchre.spec import spec_from_estimates, write_spec
from uarchre.backends.platform import boom

# tag -> (config, coarse grid, depth); coarse step ~ROB/8, the protocol adds step 1 near the knee.
# depth 2: a DRAM miss (~68 cycles) is about the 2 x ROB / decodeWidth = 64 cycles to dispatch
# an iteration at the knee, so depth-1 steps are small (1.32x on Small, under the 1.4x threshold)
CAMPAIGN = {
    "small-d1": ("SmallBoomV3Config", "4:48:4", 1),  # run with --iters 100 --probe-adds 4096
    "small-d2": ("SmallBoomV3Config", "4:48:4", 2),
    "medium-d2": ("MediumBoomV3Config", "8:96:8", 2),
    "large-d2": ("LargeBoomV3Config", "16:128:8", 2),
    "mega-d2": ("MegaBoomV3Config", "32:168:8", 2),
}


def parse_range(s):
    a, b, step = (int(v) for v in s.split(":"))
    return list(range(a, b + 1, step))


def run_one(
    config,
    tag,
    fillers,
    ns,
    depth,
    outer_reps,
    seed,
    jobs,
    iters,
    reps,
    region_mb,
    chain="preload",
    probe_adds=1024,
    plot=True,
):
    outdir = os.path.join(ROOT, "results", "boom", tag)
    os.makedirs(outdir, exist_ok=True)
    truth = boom.boom_truth(config)
    sims = []

    def measure(filler, n, rep):
        k = window_probe(filler, n, depth=depth)
        src = boom.emit_baremetal(
            k,
            iters=iters,
            reps=reps,
            seed=seed + rep,
            region_bytes=region_mb << 20,
            chain=chain,
            probe_adds=probe_adds,
        )
        elf = boom.compile_baremetal(src, k.name + ("" if chain == "preload" else f"_{chain}"))
        t = time.time()
        out, cycles = boom.run_sim(elf, config)
        sims.append(
            dict(filler=filler, n=n, outer_rep=rep, sim_cycles=cycles, wall_s=round(time.time() - t, 1))
        )
        return boom.parse_output(out)

    min_jump = 1 + 0.4 / depth
    steps = 1 if depth == 1 else 2
    t0 = time.time()
    records, repsd = run_protocol(
        measure,
        fillers,
        ns,
        outer_reps=outer_reps,
        seed=seed,
        min_jump=min_jump,
        expected_steps=steps,
        jobs=jobs,
        log=lambda m: print(m, flush=True),
    )
    knees = {f: bootstrap_knee(repsd[f], min_jump=min_jump, seed=seed, expected_steps=steps) for f in fillers}
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
                        depth,
                        "miss",
                        r.n,
                        r.outer_rep,
                        i,
                        c,
                        r.result.clock_ratio_pre,
                        r.result.clock_ratio_post,
                        seed + r.outer_rep,
                        r.order_index,
                        r.t_ns,
                    ]
                )
    rows = [
        {"filler": f, "n": n, "cycles_per_iter": statistics.median(v)}
        for f in fillers
        for n, v in sorted(repsd[f].items())
    ]
    with open(os.path.join(outdir, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filler", "n", "cycles_per_iter"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "sims.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filler", "n", "outer_rep", "sim_cycles", "wall_s"])
        w.writeheader()
        w.writerows(sorted(sims, key=lambda d: (d["filler"], d["n"], d["outer_rep"])))

    ctx = {"emitter": "riscv64", "platform": "boom", "depth": depth, "delay": "miss"}
    est = attribute(knees, ctx)
    report = {
        "platform": "boom",
        "tag": tag,
        "config": config,
        "context": ctx,
        "chipyard": boom.CHIPYARD,
        "harness": {
            "iters": iters,
            "inner_reps": reps,
            "probe_adds": probe_adds,
            "chain_region_bytes": region_mb << 20,
            "chain": chain,
            "memory": "SimDRAM + DRAMSim2 DDR3-1333, 500 MHz core",
        },
        "protocol": {
            "outer_reps": outer_reps,
            "seed": seed,
            "coarse": f"{ns[0]}:{ns[-1]}:{ns[1] - ns[0]}",
            "points": len(records),
        },
        "truth": truth,
        "knees": {f: k.to_dict() for f, k in knees.items()},
        "estimates": est,
        "elapsed_s": round(time.time() - t0, 1),
        "sim_wall_s_total": round(sum(s["wall_s"] for s in sims), 1),
    }
    with open(os.path.join(outdir, "knees.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    spec = spec_from_estimates(
        f"boom/{config}", "riscv64", est, os.path.join(outdir, "sweep.csv"), documented=truth
    )
    write_spec(os.path.join(outdir, "spec.yaml"), spec)

    print("\nestimates:")
    for s, e in est.items():
        t = f"  (truth {truth[s]})" if s in truth else ""
        print(
            f"  {s:18s} {str(e['value']):>8s} ci {e['ci']}  offset {e['offset']['value']} "
            f"[{e['offset']['calibration']}]  {e['status']} {' '.join(e['anomalies'])}{t}"
        )

    if plot:
        from uarchre import viz

        plt = viz.setup()
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, f in enumerate(fillers):
            pts = [(r["n"], r["cycles_per_iter"]) for r in rows if r["filler"] == f]
            ax.plot(*zip(*pts), label=f, **viz.series(i), markersize=4)
            if knees[f].knee is not None:
                ax.axvline(knees[f].knee, ls=":", color=viz.INK2)
        ax.axvline(truth["rob"], color=viz.MUTED, lw=1, ls=(0, (2, 2)))
        ax.set(
            xlabel="filler count N", ylabel="cycles / iteration", title=f"window probe: BOOM {config} ({tag})"
        )
        ax.legend()
        fig.savefig(os.path.join(outdir, "sweep.png"), dpi=120)
    print(f"\nwrote {outdir}/")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SmallBoomV3Config", choices=sorted(boom.BOOM_V3))
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true", help="run every CAMPAIGN entry whose simulator exists")
    ap.add_argument("--only", default="", help="with --all: comma-separated CAMPAIGN tags")
    ap.add_argument("--fillers", default="nop")
    ap.add_argument("--n", default="4:48:4")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--outer-reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3, help="inner reps per launch")
    ap.add_argument("--region-mb", type=int, default=2, help="address span of each chain")
    ap.add_argument("--chain", default="preload", choices=boom.CHAIN_MODES)
    ap.add_argument("--probe-adds", type=int, default=1024, help="clock-probe adds before and after the reps")
    args = ap.parse_args()
    fillers = args.fillers.split(",")
    if args.all:
        only = set(filter(None, args.only.split(",")))
        for tag, (config, n, depth) in CAMPAIGN.items():
            if (only and tag not in only) or not os.path.exists(boom.sim_path(config)):
                if not only or tag in only:
                    print(f"skip {tag}: {config} not built (validation/boom/build.sh {config})")
                continue
            run_one(
                config,
                tag,
                fillers,
                parse_range(n),
                depth,
                args.outer_reps,
                args.seed,
                args.jobs,
                args.iters,
                args.reps,
                args.region_mb,
                args.chain,
                args.probe_adds,
            )
    else:
        if not args.tag:
            sys.exit("--tag is required without --all")
        run_one(
            args.config,
            args.tag,
            fillers,
            parse_range(args.n),
            args.depth,
            args.outer_reps,
            args.seed,
            args.jobs,
            args.iters,
            args.reps,
            args.region_mb,
            args.chain,
            args.probe_adds,
        )


if __name__ == "__main__":
    main()
