#!/usr/bin/env python3
"""Window-probe sweeps on Apple silicon, timing only (P5.2, P0.6).

  python3 targets/m1/m1_sweep.py --tag m1-rob-d2 --fillers nop --depth 2 --n 1600:2800:32
  python3 targets/m1/m1_sweep.py --tag m1-lq --fillers load --depth 2 --n 64:192:8

Same protocol and outputs as ./sweep.py, in core cycles from a per-launch add-chain clock
probe. Setup, run list and caveats: README.md.
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.backends import RunResult  # noqa: E402
from uarchre.backends.isa.aarch64 import AArch64  # noqa: E402
from uarchre.backends.isa_base import emit_c, run_args, parse_output  # noqa: E402
from uarchre.backends.platform.native import compile_source  # noqa: E402
from uarchre.kernels.window import window_probe  # noqa: E402
from uarchre.analysis.protocol import run_protocol  # noqa: E402
from uarchre.analysis.bootstrap import bootstrap_knee  # noqa: E402
from uarchre.analysis.knee import attribute  # noqa: E402
from uarchre.spec import spec_from_estimates, write_spec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native import MACOS, machine_info, preflight  # noqa: E402


class M1Runner:
    """One launch per call; returns a RunResult in core cycles."""

    def __init__(self, args):
        self.args = args
        self.em = AArch64("mach" if MACOS else "cntvct")
        self.cc = args.cc or ("clang" if MACOS else "cc")
        self.attempts = []  # every attempt, for sweep_raw.csv

    def compile(self, k):
        return compile_source(
            emit_c(k, self.em),
            k.name + ("_mach" if MACOS else "_cntvct"),
            "aarch64",
            cc=self.cc,
            cflags=("-O1",),
            static=self.args.static,
        )

    def launch(self, k, seed):
        a = self.args
        argv = run_args(self.em, k, a.chain_mb << 20, a.iters, a.inner_reps, a.probe_adds, seed)
        cmd = [self.compile(k), *argv]
        if a.exec_prefix:
            cmd = a.exec_prefix.split() + cmd
        if a.pin_cpu is not None and not MACOS:
            cmd = ["taskset", "-c", str(a.pin_cpu)] + cmd
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        r = parse_output(out)
        hz = re.search(r"^counter_hz ([\d.]+)", out, re.M)
        return r, float(hz.group(1)) if hz else None

    def __call__(self, filler, n, rep):
        a = self.args
        k = window_probe(filler, n, depth=a.depth, delay=a.delay)
        for attempt in range(a.retries + 1):
            r, hz = self.launch(k, a.seed + rep)
            pre, post = r.clock_ratio_pre, r.clock_ratio_post
            ratio = (pre + post) / 2 if pre and post else None
            drift = abs(pre - post) / ratio if ratio else None
            ghz = hz / ratio / 1e9 if (hz and ratio) else None
            ok = drift is not None and drift <= a.max_drift and ghz is not None and ghz >= a.min_ghz
            self.attempts.append(
                dict(
                    filler=filler,
                    n=n,
                    outer_rep=rep,
                    attempt=attempt,
                    ticks=r.cycles,
                    ratio_pre=pre,
                    ratio_post=post,
                    counter_hz=hz,
                    ghz=ghz,
                    drift=drift,
                    accepted=ok,
                    t_ns=time.time_ns(),
                )
            )
            if ok:
                break
            time.sleep(a.pause)
        # convert ticks to core cycles with this launch's own probe
        cycles = [t / ratio for t in r.cycles] if ratio else list(r.cycles)
        return RunResult(cycles, pre, post)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fillers", default="nop")
    ap.add_argument("--n", default="1600:2800:32", help="coarse grid lo:hi:step")
    ap.add_argument("--depth", type=int, default=2, help="delay ops per group; NOP fillers need >= 2 on M1")
    ap.add_argument("--delay", default="miss", choices=["miss", "fp", "int"])
    ap.add_argument(
        "--chain-mb", type=int, default=256, help="per chain; M1 needs >= 256 MiB for DRAM misses"
    )
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--inner-reps", type=int, default=5)
    ap.add_argument("--outer-reps", type=int, default=5)
    ap.add_argument("--probe-adds", type=int, default=None, help="clock probe length (default 4M adds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ghz", type=float, default=2.8, help="below this a launch is not a P-core run")
    ap.add_argument("--max-drift", type=float, default=0.02)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds before a retry")
    ap.add_argument(
        "--pin-cpu", type=int, default=None, help="Linux only: taskset CPU (Asahi M1 P-cores: 4-7)"
    )
    ap.add_argument("--cc", default=None)
    ap.add_argument("--static", action="store_true", help="link statically (Linux only)")
    ap.add_argument(
        "--exec-prefix",
        default="",
        help="functional smoke test on another host, e.g. 'qemu-aarch64' with --static and "
        "--cc aarch64-linux-gnu-gcc; its timings mean nothing",
    )
    ap.add_argument("--force", action="store_true", help="run despite failed preflight checks (recorded)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    a, b, step = (int(v) for v in args.n.split(":"))
    ns = list(range(a, b + 1, step))
    fillers = args.fillers.split(",")
    info = machine_info()
    info["preflight_problems"] = preflight(info, args.force)
    outdir = os.path.join(ROOT, "results", "m1", args.tag)
    os.makedirs(outdir, exist_ok=True)
    runner = M1Runner(args)

    print(f"compiling {len(fillers)} x {len(ns)} kernels with {runner.cc} ...", flush=True)
    for f in fillers:  # compile up front so compile time never lands between timed runs
        for n in ns:
            runner.compile(window_probe(f, n, depth=args.depth, delay=args.delay))

    min_jump = 1 + 0.4 / args.depth if args.delay == "miss" else 1.3
    steps = 1 if args.depth == 1 else 2
    t0 = time.time()
    records, reps = run_protocol(
        runner,
        fillers,
        ns,
        outer_reps=args.outer_reps,
        seed=args.seed,
        min_jump=min_jump,
        expected_steps=steps,
        jobs=1,
        log=lambda m: print(m, flush=True),
    )
    knees = {
        f: bootstrap_knee(reps[f], min_jump=min_jump, seed=args.seed, expected_steps=steps) for f in fillers
    }

    with open(os.path.join(outdir, "sweep_raw.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "filler",
                "depth",
                "delay",
                "n",
                "outer_rep",
                "attempt",
                "accepted",
                "inner_rep",
                "ticks",
                "cycles",
                "clock_ratio_pre",
                "clock_ratio_post",
                "counter_hz",
                "ghz",
                "drift",
                "t_ns",
            ]
        )
        for t in runner.attempts:
            ratio = (t["ratio_pre"] + t["ratio_post"]) / 2 if t["ratio_pre"] and t["ratio_post"] else None
            for i, tk in enumerate(t["ticks"]):
                w.writerow(
                    [
                        t["filler"],
                        args.depth,
                        args.delay,
                        t["n"],
                        t["outer_rep"],
                        t["attempt"],
                        t["accepted"],
                        i,
                        tk,
                        tk / ratio if ratio else "",
                        t["ratio_pre"],
                        t["ratio_post"],
                        t["counter_hz"],
                        t["ghz"],
                        t["drift"],
                        t["t_ns"],
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

    ctx = {"emitter": "aarch64", "platform": "m1", "depth": args.depth, "delay": args.delay}
    est = attribute(knees, ctx)
    accepted = [t for t in runner.attempts if t["accepted"]]
    ghz = [t["ghz"] for t in runner.attempts if t["ghz"]]
    conditions = {
        "attempts": len(runner.attempts),
        "accepted": len(accepted),
        "rejected": len(runner.attempts) - len(accepted),
        "ghz_median": statistics.median(ghz) if ghz else None,
        "ghz_min": min(ghz) if ghz else None,
        "ghz_max": max(ghz) if ghz else None,
        "load_avg_end": os.getloadavg()[0],
    }
    report = {
        "platform": "m1",
        "tag": args.tag,
        "context": ctx,
        "machine": info,
        "conditions": conditions,
        "protocol": {
            "outer_reps": args.outer_reps,
            "inner_reps": args.inner_reps,
            "iters": args.iters,
            "chain_mb": args.chain_mb,
            "seed": args.seed,
            "coarse": args.n,
            "points": len(records),
        },
        "knees": {f: k.to_dict() for f, k in knees.items()},
        "estimates": est,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(outdir, "knees.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(outdir, "machine.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    write_spec(
        os.path.join(outdir, "spec.yaml"),
        spec_from_estimates(f"m1/{args.tag}", "aarch64", est, os.path.join(outdir, "sweep.csv")),
    )

    for f, k in knees.items():
        print(
            f"{f}: knee={k.knee} ci={k.ci} ({k.ci_method}, {k.n_reps} reps) jump={k.jump_ratio:.2f} "
            f"{k.anomaly or ''} {' '.join(k.flags)}"
        )
    print(
        f"launches: {conditions['accepted']} accepted, {conditions['rejected']} rejected; "
        f"clock {conditions['ghz_min']}-{conditions['ghz_max']} GHz"
    )
    for s, e in est.items():
        print(
            f"  {s:18s} {str(e['value']):>8s} ci {e['ci']}  [{e['offset']['calibration']}]  {e['status']} "
            f"{' '.join(e['anomalies'])}"
        )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for f in fillers:
            pts = [(r["n"], r["cycles_per_iter"]) for r in rows if r["filler"] == f]
            ax.plot(*zip(*pts), marker=".", label=f)
            if knees[f].knee is not None:
                ax.axvline(knees[f].knee, ls=":", color="gray")
        ax.set(
            xlabel="filler count N", ylabel="core cycles / iteration", title=f"window probe: m1 / {args.tag}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sweep.png"), dpi=120)
    print(f"\nwrote {os.path.relpath(outdir, ROOT)}/")


if __name__ == "__main__":
    main()
