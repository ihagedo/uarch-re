#!/usr/bin/env python3
"""Native x86 window-probe campaign with launch hygiene (P0.4, P5.1, P4.5).

sweep.py's native path for a shared hybrid host: --cpu takes a taskset list ("2,14" lets the scheduler
pick a P or E core), launches with a busy SMT sibling or preemption are retried (all attempts go to
launches.csv), and v:<variant> fillers come from uarchre/kernels/x86_micro.py. Writes results/native/<tag>/.

  targets/x86/window_campaign.py --tag x86-p-miss-d1 --cpu 2 --core p --n 16:640:16 --depth 1
  targets/x86/window_campaign.py --tag x86-p45-units --cpu 2 --core p --fillers v:nop8,v:br_nop7 --n 4:100:4
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)

import numpy as np  # noqa: E402
from uarchre.kernels.window import window_probe  # noqa: E402
from uarchre.kernels.x86_micro import VariantX86, VARIANTS  # noqa: E402
from uarchre.analysis.knee import attribute, find_knee  # noqa: E402
from uarchre.analysis.bootstrap import bootstrap_knee  # noqa: E402
from uarchre.analysis.protocol import run_protocol  # noqa: E402
from uarchre.analysis import offsets as offsets_mod  # noqa: E402
from uarchre.backends.isa_base import emit_c, run_args, parse_output  # noqa: E402
from uarchre.backends.platform.native import compile_kernel, compile_source, EMITTERS  # noqa: E402
from uarchre.spec import spec_from_estimates, write_spec  # noqa: E402
import hygiene  # noqa: E402
import published  # noqa: E402

DRIVER_CPUS = set(range(4, 20)) - {14}  # keep the driver and gcc off the measured cores


def parse_range(s):
    a, b, step = (int(v) for v in s.split(":"))
    return list(range(a, b + 1, step))


_bins = {}


def binary(key, n, depth, delay):
    if key.startswith("v:"):
        var = key[2:]
        k = window_probe("int", n, depth=depth, delay=delay)
        k.name = f"{k.name}_{var}"
        if k.name not in _bins:
            _bins[k.name] = compile_source(emit_c(k, VariantX86(var)), k.name)
        return k, _bins[k.name]
    k = window_probe(key, n, depth=depth, delay=delay)
    if k.name not in _bins:
        _bins[k.name] = compile_kernel(k, "x86_64")
    return k, _bins[k.name]


def toy_offsets(fillers, depth, delay, cache):
    """Toy offsets for an ALU-delay depth that offsets.yaml lacks, fitted as in
    validation/toy/calibrate_offsets.py (5 sizes). attribute() still reports
    them as uncalibrated."""
    from uarchre.backends.platform.toy import run_toy, ToyConfig
    from uarchre.analysis.knee import STRUCT

    if os.path.exists(cache):
        return json.load(open(cache))
    big = dict(rob=512, int_prf=32 + 480, fp_prf=32 + 480, lq=256, sq=256)
    knob = {"nop": "rob", "int": "int_prf", "fp": "fp_prf", "load": "lq", "store": "sq"}
    entries = []
    for f in fillers:
        if f not in knob:
            continue
        diffs = []
        for size in [s for s in (32, 48, 64, 96, 128, 160, 192, 256) if s >= 3 * depth][:5]:
            cfg = ToyConfig(**{**big, knob[f]: size + (32 if knob[f].endswith("prf") else 0)})
            ns = list(range(max(2, size - 2 * depth - 16), size + 7))
            ys = [run_toy(window_probe(f, n, depth=depth, delay=delay), cfg) for n in ns]
            k = find_knee(ns, ys, min_jump=1.3, expected_steps=2)
            if k.knee is not None:
                diffs.append(cfg.headroom()[STRUCT[f]] - k.knee)
        if len(diffs) >= 3:
            entries.append(
                {
                    "id": f"abstract.toy.{f}.d{depth}.{delay}",
                    "emitter": "abstract",
                    "platform": "toy",
                    "filler": f,
                    "depth": depth,
                    "delay": delay,
                    "offset": round(float(np.median(diffs))),
                    "offset_sd": round(float(np.std(diffs)), 3),
                    "n_configs": len(diffs),
                    "evidence": "targets/x86/window_campaign.py toy_offsets (in-memory)",
                }
            )
    json.dump(entries, open(cache, "w"), indent=1)
    return entries


def per_rep_knees(reps, min_jump, steps):
    """Knee of each outer rep's own curve (run-to-run spread)."""
    nrep = min(len(v) for v in reps.values())
    out = []
    for r in range(nrep):
        ns = sorted(reps)
        k = find_knee(ns, [reps[n][r] for n in ns], min_jump, steps)
        out.append(k.knee)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpu", required=True, help="taskset CPU list, e.g. 2 or 14 or 2,14")
    ap.add_argument(
        "--core", choices=["p", "e", "float"], required=True, help="which published truth applies"
    )
    ap.add_argument("--fillers", default="nop,int,fp,load,store")
    ap.add_argument("--n", default="16:640:16")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--delay", default="miss", choices=["miss", "fp", "int"])
    ap.add_argument("--chain-mb", type=int, default=256, help="per chain; two chains per binary")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--inner-reps", type=int, default=5)
    ap.add_argument("--outer-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max-sibling", type=float, default=0.10, help="retry if the SMT sibling was busier")
    ap.add_argument("--min-share", type=float, default=0.90, help="retry if cpu/wall fell below")
    ap.add_argument("--trace", action="store_true", help="record which CPUs each launch ran on")
    ap.add_argument(
        "--no-wait-own",
        action="store_true",
        help="wait only for the SMT sibling, not for the measured CPUs to be idle",
    )
    ap.add_argument(
        "--quiet-timeout",
        type=float,
        default=20.0,
        help="max seconds to wait for an idle SMT sibling before each launch",
    )
    args = ap.parse_args()

    try:
        os.sched_setaffinity(0, DRIVER_CPUS)
    except OSError:
        pass
    fillers = args.fillers.split(",")
    ns = parse_range(args.n)
    outdir = os.path.join("results", "native", args.tag)
    os.makedirs(outdir, exist_ok=True)
    cpus = [int(c) for c in args.cpu.split(",")]
    watch = sorted({s for c in cpus for s in hygiene.siblings(c)} - set(cpus)) + cpus
    sib = sorted({s for c in cpus for s in hygiene.siblings(c)} - set(cpus))
    alu = args.delay != "miss"
    min_jump = 1.3 if alu else 1 + 0.4 / args.depth
    steps = 2 if (alu or args.depth > 1) else 1
    launches = []

    def measure(key, n, rep):
        k, b = binary(key, n, args.depth, args.delay)
        cmd = [
            "taskset",
            "-c",
            args.cpu,
            b,
            *run_args(
                EMITTERS["x86_64"], k, args.chain_mb << 20, args.iters, args.inner_reps, None, args.seed + rep
            ),
        ]
        for attempt in range(args.retries + 1):
            # the SMT sibling and the measured CPU itself (other users' unpinned work lands there too)
            waited = hygiene.wait_quiet(
                sib + ([] if args.no_wait_own else cpus), args.max_sibling, timeout=args.quiet_timeout
            )
            load1 = os.getloadavg()[0]
            out, info = hygiene.run_monitored(cmd, watch, trace=args.trace)
            sib_busy = max([info["busy"].get(s, 0.0) for s in sib] or [0.0])
            clean = sib_busy <= args.max_sibling and info["share"] >= args.min_share
            res = parse_output(out)
            launches.append(
                {
                    "key": key,
                    "n": n,
                    "outer_rep": rep,
                    "attempt": attempt,
                    "clean": clean,
                    "sibling_busy": sib_busy,
                    "waited_s": round(waited, 2),
                    "load1": load1,
                    **{k2: v for k2, v in info.items() if k2 != "busy"},
                    "busy": json.dumps(info["busy"]),
                    "median": res.median,
                    "ratio_pre": res.clock_ratio_pre,
                    "ratio_post": res.clock_ratio_post,
                    "t": time.time(),
                }
            )
            if clean:
                break
            time.sleep(0.3 * (attempt + 1))
        return res

    t0 = time.time()
    log = open(os.path.join(outdir, "run.log"), "w")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    say(
        f"{args.tag}: cpu {args.cpu} (sibling {sib}) fillers {fillers} n {args.n} depth {args.depth} "
        f"delay {args.delay} chain {args.chain_mb} MB x2"
    )
    records, reps = run_protocol(
        measure,
        fillers,
        ns,
        outer_reps=args.outer_reps,
        seed=args.seed,
        min_jump=min_jump,
        expected_steps=steps,
        jobs=1,
        log=say,
    )
    knees = {
        f: bootstrap_knee(reps[f], min_jump=min_jump, seed=args.seed, expected_steps=steps) for f in fillers
    }
    rep_knees = {f: per_rep_knees(reps[f], min_jump, steps) for f in fillers}
    for f, k in knees.items():
        say(
            f"{f}: knee={k.knee} ci={k.ci} ({k.ci_method}, {k.n_reps} reps) jump={k.jump_ratio:.2f} "
            f"q={k.quality:.2f} {k.anomaly or ''} {' '.join(k.flags)} per-rep {rep_knees[f]}"
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
    with open(os.path.join(outdir, "launches.csv"), "w", newline="") as fh:
        keys = list(launches[0].keys()) + (
            ["cpus_seen"] if args.trace and "cpus_seen" not in launches[0] else []
        )
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for L in launches:
            w.writerow({**L, "cpus_seen": json.dumps(L.get("cpus_seen", []))} if args.trace else L)

    # standard fillers -> structure estimates (toy fallback offsets, flagged uncalibrated)
    std = {f: k for f, k in knees.items() if not f.startswith("v:")}
    ctx = {"emitter": "x86_64", "platform": "native", "depth": args.depth, "delay": args.delay}
    table = offsets_mod.load()
    if alu:
        have = {(e["filler"], e["depth"], e["delay"]) for e in table if e["emitter"] == "abstract"}
        need = [f for f in std if (f, args.depth, args.delay) not in have]
        if need:
            cache = os.path.join("results", "native", f"x86-toy-offsets-d{args.depth}-{args.delay}.json")
            table = table + toy_offsets(need, args.depth, args.delay, cache)
    have = {
        e["filler"]
        for e in table
        if e["emitter"] == "abstract" and e["depth"] == args.depth and e["delay"] == args.delay
    }
    no_off = [f for f in std if f not in have]  # the toy found no clean knee for these either
    est = attribute({f: k for f, k in std.items() if f not in no_off}, ctx, offsets=table) if std else {}
    for f in no_off:
        est[f"raw:{f}"] = {
            "filler": f,
            "value": None,
            "ci": None,
            "knee": std[f].knee,
            "status": f"no toy offset for {f} at depth {args.depth} delay {args.delay}",
            "offset": {"value": None, "sd": None, "calibration": None},
            "anomalies": [],
        }

    rows_a8 = [
        {
            "key": r.key,
            "n": r.n,
            "outer_rep": r.outer_rep,
            "order_index": r.order_index,
            "value": r.result.median,
        }
        for r in records
    ]
    a8 = hygiene.drift_rules(
        rows_a8, [(r.result.clock_ratio_pre, r.result.clock_ratio_post) for r in records]
    )
    last = {}
    for L in launches:
        last[(L["key"], L["n"], L["outer_rep"])] = L
    hyg = {
        "launches": len(launches),
        "points": len(records),
        "retried": sum(1 for L in launches if L["attempt"] > 0),
        "unclean_final": sum(1 for L in last.values() if not L["clean"]),
        "unclean_attempts": sum(1 for L in launches if not L["clean"]),
        "sibling_busy_max": max(L["sibling_busy"] for L in launches),
        "waited_s_total": round(sum(L["waited_s"] for L in launches), 1),
        "load1": {
            "min": min(L["load1"] for L in launches),
            "median": float(np.median([L["load1"] for L in launches])),
            "max": max(L["load1"] for L in launches),
        },
        "share_min": min(L["share"] for L in launches),
        "clock_ratio": {
            "pre_median": float(np.median([r.result.clock_ratio_pre for r in records])),
            "post_median": float(np.median([r.result.clock_ratio_post for r in records])),
            "pre_range": [
                float(min(r.result.clock_ratio_pre for r in records)),
                float(max(r.result.clock_ratio_pre for r in records)),
            ],
        },
        "a8": a8,
    }
    truth = published.truth(args.core)
    report = {
        "platform": "native",
        "tag": args.tag,
        "cpu": args.cpu,
        "core": args.core,
        "context": ctx,
        "protocol": {
            "outer_reps": args.outer_reps,
            "inner_reps": args.inner_reps,
            "iters": args.iters,
            "seed": args.seed,
            "coarse": args.n,
            "points": len(records),
            "chain_mb_each": args.chain_mb,
            "min_jump": min_jump,
            "expected_steps": steps,
        },
        "truth": truth,
        "truth_sources": published.SOURCES.get(args.core),
        "knees": {f: k.to_dict() for f, k in knees.items()},
        "per_rep_knees": rep_knees,
        "estimates": est,
        "hygiene": hyg,
        "elapsed_s": round(time.time() - t0, 1),
        "variants": {f: VARIANTS[f[2:]][2] for f in fillers if f.startswith("v:")},
    }
    with open(os.path.join(outdir, "knees.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    if any(not k.startswith("raw:") for k in est):
        spec = spec_from_estimates(
            f"native/{args.tag}",
            "x86_64",
            est,
            os.path.join(outdir, "sweep.csv"),
            documented={k: v for k, v in truth.items() if k in ("rob", "lq", "sq")},
        )
        write_spec(os.path.join(outdir, "spec.yaml"), spec)
    say("\nestimates:")
    for s, e in est.items():
        t = f"  (published {truth[s]})" if s in truth else ""
        say(
            f"  {s:18s} {str(e['value']):>8s} ci {e['ci']}  offset {e['offset']['value']} "
            f"[{e['offset']['calibration']}]  {e['status']} {' '.join(e['anomalies'])}{t}"
        )
    say(f"hygiene: {json.dumps({k: v for k, v in hyg.items() if k != 'a8'})}")
    say(f"A8: {json.dumps(a8)}")
    say(f"wrote {outdir}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
