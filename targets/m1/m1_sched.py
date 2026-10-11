#!/usr/bin/env python3
"""Scheduler queues, queue pairing, dispatch probe (P1.5) and retire bandwidth (P1.7) on Apple silicon.

  python3 targets/m1/m1_sched.py [P1.5] [P1.7] [--parts sizes,pairing,dispatch] [--fp-filler both]
  python3 targets/m1/m1_sched.py --gem5-smoke --tag smoke

A launch times every point of one curve (scheduler.multi_source) in a fresh random order; ticks
become core cycles with the runner's clock bracket around that launch.
"""

import argparse
import os
import statistics
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native  # noqa: E402

sys.path.insert(0, os.path.join(native.ROOT, "validation", "gem5"))
from sched_common import hinge, hockey, ols, sharing_verdict, warm, width  # noqa: E402
from uarchre.analysis.bootstrap import bootstrap_knee  # noqa: E402
from uarchre.analysis.knee import find_knee  # noqa: E402
from uarchre.analysis.protocol import _fine_grid  # noqa: E402
from uarchre.kernels.scheduler import (  # noqa: E402
    A64AsmEmitter,
    dispatch_probe,
    multi_source,
    nop_stream,
    parse_multi,
    retire_drain,
    sched_probe,
)

FAMILY = "sched"
ISA = "aarch64"
LINE = 128
MIN_JUMP = 1.3
SHARING_MIN_JUMP = 1.2  # a preload that spills into the probe's queue raises the low plateau

# coarse grids (lo, hi, step): published queues plus dispatch buffer +-30 %, wide where unpublished
SIZES = {
    "add": (8, 256, 8),
    "mul": (8, 64, 4),
    "fp": (16, 192, 8),
    "fadd": (16, 192, 8),
    "load": (8, 96, 4),
    "store": (8, 96, 4),
}
# filler class -> (backend.scheduler_queues field, queues of that size it can use, dispatch buffer)
QUEUES = {
    "add": (None, 0, "int"),
    "mul": ("mul", 2, "mul"),
    "fp": ("fp", 4, None),
    "fadd": ("fp", 4, None),
    "load": ("ls", 1, "ls"),  # ls is the total, 4 x 12
    "store": ("ls", 1, "ls"),
}
PAIRS = [  # (preload class X, probe class Y, preloaded X ops), m below X's own capacity
    ("mul", "add", [0, 4, 8, 12, 16, 20]),
    ("add", "mul", [0, 8, 16, 24, 32, 40, 48, 56, 64]),
    ("fp", "add", [0, 16, 32, 48]),
    ("store", "load", [0, 4, 8, 12, 16, 20]),
    ("load", "store", [0, 8, 16, 24, 32, 40]),
]
# S saturating ops, then M rename-only nops; S x (1 - rate / 8) must exceed what waits behind rename
DISPATCH = {"add": [768, 800, 832], "mul": [120, 128, 136]}
DISPATCH_M = list(range(0, 449, 16))
DRAIN_K = list(range(0, 1025, 64))
DRAIN = {"links": 24, "exhaust": 528}  # 24 fsqrt (~312 cycles) outlast the fill; 528 adds > int PRF
STREAM_K = [512, 1024, 1536, 2048]
ITERS = {"sched": 11000, "dispatch": 56000, "drain": 12000, "stream": 60000}  # >= ~1 ms per rep
REPS = 4  # inner reps per point; rep 0 warms the code

SMOKE = {
    "sizes": {
        **{c: (40, 240, 40) for c in ("add", "mul", "fp", "fadd")},
        "load": (8, 48, 8),
        "store": (8, 48, 8),
    },
    "pairs": [("mul", "add", [0, 32]), ("store", "load", [0, 8])],
    "dispatch": {"add": [160]},
    "dispatch_m": list(range(0, 193, 32)),
    "drain_k": [0, 32, 64, 96],
    "drain": {"links": 12, "exhaust": 256},
    "stream_k": [64, 128, 192, 256],
    "iters": {k: 40 for k in ITERS},
    "reps": 2,
    "outer_reps": 2,
    "chain": 1 << 20,
    "fine": False,
    "jobs": 8,
}
DEFAULTS = {
    "parts": "sizes,pairing,dispatch",
    "fp_filler": "fmov",
    "outer_reps": 3,
    "chain_mb": 256,
    "seed": 0,
}


def opt(args, k):
    v = getattr(args, k, None)  # m1_kernels.py passes only the common options
    return DEFAULTS[k] if v is None else v


def log(*a):
    print(*a, flush=True)


def config(r, args):
    if r.smoke:
        c = dict(SMOKE)
    else:
        c = {
            "sizes": SIZES,
            "pairs": PAIRS,
            "dispatch": DISPATCH,
            "dispatch_m": DISPATCH_M,
            "drain_k": DRAIN_K,
            "drain": DRAIN,
            "stream_k": STREAM_K,
            "iters": ITERS,
            "reps": REPS,
            "outer_reps": opt(args, "outer_reps"),
            "chain": opt(args, "chain_mb") << 20,
            "fine": True,
            "jobs": 1,
        }
    f = opt(args, "fp_filler")
    c["fp"] = ["fp", "fadd"] if r.smoke or f == "both" else (["fadd"] if f == "fadd" else ["fp"])
    c["seed"] = opt(args, "seed")
    return c


def pmap(f, xs, jobs):
    if jobs <= 1:
        return [f(x) for x in xs]
    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(f, xs))


def launch(r, cfg, label, kernels, iters, chain, seed):
    """Time `kernels` in one launch: (warm core cycles/iter per kernel, launch record)."""
    src = multi_source(label, kernels, em=A64AsmEmitter(r.counter))
    out, ratio = r.launch(r.compile(src, label), [chain, iters, cfg["reps"], seed], label=label)
    res = parse_multi(out, len(kernels))
    if not all(res.values()):
        raise RuntimeError(f"{label}: points missing from the output:\n{out[-1500:]}")
    att = next((a for a in reversed(r.attempts) if a["label"] == label), {})
    rec = {"label": label, "ratio": ratio, "accepted": att.get("accepted", True), "ghz": att.get("ghz")}
    rec["ticks"] = [res[i] for i in range(len(kernels))]
    return [warm(res[i]) / ratio for i in range(len(kernels))], rec


def batch(r, cfg, label, make, xs, rng, iters, seed, depth):
    """One pass over xs in random order, split so that no launch wraps its chase chains."""
    xs = [xs[i] for i in rng.permutation(len(xs))]
    chain = cfg["chain"] if depth else 1 << 20
    per = 64 if not depth else max(1, int(0.9 * (chain // LINE) / (depth * iters * cfg["reps"])))
    vals, recs = {}, []
    for c in range(0, len(xs), per):
        part = xs[c : c + per]
        v, rec = launch(r, cfg, f"{label}_{c // per}", [make(x) for x in part], iters, chain, seed)
        vals.update(zip(part, v))
        recs.append({"x": part, **rec})
        g = f", {rec['ghz']:.2f} GHz" if rec["ghz"] else ""
        log(f"  {rec['label']}: {len(part)} points{g}{'' if rec['accepted'] else ', not accepted'}")
    return vals, recs


def rng_for(cfg, key):
    return np.random.default_rng([cfg["seed"], zlib.crc32(key.encode())])


def knee_curve(r, cfg, key, make, grid, min_jump):
    """Coarse pass, a step-1 pass around the provisional knee, then outer reps over both."""
    lo, hi, step = grid
    ns = list(range(lo, hi + 1, step))
    rng = rng_for(cfg, key)
    reps, recs = {}, []

    def run(xs, phase, rep):
        vals, rs = batch(
            r, cfg, f"sched_{key}_{phase}", make, xs, rng, cfg["iters"]["sched"], cfg["seed"] + rep, 1
        )
        for n, v in vals.items():
            reps.setdefault(n, []).append(v)
        recs.extend(rs)

    run(ns, "c", 0)
    prov = find_knee(ns, [reps[n][0] for n in ns], min_jump)
    fine = _fine_grid(ns, prov, step) if cfg["fine"] else []
    if fine:
        run(fine, "f", 0)
    for rep in range(1, cfg["outer_reps"]):
        run(ns + fine, f"r{rep}", rep)
    k = bootstrap_knee(reps, min_jump=min_jump, seed=cfg["seed"])
    raw = {"grid": list(grid), "fine": fine, "reps": reps, "knee": k.to_dict(), "launches": recs}
    return k, raw


def clock_flags(raw):
    return [] if all(x["accepted"] for x in raw["launches"]) else ["clock.unaccepted_launch"]


def cite(*paths):
    """firestorm.yaml fields with their citation keys."""
    out = []
    for path in paths:
        keys = (native.published(path) or {}).get("sources", [])
        out.append(f"{path} ({', '.join(keys)})" if keys else path)
    return ", ".join(out)


def queue_published(cls):
    """(entries the class can use per firestorm.yaml or None, note, yaml fields)."""
    q, count, db = QUEUES[cls]
    node = native.published(f"backend.scheduler_queues.{q}") if q else None
    dbn = native.published(f"decoupling.dispatch_buffers.{db}") if db else None
    src = [f"backend.scheduler_queues.{q}"] if node else []
    if node:
        pub = node["value"] * count
        note = f"published {f'{count} x ' if count > 1 else ''}{node['value']} ({node.get('note', '')})"
    else:
        pub, note = None, "queue sizes not in firestorm.yaml"
    if dbn:
        src.append(f"decoupling.dispatch_buffers.{db}")
        note += f", dispatch buffer {dbn['value']}"
    return pub, note, cite(*src)


FILLER_NOTE = {"fp": "filler fmov d, x", "fadd": "fadds behind one fmov per group"}


def sizes(r, cfg):
    classes = ["add", "mul", *cfg["fp"], "load", "store"]

    def one(c):
        k, raw = knee_curve(r, cfg, c, lambda n: sched_probe(c, n, isa=ISA), cfg["sizes"][c], MIN_JUMP)
        log(f"P1.5 sizes {c}: knee {k.knee} ci {k.ci} ({k.n_reps} reps) {k.anomaly or ''}")
        return k, raw

    rows, raw = [], {}
    for c, (k, rw) in zip(classes, pmap(one, classes, cfg["jobs"])):
        pub, pnote, src = queue_published(c)
        offset = 0
        note = f"raw knee {k.knee}, offset {offset} applied (gem5: +2, its IQ holds the two miss loads; "
        note += f"the M1 offset needs ground truth); {pnote}"
        note += "; paired queues read as their sum"
        if c in FILLER_NOTE:
            note += f"; {FILLER_NOTE[c]}"
        rows.append(
            {
                "structure": f"scheduler.{c}",
                "value": None if k.knee is None else k.knee + offset,
                "ci": None if k.ci is None else [k.ci[0] + offset, k.ci[1] + offset],
                "unit": "entries",
                "published": pub,
                "source": src,
                "flags": k.flags + clock_flags(rw) + (["knee.none"] if k.knee is None else []),
                "note": note + (f"; {k.anomaly}" if k.anomaly else ""),
            }
        )
        raw[c] = {**rw, "offset": offset}
    return rows, raw


def break_ci(ms, knees, rng, n=400):
    """95 % interval of the pairing break from the per-m bootstrap knees."""
    if any(k.samples is None for k in knees):
        return None
    bs = []
    for _ in range(n):
        pts = [(m, k.samples[rng.integers(len(k.samples))]) for m, k in zip(ms, knees)]
        pts = [(m, v) for m, v in pts if np.isfinite(v)]
        if len(pts) >= 2:
            bs.append(hockey(*zip(*pts))[0])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else None


def pairing(r, cfg):
    curves = [(x, y, m) for x, y, ms in cfg["pairs"] for m in ms]

    def one(cv):
        x, y, m = cv
        make = lambda n: sched_probe(y, n, x, m, ISA)
        k, raw = knee_curve(r, cfg, f"{x}{m}_{y}", make, cfg["sizes"][y], SHARING_MIN_JUMP)
        log(f"P1.5 pairing {x}{m} -> {y}: knee {k.knee} ci {k.ci}")
        return k, raw

    res = dict(zip(curves, pmap(one, curves, cfg["jobs"])))
    rows, raw = [], {}
    for x, y, ms in cfg["pairs"]:
        got = [(m, res[(x, y, m)][0]) for m in ms if res[(x, y, m)][0].knee is not None]
        rw = {"ms": ms, "curves": {m: res[(x, y, m)][1] for m in ms}}
        launches = [rec for m in ms for rec in res[(x, y, m)][1]["launches"]]
        row = {
            "structure": f"scheduler.pairing.{x}->{y}",
            "value": None,
            "ci": None,
            "unit": f"{x} ops",
            "published": None,
            "source": "",
            "flags": clock_flags({"launches": launches}),
            "note": f"no {y} knee at any m",
        }
        if len(got) >= 2:
            gm, gk = zip(*got)
            b, c, slope = hockey(gm, [k.knee for k in gk])
            row["value"] = float(b)
            row["ci"] = break_ci(gm, gk, rng_for(cfg, f"{x}_{y}_break"))
            row["note"] = (
                f"{sharing_verdict(ms, b)}: {y} knee {[k.knee for k in gk]} at m = {list(gm)}, "
                f"slope {slope:.2f} past the break"
            )
            if b >= max(ms):
                row["flags"].append("pairing.no_break")
            rw.update(knees=[k.knee for k in gk], break_=b, level=c, slope=slope)
        rows.append(row)
        raw[f"{x}->{y}"] = rw
    return rows, raw


DISPATCH_DB = {"add": "int", "mul": "mul", "fp": None}


def dispatch(r, cfg):
    jobs = [(x, s) for x, blocks in cfg["dispatch"].items() for s in blocks]
    ms = cfg["dispatch_m"]

    def one(job):
        x, s = job
        rng, reps, recs = rng_for(cfg, f"disp_{x}{s}"), {}, []
        make = lambda m: dispatch_probe(s, m, x=x, y="nop", isa=ISA)
        for rep in range(cfg["outer_reps"]):
            vals, rs = batch(
                r,
                cfg,
                f"sched_disp_{x}{s}_r{rep}",
                make,
                ms,
                rng,
                cfg["iters"]["dispatch"],
                cfg["seed"] + rep,
                0,
            )
            for m, v in vals.items():
                reps.setdefault(m, []).append(v)
            recs.extend(rs)
        y = np.array([statistics.median(reps[m]) for m in ms])
        f, base, slope = hinge(np.asarray(ms, float), y)
        fit = {"x": x, "S": s, "ms": ms, "values": y.tolist(), "reps": reps, "launches": recs}
        fit.update(
            free=f, base=base, slope=slope, rate_x=s / base, t_free=f * slope, q_raw=f * slope * s / base
        )
        fit["rate_y"] = 1 / slope if slope > 0 else None
        log(f"P1.5 dispatch {x} S={s}: free {f:.0f} nops, {s / base:.2f} {x}/cycle -> {fit['q_raw']:.1f}")
        return fit

    fits = pmap(one, jobs, cfg["jobs"])
    rows, raw = [], {}
    for x in cfg["dispatch"]:
        fx = [f for f in fits if f["x"] == x]
        q = [f["q_raw"] for f in fx]
        db = DISPATCH_DB.get(x)
        node = native.published(f"decoupling.dispatch_buffers.{db}") if db else None
        flags = clock_flags({"launches": [rec for f in fx for rec in f["launches"]]})
        if any(f["free"] >= 0.9 * (max(ms) - 16) for f in fx):
            flags.append("fit.no_hinge")
        if any(f["free"] < 8 for f in fx):
            flags.append("fit.no_free_run")
        rate = statistics.mean(f["rate_x"] for f in fx)
        rows.append(
            {
                "structure": f"dispatch.{x}",
                "value": float(np.mean(q)),
                "ci": [float(min(q)), float(max(q))],
                "unit": "entries",
                "published": node["value"] if node else None,
                "source": cite(f"decoupling.dispatch_buffers.{db}") if node else "",
                "flags": flags,
                "note": f"free nops x {rate:.2f} {x}/cycle behind S = {[f['S'] for f in fx]} (range over S); "
                f"t0 = 0 applied (pipeline refill, uncalibrated on M1); counts each {x} queued after rename",
            }
        )
        raw[x] = fx
    return rows, raw


def run_sched(r, args):
    cfg = config(r, args)
    parts = opt(args, "parts").split(",")
    rows, raw = [], {"config": {k: v for k, v in cfg.items() if k != "jobs"}}
    for name, fn in (("sizes", sizes), ("pairing", pairing), ("dispatch", dispatch)):
        if name in parts:
            rs, rw = fn(r, cfg)
            rows += rs
            raw[name] = rw
    return raw, rows


def slope_curve(r, cfg, key, make, ks):
    """Outer reps of one curve; least-squares slope of cycles/iter against K."""
    rng, reps, recs = rng_for(cfg, key), {}, []
    for rep in range(cfg["outer_reps"]):
        vals, rs = batch(
            r, cfg, f"sched_{key}_r{rep}", make, ks, rng, cfg["iters"][key], cfg["seed"] + rep, 0
        )
        for k, v in vals.items():
            reps.setdefault(k, []).append(v)
        recs.extend(rs)
    y = [statistics.median(reps[k]) for k in ks]
    b, a, se = ols(ks, y)
    w, ci = width(b, se, 1)
    h = len(ks) // 2
    lo, hi = ols(ks[: h + 1], y[: h + 1])[0], ols(ks[h:], y[h:])[0]
    flags = [] if lo > 0 and 0.67 < hi / lo < 1.5 else ["shape.slope_change"]
    flags += clock_flags({"launches": recs})
    log(f"P1.7 {key}: 1/slope {w:.2f} ci [{ci[0]:.2f}, {ci[1]:.2f}]")
    fit = {"ks": ks, "values": y, "reps": reps, "slope": b, "intercept": a, "slope_se": se}
    return {**fit, "rate": w, "ci": ci, "half_slopes": [lo, hi], "flags": flags, "launches": recs}


def run_retire(r, args):
    cfg = config(r, args)
    d = cfg["drain"]
    curves = {
        "drain": (lambda k: retire_drain(k, d["links"], d["exhaust"], ISA), cfg["drain_k"]),
        "stream": (lambda k: nop_stream(k, ISA), cfg["stream_k"]),
    }
    fits = dict(zip(curves, pmap(lambda kv: slope_curve(r, cfg, kv[0], *kv[1]), curves.items(), cfg["jobs"])))
    rate = native.published("backend.retire.retire_rate")["value"]
    per_row = native.published("backend.retire.group_max_uops")["value"]
    dr, st = fits["drain"], fits["stream"]
    ks = cfg["drain_k"]
    rows = [
        {
            "structure": "retire.drain",
            "value": dr["rate"],
            "ci": dr["ci"],
            "unit": "nops/cycle",
            "published": rate * per_row,
            "source": cite("backend.retire.retire_rate", "backend.retire.group_max_uops"),
            "flags": dr["flags"],
            "note": f"1 / slope, K = {ks[0]}..{ks[-1]} nops between {d['links']} fsqrt, {d['exhaust']} adds",
        },
        {
            "structure": "retire.rate",
            "value": dr["rate"] / per_row,
            "ci": [c / per_row for c in dr["ci"]],
            "unit": "rows/cycle",
            "published": rate,
            "source": cite("backend.retire.retire_rate"),
            "flags": dr["flags"],
            "note": f"drain rate / {per_row} nops per retire row",
        },
        {
            "structure": "nop_stream",
            "value": st["rate"],
            "ci": st["ci"],
            "unit": "nops/cycle",
            "published": native.published("frontend.fetch_width")["value"],
            "source": cite("frontend.fetch_width"),
            "flags": st["flags"],
            "note": f"1 / slope over K = {cfg['stream_k']} nops per iteration: the narrowest stage",
        },
    ]
    raw = {"config": {k: v for k, v in cfg.items() if k != "jobs"}, **fits}
    return raw, rows


POCS = {"P1.5": run_sched, "P1.7": run_retire}
EST_MIN = {"P1.5": 8, "P1.7": 1}


def build_parser():
    ap = native.add_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("pocs", nargs="*", metavar="POC", help=f"default: all of {', '.join(POCS)}")
    ap.add_argument("--parts", default=DEFAULTS["parts"], help="P1.5 parts to run")
    ap.add_argument(
        "--fp-filler",
        choices=["fmov", "fadd", "both"],
        default=DEFAULTS["fp_filler"],
        help="fp queue filler: fmov d, x, or fadds behind one fmov per group",
    )
    ap.add_argument("--outer-reps", type=int, default=DEFAULTS["outer_reps"])
    ap.add_argument(
        "--chain-mb",
        type=int,
        default=DEFAULTS["chain_mb"],
        help="per chain; M1 needs >= 256 MiB for DRAM misses",
    )
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    bad = [p for p in a.pocs if p not in POCS]
    if bad:
        ap.error(f"unknown PoC {bad}; choose from {list(POCS)}")
    r = native.Runner(a)
    for poc in a.pocs or list(POCS):
        raw, summary = POCS[poc](r, a)
        log(f"{FAMILY} {poc}: wrote {r.save(FAMILY, poc, raw, summary)}")
        for row in summary:
            v, ci, flags = row["value"], row["ci"], " ".join(row["flags"])
            v = f"{v:.2f}" if isinstance(v, float) else v
            ci = [round(c, 2) for c in ci] if ci else ci
            log(f"  {row['structure']}: {v} {row['unit']} ci {ci} published {row['published']} {flags}")
            log(f"    {row['note']}")


if __name__ == "__main__":
    main()
