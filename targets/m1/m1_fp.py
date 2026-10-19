#!/usr/bin/env python3
"""Frontend predictor and decoupling kernels on Apple silicon (P3.4-P3.6, P3.8, P3.9), timing only.

  python3 targets/m1/m1_fp.py [P3.4 P3.9 ...] [--tag m1]       # all five by default
  python3 targets/m1/m1_fp.py P3.9 --serializer "isb"           # another rename barrier for P3.9
  python3 targets/m1/m1_fp.py --gem5-smoke --tag smoke          # tiny sweeps under gem5 ARM

Kernels: uarchre/kernels/predictors.py; knees as in validation/gem5/fp_*.py. Each timed point spans
>= --min-ms; tick counts become core cycles with each launch's clock probe.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native  # noqa: E402

sys.path.insert(0, os.path.join(native.ROOT, "validation", "gem5"))
from fp_common import depth_marginal, history_frac, history_knee, site_knee  # noqa: E402
from uarchre.analysis.anomaly import band_knee  # noqa: E402
from uarchre.analysis.knee import find_knee  # noqa: E402
from uarchre.kernels import predictors as kp  # noqa: E402

FAMILY = "fp"
A64 = "aarch64"
GHZ = 3.2  # Firestorm clock, only to size iterations
REPS = 3  # timed reps per point inside a launch

HIST_GRID = list(range(0, 257, 8))
HIST_SEEDS = (0, 1, 2)
IND_SIZES = sorted({round(8 * 2 ** (i / 4)) for i in range(45)})  # 8..16384 sites, 4 per octave
IND_TARGETS = (1, 2)
RAS_DMAX = 256
RAS_SEEDS = (0, 1, 2)
# 128 B lines: 1024 (128 KiB) fit the 192 KiB L1i, 4096 and 7168 (512, 896 KiB) miss it and hit L2
FD_LINE, FD_LINES, FD_SEEDS = 128, (1024, 4096, 7168), (1, 2, 3)
# landing lines 16 KiB apart share an L1i set (copies > ways); L2 is physically indexed, so they hit
# there, and six descending hops make the hidden miss time visible next to the queue's drain time
FQ_LAYOUT = dict(line_bytes=128, slot=16 << 10, copies=32, spread=3, hops=6)
FQ_PACK, FQ_GRID, FQ_CHAINS = 8, list(range(0, 513, 16)), (16, 18, 20)
FQ_SERIALIZER, FQ_MIN_JUMP, FSQRT = "mrs x9, fpsr", 1.15, 13


def budget(args):
    """Core cycles one timed point should span."""
    return getattr(args, "min_ms", 1.0) * 1e-3 * GHZ * 1e9


def iters_for(args, cycles_per_unit):
    return max(1, math.ceil(budget(args) / cycles_per_unit))


def to_cycles(pts, ratio):
    return {k: [x / ratio for x in v] for k, v in pts.items()}


def measure(r, src, name, label):
    """Compile and launch one harness: ({key: [cycles per unit, per rep]}, ticks per cycle)."""
    out, ratio = r.launch(r.compile(src, name), label=label)
    pts = kp.parse_points(out)
    if not pts:
        raise RuntimeError(f"{label}: no points in output:\n{out[-2000:]}")
    return to_cycles(pts, ratio), ratio


def octaves(sizes, cost):
    """Sizes grouped by octave of their per-unit cost: [(sizes, cost of the cheapest)]."""
    groups = {}
    for s in sorted(sizes):
        groups.setdefault(int(math.log2(cost(s))), []).append(s)
    return [(g, cost(g[0])) for _, g in sorted(groups.items())]


def med(v):
    return float(np.median(v))


def row(structure, k, unit, published, source, note, scale=1, offset=0, flags=()):
    """Summary row from a Knee: value = knee * scale + offset."""
    found = k is not None and k.knee is not None
    return {
        "structure": structure,
        "value": k.knee * scale + offset if found else None,
        "ci": [k.ci[0] * scale + offset, k.ci[1] * scale + offset] if found and k.ci else None,
        "unit": unit,
        "published": published,
        "source": source,
        "flags": (list(k.flags) if k is not None else []) + ([] if found else ["no_knee"]) + list(flags),
        "note": note if found else f"{note}; {getattr(k, 'anomaly', None) or 'no knee'}",
    }


def history_summary(raw):
    seeds = sorted({x["seed"] for x in raw})
    per_seed = [history_frac([x for x in raw if x["seed"] == s]) for s in seeds]
    k, ns = history_knee(per_seed)
    node = native.published("frontend.direction_predictor")
    pub, src = node["history_bits"], ", ".join(node["sources"])
    span = f"N {ns[0]}-{ns[-1]}" if ns else "no N with T_rand > T_const"
    note = f"last N on the low segment of 1 + f, + 1; {len(seeds)} LFSR seeds, {span}"
    return [row("frontend.direction_predictor.history", k, "taken branches", pub, src, note, offset=1)]


def run_history(r, args):
    grid, seeds = ([0, 8, 16, 24, 32], (0, 1)) if r.smoke else (HIST_GRID, HIST_SEEDS)
    iters = 100 if r.smoke else iters_for(args, 12)  # N = 0: ~12 cycles per iteration
    raw = []

    def sweep(ns, phase):
        for s in seeds:
            src = kp.history_source(
                ns, iters=iters, warm=iters, reps=REPS, seed=s, isa=A64, counter=r.counter
            )
            pts, ratio = measure(r, src, f"m1fp_hist_s{s}", f"P3.4 {phase} seed {s}")
            for n in ns:
                p = {key: v for key, v in pts.items() if int(key.split(":")[1]) == n}
                raw.append({"seed": s, "n": n, "phase": phase, "iters": iters, "ratio": ratio, "pts": p})

    sweep(grid, "coarse")
    k, ns = history_knee([history_frac([x for x in raw if x["seed"] == s]) for s in seeds])
    if k.knee is not None and not r.smoke and ns.index(k.knee) + 1 < len(ns):
        a, b = int(k.knee), ns[ns.index(k.knee) + 1]
        if b - a > 1:
            sweep(list(range(a + 1, b)), "fine")
    return raw, history_summary(raw)


def indirect_summary(raw):
    out = []
    for t in sorted({x["targets"] for x in raw}):
        T = {x["s"]: x["cyc"] for x in raw if x["targets"] == t}
        k, margs = site_knee(T)
        note = (
            f"raw knee {k.knee} sites x {t} target(s), not corrected for the estimator's bias "
            f"(reads 3-4 sites high on gem5); S {min(T)}-{max(T)}"
        )
        out.append(
            row(f"frontend.indirect_predictor.capacity (T={t})", k, "entries", None, "", note, scale=t)
        )
        if out[-1]["ci"]:
            out[-1]["ci"][1] += t - 1
    return out


def run_indirect(r, args):
    sizes = [8, 16, 32, 64, 128, 256, 384, 512] if r.smoke else IND_SIZES
    smax, warm = max(sizes), 2 if r.smoke else 16
    raw = []

    def sweep(t, ss, phase):
        for chunk, c0 in octaves(ss, lambda s: t * (2 * s + 30)):
            rounds = 2 if r.smoke else iters_for(args, c0)
            src = kp.indirect_source(
                smax, chunk, targets=t, rounds=rounds, warm=warm, reps=REPS, isa=A64, counter=r.counter
            )
            pts, ratio = measure(r, src, f"m1fp_ind_t{t}", f"P3.5 {phase} T={t} S {chunk[0]}-{chunk[-1]}")
            for s in chunk:
                raw.append(
                    {
                        "targets": t,
                        "s": s,
                        "phase": phase,
                        "rounds": rounds,
                        "ratio": ratio,
                        "cyc": pts[str(s)],
                    }
                )

    for t in IND_TARGETS:
        sweep(t, sizes, "coarse")
        T = {x["s"]: x["cyc"] for x in raw if x["targets"] == t}
        k, _ = site_knee(T)
        if k.knee is not None and not r.smoke:
            ss = sorted(T)
            i = ss.index(int(k.knee))
            lo, hi = ss[max(0, i - 1)], ss[min(len(ss) - 1, i + 2)]
            fine = [s for s in range(lo, hi + 1, max(1, (hi - lo) // 48)) if s not in T]
            if fine:
                sweep(t, fine, "fine")
    return raw, indirect_summary(raw)


def ras_summary(raw):
    per_seed = [depth_marginal({int(d): med(v) for d, v in x["T"].items()}) for x in raw]
    ds = sorted(per_seed[0])
    k = band_knee({d: [m[d] for m in per_seed] for d in ds})
    note = f"D 1-{ds[-1]}, {len(raw)} placement seeds; the harness call is level 1"
    if k.knee is not None:
        m = {d: med([p[d] for p in per_seed]) for d in ds}
        lo = med([m[d] for d in ds if d <= k.knee])
        hi = med([m[d] for d in ds if d > k.knee]) if ds[-1] > k.knee else float("nan")
        note += f"; {lo:.1f} -> {hi:.1f} cycles per level"
    pub = (native.published("frontend.ras_depth") or {}).get("value")
    return [row("frontend.ras_depth", k, "entries", pub, "", note)]


def run_ras(r, args):
    dmax, seeds = (32, (0, 1)) if r.smoke else (RAS_DMAX, RAS_SEEDS)
    raw = []
    for s in seeds:
        T, ratios = {}, []
        for chunk, c0 in octaves(range(1, dmax + 1), lambda d: 2 * d + 8):
            iters = 8 if r.smoke else iters_for(args, c0)
            src = kp.ras_source(
                dmax, chunk, warm=4, iters=iters, reps=REPS, seed=s, isa=A64, counter=r.counter
            )
            pts, ratio = measure(r, src, f"m1fp_ras_s{s}", f"P3.6 seed {s} D {chunk[0]}-{chunk[-1]}")
            T.update({d: pts[str(d)] for d in chunk})
            ratios.append(ratio)
        raw.append(
            {"seed": s, "dmax": dmax, "ratios": ratios, "T": {str(d): v for d, v in sorted(T.items())}}
        )
    return raw, ras_summary(raw)


def fdip_summary(raw):
    """Lines in flight = L1i miss cost behind a mispredicted return every line / steady miss cost."""
    c = {(x["seed"], x["lines"], x["period"]): med(x["cyc"]) for x in raw}
    seeds, lines = sorted({x["seed"] for x in raw}), sorted({x["lines"] for x in raw})
    ref, big = lines[0], lines[-1]
    miss = {(n, p): [c[(s, n, p)] - c[(s, ref, p)] for s in seeds] for n in lines[1:] for p in (0, 1)}
    each = {n: [b / max(a, 0.25) for a, b in zip(miss[(n, 0)], miss[(n, 1)])] for n in lines[1:]}
    barrier = med([c[(s, ref, 1)] - c[(s, ref, 0)] for s in seeds])
    lif, per_size = each[big], {n: round(med(v), 2) for n, v in each.items()}
    flags = ["barrier.predicted"] if barrier < 3 else []
    flags += ["fdip.no_l1i_miss"] if med(miss[(big, 1)]) < 5 else []  # the large footprint is no slower
    ok = not flags
    value = med(lif) if ok else None
    if value is not None:
        flags.append("fdip.present" if value > 1.5 else "fdip.absent")
    node = native.published("decoupling.fdip_signature") or {}
    pub = None if node.get("present") is None else ("present" if node["present"] else "absent")
    l2 = native.published("memory.l2") or {}
    note = (
        f"{big} vs {ref} lines of {FD_LINE} B; L1i miss {med(miss[(big, 0)]):.1f} cycles per line steady, "
        f"{med(miss[(big, 1)]):.1f} with a mispredicted return every line ({barrier:.1f} cycles per return); "
        f"lines in flight per footprint {per_size}"
    )
    return [
        {
            "structure": "decoupling.fdip_signature",
            "value": value,
            "ci": [min(lif), max(lif)] if ok else None,
            "unit": "lines in flight",
            "published": pub,
            "source": ", ".join(node.get("sources", [])),
            "flags": flags,
            "note": note,
        },
        {
            "structure": "decoupling.l1i_miss_cycles",
            "value": med(miss[(big, 0)]),
            "ci": [min(miss[(big, 0)]), max(miss[(big, 0)])],
            "unit": "cycles",
            "published": l2.get("latency_cycles"),
            "source": ", ".join(l2.get("sources", [])),
            "flags": [],
            "note": "steady cost of one L1i miss per line; published value is the L2 load latency",
        },
    ]


def run_fdip(r, args):
    lines, seeds = ((128, 512), (1,)) if r.smoke else (FD_LINES, FD_SEEDS)
    raw = []
    for s in seeds:
        for n in lines:
            iters = 1 if r.smoke else iters_for(args, 3 * n)  # >= ~3 cycles per line
            src = kp.fdip_source(
                [1, 0],
                n_lines=n,
                iters=iters,
                reps=REPS,
                seed=s,
                isa=A64,
                counter=r.counter,
                line_bytes=FD_LINE,
            )
            pts, ratio = measure(r, src, f"m1fp_fd_s{s}", f"P3.8 seed {s} {n} lines")
            for p in (1, 0):
                raw.append(
                    {"seed": s, "lines": n, "period": p, "iters": iters, "ratio": ratio, "cyc": pts[str(p)]}
                )
    return raw, fdip_summary(raw)


def fetchq_knee(raw):
    chains = sorted({x["chain"] for x in raw})
    curve = {c: {x["b"]: med(x["cyc"]) for x in raw if x["chain"] == c} for c in chains}
    bs = sorted(set.intersection(*[set(v) for v in curve.values()]))
    k = band_knee({b: [curve[c][b] for c in chains] for b in bs}, min_jump=FQ_MIN_JUMP)
    per = {c: find_knee(bs, [curve[c][b] for b in bs], min_jump=FQ_MIN_JUMP).knee for c in chains}
    return k, per, bs


def fetchq_summary(raw):
    k, per, bs = fetchq_knee(raw)
    note = (
        f"knee = all buffering between I-cache and rename (gem5: fetch queue + 26); serializer "
        f"'{raw[0]['serializer']}'; per chain {per}; B {bs[0]}-{bs[-1]}"
    )
    out = row("frontend.fetch_to_rename_buffer", k, "instructions", None, "", note)
    if out["value"] is None:
        out["note"] += "; if flat, the serializer did not hold rename: retry with --serializer"
    return [out]


def run_fetchq(r, args):
    ser = getattr(args, "serializer", None) or FQ_SERIALIZER
    grid, chains = ([0, 32, 64, 96], (16,)) if r.smoke else (FQ_GRID, FQ_CHAINS)
    raw = []

    def sweep(bs, phase):
        for c in chains:
            laps = 2 if r.smoke else iters_for(args, FQ_LAYOUT["copies"] * c * FSQRT)
            for i in range(0, len(bs), FQ_PACK):
                chunk = bs[i : i + FQ_PACK]
                src = kp.fetchq_source(
                    chunk,
                    chain=c,
                    stride=FQ_PACK * FQ_LAYOUT["slot"],
                    iters=laps,
                    reps=REPS,
                    isa=A64,
                    counter=r.counter,
                    serializer=ser,
                    **FQ_LAYOUT,
                )
                pts, ratio = measure(
                    r, src, f"m1fp_fq_c{c}", f"P3.9 {phase} chain {c} B {chunk[0]}-{chunk[-1]}"
                )
                for b in chunk:
                    x = {"chain": c, "b": b, "phase": phase, "laps": laps, "ratio": ratio, "serializer": ser}
                    raw.append({**x, "cyc": pts[str(b)]})

    sweep(grid, "coarse")
    k, _, bs = fetchq_knee(raw)
    if k.knee is not None and not r.smoke:
        fine = [b for b in range(int(k.knee) + 1, int(k.knee) + 16) if b not in bs]
        if fine:
            sweep(fine, "fine")
    return raw, fetchq_summary(raw)


POCS = {"P3.4": run_history, "P3.5": run_indirect, "P3.6": run_ras, "P3.8": run_fdip, "P3.9": run_fetchq}
EST_MIN = {"P3.4": 1, "P3.5": 2, "P3.6": 1, "P3.8": 1, "P3.9": 2}


def parser():
    ap = native.add_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("pocs", nargs="*", help=f"any of {' '.join(POCS)} (default all)")
    ap.add_argument("--min-ms", type=float, default=1.0, help="shortest timed region")
    ap.add_argument(
        "--serializer", default=None, help=f"P3.9 rename barrier, ';' between instructions ({FQ_SERIALIZER})"
    )
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    bad = [p for p in a.pocs if p not in POCS]
    if bad:
        ap.error(f"unknown PoC {' '.join(bad)}; choose from {' '.join(POCS)}")
    r = native.Runner(a)
    for poc in a.pocs or list(POCS):
        raw, summary = POCS[poc](r, a)
        print(f"{FAMILY} {poc}: wrote {r.save(FAMILY, poc, raw, summary)}")
        for s in summary:
            print(
                f"  {s['structure']}: {s['value']} {s['unit']} ci {s['ci']}  published {s['published']}  "
                f"{' '.join(s['flags'])}\n    {s['note']}"
            )


if __name__ == "__main__":
    main()
