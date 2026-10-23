#!/usr/bin/env python3
"""Frontend kernels on Apple silicon, timing only: P3.1 width, P3.2 L1i, P3.3 BTB levels, P3.7 mispredicts.

    python3 targets/m1/m1_fe.py [P3.1 P3.1-page P3.2 P3.2-ways P3.3 P3.7] [--tag m1] [--gem5-smoke]

Kernels and analysis are the gem5 campaigns' (validation/gem5/fe_*.py). A point is several launches in
shuffled order; capacity searches run serially with tau floored at TAU_SIGMAS x the bootstrapped noise.
No instruction counter: the mispredict penalty assumes half the random branches mispredict.
"""

import argparse
import dataclasses
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native  # noqa: E402

sys.path.insert(0, os.path.join(native.ROOT, "validation", "gem5"))
from fe_common import capacity_search, geom_grid  # noqa: E402
from uarchre.kernels.frontend import A64, harness_c, jump_chain, lap_excess, onset, parse_reps  # noqa: E402
from uarchre.kernels.frontend import page_loop, random_branches, step_period, straight_nops  # noqa: E402
from uarchre.kernels.frontend import stride_chain_source, width_estimate, width_loop  # noqa: E402
from uarchre.kernels.predictors import parse_points  # noqa: E402

FAMILY = "fe"
FILLERS = ("nop", "mix")
TAU_SIGMAS = 5
N_BOOT = 1000

# work: core cycles per timed rep at the peak rate (2^23: 2.6 ms at 3.2 GHz); rep 0 of a launch is dropped.
# l1i grid and step in bytes. page: the loop starts `split` instructions before a page boundary, or before
# an `inside` boundary for the control; extra_L: lengths both run below the rename limit
M1 = {
    "reps": 9,
    "outer": 5,
    "width": {"Ls": list(range(2, 17)), "laps": 1 << 23},
    "page": {
        "Ls": list(range(2, 65)),
        "laps": 1 << 22,
        "page": 16 << 10,
        "inside": 2 << 10,
        "split": 1,
        "extra_L": 8,
    },
    "l1i": {
        "grid": geom_grid(48 << 10, 512 << 10, 4, 128),
        "work": 1 << 23,
        "min_laps": 16,
        "step": 64,
        "kary": 12,
    },
    # N <= 24 lines on N pages, far below the iTLB; the jumps take 16 word slots, so few share a BTB set
    "ways": {
        "strides": [16 << 10, 32 << 10, 64 << 10],
        "ns": list(range(2, 25)),
        "work": 1 << 22,
        "reps": 5,
        "launches": 8,
    },
    "jump": {"n": 512, "seeds": [0, 1, 2], "laps": 1 << 14},
    "btb": {
        "grid": geom_grid(128, 4096, 4),
        "top": 1 << 17,
        "per_octave": 4,
        "work": 1 << 23,
        "min_laps": 16,
        "steps": [1, 32],
        "kary": 12,
        "seeds": [1, 2],
    },
    "mispredict": {"words": 16384, "seeds": list(range(8)), "pattern_launches": 3, "pattern_laps": 16},
}
# gem5's default O3: 32 kB 8-way L1i (4 KiB a way), an effective 1024-entry BTB
SMOKE = {
    "reps": 3,
    "outer": 1,
    "width": {"Ls": [2, 4, 8, 9, 12, 16], "laps": 4000},
    "page": {**M1["page"], "Ls": [2, 4, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 26], "laps": 2000},
    "l1i": {
        "grid": [n << 7 for n in (64, 96, 128, 192, 256, 384, 512)],
        "work": 1 << 15,
        "min_laps": 4,
        "step": 4096,
        "kary": 3,
    },
    "ways": {
        **M1["ways"],
        "strides": [2 << 10, 4 << 10, 8 << 10],
        "ns": list(range(2, 22)),
        "work": 1 << 10,
        "reps": 2,
        "launches": 1,
    },
    "jump": {"n": 512, "seeds": [0], "laps": 64},
    "btb": {
        "grid": geom_grid(256, 4096, 2),
        "top": 1 << 15,
        "per_octave": 2,
        "work": 1 << 15,
        "min_laps": 4,
        "steps": [64, 512],
        "kary": 3,
        "seeds": [],
    },
    "mispredict": {"words": 64, "seeds": [0, 1], "pattern_launches": 1, "pattern_laps": 1},
}


def plan(r):
    return SMOKE if r.smoke else M1


def published(path, key="value"):
    """(value, "path.key (sources)") of a spec/cores/firestorm.yaml field; `path` may end in [i]."""
    head, _, idx = path.partition("[")
    node = native.published(head)
    if idx:
        node = node[int(idx.rstrip("]"))] if isinstance(node, list) else None
    if not isinstance(node, dict):
        return node, path
    src = ", ".join(node.get("sources", []))
    return node.get(key), f"{path}.{key}" + (f" ({src})" if src else "")


def row(structure, value, ci, unit, pub, flags=(), note=""):
    return {
        "structure": structure,
        "value": value,
        "ci": ci,
        "unit": unit,
        "published": pub[0],
        "source": pub[1],
        "flags": list(flags),
        "note": note,
    }


def convert(out, ratio, units, reps, name=""):
    """Harness stdout -> dict(ticks of every rep, ratio, cpu = core cycles per unit of each measured rep)."""
    got = parse_reps(out)
    if len(got) != reps:
        raise RuntimeError(f"{name}: expected {reps} reps, got:\n{out[-1500:]}")
    ticks = [c for c, _ in got]
    meas = ticks[1:] if reps > 1 else ticks
    return {"ticks": ticks, "ratio": ratio, "units": units, "cpu": [t / ratio / units for t in meas]}


def point(r, k, laps, reps, outer=1, seeds=(0,), mode=0):
    """`outer` launches of kernel k, launch i with seeds[i % len(seeds)]; one dict per launch."""
    b = r.compile(harness_c(k, r.counter), k.name)
    res = []
    for o in range(outer):
        seed = seeds[o % len(seeds)]
        out, ratio = r.launch(b, [laps, reps, seed, mode], label=k.name)
        d = convert(out, ratio, k.units_per_lap * laps, reps, k.name)
        d.update(seed=seed, mode=mode, laps=laps, accepted=bool(r.attempts[-1]["accepted"]))
        res.append(d)
    return res


def accepted(launches):
    return [x for x in launches if x["accepted"]] or launches


def medians(launches):
    """Median cycles per unit of each accepted launch's measured reps."""
    return [statistics.median(x["cpu"]) for x in accepted(launches)]


def value(launches, scale=1.0):
    return scale * statistics.median(medians(launches))


def unaccepted(points):
    n = sum(not x["accepted"] for launches in points for x in launches)
    return [f"unaccepted: {n} launches outside --min-ghz/--max-drift"] if n else []


def boot(launches, scale=1.0, seed=0, n_boot=N_BOOT):
    """Bootstrap replicates of `value`, resampling the accepted launches."""
    a = scale * np.asarray(medians(launches), float)
    rng = np.random.default_rng(seed)
    return np.median(a[rng.integers(0, len(a), (n_boot, len(a)))], axis=1)


def search(r, make, laps_for, grid, scale, step, kary, tag):
    """fe_common.capacity_search on hardware. Returns (result, {n: launches}); the result also holds
    `ci`, the 95 % bootstrap interval of the capacity, and `no_onset`, the share of resamples without one."""
    p = plan(r)
    pts = {}

    def measure(n):
        if n not in pts:
            pts[n] = point(r, make(n), laps_for(n), p["reps"], p["outer"])
            print(f"  {tag} n={n}: {value(pts[n], scale):.4f}", flush=True)
        return value(pts[n], scale)

    for n in random.Random(0).sample(grid, len(grid)):
        measure(n)
    y = {n: measure(n) for n in grid}
    lo, hi = min(y.values()), max(y.values())
    flat = [n for n in grid if y[n] <= lo + 0.25 * (hi - lo)]
    tau_min = TAU_SIGMAS * max(n * float(np.std(boot(pts[n], scale, seed=n))) for n in flat)
    res = capacity_search(measure, grid, step=step, kary=kary, tag=tag, jobs=1, tau_min=tau_min)
    res.update(tau_min=tau_min, ci=None, no_onset=None)
    br = res.get("bracket")
    if not br or br[0] is None or len(res["base"]) < 3:
        return res, pts
    ns = sorted(pts)
    Y = np.stack([boot(pts[n], scale, seed=n) for n in ns], axis=1)
    found = [onset(ns, lap_excess(ns, yb, res["base"]), res["tau"]) for yb in Y]
    found = [b for b in found if b and b[0] is not None]
    res["no_onset"] = 1 - len(found) / len(Y)
    if found:
        lo_q = float(np.quantile([b[0] for b in found], 0.025, method="lower"))
        hi_q = float(np.quantile([b[1] - 1 for b in found], 0.975, method="higher"))
        res["ci"] = [min(lo_q, br[0]), max(hi_q, br[1] - 1)]
    return res, pts


def search_flags(res):
    f = [f"search: {res['error']}"] if res.get("error") else []
    return f + (["noise.unstable_onset"] if (res.get("no_onset") or 0) > 0.05 else [])


def capacity(res):
    br = res.get("bracket")
    return br[0] if br and br[0] is not None else None


def plain(res):
    """capacity_search result with JSON-friendly keys."""
    return {k: ({str(n): y for n, y in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}


def run_width(r, args):
    """P3.1: width_loop(L), L = 2..16 per filler; width = round(peak L / cycles), a staircase is fetch."""
    p = plan(r)
    w = p["width"]
    jobs = [(f, L) for f in FILLERS for L in w["Ls"]]
    pts = {}
    for f, L in random.Random(0).sample(jobs, len(jobs)):
        pts[f, L] = point(r, width_loop(L, filler=f, isa=A64), w["laps"], p["reps"], p["outer"])
    pub = published("frontend.fetch_width")
    summary, est = [], {}
    for f in FILLERS:
        cpi = [value(pts[f, L]) for L in w["Ls"]]
        e = width_estimate(w["Ls"], cpi)
        per = [
            width_estimate(w["Ls"], [statistics.median(pts[f, L][o]["cpu"]) for L in w["Ls"]])["width"]
            for o in range(p["outer"])
        ]
        est[f] = {**e, "cpi": cpi, "per_launch": per}
        flags = unaccepted(pts[f, L] for L in w["Ls"]) + ([] if e["shape"] == "staircase" else ["shape.ramp"])
        note = (
            f"filler {f}: peak {e['peak_ipc']:.2f} instr/cycle, {e['shape']} (staircase: fetch; ramp: decode)"
        )
        summary.append(
            row(
                f"frontend.fetch_width[{f}]",
                e["width"],
                [min(per), max(per)],
                "instr/cycle",
                pub,
                flags,
                note,
            )
        )
    raw = {
        "plan": w,
        "points": [{"filler": f, "L": L, "launches": pts[f, L]} for f, L in jobs],
        "estimates": est,
    }
    return raw, summary


def run_page(r, args):
    """P3.1-page: page_loop(L) starting just before a page boundary vs the same loop before a boundary
    inside a page; step_period of the crossing curve is the fetch block, the gap at small L the page cost."""
    p = plan(r)
    g, line = p["page"], native.l1i_line(args)
    if g["inside"] % line or g["page"] % g["inside"]:
        raise ValueError(f"control boundary {g['inside']} B: not a line boundary inside a page")
    offs = {"cross": g["page"] - 4 * g["split"], "inside": g["inside"] - 4 * g["split"]}
    Ls, pts = g["Ls"], {}
    jobs = [(w, L) for w in offs for L in Ls]
    for w, L in random.Random(0).sample(jobs, len(jobs)):
        pts[w, L] = point(r, page_loop(L, offs[w], g["page"], isa=A64), g["laps"], p["reps"], p["outer"])
    rate = published("backend.rename_width")[0]

    def curve(w, o=None):
        return [value(pts[w, L]) if o is None else statistics.median(pts[w, L][o]["cpu"]) for L in Ls]

    ok = [o for o in range(p["outer"]) if all(x[o]["accepted"] for x in pts.values())] or range(p["outer"])
    fit = step_period(Ls, curve("cross"), curve("inside"), rate)
    per = [step_period(Ls, curve("cross", o), curve("inside", o), rate) for o in ok]
    small = [i for i, L in enumerate(Ls) if L <= g["extra_L"]]
    extra = [
        statistics.median(c[i] - n[i] for i in small)
        for c, n in ((curve("cross", o), curve("inside", o)) for o in ok)
        if small
    ]
    periods = [f["period"] for f in per]
    found = [x for x in periods if x]
    flags = unaccepted(pts.values())
    if len(set(periods + [fit["period"]])) > 1:
        flags.append("period: launches disagree")
    h = fit["heights"]
    note = (
        f"loop from {4 * g['split']} B before a {g['page'] >> 10} KiB page, so the blocks after it align; "
        f"steps every 16: 16 per fetch, every 8: 8 wide; heights {h[8]:.2f} per 8, {h[16]:.2f} per 16"
    )
    summary = [
        row(
            "frontend.fetch_width[page]",
            fit["period"],
            [min(found), max(found)] if found else None,
            "instr/fetch",
            published("frontend.fetch_width"),
            flags,
            note,
        ),
        row(
            "frontend.page_cross_cycles",
            statistics.median(extra) if extra else None,
            [min(extra), max(extra)] if extra else None,
            "cycles",
            (None, ""),
            unaccepted(pts.values()),
            f"crossing minus inside a page at L <= {g['extra_L']}: 1 if a fetch stops at the page",
        ),
    ]
    raw = {
        "plan": g,
        "line": line,
        "rate": rate,
        "points": [{"where": w, "L": L, "offset": offs[w], "launches": pts[w, L]} for w, L in jobs],
        "fit": fit,
        "per_launch": per,
        "extra": extra,
    }
    return raw, summary


def run_l1i(r, args):
    """P3.2: straight_nops over n L1i lines (capacity); jump_chain cycles per taken jump."""
    p = plan(r)
    c, j = p["l1i"], p["jump"]
    line = native.l1i_line(args)
    res, pts = search(
        r,
        lambda n: straight_nops(line * n, isa=A64),
        lambda n: max(c["min_laps"], c["work"] * 32 // (line * n)),  # 32 B of code per cycle at 8 wide
        sorted({b // line for b in c["grid"]}),
        line,
        max(1, c["step"] // line),
        c["kary"],
        "l1i",
    )
    jumps = {
        s: point(r, jump_chain(j["n"], s, isa=A64), j["laps"], p["reps"], p["outer"]) for s in j["seeds"]
    }
    cap = capacity(res)
    kib = line / 1024
    y_in = statistics.median(res["points"][n] for n in res["base"])
    note = f"{cap} lines of {line} B; " if cap else ""
    note += f"cycles per line: {y_in:.2f} inside, {res['points'][max(pts)]:.2f} at {max(pts)} lines"
    per_seed = [value(v) for v in jumps.values()]
    tbpc, src = published("frontend.taken_branches_per_cycle")
    summary = [
        row(
            "frontend.l1i.size_kib",
            cap * kib if cap else None,
            [x * kib for x in res["ci"]] if res["ci"] else None,
            "KiB",
            published("frontend.l1i", "size_kib"),
            unaccepted(pts.values()) + search_flags(res),
            note,
        ),
        row(
            "frontend.taken_jump_cycles",
            statistics.median(per_seed),
            [min(per_seed), max(per_seed)],
            "cycles",
            (1 / tbpc if tbpc else None, f"1 / {src}"),
            unaccepted(jumps.values()),
            f"jump_chain({j['n']}), orders {j['seeds']}: taken jumps to new lines inside the L0 BTB and L1i",
        ),
    ]
    raw = {
        "plan": {"l1i": c, "jump": j, "line": line},
        "search": plain(res),
        "points": {str(n): v for n, v in sorted(pts.items())},
        "jump": {str(s): v for s, v in jumps.items()},
    }
    return raw, summary


WAYS_NOTES = {
    16 << 10: "12: 12 ways or 6 ways virtually indexed; 6-12, moving by launch: 6 ways physically indexed",
    32 << 10: "6: 6 ways; 12: 12 ways; 6-12, moving by launch: 6 ways physically indexed",
    64 << 10: "control: as 32 KiB unless a way holds more than 32 KiB",
}


def ways_knee(ns, cpu):
    """capacity_search over one launch's cycles per jump {N: [reps]}: (last N without misses, result)."""
    y = {n: statistics.median(cpu[n]) for n in ns}
    lo, hi = min(y.values()), max(y.values())
    flat = [n for n in ns if y[n] <= lo + 0.25 * (hi - lo)]
    tau_min = TAU_SIGMAS * max(n * float(np.std(cpu[n])) for n in flat)
    res = capacity_search(y.__getitem__, sorted(ns), step=1, log=None, jobs=1, tau_min=tau_min)
    return capacity(res), res


def run_ways(r, args):
    """P3.2-ways: stride_chain_source, N one-jump lines S apart on fresh pages per launch; the knee in N is
    the lines one L1i set holds (ways if S >= a way, twice that at half a way), one knee per launch."""
    p = plan(r)
    w, line = p["ways"], native.l1i_line(args)
    b = r.compile(stride_chain_source(r.counter), "fe_ways")
    jobs = [(s, seed) for s in w["strides"] for seed in range(w["launches"])]
    launches = {s: [] for s in w["strides"]}
    for s, seed in random.Random(0).sample(jobs, len(jobs)):
        ns = random.Random(seed).sample(w["ns"], len(w["ns"]))
        out, ratio = r.launch(b, [w["work"], w["reps"], seed, s, line, *ns], label=f"fe_ways_s{s}")
        got = parse_points(out)
        if sorted(got) != sorted(map(str, w["ns"])) or any(len(v) != w["reps"] for v in got.values()):
            raise RuntimeError(f"fe_ways: expected {w['reps']} reps per N, got:\n{out[-1500:]}")
        cpu = {n: [t / ratio for t in got[str(n)]] for n in w["ns"]}
        launches[s].append(
            {"seed": seed, "ratio": ratio, "accepted": bool(r.attempts[-1]["accepted"]), "cpu": cpu}
        )
    summary, lo, hi = [], min(w["ns"]), max(w["ns"])
    for s in w["strides"]:
        ls, flags, knees = accepted(launches[s]), unaccepted([launches[s]]), []
        for x in ls:
            x["knee"], res = ways_knee(w["ns"], x["cpu"])
            x["search"] = plain(res)
            flags += search_flags(res)
            knees += [x["knee"]] if x["knee"] is not None else []
        y = {n: statistics.median(statistics.median(x["cpu"][n]) for x in ls) for n in (lo, hi)}
        note = WAYS_NOTES.get(s, "ways at a stride of a way or more, twice that at half a way")
        note += f"; cycles per jump {y[lo]:.2f} at N={lo}, {y[hi]:.2f} at {hi}"
        summary.append(
            row(
                f"frontend.l1i.ways[{s >> 10}KiB]",
                statistics.median(knees) if knees else None,
                [min(knees), max(knees)] if knees else None,
                "lines",
                published("frontend.l1i", "ways"),
                list(dict.fromkeys(flags)),
                note,
            )
        )
    return {"plan": w, "line": line, "launches": {str(s): v for s, v in launches.items()}}, summary


def placement(r, res, laps_for, seeds, reps):
    """Re-measure the plateau and the bracket ends with other chain orders (gem5 fe_btb.py's check)."""
    lo, hi = res["bracket"]
    ns = sorted(set(res["base"]) | {lo, hi})
    out = {}
    for s in seeds:
        ys = [value(point(r, jump_chain(n, s, isa=A64), laps_for(n), reps)) for n in ns]
        exc = dict(zip(ns, lap_excess(ns, ys, res["base"])))
        out[str(s)] = {
            "consistent": bool(exc[lo] <= res["tau"] < exc[hi]),
            "points": dict(zip(map(str, ns), ys)),
        }
    return out


def run_btb(r, args):
    """P3.3: jump_chain(n) cycles per taken jump; one capacity search per tier, each grid past the last."""
    p = plan(r)
    b = p["btb"]

    def laps_for(n):
        return max(b["min_laps"], b["work"] // n)

    grid, levels, summary = b["grid"], [], []
    for i, step in enumerate(b["steps"]):
        if len(grid) < 4:
            break
        res, pts = search(
            r, lambda n: jump_chain(n, 0, isa=A64), laps_for, grid, 1, step, b["kary"], f"btb{i}"
        )
        cap = capacity(res)
        seeds = placement(r, res, laps_for, b["seeds"], p["reps"]) if cap and b["seeds"] else {}
        flags = unaccepted(pts.values()) + search_flags(res)
        if any(not s["consistent"] for s in seeds.values()):
            flags.append("placement: other chain orders disagree")
        base = [res["points"][n] for n in res["base"]]
        top = max(pts)
        note = f"{cap * 4 / 1024:.1f} KiB of jumps; " if cap else ""
        note += f"{res['points'][top]:.2f} cycles per jump at {top} jumps"
        summary.append(
            row(
                f"frontend.btb_levels[{i}].entries",
                cap,
                res["ci"],
                "entries",
                published(f"frontend.btb_levels[{i}]", "entries"),
                flags,
                note,
            )
        )
        summary.append(
            row(
                f"frontend.btb_levels[{i}].taken_bubble_cycles",
                statistics.median(base) - 1 if base else None,
                [min(base) - 1, max(base) - 1] if base else None,
                "cycles",
                published(f"frontend.btb_levels[{i}]", "taken_bubble_cycles"),
                flags,
                f"plateau {res['base'][0]}-{res['base'][-1]} jumps" if base else "",
            )
        )
        points = {str(n): v for n, v in sorted(pts.items())}
        levels.append({"grid": grid, "search": plain(res), "points": points, "seeds": seeds})
        grid = geom_grid(2 * res["bracket"][1] if cap else grid[-1], b["top"], b["per_octave"])
    return {"plan": b, "levels": levels}, summary


def run_mispredict(r, args):
    """P3.7: random_branches, random data (fresh per rep, one lap) vs 0xAAAA...; penalty = gap / 0.5."""
    p = plan(r)
    m = p["mispredict"]
    k = random_branches(m["words"], isa=A64)
    k = dataclasses.replace(k, data_words=max(k.data_words, p["reps"] * m["words"] + 1))
    jobs = [(2, 0)] * m["pattern_launches"] + [(1, s) for s in m["seeds"]]
    launches = {1: [], 2: []}
    for mode, seed in random.Random(0).sample(jobs, len(jobs)):
        laps = m["pattern_laps"] if mode == 2 else 1
        launches[mode] += point(r, k, laps, p["reps"], seeds=(seed,), mode=mode)
    pattern = value(launches[2])
    rnd = medians(launches[1])
    pens = [(c - pattern) / 0.5 for c in rnd]
    note = (
        f"cycles per branch: random {statistics.median(rnd):.3f}, pattern {pattern:.3f}; seeds {m['seeds']}"
    )
    summary = [
        row(
            "frontend.mispredict_penalty",
            statistics.median(pens),
            [min(pens), max(pens)],
            "cycles",
            published("frontend.mispredict_penalty"),
            unaccepted(launches.values()),
            note,
        )
    ]
    raw = {"plan": m, "random": launches[1], "pattern": launches[2], "penalties": pens}
    return raw, summary


POCS = {
    "P3.1": run_width,
    "P3.1-page": run_page,
    "P3.2": run_l1i,
    "P3.2-ways": run_ways,
    "P3.3": run_btb,
    "P3.7": run_mispredict,
}
EST_MIN = {"P3.1": 1, "P3.1-page": 2, "P3.2": 2, "P3.2-ways": 2, "P3.3": 6, "P3.7": 1}


def fmt(row):
    v, ci = row["value"], row["ci"]
    v = "-" if v is None else f"{v:.4g}"
    ci = "-" if not ci else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
    flags = f"  {'; '.join(row['flags'])}" if row["flags"] else ""
    return (
        f"  {row['structure']:40s} {v:>8s} {row['unit']:12s} ci {ci:20s} published {row['published']}{flags}"
    )


def main(argv=None):
    ap = native.add_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("pocs", nargs="*", metavar="PoC", help=f"any of {' '.join(POCS)} (default all)")
    a = ap.parse_args(argv)
    unknown = [x for x in a.pocs if x not in POCS]
    if unknown:
        ap.error(f"unknown PoC {' '.join(unknown)}; choose from {' '.join(POCS)}")
    r = native.Runner(a)
    for poc in a.pocs or list(POCS):
        print(f"{FAMILY} {poc}" + ("" if r.smoke else f" (~{EST_MIN[poc]} min)"), flush=True)
        raw, summary = POCS[poc](r, a)
        print(f"{FAMILY} {poc}: wrote {os.path.relpath(r.save(FAMILY, poc, raw, summary), native.ROOT)}")
        for x in summary:
            print(fmt(x), flush=True)


if __name__ == "__main__":
    main()
