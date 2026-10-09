"""Anomaly-rule fixtures (P0.3). Builders return (name, RunSet, expected rule ids).

synth   single synthetic curves: synth.CASES plus SYNTH_EXTRA
cross   multi-curve RunSets for the cross-curve rules, with negative controls
toy     toy_cases.yaml on the toy model (slow; validation/toy/mech_fixtures.py caches the curves)
pub     firestorm_published.yaml: published knees, A4 and A5 must fire
gem5    existing results/gem5 runs: only offset.uncalibrated or offset.missing may fire
"""

import json
import os
import re

import numpy as np
import yaml

from tests.fixtures.synth import synth_curve, grid, CASES
from uarchre.analysis import offsets as _offsets
from uarchre.analysis.anomaly import Curve, RunSet, probe_params, load_results

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOY_CTX = {"emitter": "abstract", "platform": "toy"}
TOY_CASES = os.path.join(HERE, "toy_cases.yaml")
FIRESTORM = os.path.join(HERE, "firestorm_published.yaml")


def offsets_table(kind=None):
    """offsets.yaml + the toy branch/ALU entries (toy_offsets_extra.yaml);
    kind="textbook" replaces the toy NOP entries with the old 2 x depth model."""
    with open(os.path.join(HERE, "toy_offsets_extra.yaml")) as fh:
        table = _offsets.load() + (yaml.safe_load(fh) or [])
    if kind == "textbook":
        table = [e for e in table if not (e["platform"] == "toy" and e["filler"] == "nop")]
        table += [
            {
                "id": f"textbook.nop.d{d}.miss",
                "emitter": "abstract",
                "platform": "toy",
                "filler": "nop",
                "depth": d,
                "delay": "miss",
                "offset": 2 * d,
                "offset_sd": 0.0,
            }
            for d in (1, 2, 3)
        ]
    return table


# ---------------------------------------------------------------- synthetic single curves


def two_step(ns, k1, k2, lo, mid, hi, n_reps=3, noise_sd=0.0, seed=0):
    """lo up to k1, mid up to k2, hi after: two stacked mechanisms."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        y = lo if n <= k1 else mid if n <= k2 else hi
        out[n] = [float(v) for v in y * (1 + noise_sd * rng.standard_normal(n_reps))]
    return out


SYNTH_EXTRA = [
    dict(
        name="double_step",
        knee=64,
        filler="store",
        reps=lambda: two_step(
            sorted(set(range(2, 100, 4)) | set(range(56, 73))), 8, 64, 200, 300, 600, noise_sd=0.01, n_reps=3
        ),
        rules=["shape.extra_step"],
    ),
]


def synth_ns(case):
    ns = set(grid(case["knee"]))
    ns |= {n for n, _ in case["kw"].get("outliers", ())}
    return sorted(ns)


def synth_cases():
    out = []
    for c in CASES:
        reps = synth_curve(synth_ns(c), c["knee"], **c["kw"])
        rs = RunSet("synth", c["name"], ctx=dict(TOY_CTX), offsets=offsets_table())
        rs.add(Curve.from_reps("nop", reps, depth=c.get("expected_steps", 1), n_boot=1000))
        out.append((c["name"], rs, sorted(c["rules"])))
    for c in SYNTH_EXTRA:
        rs = RunSet("synth", c["name"], ctx=dict(TOY_CTX), offsets=offsets_table())
        rs.add(Curve.from_reps(c["filler"], c["reps"](), n_boot=1000))
        out.append((c["name"], rs, sorted(c["rules"])))
    return out


# ---------------------------------------------------------------- synthetic cross-curve RunSets


def _c(filler, knee, lo=300.0, hi=600.0, depth=1, delay="miss", n_reps=3, noise_sd=0.0, seed=0, slope=0.0):
    ns = grid(knee)
    return Curve.from_reps(
        filler,
        synth_curve(ns, knee, lo, hi, n_reps=n_reps, noise_sd=noise_sd, seed=seed, slope=slope),
        depth=depth,
        delay=delay,
        n_boot=500,
    )


def cross_cases():
    T = offsets_table()
    cases = [
        # A3
        ("tie_synth", [_c("nop", 94), _c("int", 92)], {}, ["tie.rob"]),
        ("tie_separated_synth", [_c("nop", 94), _c("int", 58)], {}, []),
        # A4: Firestorm-shaped ratio; classic shapes; gem5 defaults (ROB 64, LQ = SQ = 32), no branch curve
        (
            "retire_ratio_synth",
            [_c("nop", 326), _c("load", 46), _c("store", 46), _c("branch", 46)],
            {},
            ["ratio.retire_group"],
        ),
        ("classic_ratio_synth", [_c("nop", 126), _c("load", 46), _c("store", 38), _c("branch", 126)], {}, []),
        ("lq_eq_sq_no_branch_synth", [_c("nop", 62), _c("load", 30), _c("store", 30)], {}, []),
        # A5
        (
            "delay_disagree_synth",
            [_c("store", 39), _c("store", 120, depth=8, delay="fp")],
            {},
            ["delay.disagreement"],
        ),
        ("delay_agree_synth", [_c("store", 24), _c("store", 24, depth=8, delay="fp")], {}, []),
        # A7: x86-like d=1 400 vs d=2 496 (offset-corrected 402 vs 499)
        (
            "depth_inconsistent_synth",
            [_c("nop", 400, lo=800, hi=1600), _c("nop", 496, depth=2, lo=1600, hi=2400)],
            {},
            ["offset.depth_inconsistent"],
        ),
        ("depth_consistent_synth", [_c("nop", 62), _c("nop", 61, depth=2, lo=600, hi=1200)], {}, []),
        # A10: 2 x 126 / 4 = 63 cycles vs a 100-cycle plateau
        (
            "throughput_synth",
            [_c("nop", 126, lo=100, hi=200)],
            {"widths": {"default": 4}},
            ["throughput.bound"],
        ),
        ("not_throughput_synth", [_c("nop", 126, lo=300, hi=600)], {"widths": {"default": 4}}, []),
        # A11: gem5 int filler at depth 2 has only the toy (analytic) offset
        (
            "uncalibrated_synth",
            [_c("int", 62, depth=2, lo=600, hi=1200)],
            {"ctx": {"emitter": "riscv64", "platform": "gem5"}},
            ["offset.uncalibrated"],
        ),
        # A11b: a flags filler has no offset anywhere
        (
            "missing_offset_synth",
            [_c("flags", 62)],
            {"ctx": {"emitter": "x86_64", "platform": "native"}},
            ["offset.missing"],
        ),
    ]
    out = []
    for name, curves, kw, expect in cases:
        rs = RunSet(
            "synth",
            name,
            ctx=kw.get("ctx", dict(TOY_CTX)),
            widths=kw.get("widths"),
            clock=kw.get("clock", []),
            offsets=T,
        )
        for c in curves:
            rs.add(c)
        out.append((name, rs, sorted(expect)))
    return out


# ---------------------------------------------------------------- toy configurations


def toy_cases(path=TOY_CASES):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _range(s):
    a, b, step = (int(v) for v in s.split(":"))
    return list(range(a, b + 1, step))


def toy_cfg(case):
    from uarchre.backends.platform.toy import ToyConfig

    cfg = dict(case["cfg"])
    if "sq_levels" in cfg:
        cfg["sq_levels"] = tuple(cfg["sq_levels"])
    return ToyConfig(**cfg)


def curve_spec(c):
    return c["filler"], int(c.get("depth", 1)), c.get("delay", "miss")


def toy_curve(job):
    """Pool worker: (case_name, case, curve) -> (case_name, key, reps); one outer
    rep of the two-phase protocol (the toy is deterministic)."""
    from uarchre.analysis.protocol import run_protocol
    from uarchre.backends.platform.toy import run_toy
    from uarchre.kernels.window import window_probe

    name, case, c = job
    cfg = toy_cfg(case)
    f, d, dl = curve_spec(c)
    min_jump, steps = probe_params(d, dl)
    measure = lambda key, n, rep: run_toy(window_probe(f, n, depth=d, delay=dl), cfg)
    _, reps = run_protocol(
        measure, ["k"], _range(c["n"]), outer_reps=1, min_jump=min_jump, expected_steps=steps
    )
    return name, (f, d, dl), {int(n): v for n, v in reps["k"].items()}


def toy_jobs(cases=None):
    cases = toy_cases() if cases is None else cases
    return [(case["name"], case, c) for case in cases for c in case["curves"]]


def toy_runset(case, reps_by_key):
    rs = RunSet(
        "toy",
        case["name"],
        ctx=dict(TOY_CTX),
        widths={"default": case["cfg"].get("width", 4)},
        offsets=offsets_table(case.get("offsets")),
    )
    for c in case["curves"]:
        key = curve_spec(c)
        rs.add(Curve.from_reps(key[0], reps_by_key[key], depth=key[1], delay=key[2], n_boot=200))
    return rs


def toy_runsets(curves, cases=None):
    """curves: {case_name: {(f, d, delay): reps}} (from toy_curve / a cache)."""
    cases = toy_cases() if cases is None else cases
    return [(case["name"], toy_runset(case, curves[case["name"]]), sorted(case["expect"])) for case in cases]


def run_toy_curves(jobs=6, cases=None):
    from multiprocessing import Pool

    out = {}
    with Pool(jobs) as pool:
        for name, key, reps in pool.imap_unordered(toy_curve, toy_jobs(cases)):
            out.setdefault(name, {})[key] = reps
    return out


def save_curves(curves, path):
    with open(path, "w") as fh:
        json.dump(
            {
                name: {"/".join(map(str, k)): {str(n): v for n, v in reps.items()} for k, reps in cs.items()}
                for name, cs in curves.items()
            },
            fh,
        )


def load_curves(path):
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for name, cs in raw.items():
        for k, reps in cs.items():
            f, d, dl = k.split("/")
            out.setdefault(name, {})[(f, int(d), dl)] = {int(n): v for n, v in reps.items()}
    return out


# ---------------------------------------------------------------- published Firestorm knees


def firestorm(path=FIRESTORM):
    with open(path) as fh:
        fx = yaml.safe_load(fh)
    rs = RunSet(
        fx["platform"],
        fx["core"],
        ctx={"emitter": fx["emitter"], "platform": fx["platform"]},
        widths=fx.get("widths"),
    )
    off = {"value": float(fx["offset"]), "sd": 0.0, "calibration": "published", "calibrated": True}
    for c in fx["curves"]:
        rs.add(
            Curve.from_knee(
                c["filler"],
                c["knee"],
                depth=c.get("depth", 1),
                delay=c["delay"],
                offset=dict(off),
                source=c["src"],
            )
        )
    return fx["core"], rs, sorted(fx["expect"])


# ---------------------------------------------------------------- gem5 results

GEM5 = os.path.join(ROOT, "results", "gem5")
GEM5_WIDTH = 8  # O3 default fetch/decode/rename/dispatch/issue width (validation/gem5/config_o3.py)


def gem5_runset(name, tags):
    return load_results(
        [os.path.join(GEM5, t) for t in tags],
        platform="gem5",
        core=name,
        ctx={"emitter": "riscv64", "platform": "gem5"},
        widths={"default": GEM5_WIDTH},
        offsets=offsets_table(),
    )


def gem5_rob_cases():
    """P1.1 campaign: one RunSet per ROB size (depths 1 and 2), plus the protocol run."""
    if not os.path.isdir(GEM5):
        return []
    out = []
    sizes = sorted(
        {
            int(t.split("rob")[1].split("-")[0])
            for t in os.listdir(GEM5)
            if re.fullmatch(r"p11-rob\d+-d[12]", t) and os.path.exists(os.path.join(GEM5, t, "knees.json"))
        }
    )
    for r in sizes:
        tags = [
            t
            for t in (f"p11-rob{r}-d1", f"p11-rob{r}-d2")
            if os.path.exists(os.path.join(GEM5, t, "knees.json"))
        ]
        out.append((f"gem5 p11 rob{r}", gem5_runset(f"p11-rob{r}", tags), []))
    if os.path.exists(os.path.join(GEM5, "proto-rob64-d2", "knees.json")):
        out.append(("gem5 proto rob64 d2", gem5_runset("proto-rob64-d2", ["proto-rob64-d2"]), []))
    return out


def gem5_campaign_cases():
    """Finished validation/gem5/backend_campaign.py runs (P1.1-P1.3), with a P1.3
    size's miss and fp12 tags in one RunSet. gem5 allocates the LSQ at dispatch,
    so only offset.uncalibrated / offset.missing may fire."""
    if not os.path.isdir(GEM5):
        return []
    done = sorted(
        t
        for t in os.listdir(GEM5)
        if t.startswith(("p11-", "p12-", "p13-")) and os.path.exists(os.path.join(GEM5, t, "knees.json"))
    )
    groups = {}
    for t in done:
        groups.setdefault(
            t[: -len("-fp12")] if t.endswith("-fp12") and t.startswith("p13-") else t, []
        ).append(t)
    out = []
    for name, tags in sorted(groups.items()):
        rs = gem5_runset(name, tags)
        rs.resolve_offsets()
        unc = any(
            c.offset is not None and not c.offset["calibrated"] for c in rs.curves.values() if c.has_knee()
        )
        miss = any(c.offset is None for c in rs.curves.values() if c.has_knee())
        out.append(
            (
                f"gem5 {name}" + ("+fp12" if len(tags) > 1 else ""),
                rs,
                sorted((["offset.uncalibrated"] if unc else []) + (["offset.missing"] if miss else [])),
            )
        )
    return out


def all_fast_cases():
    """Everything that needs no simulation: synthetic, cross, published, gem5."""
    return (
        [("synth",) + c for c in synth_cases()]
        + [("cross",) + c for c in cross_cases()]
        + [("pub",) + firestorm()]
        + [("gem5",) + c for c in gem5_rob_cases() + gem5_campaign_cases()]
    )
