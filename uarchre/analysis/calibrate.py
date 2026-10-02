"""Offset calibration on ground truth (P0.1).

Fits truth = slope * knee + intercept per (emitter, platform, filler, depth, delay) over finished
sweeps (results/<platform>/<tag>/knees.json) and scores each configuration leave-one-out, so the
reported error is a prediction. Pass: slope in [0.98, 1.02] and offset sd <= 0.5.
"""

import glob
import json
import os

import numpy as np
import yaml

from . import offsets as _offsets
from .knee import STRUCT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMITTER = {"toy": "abstract", "gem5": "riscv64", "native": "x86_64"}


def tolerance(truth):
    return max(2.0, 0.02 * truth)


def load_points(pattern, platform="gem5"):
    """[{tag, filler, depth, delay, emitter, truth, knee, ci, quality, anomaly}] from
    results/<platform>/<pattern>/knees.json (glob)."""
    pts = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", platform, pattern, "knees.json"))):
        with open(path) as fh:
            kj = json.load(fh)
        ctx = kj.get("context") or {"depth": 2 if "-d2" in path else 1, "delay": "miss"}
        for filler, k in kj["knees"].items():
            key = STRUCT.get(filler)
            if key is None or kj.get("truth") is None or key not in kj["truth"]:
                continue
            pts.append(
                dict(
                    tag=kj["tag"],
                    filler=filler,
                    depth=ctx["depth"],
                    delay=ctx["delay"],
                    emitter=ctx.get("emitter", EMITTER[platform]),
                    platform=platform,
                    truth=kj["truth"][key],
                    knee=k["knee"],
                    ci=k["ci"],
                    quality=k["quality"],
                    anomaly=k["anomaly"],
                    flags=k.get("flags", []),
                    n_reps=k.get("n_reps", 1),
                    curve_path=os.path.join(os.path.dirname(path), "sweep.csv"),
                )
            )
    return pts


def fit(points):
    """Fit one group. Returns dict(slope, intercept, offset, offset_sd, n, ok, loo=[...])."""
    pts = [p for p in points if p["knee"] is not None]
    if len(pts) < 3:
        return None
    knee = np.array([p["knee"] for p in pts], float)
    truth = np.array([p["truth"] for p in pts], float)
    slope, intercept = np.polyfit(knee, truth, 1)
    diff = truth - knee
    loo = []
    for i in range(len(pts)):
        others = np.delete(diff, i)
        off = float(np.median(others))
        loo.append(
            dict(
                tag=pts[i]["tag"],
                truth=truth[i],
                knee=knee[i],
                offset=off,
                estimate=knee[i] + off,
                error=knee[i] + off - truth[i],
                ci=[c + off for c in pts[i]["ci"]] if pts[i]["ci"] else None,
            )
        )
    sd = float(np.std(diff))
    return dict(
        slope=float(slope),
        intercept=float(intercept),
        offset=float(np.median(diff)),
        offset_sd=sd,
        n=len(pts),
        ok=bool(0.98 <= slope <= 1.02 and sd <= 0.5),
        loo=loo,
    )


def groups(points):
    out = {}
    for p in points:
        out.setdefault((p["emitter"], p["platform"], p["filler"], p["depth"], p["delay"]), []).append(p)
    return out


def write_offsets(fits, evidence, calibrated_on="gem5-o3-v25.1 (RISC-V O3)"):
    """Merge passing fits {group key: fit} into offsets.yaml (by id)."""
    entries = []
    for (emitter, platform, filler, depth, delay), f in fits.items():
        if not f or not f["ok"]:
            continue
        entries.append(
            {
                "id": f"{emitter}.{platform}.{filler}.d{depth}.{delay}",
                "emitter": emitter,
                "platform": platform,
                "calibrated_on": calibrated_on,
                "filler": filler,
                "depth": depth,
                "delay": delay,
                "offset": round(f["offset"]),
                "offset_sd": round(f["offset_sd"], 3),
                "slope": round(f["slope"], 4),
                "n_configs": f["n"],
                "evidence": evidence,
            }
        )
    ids = {e["id"] for e in entries}
    table = [e for e in _offsets.load() if e["id"] not in ids]
    with open(_offsets.PATH) as fh:
        header = "".join(line for line in fh if line.startswith("#"))
    with open(_offsets.PATH, "w") as fh:
        fh.write(header)
        yaml.safe_dump(table + entries, fh, sort_keys=False)
    return entries
