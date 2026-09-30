"""Calibrated knee-to-size offsets (P0.1): structure size = knee + offset.

The offset counts the entries the miss loads and loop overhead hold, per emitter, filler, depth and
delay (toy and gem5 give 2, 3 at depths 1, 2, not the textbook 2 * depth). `lookup()` takes the exact
context's entry, else the toy's for the same filler, depth and delay, marked uncalibrated.
"""

import os

import yaml

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets.yaml")
KEYS = ("emitter", "platform", "depth", "delay")
# the toy executes the abstract op stream directly (one-op loop branch)
TOY_CTX = {"emitter": "abstract", "platform": "toy", "depth": 1, "delay": "miss"}


def load(path=PATH):
    with open(path) as fh:
        return yaml.safe_load(fh) or []


def _find(table, filler, ctx):
    for e in table:
        if e["filler"] == filler and all(e[k] == ctx[k] for k in KEYS):
            return e
    return None


def offset_analytic(filler, depth, delay="miss", table=None):
    """The toy model's offset for this filler, depth and delay."""
    e = _find(load() if table is None else table, filler, {**TOY_CTX, "depth": depth, "delay": delay})
    if e is None:
        raise KeyError(
            f"no toy offset for {filler} depth {depth} delay {delay}: "
            f"run validation/toy/calibrate_offsets.py for it"
        )
    return e


def lookup(filler, ctx, table=None):
    """{value, sd, calibration, calibrated} for one filler in context `ctx`;
    value None when not even the toy has an offset for it."""
    table = load() if table is None else table
    e = _find(table, filler, ctx)
    if e is not None:
        return {
            "value": e["offset"],
            "sd": e.get("offset_sd", 0.0),
            "calibration": e["id"],
            "calibrated": True,
        }
    try:
        e = offset_analytic(filler, ctx["depth"], ctx["delay"], table)
    except KeyError:
        return {"value": None, "sd": 0.0, "calibration": "none", "calibrated": False}
    return {
        "value": e["offset"],
        "sd": e.get("offset_sd", 0.0),
        "calibration": f"analytic:{e['id']}",
        "calibrated": False,
    }
