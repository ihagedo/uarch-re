"""Knee detection for window-probe sweeps, plus cross-filler attribution.

The knee is the change point of the best two-segment least-squares fit to cycles/iter vs filler count
N, read as the last N on the low segment. A depth >= 2 curve is a staircase; `expected_steps` > 1 moves
the knee back to its onset so every depth reads the same point. `attribute()` turns the knees of
several fillers into structure estimates with calibrated offsets.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from . import offsets as _offsets


@dataclass
class Knee:
    knee: float | None
    ci: list[float] | None
    jump_ratio: float
    quality: float
    anomaly: str | None = None
    flags: list[str] = field(default_factory=list)
    ci_method: str = "bracket"
    ci_level: float | None = None
    n_reps: int = 1
    resolution: list[float] | None = None
    frac_no_knee: float | None = None
    samples: np.ndarray | None = field(default=None, repr=False)  # bootstrap knees

    def to_dict(self):
        d = asdict(self)
        d.pop("samples")
        return d


def _prefix_fits(x, Y):
    """Least-squares line through the first m points, m = 0..n, for every row
    of Y at once (running sums, centered for precision). x: (n,), Y: (B, n).
    Returns (sse, val), each (B, n + 1): the line's SSE (0 below 2 points) and
    its value at (x[m-1] + x[m]) / 2, the gap after its last point."""
    n = len(x)
    x0, y0 = x.mean(), Y.mean(axis=1, keepdims=True)
    xc, Yc = x - x0, Y - y0

    def cs(a):
        a = np.broadcast_to(a, Y.shape)
        return np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)

    m = np.arange(n + 1, dtype=float)
    sx, sxx = cs(xc), cs(xc * xc)
    sy, syy, sxy = cs(Yc), cs(Yc * Yc), cs(Yc * xc)
    at = np.full(n + 1, np.nan)
    at[1:n] = (xc[:-1] + xc[1:]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cxx = sxx - sx * sx / m
        cxy = sxy - sx * sy / m
        cyy = syy - sy * sy / m
        slope = np.where(cxx > 0, cxy / np.where(cxx > 0, cxx, 1), 0)
        sse = cyy - slope * cxy
        val = y0 + sy / m + slope * (at - sx / m)
    sse[:, :2] = 0
    return np.maximum(sse, 0), val


def _split(x, Y):
    """Best two-segment split per row, left = first k points, k in [2, n - 2]: (k, quality, jump).

    jump = max(right line / left line between x[k-1] and x[k], mean of 3 points after / 3 before);
    the second covers a 2-point segment at the grid's edge, where the line ratio extrapolates wildly."""
    n = len(x)
    pre, pre_val = _prefix_fits(x, Y)
    suf, suf_val = (a[:, ::-1] for a in _prefix_fits(x[::-1], Y[:, ::-1]))  # [:, k] = points k..n-1
    two = pre[:, 2 : n - 1] + suf[:, 2 : n - 1]
    k = 2 + np.argmin(two, axis=1)
    rows = np.arange(len(k))
    sse2 = two[rows, k - 2]
    sse1 = pre[:, n]
    lo, hi = pre_val[rows, k], suf_val[rows, k]
    c = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Y, axis=1)], axis=1)
    a, b = np.maximum(k - 3, 0), np.minimum(k + 3, n)
    lo3 = (c[rows, k] - c[rows, a]) / (k - a)
    hi3 = (c[rows, b] - c[rows, k]) / (b - k)
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.where(sse1 > 0, 1 - sse2 / sse1, 0.0)
        jump = np.maximum(np.where(lo > 0, hi / lo, 1.0), np.where(lo3 > 0, hi3 / lo3, 1.0))
    return k, quality, jump


def _onset(y, k, expected_steps):
    """Move a split back over a staircase's intermediate sub-steps."""
    lo = np.median(y[max(0, k - 5) : k])
    hi = np.median(y[k : k + 5])
    thr = lo + (hi - lo) * 0.5 / expected_steps
    while k > 1 and y[k - 1] > thr:
        k -= 1
    return k


def knees_rows(x, Y, min_jump=1.3, expected_steps=1, split=False):
    """Knee split for every row of Y (sorted x). Returns (k, jump, quality),
    with k = -1 where the step is smaller than `min_jump`; with `split`, also
    the two-segment split before any onset correction."""
    k, quality, jump = _split(x, Y)
    rows = np.arange(Y.shape[0])
    ks = k
    if expected_steps > 1:
        k = np.array([_onset(Y[r], kk, expected_steps) for r, kk in zip(rows, k)])
    k = np.where(jump < min_jump, -1, k)
    return (k, jump, quality, ks) if split else (k, jump, quality)


def find_knee(ns, ys, min_jump=1.3, expected_steps=1) -> Knee:
    x = np.asarray(ns, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < 4:
        return Knee(None, None, 1.0, 0.0, "too few points")

    k, jump, quality, ks = (
        float(v[0]) for v in knees_rows(x, y[None, :], min_jump, expected_steps, split=True)
    )
    k, ks = int(k), int(ks)
    if k < 0:
        return Knee(None, None, jump, quality, f"no step: max ratio {jump:.2f} < {min_jump}")

    # capacity = last N that still fits; the true value lies in [x0, x1 - 1]
    x0, x1 = x[k - 1], x[k]

    anomaly, flags = None, []
    # a second step of comparable size on the high side means two mechanisms;
    # checked from the split, so a staircase's own sub-steps don't count
    if ks + 4 < n:
        tail = y[ks:]
        rel = np.diff(tail) / tail[:-1]
        if np.any(rel > 0.4):
            anomaly = "second step after the knee"
            flags.append("shape.extra_step")
    if quality < 0.6:
        anomaly = (anomaly + "; " if anomaly else "") + f"poor two-segment fit ({quality:.2f})"
        flags.append("shape.poor_fit")

    ci = [float(x0), float(x1 - 1)]
    return Knee(float(x0), ci, jump, quality, anomaly, flags, resolution=list(ci))


STRUCT = {
    "nop": "rob",
    "int": "int_prf_headroom",
    "fp": "fp_prf_headroom",
    "load": "lq",
    "store": "sq",
    "branch": "rob_branch",
    "flags": "flags_prf_headroom",
    "mix": "shared_window",
}

# "info" leaves the value standing; "ambiguous" and "anomaly" replace it with a
# bound in the spec. anomaly.py holds the full rule registry.
SEVERITY = {
    "shape.extra_step": "anomaly",
    "shape.poor_fit": "anomaly",
    "tie.rob": "ambiguous",
    "noise.unstable_knee": "anomaly",
    "offset.uncalibrated": "info",
    "offset.missing": "anomaly",
}


def attribute(
    knees: dict[str, Knee], ctx: dict | None = None, offsets=None, tol: float = 0.08
) -> dict[str, dict]:
    """Turn per-filler knees into structure estimates with findings.

    `ctx` (emitter, platform, depth, delay) selects the offset, default the toy at depth 1 with a
    miss delay. The reported CI widens the knee CI by 2 sd of the offset.
    """
    ctx = {**_offsets.TOY_CTX, **(ctx or {})}
    table = _offsets.load() if offsets is None else offsets
    off = {f: _offsets.lookup(f, ctx, table) for f in knees}
    out = {}
    rob = knees.get("nop")
    rob_size = (
        rob.knee + off["nop"]["value"]
        if rob and rob.knee is not None and off["nop"]["value"] is not None
        else None
    )
    for filler, kn in knees.items():
        struct = STRUCT[filler]
        o = off[filler]
        entry = {
            "filler": filler,
            "value": None,
            "ci": None,
            "status": "unmeasured",
            "knee": kn.knee,
            "offset": {k: o[k] for k in ("value", "sd", "calibration")},
            "ci_method": kn.ci_method,
            "ci_level": kn.ci_level,
            "n_reps": kn.n_reps,
            "anomalies": list(kn.flags),
        }
        if kn.knee is None:
            entry["status"] = "no knee: " + (kn.anomaly or "")
            out[struct] = entry
            continue
        if o["value"] is None:  # no offset anywhere: the knee is all we have
            entry["anomalies"].append("offset.missing")
            o = {**o, "value": 0.0}
            entry["offset"]["value"] = None
        elif not o["calibrated"]:
            entry["anomalies"].append("offset.uncalibrated")
        size = kn.knee + o["value"]
        ci = [kn.ci[0] + o["value"] - 2 * o["sd"], kn.ci[1] + o["value"] + 2 * o["sd"]]
        res = kn.resolution or kn.ci
        entry.update(
            value=round(size, 1),
            ci=ci,
            status="measured",
            resolution=[res[0] + o["value"], res[1] + o["value"]],
        )
        if kn.anomaly:
            entry["status"] = "anomaly: " + kn.anomaly
        if filler != "nop" and rob_size is not None:
            if size >= rob_size * (1 - tol):
                entry["anomalies"].append("tie.rob")
                entry["status"] = (
                    f"ambiguous: knee {size:.0f} is within {tol:.0%} of ROB "
                    f"{rob_size:.0f}; may be ROB-limited, report as >= {size:.0f}"
                )
        out[struct] = entry
    return out
