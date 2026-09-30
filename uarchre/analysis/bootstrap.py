"""Bootstrap CI for a window-probe knee (P0.2).

`reps` is {N: [median of one launch's inner reps, per outer rep]}. Each resample redraws every N's
reps and refits; the CI runs from the lower quantile of bracket starts to the upper quantile of bracket
ends, so it never claims more than the grid resolves. One rep gives the bracket alone. If more than
UNSTABLE of the resamples find no step, the knee is flagged and the CI widens to the step-1 run around it.
"""

import numpy as np

from .knee import Knee, find_knee, knees_rows

UNSTABLE = 0.05


def _fine_run(x, i):
    """Bounds of the contiguous step-1 run of x containing index i."""
    a = b = i
    while a > 0 and x[a - 1] == x[a] - 1:
        a -= 1
    while b + 1 < len(x) and x[b + 1] == x[b] + 1:
        b += 1
    return float(x[a]), float(x[b])


def bootstrap_knee(reps, *, n_boot=2000, level=0.95, min_jump=1.3, seed=0, expected_steps=1) -> Knee:
    ns = sorted(reps)
    x = np.array(ns, dtype=float)
    R = [np.asarray(reps[n], dtype=float) for n in ns]
    point = find_knee(x, [np.median(r) for r in R], min_jump, expected_steps)
    point.n_reps = min(len(r) for r in R)
    point.resolution = point.ci
    if point.n_reps < 2 or len(x) < 4:
        return point

    rng = np.random.default_rng(seed)
    Y = np.empty((n_boot, len(x)))
    for j, r in enumerate(R):
        Y[:, j] = np.median(r[rng.integers(0, len(r), size=(n_boot, len(r)))], axis=1)
    k, _, _ = knees_rows(x, Y, min_jump, expected_steps)
    found = k >= 0
    point.ci_method, point.ci_level = "bootstrap", level
    point.frac_no_knee = float(1 - found.mean())
    point.samples = np.where(found, x[np.maximum(k, 1) - 1], np.nan)
    if point.frac_no_knee > UNSTABLE:
        point.flags.append("noise.unstable_knee")
    if point.knee is None or not found.any():
        return point

    alpha = 1 - level
    lo = float(np.quantile(x[k[found] - 1], alpha / 2, method="lower"))
    hi = float(np.quantile(x[k[found]] - 1, 1 - alpha / 2, method="higher"))
    lo, hi = min(lo, point.ci[0]), max(hi, point.ci[1])
    if point.frac_no_knee > UNSTABLE:
        a, b = _fine_run(x, int(np.searchsorted(x, point.knee)))
        lo, hi = min(lo, a), max(hi, b)
    point.ci = [lo, hi]
    return point
