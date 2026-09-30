"""Synthetic window-probe curves whose `knee` is the last N on the low plateau.

    step       lo up to `knee`, hi after it
    sigmoid    logistic centred on knee + 0.5, 10-90 % rise over `width` N
    staircase  `steps` equal sub-steps from knee + 1, one every max(1, width) N (toy depth >= 2)

Noise is value * (1 + noise_sd * N(0, 1)) per repetition. `outliers` ((n, factor), ...) scale
repetition 0; `drift` (from_rep, factor) scales every repetition from `from_rep` on.
"""

import math

import numpy as np


def synth_curve(
    ns,
    knee,
    lo,
    hi,
    *,
    shape="step",
    width=0,
    steps=1,
    slope=0.0,
    noise_sd=0.0,
    outliers=(),
    drift=None,
    n_reps=5,
    seed=0,
):
    """Return {n: [value per repetition]}."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        if shape == "step":
            y = lo if n <= knee else hi
        elif shape == "sigmoid":
            s = max(width, 1e-9) / (2 * math.log(9))
            y = lo + (hi - lo) / (1 + math.exp(-(n - knee - 0.5) / s))
        elif shape == "staircase":
            gap = max(1, width)
            done = 0 if n <= knee else min(steps, 1 + (n - knee - 1) // gap)
            y = lo + (hi - lo) * done / steps
        else:
            raise ValueError(shape)
        y += slope * max(0, n - knee)
        reps = y * (1 + noise_sd * rng.standard_normal(n_reps))
        out[n] = [float(v) for v in reps]
    for n, factor in outliers:
        out[n][0] *= factor
    if drift:
        r0, factor = drift
        for v in out.values():
            for r in range(r0, n_reps):
                v[r] *= factor
    return out


def grid(knee, coarse_step=4, span=40, pad=8):
    """Coarse grid over knee +- span (offset from the knee, as in a real sweep)
    plus step 1 over knee +- pad."""
    coarse = range(knee - span + 1, knee + span + 1, coarse_step)
    fine = range(knee - pad, knee + pad + 1)
    return sorted(set(coarse) | set(fine))


# Curves the anomaly rules must classify; `rules` lists the rule ids each case
# fires. Plateaus are ~1 and ~2 miss latencies of 300 cycles.
CASES = [
    dict(name="clean_step", knee=62, kw=dict(lo=300, hi=600), rules=[]),
    dict(name="noisy_step", knee=62, kw=dict(lo=300, hi=600, noise_sd=0.03), rules=[]),
    dict(
        name="staircase_d2",
        knee=61,
        kw=dict(lo=600, hi=1200, shape="staircase", steps=2),
        expected_steps=2,
        rules=[],
    ),
    dict(
        name="sigmoid_w10",
        knee=62,
        kw=dict(lo=300, hi=600, shape="sigmoid", width=10, noise_sd=0.01),
        rules=["jitter.band"],
    ),
    dict(name="soft", knee=62, kw=dict(lo=300, hi=600, shape="sigmoid", width=60), rules=["shape.poor_fit"]),
    dict(name="tiny_jump", knee=62, kw=dict(lo=300, hi=360, noise_sd=0.02), rules=["noise.unstable_knee"]),
    dict(name="drift", knee=62, kw=dict(lo=300, hi=600, drift=(3, 1.05)), rules=["drift.reps"]),
    dict(name="outlier", knee=62, kw=dict(lo=300, hi=600, outliers=((50, 3.0),)), rules=["noise.outlier"]),
]
