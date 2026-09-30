"""Knee detection, bootstrap CIs and the sweep protocol on synthetic curves.

python3 -m unittest tests.test_analysis
"""

import unittest

import numpy as np

from tests.fixtures.synth import synth_curve, grid
from uarchre.analysis.knee import find_knee, _split
from uarchre.analysis.bootstrap import bootstrap_knee
from uarchre.analysis.protocol import run_protocol
from uarchre.backends import RunResult


def reference_split(x, y):
    """Plain per-split np.linalg.lstsq loop; returns (k, quality)."""

    def sse(xs, ys):
        if len(xs) < 2:
            return 0.0
        A = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(np.sum((A @ coef - ys) ** 2))

    n = len(x)
    best = (np.inf, None)
    for k in range(2, n - 1):
        s = sse(x[:k], y[:k]) + sse(x[k:], y[k:])
        if s < best[0]:
            best = (s, k)
    s1 = sse(x, y)
    return best[1], 1 - best[0] / s1 if s1 > 0 else 0.0


def medians(reps):
    ns = sorted(reps)
    return ns, [float(np.median(reps[n])) for n in ns]


class TestSplit(unittest.TestCase):
    def test_matches_reference_loop(self):
        rng = np.random.default_rng(1)
        for t in range(200):
            x = np.sort(rng.choice(np.arange(1, 200), size=rng.integers(5, 40), replace=False)).astype(float)
            knee = rng.integers(int(x[1]), int(x[-3]) + 1)
            y = np.where(x <= knee, 300.0, 300.0 * rng.uniform(1.1, 3)) * (
                1 + 0.05 * rng.standard_normal(len(x))
            )
            k, q, _ = _split(x, y[None])
            rk, rq = reference_split(x, y)
            self.assertEqual(int(k[0]), rk, t)
            self.assertAlmostEqual(float(q[0]), rq, places=6)


class TestSynth(unittest.TestCase):
    def test_shapes_put_the_knee_where_asked(self):
        ns = list(range(40, 90))
        for kw, steps in [
            ({}, 1),
            (dict(shape="sigmoid", width=10), 1),
            (dict(shape="staircase", steps=2), 2),
            (dict(shape="staircase", steps=3), 2),
        ]:
            ns_, ys = medians(synth_curve(ns, 62, 300, 600, n_reps=1, **kw))
            self.assertEqual(find_knee(ns_, ys, expected_steps=steps).knee, 62, kw)

    def test_seeded(self):
        a = synth_curve(range(10), 5, 1, 2, noise_sd=0.1, seed=3)
        self.assertEqual(a, synth_curve(range(10), 5, 1, 2, noise_sd=0.1, seed=3))
        self.assertNotEqual(a, synth_curve(range(10), 5, 1, 2, noise_sd=0.1, seed=4))

    def test_staircase_extra_steps_are_not_a_second_mechanism(self):
        ns, ys = medians(synth_curve(range(30, 90), 61, 600, 1200, shape="staircase", steps=3, n_reps=1))
        k = find_knee(ns, ys, expected_steps=2)
        self.assertEqual(k.knee, 61)
        self.assertNotIn("shape.extra_step", k.flags)

    def test_step_at_the_grid_edge(self):
        """window_sweep 'separated' case: one point below the step on the coarse grid."""
        from uarchre.analysis.sweep import sweep_refine

        run = lambda n: 400 if n <= 58 else 800 + n
        ns = list(range(57, 129, 4))
        self.assertIsNotNone(find_knee(ns, [run(n) for n in ns]).knee)  # coarse pass must see the step
        self.assertEqual(sweep_refine(run, ns)[2].knee, 58)

    def test_ramp_is_not_a_step(self):
        ns = list(range(1, 80))
        self.assertIsNone(find_knee(ns, [300 + 4 * n for n in ns]).knee)


class TestBootstrap(unittest.TestCase):
    def test_single_rep_is_a_bracket(self):
        k = bootstrap_knee(synth_curve(range(40, 90, 4), 62, 300, 600, n_reps=1))
        self.assertEqual(k.ci_method, "bracket")
        self.assertEqual(k.ci, [60.0, 63.0])  # coarse grid: last below 60, first above 64

    def test_deterministic_reps_give_zero_width(self):
        k = bootstrap_knee(synth_curve(grid(62), 62, 300, 600, n_reps=3))
        self.assertEqual((k.ci_method, k.ci, k.frac_no_knee), ("bootstrap", [62.0, 62.0], 0.0))

    def test_tiny_jump_is_unstable(self):
        k = bootstrap_knee(synth_curve(grid(62), 62, 300, 370, noise_sd=0.06, n_reps=5, seed=2))
        self.assertIn("noise.unstable_knee", k.flags)

    def test_coverage(self):
        """P0.2: the 95 % interval must not under-cover. Exact knees over-cover, so
        only the floor is checked; the hardest case lands in the 93-97 % band."""
        cases = [
            ("step", dict(lo=300, hi=450, noise_sd=0.25), 1),
            ("sigmoid", dict(lo=300, hi=600, shape="sigmoid", width=10, noise_sd=0.10), 1),
            ("staircase", dict(lo=600, hi=1200, shape="staircase", steps=2, noise_sd=0.15), 2),
        ]
        for name, kw, steps in cases:
            hit = 0
            for t in range(500):
                reps = synth_curve(grid(62), 62, n_reps=3, seed=t, **kw)
                k = bootstrap_knee(reps, n_boot=1000, seed=t, expected_steps=steps)
                hit += k.ci is not None and k.ci[0] <= 62 <= k.ci[1]
            with self.subTest(shape=name):
                self.assertGreaterEqual(hit / 500, 0.93, f"{name}: coverage {hit / 500:.3f}")


class TestProtocol(unittest.TestCase):
    def test_two_phase_shuffled(self):
        calls = []

        def measure(key, n, rep):
            calls.append((key, n, rep))
            y = (300 if n <= {"a": 62, "b": 30}[key] else 600) * (1 + 0.01 * rep)
            return RunResult([y, y + 1])

        records, reps = run_protocol(measure, ["a", "b"], list(range(4, 100, 8)), outer_reps=3, seed=1)
        self.assertEqual(sorted(r.order_index for r in records), list(range(len(records))))
        rep0 = [c for c in calls if c[2] == 0]
        coarse = [c for c in rep0 if c[1] in range(4, 100, 8)]
        self.assertNotEqual(
            [c[1] for c in coarse if c[0] == "a"], sorted(c[1] for c in coarse if c[0] == "a")
        )
        self.assertEqual(len({c[0] for c in coarse[:6]}), 2, "keys should interleave")
        for key, knee in (("a", 62), ("b", 30)):
            for n in range(knee - 8, knee + 9):  # the step-1 grid around the knee
                self.assertEqual(len(reps[key][n]), 3, (key, n))
            ns, ys = medians(reps[key])
            self.assertEqual(find_knee(ns, ys).knee, knee)
        self.assertEqual(reps["a"][4], [300.5, 303.5, 306.5])  # per-launch medians, rep order


if __name__ == "__main__":
    unittest.main()
