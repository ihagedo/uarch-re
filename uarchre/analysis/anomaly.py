"""Anomaly rules A1-A12: named reasons not to trust a knee.

`evaluate()` runs every rule on a RunSet; `apply()` turns a flagged value into a bound.
Thresholds are in thresholds.yaml.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import os

import numpy as np
import yaml

from . import offsets as _offsets
from .bootstrap import bootstrap_knee
from .knee import Knee, STRUCT as _STRUCT, _split

THRESHOLDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.yaml")

STRUCT = {"branch": "rob_branch", "flags": "flags_prf_headroom", **_STRUCT}  # knee.STRUCT wins
FAILABLE = ("load", "store", "branch")
ALU = ("fp", "int", "alu")
DETERMINISTIC = ("toy", "gem5")  # platforms where rule A8 (drift) does not apply

RULE_IDS = {
    "A1": "shape.extra_step",
    "A2": "shape.poor_fit",
    "A3": "tie.rob",
    "A4": "ratio.retire_group",
    "A5": "delay.disagreement",
    "A6": "jitter.band",
    "A7": "offset.depth_inconsistent",
    "A8": "drift.reps",
    "A9": "noise.outlier",
    "A10": "throughput.bound",
    "A11": "offset.uncalibrated",
    "A12": "noise.unstable_knee",
    "A11b": "offset.missing",
}
CODE = {v: k for k, v in RULE_IDS.items()}
SEVERITY = {
    "shape.extra_step": "anomaly",
    "shape.poor_fit": "anomaly",
    "tie.rob": "ambiguous",
    "ratio.retire_group": "anomaly",
    "delay.disagreement": "ambiguous",
    "jitter.band": "info",
    "offset.depth_inconsistent": "anomaly",
    "drift.reps": "anomaly",
    "noise.outlier": "info",
    "throughput.bound": "anomaly",
    "offset.uncalibrated": "info",
    "noise.unstable_knee": "anomaly",
    "offset.missing": "anomaly",
}

Key = tuple[str, int, str]  # (filler, depth, delay)


def load_thresholds(path=THRESHOLDS):
    with open(path) as fh:
        return yaml.safe_load(fh)


def probe_params(depth, delay="miss"):
    """(min_jump, expected_steps) as sweep.py uses them for this depth and delay."""
    min_jump = 1 + 0.4 / depth if delay == "miss" else 1.3
    return min_jump, 1 if depth == 1 else 2


def is_alu(delay):
    return delay in ALU


# --------------------------------------------------------------------------- data


@dataclass
class Curve:
    """One sweep of one filler under one condition.

    `reps` is {N: [median per outer rep]}, None for published knees (cross-curve rules only).
    `offset` is set by RunSet.resolve_offsets(); `order` maps (N, outer_rep) to order_index (A8c)."""

    filler: str
    depth: int = 1
    delay: str = "miss"
    reps: dict[int, list[float]] | None = None
    knee: Knee | None = None
    expected_steps: int = 1
    offset: dict | None = None
    order: dict[tuple[int, int], int] | None = None
    source: str = ""

    @classmethod
    def from_reps(cls, filler, reps, depth=1, delay="miss", *, n_boot=2000, seed=0, **kw):
        min_jump, steps = probe_params(depth, delay)
        reps = {int(n): [float(v) for v in vs] for n, vs in reps.items()}
        k = bootstrap_knee(reps, n_boot=n_boot, seed=seed, min_jump=min_jump, expected_steps=steps)
        return cls(filler, depth, delay, reps, k, steps, **kw)

    @classmethod
    def from_knee(cls, filler, knee, depth=1, delay="miss", **kw):
        """Knee-level data: `knee` is a number or a [lo, hi] range."""
        lo, hi = (knee, knee) if np.isscalar(knee) else knee
        k = Knee(
            float(lo + hi) / 2, [float(lo), float(hi)], float("nan"), float("nan"), ci_method="published"
        )
        return cls(filler, depth, delay, None, k, 1, **kw)

    @property
    def key(self) -> Key:
        return (self.filler, self.depth, self.delay)

    @property
    def structure(self):
        return STRUCT.get(self.filler, self.filler)

    def median_curve(self):
        ns = sorted(self.reps)
        return np.array(ns, float), np.array([np.median(self.reps[n]) for n in ns], float)

    def has_knee(self):
        return self.knee is not None and self.knee.knee is not None

    def estimate(self):
        """(value, [lo, hi]) offset-corrected like attribute(), or None."""
        if not self.has_knee():
            return None
        o = self.offset or {"value": 0.0, "sd": 0.0}
        k = self.knee
        ci = k.ci or [k.knee, k.knee]
        return k.knee + o["value"], [ci[0] + o["value"] - 2 * o["sd"], ci[1] + o["value"] + 2 * o["sd"]]

    def plateau(self):
        """Cycles per iteration on the median curve at the knee."""
        if self.reps is None or not self.has_knee():
            return None
        x, y = self.median_curve()
        i = int(np.searchsorted(x, self.knee.knee, side="right")) - 1
        return float(y[max(i, 0)])


@dataclass
class RunSet:
    """Everything measured on one core in one session.

    `ctx` selects offsets (emitter, platform); `widths` are issue widths per filler ("default" for any,
    rule A10); `clock` is one {"pre", "post", "rep", "filler"} dict per launch."""

    platform: str
    core: str
    curves: dict[Key, Curve] = field(default_factory=dict)
    clock: list[dict] = field(default_factory=list)
    widths: dict[str, float] | None = None
    ctx: dict = field(default_factory=dict)
    offsets: list | None = None
    deterministic: bool | None = None

    def __post_init__(self):
        if self.deterministic is None:  # counter = core cycles, output fixed per seed
            self.deterministic = self.platform in DETERMINISTIC

    def add(self, curve: Curve):
        self.curves[curve.key] = curve
        return curve

    def resolve_offsets(self, force=False):
        table = _offsets.load() if self.offsets is None else self.offsets
        for c in self.curves.values():
            if c.offset is not None and not force:
                continue
            ctx = {**_offsets.TOY_CTX, **self.ctx, "depth": c.depth, "delay": c.delay}
            try:
                c.offset = _offsets.lookup(c.filler, ctx, table)
            except KeyError:  # no usable entry
                c.offset = None
            if c.offset is not None and c.offset.get("value") is None:
                c.offset = None
        return self

    def by_filler(self, filler):
        return [c for c in self.curves.values() if c.filler == filler]

    def width(self, filler):
        if not self.widths:
            return None
        return self.widths.get(filler, self.widths.get("default"))


@dataclass
class Finding:
    rule: str
    severity: str
    structure: str | None
    message: str
    evidence: dict = field(default_factory=dict)
    curves: list[Key] = field(default_factory=list)

    @property
    def code(self):
        return CODE.get(self.rule, "?")

    def to_dict(self):
        return {
            "rule": self.rule,
            "code": self.code,
            "severity": self.severity,
            "structure": self.structure,
            "message": self.message,
            "curves": [list(k) for k in self.curves],
            "evidence": _plain(self.evidence),
        }


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _finding(rule, structure, message, curves=(), **evidence):
    return Finding(rule, SEVERITY[rule], structure, message, evidence, list(curves))


def _ckey(c):
    return f"{c.filler}/d{c.depth}/{c.delay}"


# --------------------------------------------------------------------------- shape helpers


def transitions(x, y, min_rise=0.4, plateau_pts=3, plateau_flat=0.2):
    """Steps in a median curve grouped into transitions (A1): [(x_last_low, x_first_high, rise), ...].

    Recursive binary segmentation with the knee fit; the parameters are described in thresholds.yaml."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    splits = {}

    def rec(a, b, depth):
        if b - a < 4 or depth > 6:
            return
        k = int(_split(x[a:b], y[None, a:b])[0][0])
        lo = float(np.median(y[a + max(0, k - 3) : a + k]))
        hi = float(np.median(y[a + k : a + min(b - a, k + 3)]))
        rise = hi / lo - 1 if lo > 0 else 0.0
        if rise <= min_rise:
            return
        splits[a + k] = rise
        rec(a, a + k, depth + 1)
        rec(a + k, b, depth + 1)

    rec(0, len(x), 0)
    groups = []
    for s in sorted(splits):
        if groups:
            seg = y[groups[-1][-1] : s]
            flat = len(seg) >= plateau_pts and seg.min() > 0 and seg.max() / seg.min() - 1 < plateau_flat
            if not flat:
                groups[-1].append(s)
                continue
        groups.append([s])
    out = []
    for g in groups:
        a, b = g[0], g[-1]
        lo = float(np.median(y[max(0, a - 3) : a]))
        hi = float(np.median(y[b : b + 3]))
        out.append((float(x[a - 1]), float(x[b]), hi / lo - 1 if lo > 0 else 0.0))
    return out


def transition_width(x, y, knee):
    """10-90 % width of the step at `knee` (A6), in N.

    The plateau lines are fitted more than max(2, 2 % of knee) away, so a sloped plateau is not transition."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = max(2.0, 0.02 * knee)
    left, right = x <= knee, x > knee
    lfit, rfit = left & (x < knee - h), right & (x > knee + 1 + h)
    if lfit.sum() < 2:
        lfit = left
    if rfit.sum() < 2:
        rfit = right
    if lfit.sum() < 1 or rfit.sum() < 1:
        return None

    def line(m):
        if m.sum() >= 2 and np.ptp(x[m]) > 0:
            return np.polyfit(x[m], y[m], 1)
        return np.array([0.0, float(np.median(y[m]))])

    L, R = line(lfit), line(rfit)
    xm = knee + 0.5
    J = np.polyval(R, xm) - np.polyval(L, xm)
    if J <= 0:
        return None
    low = left & (y <= np.polyval(L, x) + 0.1 * J)
    high = right & (y >= np.polyval(R, x) - 0.1 * J)
    if not low.any() or not high.any():
        return None
    return float(x[high].min() - x[low].max())


# --------------------------------------------------------------------------- rules


def a1_extra_step(rs, thr):
    """Counts transitions, not splits, so a depth >= 2 staircase is one transition. Scans the whole
    curve: a two-level SQ's smaller step sits before the main knee, where knee.py does not look."""
    t = thr["A1"]
    out = []
    for c in rs.curves.values():
        if c.reps is None or len(c.reps) < 8:
            continue
        x, y = c.median_curve()
        tr = transitions(x, y, t["min_rise"], t["plateau_pts"], t["plateau_flat"])
        if len(tr) > t["max_transitions"]:
            where = ", ".join(f"{a:g}->{b:g} (+{r:.0%})" for a, b, r in tr)
            out.append(
                _finding(
                    "shape.extra_step",
                    c.structure,
                    f"{_ckey(c)}: {len(tr)} separate steps ({where}); two mechanisms stacked",
                    [c.key],
                    transitions=tr,
                )
            )
    return out


def a2_poor_fit(rs, thr):
    out = []
    for c in rs.curves.values():
        if c.reps is None or c.knee is None or math.isnan(c.knee.quality):
            continue
        if c.knee.quality < thr["A2"]["min_quality"]:
            out.append(
                _finding(
                    "shape.poor_fit",
                    c.structure,
                    f"{_ckey(c)}: two-segment fit quality {c.knee.quality:.2f}; no clean step",
                    [c.key],
                    quality=c.knee.quality,
                )
            )
    return out


def _rob_curve(rs, c):
    cands = [n for n in rs.by_filler("nop") if n.has_knee()]
    for pref in (
        lambda n: n.depth == c.depth and n.delay == c.delay,
        lambda n: n.delay == c.delay,
        lambda n: n.depth == c.depth,
        lambda n: True,
    ):
        m = [n for n in cands if pref(n)]
        if m:
            return m[0]
    return None


def a3_tie_rob(rs, thr):
    """Also fires on CI overlap. Branch is exempt: it takes only a ROB entry, so its knee is the ROB."""
    tol = thr["A3"]["tol"]
    out = []
    for c in rs.curves.values():
        if c.filler in ("nop", "branch") or not c.has_knee():
            continue
        rob = _rob_curve(rs, c)
        if rob is None:
            continue
        (v, ci), (r, rci) = c.estimate(), rob.estimate()
        overlap = ci[0] <= rci[1] and rci[0] <= ci[1]
        if v >= (1 - tol) * r or overlap:
            out.append(
                _finding(
                    "tie.rob",
                    c.structure,
                    f"{_ckey(c)}: estimate {v:g} is within {tol:.0%} of the ROB estimate {r:g}"
                    f"{' (CIs overlap)' if overlap else ''}; may be ROB-limited, report >= {v:g}",
                    [c.key, rob.key],
                    value=v,
                    rob=r,
                    ci=ci,
                    rob_ci=rci,
                    bound={"lower": v},
                )
            )
    return out


def a4_retire_group(rs, thr):
    """NOP estimate a whole multiple of the failable fillers' common estimate.

    Each filler at its largest offset-corrected estimate (an ALU delay's depth counts links). Without a
    branch curve r >= 3: gem5's default LQ = SQ = 32 under a 64-entry ROB gives r = 2.07."""
    t = thr["A4"]
    best = {}
    for c in rs.curves.values():
        if c.has_knee() and (c.filler == "nop" or c.filler in FAILABLE):
            v = c.estimate()[0]
            if c.filler not in best or v > best[c.filler][0]:
                best[c.filler] = (v, c)
    fail = {f: best[f] for f in FAILABLE if f in best}
    if "nop" not in best or len(fail) < 2:
        return []
    nop_v, nop = best["nop"]
    ks = {f: v for f, (v, _) in fail.items()}
    R = min(ks.values())
    if R <= 0 or max(ks.values()) > (1 + t["agree"]) * R:
        return []
    r = nop_v / R
    g = round(r)
    min_ratio = t["min_ratio"] if "branch" in fail else t["min_ratio_no_branch"]
    if r < min_ratio or g < 2 or abs(r / g - 1) > t["int_tol"]:
        return []
    if any(nop_v <= t["nop_factor"] * k for k in ks.values()):
        return []
    fs = "/".join(fail)
    return [
        _finding(
            "ratio.retire_group",
            "rob",
            f"NOP estimate {nop_v:g} is {r:.2f}x the {fs} estimate {R:g}: NOP fillers count retire rows x "
            f"group size, not a ROB; candidate retire_rows ~ {R:g}, group_max_uops ~ {g} "
            f"(inferred-from-measurement)",
            [nop.key] + [c.key for _, c in fail.values()],
            ratio=r,
            group=g,
            rows=R,
            estimates=ks,
            nop=nop_v,
            branch=("branch" in fail),
            candidates={"retire_rows": R, "group_max_uops": g},
        )
    ]


def a5_delay(rs, thr):
    """Pairs a miss curve with an ALU curve of the same filler at any depth (offsets are per depth)."""
    t = thr["A5"]
    out = []
    for f in sorted({c.filler for c in rs.curves.values()}):
        miss = [c for c in rs.by_filler(f) if c.delay == "miss" and c.has_knee()]
        alu = [c for c in rs.by_filler(f) if is_alu(c.delay) and c.has_knee()]
        for m in miss:
            for a in sorted(alu, key=lambda a: (a.depth != m.depth, a.depth)):
                (vm, cm), (va, ca) = m.estimate(), a.estimate()
                half = (cm[1] - cm[0]) / 2 + (ca[1] - ca[0]) / 2
                margin = half + max(t["abs_floor"], t["rel"] * (vm + va) / 2)
                if abs(vm - va) > margin:
                    bound = {"lower": min(cm[0], ca[0]), "upper": max(cm[1], ca[1])}
                    out.append(
                        _finding(
                            "delay.disagreement",
                            STRUCT.get(f, f),
                            f"{f}: {vm:g} with a miss delay vs {va:g} with an ALU ({a.delay}) delay "
                            f"(|diff| {abs(vm - va):g} > {margin:.1f}); allocated at issue or "
                            f"competing with the delay: neither is the structure, report bound "
                            f"[{bound['lower']:g}, {bound['upper']:g}]",
                            [m.key, a.key],
                            miss=vm,
                            alu=va,
                            miss_ci=cm,
                            alu_ci=ca,
                            margin=margin,
                            bound=bound,
                        )
                    )
                    break
    return out


def a6_jitter(rs, thr):
    """ALU delays get `alu_ramp_per_link` N per link (toy fp x 8: 4 N, gem5 fsqrt x 12: 5 N)."""
    t = thr["A6"]
    out = []
    for c in rs.curves.values():
        if c.reps is None or not c.has_knee():
            continue
        x, y = c.median_curve()
        k = c.knee.knee
        lim = max(t["abs_width"], t["rel_width"] * k)
        if is_alu(c.delay):  # the delay group retires link by link: a calibrated ramp, not jitter
            lim += t["alu_ramp_per_link"] * c.depth
        w = transition_width(x, y, k)
        ciw = c.knee.ci[1] - c.knee.ci[0] if c.knee.ci_method == "bootstrap" else 0.0
        if (w is not None and w > lim) or ciw > lim:
            out.append(
                _finding(
                    "jitter.band",
                    c.structure,
                    f"{_ckey(c)}: transition {w if w is not None else float('nan'):g} N wide, CI width "
                    f"{ciw:g} (limit {lim:.1f}); tolerance is the band, not a few entries",
                    [c.key],
                    width=w,
                    ci_width=ciw,
                    limit=lim,
                )
            )
    return out


def a7_depth(rs, thr):
    """`slack` entries are allowed between the CIs: deterministic platforms give zero-width CIs."""
    slack = thr["A7"]["slack"]
    out = []
    groups = {}
    for c in rs.curves.values():
        if c.has_knee():
            groups.setdefault((c.filler, c.delay), []).append(c)
    for (f, dl), cs in sorted(groups.items()):
        cs = sorted(cs, key=lambda c: c.depth)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                (vi, ci), (vj, cj) = cs[i].estimate(), cs[j].estimate()
                gap = max(ci[0] - cj[1], cj[0] - ci[1])
                if gap > slack:
                    out.append(
                        _finding(
                            "offset.depth_inconsistent",
                            STRUCT.get(f, f),
                            f"{f}/{dl}: depth {cs[i].depth} gives {vi:g} {ci}, "
                            f"depth {cs[j].depth} gives {vj:g} {cj}; "
                            f"offset model or a throughput-limited shallow run",
                            [cs[i].key, cs[j].key],
                            estimates={cs[i].depth: vi, cs[j].depth: vj},
                            gap=gap,
                        )
                    )
    return out


def _residuals(c):
    """[(n, rep, rel_residual)] against the median curve."""
    x, y = c.median_curve()
    med = dict(zip(x, y))
    return [(n, r, v / med[n] - 1) for n, vs in c.reps.items() for r, v in enumerate(vs) if med[n] > 0]


def a8_drift(rs, thr):
    """(a) clock ratio moved within a launch, (b) outer reps differ, (c) residuals trend with run order.

    (b) needs an effect size as well as p, since native reps also change the chain layout."""
    t = thr["A8"]
    out = []
    if rs.deterministic:
        return out  # a simulator has no clock to drift; rep-to-rep spread is the seed
    bad = [
        p for p in rs.clock if p.get("pre") and p.get("post") and abs(p["post"] / p["pre"] - 1) > t["clock"]
    ]
    if bad:
        out.append(
            _finding(
                "drift.reps",
                None,
                f"clock ratio moved by > {t['clock']:.0%} within {len(bad)} launch(es); rerun them",
                [],
                clock=bad[:10],
            )
        )
    from scipy import stats

    for c in rs.curves.values():
        if c.reps is None:
            continue
        res = _residuals(c)
        nrep = min(len(v) for v in c.reps.values())
        if nrep >= 2:
            groups = [[e for n, r, e in res if r == k] for k in range(nrep)]
            meds = [float(np.median(g)) for g in groups]
            try:
                p = stats.kruskal(*groups).pvalue
            except ValueError:  # all residuals identical: no drift
                p = 1.0
            if p < t["kw_p"] and max(meds) - min(meds) > t["min_effect"]:
                out.append(
                    _finding(
                        "drift.reps",
                        c.structure,
                        f"{_ckey(c)}: outer reps differ (Kruskal-Wallis p={p:.1e}, per-rep median "
                        f"residuals {', '.join(f'{m:+.1%}' for m in meds)}); rerun the outlying reps",
                        [c.key],
                        kw_p=p,
                        rep_medians=meds,
                    )
                )
                continue
        if c.order:
            pts = [(c.order[(n, r)], e) for n, r, e in res if (n, r) in c.order]
            if len(pts) >= 10:
                rho = stats.spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
                if np.isfinite(rho) and abs(rho) > t["rho"]:
                    out.append(
                        _finding(
                            "drift.reps",
                            c.structure,
                            f"{_ckey(c)}: residuals trend with run order (Spearman rho={rho:.2f})",
                            [c.key],
                            rho=rho,
                        )
                    )
    return out


def a9_outlier(rs, thr):
    """Robust sigmas (1.4826 x MAD) against the band of the neighbouring smoothed points, so a rep
    caught mid-transition is not an outlier."""
    t = thr["A9"]
    out = []
    if rs.deterministic:
        return out  # as A8: on a simulator, rep-to-rep spread is the chain seed, not noise
    for c in rs.curves.values():
        if c.reps is None or len(c.reps) < 5:
            continue
        x, y = c.median_curve()
        n, w = len(x), t["window"] // 2
        s = np.array([np.median(y[max(0, i - w) : i + w + 1]) for i in range(n)])
        lo = np.array([s[max(0, i - 1) : i + 2].min() for i in range(n)])
        hi = np.array([s[max(0, i - 1) : i + 2].max() for i in range(n)])
        idx = {v: i for i, v in enumerate(x)}
        vals = [(nn, r, v, idx[float(nn)]) for nn, vs in c.reps.items() for r, v in enumerate(vs)]
        # each rep against its own level: a whole rep that sits high is drift (A8), not outliers
        reps_ = np.array([r for _, r, _, _ in vals])
        rel = np.array([v / s[i] - 1 for _, _, v, i in vals])
        shift = {r: float(np.median(rel[reps_ == r])) for r in set(reps_.tolist())}
        dev, resid = [], []
        for _nn, r, v, i in vals:
            v = v / (1 + shift[r])
            resid.append(v / s[i] - 1)
            dev.append((v - hi[i]) / hi[i] if v > hi[i] else (v - lo[i]) / lo[i] if v < lo[i] else 0.0)
        resid = np.array(resid)
        sigma = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), t["floor"])
        # only points with a full centred window: at the grid's edges a one-sided median
        # turns a sloped plateau into a false dip
        hits = [
            (nn, r, v) for (nn, r, v, i), d in zip(vals, dev) if w <= i < n - w and abs(d) > t["mads"] * sigma
        ]
        if hits:
            out.append(
                _finding(
                    "noise.outlier",
                    c.structure,
                    f"{_ckey(c)}: {len(hits)} repetition(s) beyond {t['mads']:g} robust sigmas "
                    f"({sigma:.1%}) of the moving median; dropped from the fit, kept in raw data",
                    [c.key],
                    points=hits[:20],
                    sigma=sigma,
                )
            )
    return out


def a10_throughput(rs, thr):
    frac = thr["A10"]["frac"]
    out = []
    for c in rs.curves.values():
        w, p = rs.width(c.filler), c.plateau()
        if not w or p is None:
            continue
        t_issue = 2 * c.knee.knee / w
        if t_issue > frac * p:
            out.append(
                _finding(
                    "throughput.bound",
                    c.structure,
                    f"{_ckey(c)}: issuing 2 x {c.knee.knee:g} fillers at {w:g}/cycle takes {t_issue:.0f} "
                    f"cycles, > {frac:.0%} of the {p:.0f}-cycle plateau; the knee may be a throughput "
                    f"limit: rerun with a larger depth",
                    [c.key],
                    issue_cycles=t_issue,
                    plateau=p,
                    width=w,
                )
            )
    return out


def a11_uncalibrated(rs, thr):
    out = []
    for c in rs.curves.values():
        if not c.has_knee():
            continue
        if c.offset is None:
            out.append(
                _finding(
                    "offset.missing",
                    c.structure,
                    f"{_ckey(c)}: no offset at all (not even the toy's) for this filler, depth "
                    f"and delay; the knee is all there is",
                    [c.key],
                    calibration=None,
                )
            )
        elif not c.offset.get("calibrated", True):
            out.append(
                _finding(
                    "offset.uncalibrated",
                    c.structure,
                    f"{_ckey(c)}: offset {c.offset['value']:g} from {c.offset['calibration']}",
                    [c.key],
                    calibration=c.offset["calibration"],
                )
            )
    return out


def a12_unstable(rs, thr):
    lim = thr["A12"]["max_frac_no_knee"]
    out = []
    for c in rs.curves.values():
        fr = c.knee.frac_no_knee if c.knee is not None else None
        if c.knee is not None and c.knee.quality < thr["A2"]["min_quality"]:
            continue  # no clean step at all: A2 says so; one cause, one finding
        if fr is not None and fr > lim:
            out.append(
                _finding(
                    "noise.unstable_knee",
                    c.structure,
                    f"{_ckey(c)}: {fr:.0%} of bootstrap resamples find no step",
                    [c.key],
                    frac_no_knee=fr,
                )
            )
    return out


Rule = Callable[[RunSet, dict], list[Finding]]
RULES: dict[str, Rule] = {
    "shape.extra_step": a1_extra_step,
    "shape.poor_fit": a2_poor_fit,
    "tie.rob": a3_tie_rob,
    "ratio.retire_group": a4_retire_group,
    "delay.disagreement": a5_delay,
    "jitter.band": a6_jitter,
    "offset.depth_inconsistent": a7_depth,
    "drift.reps": a8_drift,
    "noise.outlier": a9_outlier,
    "throughput.bound": a10_throughput,
    "offset.uncalibrated": a11_uncalibrated,
    "noise.unstable_knee": a12_unstable,
}


def evaluate(rs: RunSet, rules: dict[str, Rule] = RULES, thresholds=None) -> list[Finding]:
    """Run every rule on `rs` (offsets are resolved first)."""
    thr = load_thresholds() if thresholds is None else thresholds
    rs.resolve_offsets()
    out = []
    for rule in rules.values():
        out.extend(rule(rs, thr))
    return out


def fired(findings) -> list[str]:
    """Sorted distinct rule ids."""
    return sorted({f.rule for f in findings})


# --------------------------------------------------------------------------- integration


def runset_from_knees(
    knees: dict[str, Knee],
    reps=None,
    *,
    depth=1,
    delay="miss",
    platform="toy",
    core="",
    ctx=None,
    widths=None,
    offsets=None,
    order=None,
    clock=None,
) -> RunSet:
    """A RunSet for one sweep.py run: `knees` from bootstrap_knee per filler,
    `reps` the per-filler {N: [values]} from run_protocol."""
    rs = RunSet(platform, core, ctx=dict(ctx or {}), widths=widths, offsets=offsets, clock=list(clock or []))
    for f, k in knees.items():
        rs.add(
            Curve(
                f,
                depth,
                delay,
                (reps or {}).get(f),
                k,
                probe_params(depth, delay)[1],
                order=(order or {}).get(f),
            )
        )
    return rs


def load_results(dirs, *, platform=None, core="", ctx=None, widths=None, offsets=None, n_boot=1000) -> RunSet:
    """A RunSet from sweep.py result directories (results/<platform>/<tag>/).

    Prefers sweep_raw.csv (per-launch medians, order_index, clock probes) over sweep.csv. Depth and
    delay come from knees.json, so runs at several depths and delays combine for A4, A5 and A7."""
    import csv
    import json

    curves, clock, plat, cx = [], [], platform, dict(ctx or {})
    for d in dirs:
        with open(os.path.join(d, "knees.json")) as fh:
            meta = json.load(fh)
        mctx = meta.get("context", {})
        plat = plat or meta.get("platform")
        for k in ("emitter", "platform"):
            if k in mctx:
                cx.setdefault(k, mctx[k])
        raw = os.path.join(d, "sweep_raw.csv")
        if os.path.exists(raw):
            launches = {}
            with open(raw) as fh:
                for r in csv.DictReader(fh):
                    key = (r["filler"], int(r["depth"]), r["delay"], int(r["n"]), int(r["outer_rep"]))
                    e = launches.setdefault(
                        key,
                        {
                            "y": [],
                            "order": int(r["order_index"]),
                            "pre": r["clock_ratio_pre"],
                            "post": r["clock_ratio_post"],
                        },
                    )
                    e["y"].append(float(r["cycles_per_iter"]))
            for f, dep, dl in sorted({k[:3] for k in launches}):
                reps, order = {}, {}
                for (ff, dd, ll, n, rep), e in sorted(launches.items(), key=lambda kv: kv[0][4]):
                    if (ff, dd, ll) != (f, dep, dl):
                        continue
                    reps.setdefault(n, []).append(float(np.median(e["y"])))
                    order[(n, rep)] = e["order"]
                    if e["pre"] not in ("", "None") and e["post"] not in ("", "None"):
                        clock.append(
                            {
                                "rep": rep,
                                "n": n,
                                "filler": f,
                                "pre": float(e["pre"]),
                                "post": float(e["post"]),
                            }
                        )
                curves.append(
                    Curve.from_reps(f, reps, depth=dep, delay=dl, order=order, source=d, n_boot=n_boot)
                )
        else:
            tag = os.path.basename(os.path.normpath(d))
            dep = int(mctx.get("depth", tag.split("-d")[-1][0] if "-d" in tag else 1))
            dl = mctx.get("delay", "miss")
            with open(os.path.join(d, "sweep.csv")) as fh:
                rows = list(csv.DictReader(fh))
            for f in sorted({r["filler"] for r in rows}):
                reps = {int(r["n"]): [float(r["cycles_per_iter"])] for r in rows if r["filler"] == f}
                curves.append(Curve.from_reps(f, reps, depth=dep, delay=dl, source=d, n_boot=n_boot))
    rs = RunSet(plat or "unknown", core, ctx=cx, widths=widths, offsets=offsets, clock=clock)
    for c in curves:
        rs.add(c)
    return rs


def apply(
    estimates: dict[str, dict], findings: list[Finding], depth=None, delay=None, replace=False
) -> dict[str, dict]:
    """Merge findings into attribute()-style estimates in place (only those at depth/delay, if given).

    Rule ids go to `anomalies`, a non-info finding replaces a "measured" status, A3/A5 set `bound`
    (A5 wins). replace=True first drops knee.py's own flags, so the registry alone decides."""
    if replace:
        for e in estimates.values():
            e["anomalies"] = [a for a in e.get("anomalies", []) if a not in SEVERITY]
            e.pop("bound", None)
            if e.get("value") is not None:
                e["status"] = "measured"
    for f in findings:
        if depth is not None and f.curves and not any(k[1] == depth and k[2] == delay for k in f.curves):
            continue
        targets = [f.structure] if f.structure else list(estimates)
        for s in targets:
            e = estimates.get(s)
            if e is None:
                continue
            an = e.setdefault("anomalies", [])
            if f.rule not in an:
                an.append(f.rule)
            if f.severity != "info" and e.get("status", "measured") == "measured":
                e["status"] = f"{f.severity}: {f.message}"
            b = f.evidence.get("bound")
            if b and (f.rule == "delay.disagreement" or "bound" not in e):
                e["bound"] = dict(b)
    return estimates
