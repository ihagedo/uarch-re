"""House style for figures, plus the evidence record each PoC writes.

Categorical colors in fixed slot order, never cycled; one y-axis per panel; status colors only
for pass / fail / flag and always with a text label. `record()` writes an evidence record and
copies the figures beside it; raw data stays in results/ and the record points to it.
"""

import os
import shutil

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVIDENCE = os.path.join(ROOT, "docs", "evidence")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # secondary encoding, same slot order
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"pass": "#0ca30c", "warning": "#fab219", "flag": "#ec835a", "fail": "#d03b3b"}
BAND = "#f0efec"  # neutral fill (tolerance band)


def setup():
    """Apply the house rcParams; call before creating figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK2,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    return plt


def series(i):
    """Style kwargs for categorical slot i (color + marker + surface ring)."""
    return dict(color=SERIES[i], marker=MARKERS[i], markeredgecolor=SURFACE, markeredgewidth=1.0)


def scatter_kw(i, hollow=False):
    if hollow:
        return dict(facecolor="none", edgecolor=SERIES[i], marker=MARKERS[i], linewidth=1.3, s=44, zorder=3)
    return dict(color=SERIES[i], marker=MARKERS[i], edgecolor=SURFACE, linewidth=1.0, s=44, zorder=3)


def tolerance(truth):
    """L1/L2 pass tolerance: |error| <= max(2, 2% of truth)."""
    return np.maximum(2.0, 0.02 * np.asarray(truth, dtype=float))


def diagonal(ax, groups, *, label="configured", unit="entries", tol=True):
    """Detected vs configured, one series per group; flagged points are hollow (a bound, not a value).

    groups: [{name, truth, value, ci: [[lo, hi], ...] | None, flagged: [bool] | None, nudge: x shift}].
    """
    allv = [v for g in groups for v in list(g["truth"]) + [x for x in g["value"] if x is not None]]
    hi = max(allv) * 1.08 if allv else 1
    xs = np.linspace(0, hi, 200)
    if tol:
        t = tolerance(xs)
        ax.fill_between(xs, xs - t, xs + t, color=BAND, lw=0, label="tolerance max(2, 2%)")
    ax.plot([0, hi], [0, hi], color=AXIS, lw=1, zorder=1)
    for i, g in enumerate(groups):
        tr = np.asarray(g["truth"], float) + g.get("nudge", 0)
        val = np.array([np.nan if v is None else v for v in g["value"]], float)
        fl = np.asarray(g.get("flagged") or [False] * len(tr), bool)
        if g.get("ci") is not None:
            lo = np.array([np.nan if c is None else c[0] for c in g["ci"]], float)
            up = np.array([np.nan if c is None else c[1] for c in g["ci"]], float)
            ax.errorbar(
                tr,
                val,
                yerr=[val - lo, up - val],
                fmt="none",
                ecolor=SERIES[i],
                elinewidth=1.2,
                capsize=2.5,
                zorder=2,
            )
        if (~fl).any():
            ax.scatter(tr[~fl], val[~fl], label=g["name"], **scatter_kw(i))
        if fl.any():
            ax.scatter(tr[fl], val[fl], label=f"{g['name']} (flagged)", **scatter_kw(i, hollow=True))
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"{label} ({unit})", ylabel=f"detected ({unit})")
    ax.set_aspect("equal")


def error_panel(ax, groups, *, label="configured", unit="entries"):
    """Detected minus configured against configured, with the tolerance band."""
    allt = [v for g in groups for v in g["truth"]]
    hi = max(allt) * 1.08 if allt else 1
    xs = np.linspace(0, hi, 200)
    t = tolerance(xs)
    ax.fill_between(xs, -t, t, color=BAND, lw=0, label="tolerance")
    ax.axhline(0, color=AXIS, lw=1)
    for i, g in enumerate(groups):
        tr = np.asarray(g["truth"], float)
        val = np.array([np.nan if v is None else v for v in g["value"]], float)
        ax.scatter(tr, val - tr, label=g["name"], **scatter_kw(i))
    ax.set(xlim=(0, hi), xlabel=f"{label} ({unit})", ylabel=f"detected minus {label}")


def status_text(ax, ok, text, loc=(0.02, 0.96)):
    """A pass/fail tag in the corner: status-colored dot + ink label."""
    c = STATUS["pass" if ok else "fail"]
    ax.annotate("●", loc, xycoords="axes fraction", color=c, va="top", fontsize=11)
    ax.annotate(
        ("PASS  " if ok else "FAIL  ") + text,
        (loc[0] + 0.045, loc[1]),
        xycoords="axes fraction",
        color=INK,
        va="top",
        fontsize=9,
        fontweight="semibold",
    )


def save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    return path


def record(
    poc, *, title, claim, level, status, figures, metrics=None, method="", data=None, notes="", caption=""
):
    """Write the evidence record for `poc` (e.g. "P1.1") and copy `figures` beside it as <poc>-<n>.png.

    level is "L0".."L4" or a list; status is pass | fail | partial | control | blocked.
    """
    os.makedirs(EVIDENCE, exist_ok=True)
    slug = poc.replace(".", "_")
    out = []
    for i, f in enumerate(figures):
        dst = os.path.join(EVIDENCE, f"{slug}-{i + 1}.png")
        shutil.copyfile(f, dst)
        out.append(os.path.relpath(dst, EVIDENCE))
    rec = {
        "poc": poc,
        "title": title,
        "claim": claim,
        "level": level,
        "status": status,
        "figures": out,
        "caption": caption,
        "metrics": metrics or {},
        "method": method,
        "data": data,
        "notes": notes,
    }
    with open(os.path.join(EVIDENCE, f"{slug}.yaml"), "w") as fh:
        yaml.safe_dump(rec, fh, sort_keys=False, allow_unicode=True)
    return rec
