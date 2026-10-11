"""gem5 driver for the P1.5 (scheduler) and P1.7 (retire) campaigns.

Runs raw-asm kernels under config_o3.py (AArch64 kernels: arm_config_o3.py on the ARM build)
like gem5.run_gem5_binary, plus a timeout
(a layout that leaves an op class without a capable queue hangs dispatch), a result
cache in results/gem5/sched-cache/, and ground truth read back from config.ini.
`expected_knee` and `hockey` are the P1.5 reference model and pairing fit; `hinge`, `ols` and
`width` the dispatch-probe and retire-slope fits.
"""

import configparser
import hashlib
import json
import os
import statistics
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.backends.platform import gem5 as g5
from uarchre.backends.isa_base import emit_c, run_args, parse_output
from uarchre.backends.platform.native import EMITTERS, compile_kernel
from uarchre.kernels.scheduler import ASM_EMITTERS, kernel_isa, multi_source, parse_multi, chain_nodes_needed

CACHE = os.path.join(ROOT, "results", "gem5", "sched-cache")
OUTROOT = os.path.join(ROOT, "build", "gem5out", "sched")

# FU descriptors for custom pools (gem5 v25.1 FuncUnitConfig.py names)
FU = {
    "alu": "IntALU(count={n})",  # IntAlu
    "muldiv": "IntMultDiv(count={n})",  # IntMult, IntDiv
    "fp": "FP_ALU(count={n}), FP_MultDiv(count=2)",  # FloatAdd/Cmp/Cvt + Mult/Div/Sqrt
    "rd": "ReadPort(count={n})",  # MemRead, FloatMemRead
    "wr": "WritePort(count={n})",  # MemWrite, FloatMemWrite
    "rdwr": "RdWrPort(count={n})",  # both
    "misc": "SIMD_Unit(count=1), Matrix_Unit(count=1), System_Unit(count=1), PredALU(count=1)",
}
MISC_QUEUE = (8, [("misc", 1)])  # never probed; keeps every op class issuable


def iq_set(queues, misc=True):
    """queues: [(numEntries, [(fu, count), ...]), ...] in gem5's search order."""
    qs = list(queues) + ([MISC_QUEUE] if misc else [])
    items = []
    for size, fus in qs:
        fl = ", ".join(FU[f].format(n=c) for f, c in fus)
        items.append(f"IQUnit(numEntries={size}, fuPool=FUPool(FUList=[{fl}]))")
    return "cpu.instQueues=[" + ", ".join(items) + "]"


def _cmd(binpath, argv, params, sets, isa="riscv64"):
    tail = [g5.CONFIGS[isa], "--isa", g5.GEM5_ISA[isa], "--cmd", binpath, "--args", " ".join(map(str, argv))]
    for k, v in (params or {}).items():
        tail += [f"--{k}", str(v)]
    for s in sets:
        tail += ["--set", s]
    return tail


def run(
    kernel,
    params=None,
    sets=(),
    iters=40,
    reps=2,
    chain_bytes=4 << 20,
    seed=0,
    timeout=1200,
    source=None,
    want_outdir=False,
    isa="riscv64",
):
    """One gem5 launch of a raw-asm kernel (or a window Kernel with source="window", built
    for `isa`; raw-asm kernels carry their ISA).

    Returns a RunResult, or (RunResult, outdir) with want_outdir."""
    if source == "window":
        em = EMITTERS[isa]
        cc, cflags = g5.TOOLCHAIN[isa]
        binpath = compile_kernel(kernel, isa, cc=cc, cflags=cflags, static=True)
    else:
        isa = kernel_isa([kernel])
        em = ASM_EMITTERS[isa]
        binpath = g5.compile_c(emit_c(kernel, em), kernel.name, isa)
    argv = run_args(em, kernel, chain_bytes, iters, reps, 0, seed)
    tail = _cmd(binpath, argv, params, sets, isa)
    key = hashlib.md5(json.dumps([os.path.basename(binpath)] + tail[1:]).encode()).hexdigest()[:16]
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, key + ".out")
    outdir = os.path.join(OUTROOT, key)
    if os.path.exists(cpath):
        with open(cpath) as fh:
            out = fh.read()
    else:
        os.makedirs(outdir, exist_ok=True)
        cmd = [
            "timeout",
            str(timeout),
            "taskset",
            "-c",
            g5.CPUS,
            g5.GEM5[isa],
            "-q",
            "--outdir",
            outdir,
            *tail,
        ]
        with g5.slot():
            r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout
        if "cycles_per_iter" not in out:
            raise RuntimeError(
                f"gem5 failed ({r.returncode}) for {kernel.name} {params} {sets}:\n"
                f"{out[-1500:]}\n{r.stderr[-1500:]}"
            )
        with open(cpath + ".tmp", "w") as fh:
            fh.write(out)
        os.replace(cpath + ".tmp", cpath)
        with open(os.path.join(CACHE, key + ".json"), "w") as fh:
            json.dump(
                {
                    "kernel": kernel.name,
                    "params": params,
                    "sets": list(sets),
                    "iters": iters,
                    "reps": reps,
                    "seed": seed,
                    "outdir": outdir,
                },
                fh,
            )
    res = parse_output(out)
    return (res, outdir) if want_outdir else res


def run_multi(
    name, kernels, params=None, sets=(), iters=60, reps=3, chain_bytes=1 << 20, seed=0, timeout=3600
):
    """One gem5 launch timing every kernel in turn (kernels/scheduler.py multi_source); the
    kernels' ISA picks the gem5 build.

    Returns ({point index: [cycles per rep]}, outdir)."""
    isa = kernel_isa(kernels)
    line = ASM_EMITTERS[isa].line
    need = chain_nodes_needed(kernels, iters, reps)
    if need >= chain_bytes // line:
        raise ValueError(f"{name}: {need} chase nodes would wrap a {chain_bytes // line}-node chain")
    binpath = g5.compile_c(multi_source(name, kernels), name, isa)
    tail = _cmd(binpath, [chain_bytes, iters, reps, seed], params, sets, isa)
    key = hashlib.md5(json.dumps([os.path.basename(binpath)] + tail[1:]).encode()).hexdigest()[:16]
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, key + ".out")
    outdir = os.path.join(OUTROOT, key)
    if os.path.exists(cpath):
        with open(cpath) as fh:
            out = fh.read()
    else:
        os.makedirs(outdir, exist_ok=True)
        cmd = [
            "timeout",
            str(timeout),
            "taskset",
            "-c",
            g5.CPUS,
            g5.GEM5[isa],
            "-q",
            "--outdir",
            outdir,
            *tail,
        ]
        with g5.slot():
            r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout
        if "cycles_per_iter done" not in out:
            raise RuntimeError(
                f"gem5 failed ({r.returncode}) for {name} {params} {sets}:\n{out[-1500:]}\n{r.stderr[-1500:]}"
            )
        with open(cpath + ".tmp", "w") as fh:
            fh.write(out)
        os.replace(cpath + ".tmp", cpath)
        with open(os.path.join(CACHE, key + ".json"), "w") as fh:
            json.dump(
                {
                    "kernel": name,
                    "points": [k.name for k in kernels],
                    "params": params,
                    "sets": list(sets),
                    "iters": iters,
                    "reps": reps,
                    "seed": seed,
                    "outdir": outdir,
                },
                fh,
            )
    return parse_multi(out, len(kernels)), outdir


def config_truth(outdir):
    """Ground truth from gem5's config.ini: CPU widths/sizes and every IQ with
    its size and the op classes its FU pool accepts."""
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(os.path.join(outdir, "config.ini"))
    cpu = cp["system.cpu"]
    t = {
        k: int(cpu[k])
        for k in (
            "numROBEntries",
            "numPhysIntRegs",
            "numPhysFloatRegs",
            "LQEntries",
            "SQEntries",
            "fetchWidth",
            "decodeWidth",
            "renameWidth",
            "dispatchWidth",
            "issueWidth",
            "wbWidth",
            "commitWidth",
        )
        if k in cpu
    }
    queues = []
    for name in cpu["instQueues"].split():
        sec = cp[name]
        classes = []
        for fu in cp[sec["fuPool"]]["FUList"].split():
            if int(cp[fu]["count"]) > 0:  # gem5 ignores FU types with count 0
                classes += [cp[op]["opClass"] for op in cp[fu]["opList"].split()]
        fus = [
            {
                "count": int(cp[fu]["count"]),
                "ops": sorted({cp[op]["opClass"] for op in cp[fu]["opList"].split()}),
            }
            for fu in cp[sec["fuPool"]]["FUList"].split()
        ]
        queues.append(
            {
                "name": name.split(".")[-1],
                "numEntries": int(sec["numEntries"]),
                "fu_count": {
                    fu.split(".")[-1]: int(cp[fu]["count"]) for fu in cp[sec["fuPool"]]["FUList"].split()
                },
                "fus": fus,
                "op_classes": sorted(set(classes)),
            }
        )
    t["instQueues"] = queues
    return t


def class_capacity(truth, op_class):
    """Entries available to an op class: the sum over every queue that accepts it
    (gem5 v25.1 findIQ takes the first capable queue with a free entry)."""
    return sum(q["numEntries"] for q in truth["instQueues"] if op_class in q["op_classes"])


# probe filler class -> gem5 op class
OPCLASS = {"add": "IntAlu", "mul": "IntMult", "fp": "FloatCvt", "load": "MemRead", "store": "MemWrite"}
# gem5 ARM: fmov d, x is FloatMisc (FP_MultDiv); the queue layouts above cover it unchanged
OPCLASSES = {"riscv64": OPCLASS, "aarch64": {**OPCLASS, "fp": "FloatMisc"}}
MEMORY = ("load", "store")


def expected_knee(queues, filler, pre=None, m=0, resident_misses=2, isa="riscv64"):
    """Entries left for the filler's class once the miss loads, memory-op address
    helpers and m preload ops are placed, each in the first capable queue of
    `queues` [(size, op classes)] with room."""
    free = [s for s, _ in queues]

    def place(cls, k):
        for _ in range(k):
            for i, (_, ops) in enumerate(queues):
                if cls in ops and free[i] > 0:
                    free[i] -= 1
                    break

    opc = OPCLASSES[isa]
    place("MemRead", resident_misses)
    if filler in MEMORY or pre in MEMORY:
        place("IntAlu", 2)
    if pre:
        place(opc[pre], m)
    return sum(free[i] for i, (_, ops) in enumerate(queues) if opc[filler] in ops)


def hockey(ms, knees):
    """Fit knee(m) = c - max(0, m - b) (flat, then one-for-one). Returns (b, c,
    slope of an OLS line over m >= b). b >= max(m) means no sharing seen."""
    ms, ks = np.asarray(ms, float), np.asarray(knees, float)
    best = None
    for b in np.arange(ms.min(), ms.max() + 1):
        d = np.maximum(0, ms - b)
        c = float(np.mean(ks + d))
        sse = float(np.sum((ks - (c - d)) ** 2))
        if best is None or sse < best[0] - 1e-9:
            best = (sse, int(b), c)
    _, b, c = best
    sel = ms >= b
    slope = float(np.polyfit(ms[sel], ks[sel], 1)[0]) if sel.sum() >= 2 and np.ptp(ms[sel]) > 0 else 0.0
    return b, round(c), slope


def sharing_verdict(ms, b):
    """What a `hockey` break b over preload counts ms says about the two classes' queues."""
    if b >= max(ms):
        return "separate queues"
    if b <= min(ms):
        return "shared from the first op"
    return f"shared after {b} ops"


def warm(cycles):
    """Median over inner reps after the first (rep 0 runs the point's code cold)."""
    c = cycles[1:] if len(cycles) > 1 else cycles
    return float(statistics.median(c))


def hinge(ms, ys):
    """Least-squares flat-then-linear fit: returns (corner, base, slope)."""
    best = None
    for f in np.arange(0, ms.max() - 16 + 0.5, 0.5):
        d_ = np.maximum(0, ms - f)
        A = np.vstack([np.ones_like(ms), d_]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        sse = float(np.sum((ys - A @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(f), float(coef[0]), float(coef[1]))
    return best[1:]


def ols(x, y):
    """(slope, intercept, slope standard error) of a least-squares line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(len(x) - 2, 1)
    s2 = float(np.sum((y - A @ coef) ** 2) / dof)
    se = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
    return float(coef[0]), float(coef[1]), se


def width(slope, se, per):
    """per / slope with a 95% interval from the slope's standard error."""
    w = per / slope
    lo, hi = per / (slope + 2 * se), per / max(slope - 2 * se, 1e-9)
    return w, [lo, hi]
