"""gem5 SE-mode driver.

Any C program runs via `compile_c` + `run_gem5_binary`; the window probe uses `run_gem5_reps`.
`params` sets top-level O3 CPU params (`--numROBEntries 64`), `sets` dotted paths below `system`
(`cpu.branchPred.ras.numEntries=16`, `l2.mshrs=4`). Each gem5 process holds one of UARCHRE_GEM5_SLOTS
(16) host-wide slots and runs on UARCHRE_GEM5_CPUS (4-19), leaving P-cores 0-3 quiet for native timing.
"""

import contextlib
import fcntl
import hashlib
import os
import re
import subprocess
import time

from .native import compile_kernel, compile_source, EMITTERS
from ..isa_base import run_args, parse_output

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GEM5_SRC = os.path.join(ROOT, "validation", "gem5", "gem5-src")
GEM5 = {
    "riscv64": os.environ.get("GEM5", os.path.join(GEM5_SRC, "build", "RISCV", "gem5.opt")),
    "aarch64": os.environ.get("GEM5_ARM", os.path.join(GEM5_SRC, "build", "ARM", "gem5.opt")),
}
CONFIG = os.path.join(ROOT, "validation", "gem5", "config_o3.py")
# ARM SE mode has no generic timer: arm_config_o3.py adds the PMU that PMCCNTR_EL0 needs
CONFIGS = {"riscv64": CONFIG, "aarch64": os.path.join(ROOT, "validation", "gem5", "arm_config_o3.py")}
TOOLCHAIN = {  # cross compiler, flags; static binaries only
    "riscv64": ("riscv64-linux-gnu-gcc", ("-O1", "-march=rv64gc", "-mabi=lp64d")),
    "aarch64": ("aarch64-linux-gnu-gcc", ("-O1", "-march=armv8-a")),
}
CROSS, CFLAGS = TOOLCHAIN["riscv64"]
GEM5_ISA = {"riscv64": "riscv", "aarch64": "arm"}

SLOTS = int(os.environ.get("UARCHRE_GEM5_SLOTS", "16"))
CPUS = os.environ.get("UARCHRE_GEM5_CPUS", "4-19")
SLOT_DIR = os.path.join(
    os.environ.get("UARCHRE_SCRATCH", f"/var/tmp/{os.environ.get('USER', 'u')}/uarch-re"), "gem5-slots"
)

# gem5 defaults (RiscvO3CPU): the ground truth when not overridden
DEFAULTS = {
    "numROBEntries": 192,
    "numPhysIntRegs": 256,
    "numPhysFloatRegs": 256,
    "LQEntries": 32,
    "SQEntries": 32,
}
ARCH_INT = 32  # RISC-V x0..x31 (x0 may or may not be pinned; report both)
ARCH_FP = 32
# ArmO3CPU (validation/gem5/arm_sweep.py, P0.6): 42 int and 44 vector registers are
# architectural; scalar FP draws on the vector pool; 6 CC registers, and an AArch64
# cmp renames three of them (NZ, C, V), so flag-setting ops = (CC - 6) / 3
ARM_DEFAULTS = {
    "numROBEntries": 192,
    "numPhysIntRegs": 256,
    "numPhysVecRegs": 256,
    "LQEntries": 32,
    "SQEntries": 32,
}
ARM_ARCH = {"int": 42, "vec": 44, "cc": 6}
CC_PER_FLAGS_OP = 3


def gem5_truth(params, isa="riscv64"):
    if isa == "aarch64":
        p = {**ARM_DEFAULTS, **params}
        cc = p.get("numPhysCCRegs", 5 * p["numPhysIntRegs"])
        return {
            "rob": p["numROBEntries"],
            "int_prf_headroom": p["numPhysIntRegs"] - ARM_ARCH["int"],
            "fp_prf_headroom": p["numPhysVecRegs"] - ARM_ARCH["vec"],
            "lq": p["LQEntries"],
            "sq": p["SQEntries"],
            "flags_prf_headroom": cc - ARM_ARCH["cc"],
            "flags_ops": (cc - ARM_ARCH["cc"]) // CC_PER_FLAGS_OP,
            "int_prf": p["numPhysIntRegs"],
            "fp_prf": p["numPhysVecRegs"],
            "flags_prf": cc,
        }
    p = {**DEFAULTS, **params}
    return {
        "rob": p["numROBEntries"],
        "int_prf_headroom": p["numPhysIntRegs"] - ARCH_INT,
        "fp_prf_headroom": p["numPhysFloatRegs"] - ARCH_FP,
        "lq": p["LQEntries"],
        "sq": p["SQEntries"],
        "int_prf": p["numPhysIntRegs"],
        "fp_prf": p["numPhysFloatRegs"],
    }


@contextlib.contextmanager
def slot():
    """Hold one host-wide gem5 slot (flock on one of SLOTS files)."""
    os.makedirs(SLOT_DIR, exist_ok=True)
    while True:
        for i in range(SLOTS):
            fh = open(os.path.join(SLOT_DIR, f"slot{i}"), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(1.0)


def compile_c(src, name, isa="riscv64"):
    """Cross-compile a self-contained C source for gem5 (static); cached by hash."""
    cc, cflags = TOOLCHAIN[isa]
    return compile_source(src, name, isa, cc=cc, cflags=cflags, static=True)


def run_gem5_binary(
    binpath, argv=(), params=None, sets=(), isa="riscv64", config_args=(), check=True, timeout=3600
):
    """Run a static binary under gem5 O3; returns (stdout, outdir with stats.txt, config.ini).
    A run longer than `timeout` s (e.g. a scheduler layout that deadlocks dispatch) raises."""
    cmd_tail = [CONFIGS[isa], "--isa", GEM5_ISA[isa], "--cmd", binpath, "--args", " ".join(map(str, argv))]
    for k, v in (params or {}).items():
        cmd_tail += [f"--{k}", str(v)]
    for s in sets:
        cmd_tail += ["--set", s]
    cmd_tail += list(config_args)
    # one outdir per distinct run, so concurrent sims never share stats.txt / config.ini
    tag = hashlib.md5(" ".join(cmd_tail).encode()).hexdigest()[:10]
    outdir = os.path.join(ROOT, "build", "gem5out", f"{os.path.basename(binpath)}_{tag}")
    os.makedirs(outdir, exist_ok=True)
    cmd = ["taskset", "-c", CPUS, GEM5[isa], "-q", "--outdir", outdir, *cmd_tail]
    with slot():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gem5 timed out after {timeout} s for {binpath}") from None
    with open(os.path.join(outdir, "stderr.txt"), "w") as fh:  # gem5 panics print before the backtrace
        fh.write(r.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"gem5 failed ({r.returncode}) for {binpath}:\n{r.stdout[-2000:]}\n{_head_tail(r.stderr)}"
        )
    return r.stdout, outdir


def _head_tail(text, n=1500):
    return text if len(text) <= 2 * n else f"{text[:n]}\n[...]\n{text[-n:]}"


def run_gem5_reps(
    kernel,
    params,
    chain_bytes=8 << 20,
    iters=300,
    reps=2,
    seed=0,
    probe_adds=None,
    config_args=(),
    sets=(),
    isa="riscv64",
):
    """Run a window-probe kernel once under gem5; RunResult with every inner rep.
    gem5 is deterministic per binary and seed: vary `seed` for spread."""
    cc, cflags = TOOLCHAIN[isa]
    binpath = compile_kernel(kernel, isa, cc=cc, cflags=cflags, static=True)
    argv = run_args(EMITTERS[isa], kernel, chain_bytes, iters, reps, probe_adds, seed)
    out, outdir = run_gem5_binary(binpath, argv, params, sets, isa, config_args, check=False)
    if not re.search(r"^cycles_per_iter ", out, re.M):
        with open(os.path.join(outdir, "stderr.txt")) as fh:
            raise RuntimeError(f"gem5 failed for {kernel.name}:\n{out[-2000:]}\n{_head_tail(fh.read())}")
    return parse_output(out)


def run_gem5(kernel, params, **kw):
    """Minimum cycles/iter over the inner reps."""
    return run_gem5_reps(kernel, params, **kw).best


def stat(outdir, name):
    """One scalar from stats.txt (first match), or None."""
    with open(os.path.join(outdir, "stats.txt")) as fh:
        for line in fh:
            if line.startswith(name + " ") or line.startswith(name + "\t"):
                return float(line.split()[1])
    return None
