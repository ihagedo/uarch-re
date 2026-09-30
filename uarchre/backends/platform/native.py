"""Native driver: compile a kernel with the host gcc and run it (x86-64, rdtscp).

`compile_source` caches binaries by content hash in build/native; the gem5
driver reuses it with the cross toolchain.
"""

import os
import subprocess
import hashlib
import threading

from ..isa_base import emit_c, run_args, parse_output
from ..isa.x86_64 import X86_64
from ..isa.riscv64 import RISCV64

EMITTERS = {"x86_64": X86_64(), "riscv64": RISCV64()}
try:  # isa/aarch64.py is optional
    from ..isa.aarch64 import AArch64

    EMITTERS["aarch64"] = AArch64()
except ImportError:
    pass


def _build_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "native")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def compile_source(src, name, isa="x86_64", cc="gcc", cflags=("-O1",), static=False):
    """Compile a self-contained C source; cached by content hash in build/native."""
    h = hashlib.md5((src + cc + " ".join(cflags)).encode()).hexdigest()[:10]
    d = _build_dir()
    cpath = os.path.join(d, f"{name}_{h}.c")
    binpath = os.path.join(d, f"{name}_{h}.{isa}")
    if not os.path.exists(binpath):
        with open(cpath, "w") as f:
            f.write(src)
        cmd = [cc, *cflags, "-mavx"] if isa == "x86_64" else [cc, *cflags]
        if static:
            cmd.append("-static")
        # build under a private name, then rename: concurrent sweeps (and
        # threads within one) may compile the same kernel at once
        tmp = f"{binpath}.{os.getpid()}.{threading.get_ident()}"
        subprocess.run(cmd + [cpath, "-o", tmp], check=True)
        os.replace(tmp, binpath)
    return binpath


def compile_kernel(kernel, isa="x86_64", cc="gcc", cflags=("-O1",), static=False, **ov):
    return compile_source(emit_c(kernel, EMITTERS[isa], **ov), kernel.name, isa, cc, cflags, static)


def run_native_reps(kernel, chain_bytes=None, iters=None, reps=None, pin_cpu=None, seed=0, probe_adds=None):
    """Run once; return a RunResult with every inner rep and the clock probe."""
    binpath = compile_kernel(kernel, "x86_64")
    args = [binpath, *run_args(EMITTERS["x86_64"], kernel, chain_bytes, iters, reps, probe_adds, seed)]
    if pin_cpu is not None:
        args = ["taskset", "-c", str(pin_cpu)] + args
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return parse_output(out)


def run_native(kernel, **kw):
    """Minimum cycles/iter over the inner reps."""
    return run_native_reps(kernel, **kw).best
