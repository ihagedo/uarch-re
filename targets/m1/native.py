"""Shared plumbing for the Apple silicon runners: machine checks, compile, clocked launches.

A launch is bracketed by a dependent-add clock probe, so 24 MHz counter ticks convert to core
cycles and a launch that ran slow (E-core, throttled, migrated) is retried. --gem5-smoke runs the
same binaries under gem5 ARM with the cycle counter instead: a functional test, not a measurement.
"""

import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from uarchre.backends.isa.aarch64 import COUNTERS  # noqa: E402
from uarchre.backends.platform.native import compile_source  # noqa: E402

MACOS = platform.system() == "Darwin"
RESULTS = os.path.join(ROOT, "results", "m1")

PROBE_SRC = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
%COUNTER%
int main(int argc, char **argv) {
    uint64_t n = argc > 1 ? strtoull(argv[1], 0, 0) : (1ULL << 22), x = 0, k = n / 64, t0, t1;
    t0 = cycles();
    __asm__ __volatile__("1:\n\t" ".rept 64\n\tadd %0, %0, %1\n\t.endr\n\t" "subs %1, %1, #1\n\tb.ne 1b"
                         : "+r"(x), "+r"(k) :: "cc");
    t1 = cycles();
    __asm__ __volatile__("" :: "r"(x));
    printf("ticks_per_add %.6f\n", (double)(t1 - t0) / (double)(n / 64 * 64));
    return 0;
}
"""


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_info():
    """What identifies a product, never a machine or a person."""
    info = {"system": platform.system(), "machine": platform.machine(), "release": platform.release()}
    if MACOS:
        for k in (
            "hw.model",
            "machdep.cpu.brand_string",
            "hw.memsize",
            "hw.nperflevels",
            "sysctl.proc_translated",
            "kern.osproductversion",
            "kern.osversion",
        ):
            info[k] = sh("sysctl", "-n", k)
        levels = []
        for i in range(int(info.get("hw.nperflevels") or 0)):
            levels.append(
                {
                    k: sh("sysctl", "-n", f"hw.perflevel{i}.{k}")
                    for k in ("name", "physicalcpu", "l1icachesize", "l1dcachesize", "l2cachesize")
                }
            )
        info["perflevels"] = levels
        batt = sh("pmset", "-g", "batt")
        info["power"] = "ac" if "AC Power" in batt else ("battery" if "Battery Power" in batt else "unknown")
        info["low_power_mode"] = bool(re.search(r"lowpowermode\s+1", sh("pmset", "-g")))
    else:
        cpuinfo = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
        info["cpu_parts"] = sorted(set(re.findall(r"CPU part\s*:\s*(\S+)", cpuinfo)))
        info["compatible"] = sh("cat", "/proc/device-tree/compatible").replace("\x00", " ")
    info["load_avg_start"] = os.getloadavg()[0]
    return info


def preflight(info, force):
    problems = []
    if info["machine"] not in ("arm64", "aarch64"):
        problems.append(f"not an arm64 Python ({info['machine']}): run a native arm64 python3")
    if MACOS:
        if info.get("sysctl.proc_translated") == "1":
            problems.append("running under Rosetta")
        if info.get("power") == "battery":
            problems.append("on battery power")
        if info.get("low_power_mode"):
            problems.append("Low Power Mode is on")
    if info["load_avg_start"] > 2.0:
        problems.append(f"load average {info['load_avg_start']:.1f}: close other apps")
    if problems and not force:
        sys.exit("refusing to run: " + "; ".join(problems) + " (--force to run anyway; it is recorded)")
    return problems


def published(path, core="firestorm"):
    """A field of spec/cores/<core>.yaml by dotted path (e.g. "memory.l1d.size_kib"), or None."""
    import yaml

    with open(os.path.join(ROOT, "spec", "cores", f"{core}.yaml")) as fh:
        node = yaml.safe_load(fh)
    for key in path.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def l1i_line(args):
    """L1i line bytes: --l1i-line, else firestorm.yaml."""
    return getattr(args, "l1i_line", None) or published("frontend.l1i.line_bytes")


def add_args(ap):
    """Options every runner takes."""
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--min-ghz", type=float, default=2.8, help="slower launches are E-core or throttled")
    ap.add_argument("--max-drift", type=float, default=0.02, help="clock change across a launch")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--pin-cpu", type=int, default=None, help="Linux only (Asahi M1 P-cores: 4-7)")
    ap.add_argument("--cc", default=None)
    ap.add_argument("--l1i-line", type=int, default=None, help="L1i line bytes (default: firestorm.yaml)")
    ap.add_argument("--force", action="store_true", help="run despite failed preflight checks (recorded)")
    ap.add_argument("--gem5-smoke", action="store_true", help="tiny sweeps under gem5 ARM (no Mac needed)")
    return ap


class Runner:
    """Compile and launch AArch64 kernels; `counter` is what their sources should use."""

    def __init__(self, args):
        self.args = args
        self.smoke = args.gem5_smoke
        self.counter = "pmccntr" if self.smoke else ("mach" if MACOS else "cntvct")
        self.cc = args.cc or ("clang" if MACOS else "cc")
        self.attempts = []
        if self.smoke:
            self.info, self.problems = {"system": "gem5-smoke"}, []
        else:
            self.info = machine_info()
            self.problems = preflight(self.info, args.force)
        self._probe = None

    def compile(self, src, name):
        if self.smoke:
            from uarchre.backends.platform.gem5 import compile_c

            return compile_c(src, name, isa="aarch64")
        return compile_source(src, f"{name}_{self.counter}", "aarch64", cc=self.cc, cflags=("-O1",))

    def _exec(self, binpath, argv, timeout):
        if self.smoke:
            from uarchre.backends.platform.gem5 import run_gem5_binary

            return run_gem5_binary(binpath, [str(a) for a in argv], isa="aarch64", timeout=timeout)[0]
        cmd = [binpath, *map(str, argv)]
        if self.args.pin_cpu is not None and not MACOS:
            cmd = ["taskset", "-c", str(self.args.pin_cpu)] + cmd
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout

    def ticks_per_cycle(self):
        """(ticks per core cycle, counter Hz) from the add-chain probe; (1, None) under gem5."""
        if self.smoke:
            return 1.0, None
        if self._probe is None:
            self._probe = self.compile(PROBE_SRC.replace("%COUNTER%", COUNTERS[self.counter]), "clockprobe")
        out = self._exec(self._probe, [1 << 22], 60)
        hz = re.search(r"^counter_hz ([\d.]+)", out, re.M)
        return float(re.search(r"^ticks_per_add ([\d.]+)", out, re.M).group(1)), float(
            hz.group(1)
        ) if hz else None

    def launch(self, binpath, argv=(), timeout=1800, label=""):
        """stdout of an accepted launch and its ticks per core cycle (divide tick counts by it)."""
        a = self.args
        for attempt in range(a.retries + 1):
            pre, hz = self.ticks_per_cycle()
            out = self._exec(binpath, argv, timeout)
            post, _ = self.ticks_per_cycle()
            ratio = (pre + post) / 2
            drift = abs(pre - post) / ratio
            ghz = hz / ratio / 1e9 if hz else None
            ok = self.smoke or (drift <= a.max_drift and ghz is not None and ghz >= a.min_ghz)
            self.attempts.append(
                dict(
                    label=label,
                    attempt=attempt,
                    ratio_pre=pre,
                    ratio_post=post,
                    ghz=ghz,
                    drift=drift,
                    accepted=ok,
                    t_ns=time.time_ns(),
                )  # fmt: skip
            )
            if ok:
                return out, ratio
            time.sleep(0.5)
        return out, ratio  # kept, marked unaccepted in attempts

    def save(self, family, poc, raw, summary):
        """results/m1/<tag>/<family>/<poc>/{raw,summary,run}.json"""
        d = os.path.join(RESULTS, self.args.tag, family, poc)
        os.makedirs(d, exist_ok=True)
        run = dict(machine=self.info, preflight=self.problems, counter=self.counter, attempts=self.attempts)
        for name, obj in (("raw", raw), ("summary", summary), ("run", run)):
            with open(os.path.join(d, f"{name}.json"), "w") as fh:
                json.dump(obj, fh, indent=1, default=float)
        self.attempts = []  # each PoC's run.json holds its own launches
        return d


def pack(tag="m1"):
    """results/m1/<tag> as one .tgz to send back."""
    src = os.path.join(RESULTS, tag)
    out = os.path.join(RESULTS, f"{tag}.tgz")
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname=tag)
    return out
