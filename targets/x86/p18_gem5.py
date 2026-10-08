#!/usr/bin/env python3
"""P1.8 under gem5: does the latency / throughput harness read gem5's configured FU latencies?

The uarchre/kernels/x86_micro.py method (dependent chain for latency, independent registers
for throughput) as RV64GC inline asm under gem5 O3 (validation/gem5/config_o3.py, default FU
pool). rdcycle counts core cycles, so no clock probe is needed. Truth is gem5's
FuncUnitConfig.py, via expected().

  targets/x86/p18_gem5.py            # ~1 min; results/gem5/p1_8/micro.json
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from uarchre.backends.platform.gem5 import compile_c, run_gem5_binary  # noqa: E402

G = ["t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5", "a6", "a7"]
F = [f"ft{i}" for i in range(10)]
# name, kind, unit(i), opclass, configured (opLat, units, pipelined)
TESTS = [
    ("add_lat", "lat", lambda i: "add t2, t2, t0", "IntAlu", (1, 6, True)),
    ("add_tput", "tput", lambda i: f"add {G[i % 11]}, {G[i % 11]}, t0", "IntAlu", (1, 6, True)),
    ("mul_lat", "lat", lambda i: "mul t2, t2, t0", "IntMult", (3, 2, True)),
    ("mul_tput", "tput", lambda i: f"mul {G[i % 11]}, {G[i % 11]}, t0", "IntMult", (3, 2, True)),
    ("div_lat", "lat", lambda i: "div t2, t2, t0", "IntDiv", (20, 2, False)),
    ("div_tput", "tput", lambda i: f"div {G[i % 11]}, {G[i % 11]}, t0", "IntDiv", (20, 2, False)),
    ("fadd_lat", "lat", lambda i: "fadd.d ft0, ft0, ft10", "FloatAdd", (2, 4, True)),
    ("fadd_tput", "tput", lambda i: f"fadd.d {F[i % 10]}, {F[i % 10]}, ft10", "FloatAdd", (2, 4, True)),
    ("fmul_lat", "lat", lambda i: "fmul.d ft0, ft0, ft10", "FloatMult", (4, 2, True)),
    ("fmul_tput", "tput", lambda i: f"fmul.d {F[i % 10]}, {F[i % 10]}, ft10", "FloatMult", (4, 2, True)),
    ("fmadd_lat", "lat", lambda i: "fmadd.d ft0, ft0, ft10, ft11", "FloatMultAcc", (5, 2, True)),
    ("fdiv_lat", "lat", lambda i: "fdiv.d ft0, ft0, ft10", "FloatDiv", (12, 2, False)),
    ("fdiv_tput", "tput", lambda i: f"fdiv.d {F[i % 10]}, {F[i % 10]}, ft10", "FloatDiv", (12, 2, False)),
    ("fsqrt_lat", "lat", lambda i: "fsqrt.d ft0, ft0", "FloatSqrt", (24, 2, False)),
    ("fsqrt_tput", "tput", lambda i: f"fsqrt.d {F[i % 10]}, ft10", "FloatSqrt", (24, 2, False)),
]
UNROLL = 32
ISSUE_WIDTH = 8

C = r"""
#include <stdint.h>
#include <stdio.h>
static inline uint64_t cyc(void) {{ uint64_t c; __asm__ __volatile__("rdcycle %0" : "=r"(c)); return c; }}
static uint64_t buf[64] __attribute__((aligned(64)));
{funcs}
int main(int argc, char **argv) {{
    buf[0] = (uint64_t)buf;
    for (int r = 0; r < 3; r++) {{
{calls}
    }}
    return 0;
}}
"""
FUNC = r"""
__attribute__((noinline)) static uint64_t t_{name}(uint64_t n) {{
    uint64_t t0 = cyc();
    __asm__ __volatile__(
        "li t0, 1\n\t" "li t1, 3\n\t" "fcvt.d.l ft10, t0\n\t" "fcvt.d.l ft11, t0\n\t"
{setup}
        "1:\n\t"
{body}
        "addi %0, %0, -1\n\t" "bnez %0, 1b"
        : "+r"(n) : "r"(buf) : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5",
          "a6", "a7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11");
    return cyc() - t0;
}}
"""
CALL = (
    "        {{ uint64_t t = t_{name}({iters}); "
    'printf("{name} %d %.4f\\n", r, t / (double)({iters} * {u})); }}\n'
)


def lines(ls):
    return "\n".join(f'        "{ln}\\n\\t"' for ln in ls)


def expected(kind, lat, units, piped):
    if kind == "lat":
        return lat
    return max(1 / ISSUE_WIDTH, (1 if piped else lat) / units)


def main():
    setup = [f"li {g}, 1" for g in G] + [f"fcvt.d.l {f}, t0" for f in F]
    funcs, calls = [], []
    for name, _, unit, *_ in TESTS:
        iters = 40 if "div" in name or "sqrt" in name else 150
        funcs.append(FUNC.format(name=name, setup=lines(setup), body=lines([unit(i) for i in range(UNROLL)])))
        calls.append(CALL.format(name=name, iters=iters, u=UNROLL))
    src = C.format(funcs="".join(funcs), calls="".join(calls))
    b = compile_c(src, "p18_micro", "riscv64")
    out, outdir = run_gem5_binary(b, ())
    meas = {}
    for m in re.finditer(r"^(\w+) (\d) ([\d.]+)$", out, re.M):
        meas.setdefault(m.group(1), []).append(float(m.group(3)))
    rows = []
    for name, kind, _, opclass, (lat, units, piped) in TESTS:
        v = meas[name][-1]  # last rep: warm
        e = expected(kind, lat, units, piped)
        rows.append(
            {
                "test": name,
                "kind": kind,
                "opclass": opclass,
                "measured": v,
                "configured": e,
                "reps": meas[name],
                "opLat": lat,
                "units": units,
                "pipelined": piped,
            }
        )
        print(f"{name:12s} {kind:4s} {opclass:13s} measured {v:7.3f}  configured {e:7.3f}")
    od = os.path.join("results", "gem5", "p1_8")
    os.makedirs(od, exist_ok=True)
    json.dump(
        {"rows": rows, "gem5_outdir": outdir, "binary": b},
        open(os.path.join(od, "micro.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
