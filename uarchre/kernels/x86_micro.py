"""x86-only kernels for P1.6, P1.8 and P4.5 (L3 host checks).

`VariantX86` swaps the window probe's `int` filler for another instruction or group (VARIANTS), so
window_probe("int", n) asks what that instruction holds in the window. `MICRO`, `micro_c` and
`run_micro` are a latency/throughput harness: unrolled blocks of one pattern timed with rdtscp, in
core cycles via an `x += n` probe chain run before and after every rep (P-core clocks move ~10 %
within a run). Used by targets/x86/micro_campaign.py and targets/x86/window_campaign.py.
"""

import hashlib
import re
import statistics
import subprocess

from ..backends.isa.x86_64 import X86_64, INT_ROT
from . import Op

# ---------------------------------------------------------------------------
# 1. window-probe filler variants (replace the `int` filler)

R32 = {
    "%%r10": "%%r10d",
    "%%r11": "%%r11d",
    "%%r12": "%%r12d",
    "%%r13": "%%r13d",
    "%%r14": "%%r14d",
    "%%r15": "%%r15d",
}
_BR = "cmpq %%r8, %%r8\\n\\tjne 3f\\n\\t3:"  # never taken, macro-fuses
_NOPS = lambda k: "\\n\\t".join(["nop"] * k)

# name -> (fn(dest reg) -> asm, x86 instructions per unit, doc)
VARIANTS = {
    "lea": (lambda r: f"leaq (%%r8,%%r9), {r}", 1, "writes an int register (the project's int filler)"),
    "mov": (lambda r: f"movq %%r8, {r}", 1, "mov r64, r64 (move-elimination candidate)"),
    "mov32": (lambda r: f"movl %%r8d, {R32[r]}", 1, "mov r32, r32 (zero-extending move)"),
    "xor": (lambda r: f"xorl {R32[r]}, {R32[r]}", 1, "xor r32, r32 zeroing idiom"),
    "movimm": (lambda r: f"movl $7, {R32[r]}", 1, "mov r32, imm32"),
    "addimm": (lambda r: f"addq $1, {r}", 1, "add r64, imm8 (dependent per register)"),
    "branch": (lambda r: _BR, 2, "cmp r, r + jne (fusible pair)"),
    "nop8": (lambda r: _NOPS(8), 8, "8 nops: a unit of 8 entries"),
    "nop9": (lambda r: _NOPS(9), 9, "9 nops: a unit of 9 entries"),
    "br_nop7": (lambda r: _BR + "\\n\\t" + _NOPS(7), 9, "cmp+jne (adjacent, fusible) + 7 nops"),
    "br_split6": (
        lambda r: "cmpq %%r8, %%r8\\n\\tnop\\n\\tjne 3f\\n\\t3:\\n\\t" + _NOPS(6),
        9,
        "cmp; nop; jne (not adjacent, cannot fuse) + 6 nops",
    ),
    "mov_nop7": (lambda r: f"movq %%r8, {r}\\n\\t" + _NOPS(7), 8, "mov r64, r64 + 7 nops"),
    "store6w": (
        lambda r: f"movq %%r8, {8 * INT_ROT.index(r)}(%3)",
        1,
        "store to 6 different words of one line (the project's store filler hits one address)",
    ),
    "store6l": (
        lambda r: f"movq %%r8, {64 * (INT_ROT.index(r) % 2)}(%3)",
        1,
        "store alternating between the 2 lines of the scratch buffer",
    ),
}


class VariantX86(X86_64):
    """X86_64 window emitter whose Op.INT filler is VARIANTS[variant]."""

    def __init__(self, variant):
        self.variant = variant
        self.name = f"x86_64-{variant}"

    def filler(self, op, idx):
        if op == Op.INT:
            return VARIANTS[self.variant][0](INT_ROT[idx % len(INT_ROT)])
        return super().filler(op, idx)


# ---------------------------------------------------------------------------
# 2. latency / throughput micro harness

GPR = ["%%rax", "%%rcx", "%%rdx", "%%rsi", "%%rdi", "%%r8", "%%r9", "%%r10", "%%r11", "%%r12", "%%r13"]
G32 = {
    "%%rax": "%%eax",
    "%%rcx": "%%ecx",
    "%%rdx": "%%edx",
    "%%rsi": "%%esi",
    "%%rdi": "%%edi",
    "%%r8": "%%r8d",
    "%%r9": "%%r9d",
    "%%r10": "%%r10d",
    "%%r11": "%%r11d",
    "%%r12": "%%r12d",
    "%%r13": "%%r13d",
    "%%rbx": "%%ebx",
}
XMM = [f"%%xmm{i}" for i in range(12)]
CLOBBER = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13"] + [
    f"xmm{i}" for i in range(16)
]
SETUP = (
    ["movq $1, %%rbx"]
    + [f"movq $1, {r}" for r in GPR]
    + ["movabsq $0x3FF0000000000000, %%rax", "movq %%rax, %%xmm14", "movq %%rax, %%xmm15"]
    + [f"movq %%rax, {x}" for x in XMM]
    + ["movq $1, %%rax"]
)


class T:
    """One micro test: unit(i) is the asm of the i-th repetition (i picks the rotation register),
    `per_unit` the instructions of interest in one unit (cycles are reported per such instruction)."""

    def __init__(self, name, kind, unit, per_unit=1, setup=(), unroll=96, group="", label=None):
        self.name, self.kind, self.unit, self.per_unit = name, kind, unit, per_unit
        self.setup, self.unroll, self.group, self.label = list(setup), unroll, group, label or name

    def body(self):
        lines = []
        for i in range(self.unroll):
            u = self.unit(i)
            lines += u if isinstance(u, list) else [u]
        return lines


def _g(i):
    return GPR[i % len(GPR)]


def _x(i):
    return XMM[i % len(XMM)]


def lat(name, asm, **kw):
    return T(name, "lat", lambda i: asm, **kw)


def tput(name, fn, **kw):
    return T(name, "tput", fn, **kw)


# P1.8: latency and throughput spot checks
P18 = [
    lat("add_lat", "addq %%rbx, %%rax"),
    tput("add_tput", lambda i: f"addq %%rbx, {_g(i)}"),
    lat("lea2_lat", "leaq (%%rax,%%rbx), %%rax"),
    tput("lea2_tput", lambda i: f"leaq (%%rbx,%%rbx), {_g(i)}"),
    lat("lea3_lat", "leaq 8(%%rax,%%rbx,2), %%rax"),
    tput("lea3_tput", lambda i: f"leaq 8(%%rbx,%%rbx,2), {_g(i)}"),
    lat("shl_lat", "shlq $3, %%rax"),
    tput("shl_tput", lambda i: f"shlq $3, {_g(i)}"),
    lat("shlx_lat", "shlxq %%rbx, %%rax, %%rax"),
    lat("imul_lat", "imulq %%rbx, %%rax"),
    tput("imul_tput", lambda i: f"imulq %%rbx, {_g(i)}"),
    lat("popcnt_lat", "popcntq %%rax, %%rax"),
    tput("popcnt_tput", lambda i: f"popcntq %%rbx, {_g(i)}"),
    lat("lzcnt_lat", "lzcntq %%rax, %%rax"),
    lat("tzcnt_lat", "tzcntq %%rax, %%rax"),
    lat("bswap_lat", "bswapq %%rax"),
    lat("crc32_lat", "crc32q %%rbx, %%rax"),
    lat("cmov_lat", "cmovneq %%rbx, %%rax"),
    lat("adc_lat", "adcq %%rbx, %%rax"),
    lat(
        "div64_lat",
        ["xorl %%edx, %%edx", "divq %%rbx"],
        unroll=24,
        setup=["movq $1000, %%rax", "movq $1, %%rbx"],
        label="div64_lat (q=1000, d=1; +xor)",
    ),
    lat("load_lat", "movq (%%rax), %%rax", setup=["movq %1, %%rax"]),
    tput("load_tput", lambda i: f"movq (%1), {_g(i)}"),
    tput("store_tput", lambda i: f"movq %%rbx, {64 + 8 * (i % 8)}(%1)"),
    lat("addsd_lat", "addsd %%xmm14, %%xmm0"),
    tput("addsd_tput", lambda i: f"addsd %%xmm14, {_x(i)}"),
    lat("mulsd_lat", "mulsd %%xmm14, %%xmm0"),
    tput("mulsd_tput", lambda i: f"mulsd %%xmm14, {_x(i)}"),
    lat("fma_lat", "vfmadd231sd %%xmm14, %%xmm15, %%xmm0"),
    tput("fma_tput", lambda i: f"vfmadd231sd %%xmm14, %%xmm15, {_x(i)}"),
    lat("divsd_lat", "divsd %%xmm14, %%xmm0", unroll=48),
    tput("divsd_tput", lambda i: f"divsd %%xmm14, {_x(i)}", unroll=48),
    lat("sqrtsd_lat", "sqrtsd %%xmm0, %%xmm0", unroll=48, label="sqrtsd_lat (x=1.0)"),
    tput("sqrtsd_tput", lambda i: f"sqrtsd %%xmm15, {_x(i)}", unroll=48),
    lat("vpaddd_lat", "vpaddd %%xmm14, %%xmm0, %%xmm0"),
    tput("vpaddd_tput", lambda i: f"vpaddd %%xmm14, %%xmm15, {_x(i)}"),
    lat("vpmulld_lat", "vpmulld %%xmm14, %%xmm0, %%xmm0"),
    tput("vpmulld_tput", lambda i: f"vpmulld %%xmm14, %%xmm15, {_x(i)}"),
    lat("pshufb_lat", "pshufb %%xmm14, %%xmm0"),
    lat("movq_rt_lat", ["movq %%rax, %%xmm0", "movq %%xmm0, %%rax"], label="movq r64->xmm->r64 round trip"),
    lat(
        "cvt_rt_lat",
        ["cvtsi2sdq %%rax, %%xmm0", "cvttsd2siq %%xmm0, %%rax"],
        label="cvtsi2sd + cvttsd2si round trip",
    ),
]

# P1.6: rename width, move elimination, zeroing idioms, immediate folding
P16 = [
    tput("nop_tput", lambda i: "nop", group="width", label="nop"),
    lat("addreg_dep", "addq %%rbx, %%rax", group="fold", label="add r64, r64 (control)"),
    lat("addimm_dep", "addq $1, %%rax", group="fold", label="add r64, imm8"),
    lat("subimm_dep", "subq $1, %%rax", group="fold", label="sub r64, imm8"),
    lat("inc_dep", "incq %%rax", group="fold", label="inc r64"),
    lat("addimm32_dep", "addq $100000, %%rax", group="fold", label="add r64, imm32"),
    lat("addlimm_dep", "addl $1, %%eax", group="fold", label="add r32, imm8"),
    lat("leadisp_dep", "leaq 1(%%rax), %%rax", group="fold", label="lea r64, [r64+1]"),
    lat(
        "addimm_reg_dep",
        ["addq $1, %%rax", "addq %%rbx, %%rax"],
        per_unit=2,
        group="fold",
        label="add imm8 / add reg alternating",
    ),
    lat("shlimm_dep", "shlq $1, %%rax", group="fold", label="shl r64, 1 (control)"),
    lat(
        "mov64_dep",
        ["movq %%rax, %%rcx", "movq %%rcx, %%rax"],
        per_unit=2,
        group="move",
        label="mov r64, r64",
    ),
    lat(
        "mov32_dep",
        ["movl %%eax, %%ecx", "movl %%ecx, %%eax"],
        per_unit=2,
        group="move",
        label="mov r32, r32",
    ),
    lat(
        "movaps_dep",
        ["movaps %%xmm0, %%xmm1", "movaps %%xmm1, %%xmm0"],
        per_unit=2,
        group="move",
        label="movaps xmm, xmm",
    ),
    lat("movsame_dep", "movq %%rax, %%rax", group="move", label="mov r64, same r64"),
    tput("mov64_tput", lambda i: f"movq %%rbx, {_g(i)}", group="move", label="mov r64, r64 (independent)"),
    tput("xor_tput", lambda i: f"xorl {G32[_g(i)]}, {G32[_g(i)]}", group="zero", label="xor r32, r32"),
    tput("movimm_tput", lambda i: f"movl $7, {G32[_g(i)]}", group="zero", label="mov r32, imm32"),
    tput("addreg_tput", lambda i: f"addq %%rbx, {_g(i)}", group="zero", label="add r64, r64 (independent)"),
    lat("imul_xor", ["imulq %%rax, %%rax", "xorl %%eax, %%eax"], group="idiom", label="imul; xor eax,eax"),
    lat("imul_sub", ["imulq %%rax, %%rax", "subl %%eax, %%eax"], group="idiom", label="imul; sub eax,eax"),
    lat("imul_mov0", ["imulq %%rax, %%rax", "movl $0, %%eax"], group="idiom", label="imul; mov eax,0"),
    lat("imul_and0", ["imulq %%rax, %%rax", "andl $0, %%eax"], group="idiom", label="imul; and eax,0"),
    lat("imul_only", "imulq %%rax, %%rax", group="idiom", label="imul (latency)"),
    lat(
        "vmul_vpxor",
        ["mulsd %%xmm14, %%xmm0", "vpxor %%xmm0, %%xmm0, %%xmm0"],
        group="idiom",
        label="mulsd; vpxor x,x,x",
    ),
    lat("vmul_only", "mulsd %%xmm14, %%xmm0", group="idiom", label="mulsd (latency)"),
]

# P1.8 follow-ups on the P-core: SHLX latency, load throughput
P18X = [
    lat("shlx_lat", "shlxq %%rbx, %%rax, %%rax"),
    lat("shlx_c0_lat", "shlxq %%rcx, %%rax, %%rax", setup=["movq $0, %%rcx"], label="shlx, count 0"),
    lat("shrx_lat", "shrxq %%rbx, %%rax, %%rax", label="shrx"),
    lat("sarx_lat", "sarxq %%rbx, %%rax, %%rax", label="sarx"),
    lat("shlcl_lat", "shlq %%cl, %%rax", setup=["movq $1, %%rcx"], label="shl r64, cl"),
    lat("rorx_lat", "rorxq $1, %%rax, %%rax", label="rorx"),
    tput("shlx_tput", lambda i: f"shlxq %%rbx, {_g(i)}, {_g(i)}", label="shlx tput"),
    tput("load_tput", lambda i: f"movq (%1), {_g(i)}"),
    tput("load8_tput", lambda i: f"movq {64 * (i % 8)}(%1), {_g(i)}", label="loads from 8 lines"),
    tput("load8w_tput", lambda i: f"movq {8 * (i % 8)}(%1), {_g(i)}", label="loads from 8 words, 1 line"),
    tput("load32_tput", lambda i: f"movl (%1), {G32[_g(i)]}", label="32-bit loads"),
    tput("loadx_tput", lambda i: f"vmovq (%1), {_x(i)}", label="loads to xmm"),
]

MICRO = {t.name: t for t in P18 + P16 + [t for t in P18X if t.name not in ("shlx_lat", "load_tput")]}

C_MICRO = r"""
/* generated by uarch-re x86_micro: {names} */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

static inline uint64_t tsc(void) {{ unsigned a; __asm__ __volatile__("lfence" ::: "memory"); return __rdtscp(&a); }}
static uint64_t buf[64] __attribute__((aligned(64)));

/* the project's clock probe: x += n, 64 per block, 1 core cycle per add */
__attribute__((noinline)) static uint64_t probe(uint64_t n) {{
    uint64_t x = 0, t0 = tsc();
    __asm__ __volatile__("1:\n\t" {probe} "decq %1\n\tjnz 1b" : "+r"(x), "+r"(n) :: "cc");
    uint64_t t1 = tsc();
    __asm__ __volatile__("" :: "r"(x));
    return t1 - t0;
}}
{funcs}
int main(int argc, char **argv) {{
    int reps = argc > 1 ? atoi(argv[1]) : 7;
    uint64_t pn = 8192;                              /* probe blocks: 524288 adds */
    buf[0] = (uint64_t)buf;                          /* L1 self-pointer for the load chase */
    uint64_t w0 = tsc();                             /* ~0.2 s warm-up: let the core ramp */
    while (tsc() - w0 < 500000000ULL) probe(256);
    for (int r = 0; r < reps; r++) {{
{calls}
    }}
    return 0;
}}
"""  # noqa: E501

FUNC = r"""
__attribute__((noinline)) static uint64_t t_{name}(uint64_t n) {{
    uint64_t t0 = tsc();
    __asm__ __volatile__(
{setup}
        "1:\n\t"
{body}
        "decq %0\n\tjnz 1b"
        : "+r"(n) : "r"(buf) : "memory", "cc"{clob});
    return tsc() - t0;
}}
"""

CALL = (
    "        {{ uint64_t a = probe(pn), t = t_{name}({iters}ULL), b = probe(pn);\n"
    '          printf("{name} %d %.5f %.5f %.5f\\n", r, a / (64.0 * pn), '
    "t / (double)({iters}ULL * {count}), b / (64.0 * pn)); }}\n"
)


def _lines(ls):
    return "\n".join(f'        "{line}\\n\\t"' for line in ls)


def micro_c(tests, iters=None):
    funcs, calls = [], []
    clob = "".join(f', "{c}"' for c in CLOBBER)
    for t in tests:
        count = t.unroll * t.per_unit
        it = iters or max(200, 2_000_000 // len(t.body()))  # ~2M instructions per timing
        funcs.append(
            FUNC.format(name=t.name, setup=_lines(SETUP + t.setup), body=_lines(t.body()), clob=clob)
        )
        calls.append(CALL.format(name=t.name, iters=it, count=count))
    return C_MICRO.format(
        names=", ".join(t.name for t in tests),
        probe=" ".join(['"addq %1, %0\\n\\t"'] * 64),
        funcs="".join(funcs),
        calls="".join(calls),
    )


def compile_micro(tests, iters=None):
    from ..backends.platform.native import compile_source

    src = micro_c(tests, iters)
    h = hashlib.md5(src.encode()).hexdigest()[:8]
    return compile_source(
        src,
        f"micro_{h}",
        cflags=("-O1", "-mavx2", "-mfma", "-mpopcnt", "-mlzcnt", "-mbmi", "-mbmi2", "-msse4.2"),
    )


def run_micro(tests, cpu, reps=7, iters=None):
    """{name: [(probe_pre, ticks_per_instr, probe_post), ...]} for every rep."""
    binpath = compile_micro(tests, iters)
    cmd = ["taskset", "-c", str(cpu), binpath, str(reps)] if cpu is not None else [binpath, str(reps)]
    return parse_micro(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def parse_micro(out):
    res = {}
    for m in re.finditer(r"^(\w+) (\d+) ([\d.]+) ([\d.]+) ([\d.]+)$", out, re.M):
        res.setdefault(m.group(1), []).append(tuple(float(m.group(i)) for i in (3, 4, 5)))
    return res


def cycles(samples):
    """Core cycles per instruction for each rep: ticks / mean(probe pre, post)."""
    return [t / ((a + b) / 2) for a, t, b in samples]


def summary(samples):
    c = cycles(samples)
    return {
        "median": statistics.median(c),
        "min": min(c),
        "max": max(c),
        "n": len(c),
        "probe_drift": max(abs(b / a - 1) for a, _, b in samples),
    }
