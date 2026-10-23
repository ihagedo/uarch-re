"""Frontend predictor and decoupling kernels: P3.4 history length, P3.5 indirect capacity,
P3.6 RAS depth, P3.8 fetch-directed prefetch, P3.9 fetch queue.

Each generator returns a self-contained C source with the kernel as top-level asm: RISC-V (.option
norvc, so layouts are exact) or, with isa="aarch64", AArch64 read through `counter` (aarch64.COUNTERS).
Most time every sweep point inside one gem5 process; P3.8 and P3.9 put the swept quantity into the
code layout. Every point prints `pt <key> rep <r> cyc <cycles per unit>` for parse_points(). Used by
validation/gem5/fp_*.py.
"""

import random
import re
from collections import defaultdict

from ..backends.isa.aarch64 import COUNTERS

HEAD = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static inline uint64_t cyc(void) {
    uint64_t c; __asm__ __volatile__("rdcycle %0" : "=r"(c)); return c;
}
"""


def _toplevel(lines):
    """Top-level asm block (4-byte instructions, no compression)."""
    body = "\n".join(f'    "{line}\\n"' for line in lines)
    return f'__asm__(\n    ".option push\\n"\n    ".option norvc\\n"\n{body}\n    ".option pop\\n"\n);\n'


def _head(isa, counter):
    if isa == "riscv64":
        return HEAD
    if isa != "aarch64":
        raise ValueError(f"isa must be riscv64 or aarch64, not {isa!r}")
    head = "\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n" + COUNTERS[counter]
    return head + "static inline uint64_t cyc(void) { return cycles(); }\n"


def _asm(isa, lines):
    if isa == "riscv64":
        return _toplevel(lines)
    body = "\n".join(f'    "{line}\\n"' for line in lines)
    return f"__asm__(\n{body}\n);\n"


def _extern(isa, decl, name):
    """C declaration of an asm symbol; the AArch64 asm label keeps the name as written (Mach-O)."""
    return f"extern {decl};" if isa == "riscv64" else f'extern {decl} __asm__("{name}");'


_PT = re.compile(r"^pt (\S+) rep (\d+) cyc ([-\d.eE+]+)", re.M)


def parse_points(out):
    """{key: [cycles per unit per rep]} from the harness output (rep order kept)."""
    pts = defaultdict(list)
    for key, _, v in _PT.findall(out):
        pts[key].append(float(v))
    return dict(pts)


# ---------------------------------------------- P3.6 return-address-stack depth


def ras_source(dmax=128, depths=None, warm=4, iters=48, reps=3, seed=0, isa="riscv64", counter="pmccntr"):
    """C source for the RAS depth sweep: a chain of `dmax` functions, entered at f_{dmax-D} for depth D.

    An R-entry RAS mispredicts the outer D-R returns, so T(D) - T(D-1) steps up by one mispredict
    penalty after D = R. `seed` changes code placement only (0-15 NOPs before each function)."""
    depths = list(depths or range(1, dmax + 1))
    assert all(1 <= d <= dmax for d in depths)
    rng = random.Random(seed)
    asm = [".text", ".p2align 6"]
    for i in range(dmax):
        pad = rng.randrange(16) if seed else 0
        asm += ["nop"] * pad
        asm += [f".globl ras_f{i}", f"ras_f{i}:"]
        if i == dmax - 1:
            asm += ["ret"]
        elif isa == "aarch64":
            asm += ["str x30, [sp, #-16]!", f"bl ras_f{i + 1}", "ldr x30, [sp], #16", "ret"]
        else:
            asm += [
                "addi sp, sp, -16",
                "sd ra, 8(sp)",
                f"jal ra, ras_f{i + 1}",
                "ld ra, 8(sp)",
                "addi sp, sp, 16",
                "ret",
            ]
    decl = "\n".join(_extern(isa, f"void ras_f{i}(void)", f"ras_f{i}") for i in range(dmax))
    table = ", ".join(f"ras_f{dmax - d}" for d in depths)
    return (
        _head(isa, counter)
        + _asm(isa, asm)
        + decl
        + f"""
static void (*const entry[])(void) = {{ {table} }};
static const int depth[] = {{ {", ".join(map(str, depths))} }};
int main(void) {{
    for (unsigned i = 0; i < sizeof depth / sizeof depth[0]; i++) {{
        void (*volatile f)(void) = entry[i];
        void (*g)(void) = f;
        for (int w = 0; w < {warm}; w++) g();
        for (int r = 0; r < {reps}; r++) {{
            uint64_t t0 = cyc();
            for (int n = 0; n < {iters}; n++) g();
            uint64_t t1 = cyc();
            printf("pt %d rep %d cyc %.4f\\n", depth[i], r, (double)(t1 - t0) / {iters});
        }}
    }}
    return 0;
}}
"""
    )


# --------------------------------------------- P3.9 fetch-to-decode queue depth


def fetchq_source(
    bs,
    copies=16,
    chain=12,
    stride=64 << 10,
    iters=12,
    reps=3,
    warm=2,
    isa="riscv64",
    counter="pmccntr",
    line_bytes=64,
    slot=4096,
    spread=1,
    hops=1,
    serializer=None,
):
    """C source for up to stride/slot points (B NOPs each) of the fetch-queue sweep.

    Each block: an fsqrt chain (backend stall), a serializing read, B NOPs and a jump through `hops` lines
    that miss in L1i (and L2 on gem5); the misses stay hidden while the frontend holds the NOPs and jump."""
    bs = [bs] if isinstance(bs, int) else list(bs)
    a64 = isa == "aarch64"
    ser = serializer or ("mrs x9, fpsr" if a64 else "csrr t2, fflags")
    ser = [s.strip() for s in ser.split(";") if s.strip()]  # ';' starts a comment in Mach-O asm
    per = stride // slot
    tail = slot - line_bytes * hops - 128
    assert line_bytes % 4 == 0 and len(bs) <= per and copies * stride <= (128 << 20 if a64 else 1 << 20)
    assert 4 * (max(bs) + chain + len(ser) + 1) + line_bytes * spread * (copies - 1) < tail, "block too long"
    j = "b" if a64 else "j"
    asm = [".text", ".p2align 12"]
    for v in range(len(bs)):
        asm += [f".globl fq_kern{v}", f"fq_kern{v}:"]
        if a64:
            asm += ["fmov d1, #4.0", "mov x9, x0", "cmp x0, #0"]  # claim d1, x9 and NZCV
        else:
            asm += ["fcvt.d.w ft0, zero", "li t0, 4", "fcvt.d.w ft1, t0"]  # ft1 = 4.0, sqrt stays finite
        asm += [f"{j} fq_blk{v}_0"]
    # slot k*per + v holds block (v, k) at line spread*k and its landing lines at the slot's end, one per
    # hop, in descending order. Defaults: one L1i set (63) and one L2 set per variant, thrashed by
    # copies > 8 ways, and one SimpleBTB slot per jump; B + chain + 2 > 128 thrashes the blocks too.
    asm += [".p2align 16", "fq_region:"]
    land = lambda v, k, h: f"fq_land{v}_{k}" + (f"_{h}" if h else "")
    for k in range(copies):
        for v, b in enumerate(bs):
            page = f"fq_region + {(k * per + v) * slot}"
            nxt = f"fq_blk{v}_{k + 1}" if k + 1 < copies else f"fq_tail{v}"
            asm += [f".org {page} + {line_bytes * spread * k}", f"fq_blk{v}_{k}:"]
            asm += ["fsqrt d1, d1" if a64 else "fsqrt.d ft1, ft1"] * chain  # chain x FloatSqrt latency
            # serializing read: the next instruction waits in rename for an empty ROB (gem5: csrr is
            # serialize-after, mrs serialize-before)
            asm += ser
            asm += ["nop"] * b
            asm += [f"{j} {land(v, k, 0)}"]
            if k == 0:
                asm += [f".org {page} + {tail}", f"fq_tail{v}:"]
                if a64:
                    asm += ["subs x0, x0, #1", f"b.ne fq_blk{v}_0", "ret"]
                else:
                    asm += ["addi a0, a0, -1", f"bnez a0, fq_blk{v}_0", "ret"]
            for h in reversed(range(hops)):
                to = nxt if h == hops - 1 else land(v, k, h + 1)
                asm += [
                    f".org {page} + {slot - line_bytes * (h + 1) + 4 * (k % (line_bytes // 4))}",
                    f"{land(v, k, h)}:",
                    f"{j} {to}",
                ]
    asm += [f".org fq_region + {copies * stride}"]
    decl = "\n".join(_extern(isa, f"void fq_kern{v}(long laps)", f"fq_kern{v}") for v in range(len(bs)))
    runs = "\n".join(
        f"""    fq_kern{v}({warm});
    for (int r = 0; r < {reps}; r++) {{
        uint64_t t0 = cyc();
        fq_kern{v}({iters});
        uint64_t t1 = cyc();
        printf("pt {b} rep %d cyc %.4f\\n", r, (double)(t1 - t0) / ({iters} * {copies}));
    }}"""
        for v, b in enumerate(bs)
    )
    return (
        _head(isa, counter)
        + _asm(isa, asm)
        + decl
        + f"""
int main(void) {{
{runs}
    return 0;
}}
"""
    )


# ---------------------------------- P3.8 decoupling and fetch-directed prefetch


def fdip_order(n_lines, seed=1):
    """Traversal order: 4 KiB pages in address order, the 64 lines of each
    page in a random order (no sequential next line inside a page)."""
    rng = random.Random(seed)
    order = []
    for p in range(0, n_lines, 64):
        page = list(range(p, min(p + 64, n_lines)))
        rng.shuffle(page)
        order += page
    return order


def fdip_source(
    periods=0, n_lines=7168, iters=1, reps=2, warm=1, seed=1, isa="riscv64", counter="pmccntr", line_bytes=64
):
    """C source for one or more barrier periods (0 = none) of the run-ahead sweep.

    Each line (64 B) is NOPs and a jump to the next line in `fdip_order` (7168 lines exceed L2). Every P-th
    line ends in a `ret` the RAS mispredicts, so FDIP overlaps at most min(P, FTQ) line misses."""
    periods = [periods] if isinstance(periods, int) else list(periods)
    order = fdip_order(n_lines, seed)
    nxt = {order[i]: order[i + 1] for i in range(len(order) - 1)}
    nxt[order[-1]] = None
    a64 = isa == "aarch64"
    assert not a64 or n_lines * line_bytes < 1 << 20, "b.ne back to the first line"
    asm = [".text"]
    targets = []  # AArch64: barrier targets in traversal order, as line numbers (n_lines = tail)
    for v, period in enumerate(periods):
        barrier = [order[i] for i in range(period - 1, len(order), period)] if period else []
        targets.append([n_lines if nxt[b] is None else nxt[b] for b in barrier])
        barrier = set(barrier)
        base = f"fd{v}_lines"
        if a64:
            # x10 walks a table of barrier targets built in C (no adr in timed code); x11 rewinds it
            asm += [".p2align 12", f".globl fd_kern{v}", f"fd_kern{v}:", "str x30, [sp, #-16]!"]
            asm += ["mov x11, x1", "mov x10, x1", "cmp x0, #0", f"b fd{v}_l{order[0]}"]
            asm += [".p2align 12", f".globl {base}", f"{base}:"]
        else:
            asm += [
                ".p2align 12",
                f".globl fd_kern{v}",
                f"fd_kern{v}:",
                "addi sp, sp, -16",
                "sd ra, 8(sp)",  # barrier lines overwrite ra
                f"j fd{v}_l{order[0]}",
                ".p2align 12",
                f"{base}:",
            ]
        for line in range(n_lines):
            # distinct slots for up to 7168 jumps in a BTB on pc[2:15) (SimpleBTB with 32768 entries)
            w = 2 + (line >> 9) % 14
            tgt = f"fd{v}_l{nxt[line]}" if nxt[line] is not None else f"fd{v}_tail"
            asm += [f".org {base} + {line * line_bytes}", f"fd{v}_l{line}:"]
            if line in barrier and a64:
                asm += ["ldr x30, [x10], #8"] + ["nop"] * (w - 1) + ["ret"]
            elif line in barrier:
                asm += [f"lla ra, {tgt}"] + ["nop"] * (w - 2) + ["ret"]
            else:
                asm += ["nop"] * w + [f"{'b' if a64 else 'j'} {tgt}"]
        asm += [f".org {base} + {n_lines * line_bytes}", f"fd{v}_tail:"]
        if a64:
            asm += ["subs x0, x0, #1", "mov x10, x11", f"b.ne fd{v}_l{order[0]}", "ldr x30, [sp], #16", "ret"]
        else:
            asm += [
                "addi a0, a0, -1",
                f"bnez a0, fd{v}_l{order[0]}",
                "ld ra, 8(sp)",
                "addi sp, sp, 16",
                "ret",
            ]
    if a64:
        return _fdip_a64(asm, periods, targets, n_lines, iters, reps, warm, counter, line_bytes)
    decl = "\n".join(f"extern void fd_kern{v}(long laps);" for v in range(len(periods)))
    runs = "\n".join(
        f"""    fd_kern{v}({warm});
    for (int r = 0; r < {reps}; r++) {{
        uint64_t t0 = cyc();
        fd_kern{v}({iters});
        uint64_t t1 = cyc();
        printf("pt {p} rep %d cyc %.4f\\n", r, (double)(t1 - t0) / ({iters} * {n_lines}));
    }}"""
        for v, p in enumerate(periods)
    )
    return (
        HEAD
        + _toplevel(asm)
        + decl
        + f"""
int main(void) {{
{runs}
    return 0;
}}
"""
    )


def _fdip_a64(asm, periods, targets, n_lines, iters, reps, warm, counter, line_bytes=64):
    isa = "aarch64"
    decl = []
    for v, t in enumerate(targets):
        decl += [
            _extern(isa, f"void fd_kern{v}(long laps, void *const *bt)", f"fd_kern{v}"),
            _extern(isa, f"char fd{v}_lines[]", f"fd{v}_lines"),
            f"static const unsigned short fd{v}_bt[] = {{ {', '.join(map(str, t)) or '0'} }};",
        ]
    runs = "\n".join(
        f"""    for (unsigned i = 0; i < {len(t)}; i++) bt[i] = fd{v}_lines + {line_bytes} * fd{v}_bt[i];
    fd_kern{v}({warm}, bt);
    for (int r = 0; r < {reps}; r++) {{
        uint64_t t0 = cyc();
        fd_kern{v}({iters}, bt);
        uint64_t t1 = cyc();
        printf("pt {p} rep %d cyc %.4f\\n", r, (double)(t1 - t0) / ({iters} * {n_lines}));
    }}"""
        for v, (p, t) in enumerate(zip(periods, targets))
    )
    return (
        _head(isa, counter)
        + _asm(isa, asm)
        + "\n".join(decl)
        + f"""
int main(void) {{
    void **bt = malloc(sizeof(void *) * {max(1, *map(len, targets))});
{runs}
    return 0;
}}
"""
    )


# -------------------------------------- P3.4 direction-predictor history length


def _history_fn(name, n, variant, isa="riscv64"):
    """One loop: LFSR step; branch A on bit 0; N taken filler jumps; branch B.
    variant: corr (B on bit 0, same as A), rand (B on bit 1), const (B on 0)."""
    if isa == "aarch64":
        bit_b = {"corr": "x11", "rand": "x12", "const": "xzr"}[variant]
        a = [".p2align 6", f"{name}:", "mov x11, x10", "mov x12, x10", "cmp x0, #0", "1:"]  # claim
        a += ["eor x10, x10, x10, lsl #13", "eor x10, x10, x10, lsr #7", "eor x10, x10, x10, lsl #17"]
        a += ["and x11, x10, #1", "ubfx x12, x10, #1, #1", "cbz x11, 2f", "nop", "2:"]
        a += ["b 3f", "3:"] * n
        return a + [f"cbz {bit_b}, 4f", "nop", "4:", "subs x0, x0, #1", "b.ne 1b", "ret"]
    bit_b = {"corr": "t3", "rand": "t4", "const": "zero"}[variant]
    a = [
        ".p2align 6",
        f"{name}:",
        "1:",
        "slli t2, s1, 13",
        "xor s1, s1, t2",
        "srli t2, s1, 7",
        "xor s1, s1, t2",
        "slli t2, s1, 17",
        "xor s1, s1, t2",
        "andi t3, s1, 1",
        "srli t4, s1, 1",
        "andi t4, t4, 1",
        "beqz t3, 2f",
        "nop",
        "2:",
    ]  # A: taken iff bit 0 == 0
    for _ in range(n):
        a += ["j 3f", "3:"]  # taken filler: one history bit each
    a += [
        f"beqz {bit_b}, 4f",
        "nop",
        "4:",  # B
        "addi a0, a0, -1",
        "bnez a0, 1b",
        "ret",
    ]
    return a


def history_source(
    ns,
    iters=1000,
    warm=1000,
    reps=2,
    variants=("corr", "rand", "const"),
    seed=0,
    isa="riscv64",
    counter="pmccntr",
):
    """C source for one or more N (filler counts), each with the three B variants.

    Branch B repeats random branch A after N taken fillers, so it is predictable only while the global
    history reaches back N+1 branches; `rand` gives the unpredictable cost. `seed` picks the LFSR start."""
    ns = [ns] if isinstance(ns, int) else list(ns)
    asm = [".text"]
    for n in ns:
        for v in variants:
            asm += [f".globl hist_{v}_{n}"] + _history_fn(f"hist_{v}_{n}", n, v, isa)
    if isa == "aarch64":
        return _history_a64(asm, ns, iters, warm, reps, variants, seed, counter)
    decl = "\n".join(f"extern void hist_{v}_{n}(long iters, uint64_t seed);" for n in ns for v in variants)
    calls = "\n".join(
        f"""    {{
        run(hist_{v}_{n}, {warm});
        for (int r = 0; r < {reps}; r++) {{
            uint64_t t0 = cyc(); run(hist_{v}_{n}, {iters}); uint64_t t1 = cyc();
            printf("pt {v}:{n} rep %d cyc %.4f\\n", r, (double)(t1 - t0) / {iters});
        }}
    }}"""
        for n in ns
        for v in variants
    )
    return (
        HEAD
        + _toplevel(asm)
        + decl
        + f"""
static uint64_t state = 0x9E3779B97F4A7C15ull ^ ({seed}ull * 0xD1B54A32D192ED03ull);
/* s1 carries the LFSR state across calls (callee-saved: save/restore here) */
static void run(void (*f)(long, uint64_t), long n) {{
    register uint64_t s1 __asm__("s1") = state;
    register long a0 __asm__("a0") = n;
    __asm__ __volatile__("jalr ra, 0(%2)" : "+r"(a0), "+r"(s1) : "r"(f)
                         : "ra", "t2", "t3", "t4", "memory");
    state = s1;
}}
int main(void) {{
{calls}
    return 0;
}}
"""
    )


def _history_a64(asm, ns, iters, warm, reps, variants, seed, counter):
    isa = "aarch64"
    decl = "\n".join(_extern(isa, f"void hist_{v}_{n}(void)", f"hist_{v}_{n}") for n in ns for v in variants)
    calls = "\n".join(
        f"""    {{
        void (*volatile fv)(void) = hist_{v}_{n};
        void (*f)(void) = fv;
        s = run(f, {warm}, s);
        for (int r = 0; r < {reps}; r++) {{
            uint64_t t0 = cyc(); s = run(f, {iters}, s); uint64_t t1 = cyc();
            printf("pt {v}:{n} rep %d cyc %.4f\\n", r, (double)(t1 - t0) / {iters});
        }}
    }}"""
        for n in ns
        for v in variants
    )
    return (
        _head(isa, counter)
        + _asm(isa, asm)
        + decl
        + f"""
/* x10 carries the LFSR state across calls; f comes through a volatile, so no adrp is timed */
static inline uint64_t run(void (*f)(void), long n, uint64_t s) {{
    register uint64_t x10 __asm__("x10") = s;
    register long x0 __asm__("x0") = n;
    __asm__ __volatile__("blr %2" : "+r"(x0), "+r"(x10) : "r"(f) : "x11", "x12", "x30", "cc", "memory");
    return x10;
}}
int main(void) {{
    uint64_t s = 0x9E3779B97F4A7C15ull ^ ({seed}ull * 0xD1B54A32D192ED03ull);
{calls}
    return 0;
}}
"""
    )


# --------------------------------------------- P3.5 indirect-predictor capacity


def indirect_source(smax, sizes, targets=1, rounds=8, warm=4, reps=2, isa="riscv64", counter="pmccntr"):
    """C source for the indirect-capacity sweep (all S in one binary).

    The last S of `smax` jalr/blr sites each call a leaf rotated over `targets` addresses, so cost per call
    steps up once S*T exceeds capacity. Every branch is taken, so the direction history stays constant."""
    if isa == "aarch64":
        return _indirect_a64(smax, sizes, targets, rounds, warm, reps, counter)
    asm = [".text", ".p2align 6"]
    for t in range(targets):
        asm += [f".globl ind_leaf{t}", f"ind_leaf{t}:", "ret", ".p2align 4"]
    asm += [".p2align 6", ".globl ind_sites", "ind_sites:"]
    asm += ["jalr ra, 0(s2)"] * smax
    asm += ["ld ra, 8(sp)", "addi sp, sp, 16", "ret"]
    asm += [
        ".globl ind_enter",
        "ind_enter:",  # a0 = entry address, s2 = leaf
        "addi sp, sp, -16",
        "sd ra, 8(sp)",
        "jr a0",
    ]
    leaves = ", ".join(f"ind_leaf{t}" for t in range(targets))
    decl = "\n".join(f"extern void ind_leaf{t}(void);" for t in range(targets))
    return (
        HEAD
        + _toplevel(asm)
        + decl
        + f"""
extern char ind_sites[];
extern void ind_enter(void);
static void (*const leaf[])(void) = {{ {leaves} }};
static const int sizes[] = {{ {", ".join(map(str, sizes))} }};
static inline void run(char *entry, void (*lf)(void)) {{
    register char *a0 __asm__("a0") = entry;
    register void (*s2)(void) __asm__("s2") = lf;
    __asm__ __volatile__("call ind_enter" : "+r"(a0) : "r"(s2)
                         : "ra", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2",
                           "t3", "t4", "t5", "t6", "memory");
}}
int main(void) {{
    for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {{
        int s = sizes[i];
        char *entry = ind_sites + 4 * ({smax} - s);
        for (int w = 0; w < {warm} * {targets}; w++) run(entry, leaf[w % {targets}]);
        for (int r = 0; r < {reps}; r++) {{
            uint64_t t0 = cyc();
            for (int n = 0; n < {rounds} * {targets}; n++) run(entry, leaf[n % {targets}]);
            uint64_t t1 = cyc();
            printf("pt %d rep %d cyc %.4f\\n", s, r, (double)(t1 - t0) / ({rounds} * {targets}));
        }}
    }}
    return 0;
}}
"""
    )


def _indirect_a64(smax, sizes, targets, rounds, warm, reps, counter):
    isa = "aarch64"
    asm = [".text", ".p2align 6"]
    for t in range(targets):
        asm += [f".globl ind_leaf{t}", f"ind_leaf{t}:", "ret", ".p2align 4"]
    asm += [".p2align 6", ".globl ind_sites", "ind_sites:"]
    asm += ["blr x9"] * smax
    asm += ["ldr x30, [sp], #16", "ret"]
    asm += [".globl ind_enter", "ind_enter:", "str x30, [sp, #-16]!", "br x0"]  # x0 = entry, x9 = leaf
    leaves = ", ".join(f"ind_leaf{t}" for t in range(targets))
    decl = [_extern(isa, f"void ind_leaf{t}(void)", f"ind_leaf{t}") for t in range(targets)]
    decl += [_extern(isa, "char ind_sites[]", "ind_sites")]
    return (
        _head(isa, counter)
        + _asm(isa, asm)
        + "\n".join(decl)
        + f"""
static void (*const leaf[])(void) = {{ {leaves} }};
static const int sizes[] = {{ {", ".join(map(str, sizes))} }};
static inline void run(char *entry, void (*lf)(void)) {{
    register char *x0 __asm__("x0") = entry;
    register void (*x9)(void) __asm__("x9") = lf;
    __asm__ __volatile__("bl ind_enter" : "+r"(x0) : "r"(x9) : "x30", "memory");
}}
int main(void) {{
    /* addresses come through volatiles, so the timed loop has no adrp */
    char *volatile sv = ind_sites;
    void (*const *volatile lv)(void) = leaf;
    char *sites = sv;
    void (*const *lp)(void) = lv;
    for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {{
        int s = sizes[i];
        char *entry = sites + 4 * ({smax} - s);
        for (int w = 0; w < {warm} * {targets}; w++) run(entry, lp[w % {targets}]);
        for (int r = 0; r < {reps}; r++) {{
            uint64_t t0 = cyc();
            for (int n = 0; n < {rounds} * {targets}; n++) run(entry, lp[n % {targets}]);
            uint64_t t1 = cyc();
            printf("pt %d rep %d cyc %.4f\\n", s, r, (double)(t1 - t0) / ({rounds} * {targets}));
        }}
    }}
    return 0;
}}
"""
    )
