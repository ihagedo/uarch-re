"""Frontend predictor and decoupling kernels: P3.4 history length, P3.5 indirect capacity,
P3.6 RAS depth, P3.8 fetch-directed prefetch, P3.9 fetch queue.

Each generator returns a self-contained C source with the kernel as top-level RISC-V asm (.option
norvc, so layouts are exact). Most time every sweep point inside one gem5 process; P3.8 and P3.9 put
the swept quantity into the code layout. Every point prints `pt <key> rep <r> cyc <cycles per unit>`
for parse_points(). Used by validation/gem5/fp_*.py.
"""

import random
import re
from collections import defaultdict

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


_PT = re.compile(r"^pt (\S+) rep (\d+) cyc ([-\d.eE+]+)", re.M)


def parse_points(out):
    """{key: [cycles per unit per rep]} from the harness output (rep order kept)."""
    pts = defaultdict(list)
    for key, _, v in _PT.findall(out):
        pts[key].append(float(v))
    return dict(pts)


# ---------------------------------------------- P3.6 return-address-stack depth


def ras_source(dmax=128, depths=None, warm=4, iters=48, reps=3, seed=0):
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
        else:
            asm += [
                "addi sp, sp, -16",
                "sd ra, 8(sp)",
                f"jal ra, ras_f{i + 1}",
                "ld ra, 8(sp)",
                "addi sp, sp, 16",
                "ret",
            ]
    decl = "\n".join(f"extern void ras_f{i}(void);" for i in range(dmax))
    table = ", ".join(f"ras_f{dmax - d}" for d in depths)
    return (
        HEAD
        + _toplevel(asm)
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


def fetchq_source(bs, copies=16, chain=12, stride=64 << 10, iters=12, reps=3, warm=2):
    """C source for up to stride/4096 points (B NOPs each) of the fetch-queue sweep.

    Each block: an fsqrt chain (backend stall), a serializing CSR read, B NOPs and a jump to a line that
    misses in L1i and L2; the miss is hidden while the frontend can buffer the NOPs and the jump."""
    bs = [bs] if isinstance(bs, int) else list(bs)
    per = stride // 4096
    assert copies <= 16 and len(bs) <= per and copies * stride <= (1 << 20)
    assert 4 * (max(bs) + chain + 2) + 64 * (copies - 1) < 3904, "block must fit below the tail"
    asm = [".text", ".p2align 12"]
    for v in range(len(bs)):
        asm += [
            f".globl fq_kern{v}",
            f"fq_kern{v}:",
            "fcvt.d.w ft0, zero",
            "li t0, 4",
            "fcvt.d.w ft1, t0",  # ft1 = 4.0, sqrt chain stays finite
            f"j fq_blk{v}_0",
        ]
    # page k*per + v holds block (v, k) at offset 64k and its landing line at 4032 + 4k: one L1i set
    # (63) and one L2 set per variant, thrashed by copies > 8 ways, and one SimpleBTB slot per jump.
    # B + chain + 2 > 128 instructions would put 9+ blocks in one L1i set.
    asm += [".p2align 16", "fq_region:"]
    for k in range(copies):
        for v, b in enumerate(bs):
            page = f"fq_region + {(k * per + v) * 4096}"
            nxt = f"fq_blk{v}_{k + 1}" if k + 1 < copies else f"fq_tail{v}"
            asm += [f".org {page} + {64 * k}", f"fq_blk{v}_{k}:"]
            asm += ["fsqrt.d ft1, ft1"] * chain  # backend stall: chain x FloatSqrt latency
            asm += ["csrr t2, fflags"]  # serialize-after: next instruction waits for an empty ROB
            asm += ["nop"] * b
            asm += [f"j fq_land{v}_{k}"]
            if k == 0:
                asm += [
                    f".org {page} + 3904",
                    f"fq_tail{v}:",
                    "addi a0, a0, -1",
                    f"bnez a0, fq_blk{v}_0",
                    "ret",
                ]
            asm += [f".org {page} + {4032 + 4 * k}", f"fq_land{v}_{k}:", f"j {nxt}"]
    asm += [f".org fq_region + {copies * stride}"]
    decl = "\n".join(f"extern void fq_kern{v}(long laps);" for v in range(len(bs)))
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


def fdip_source(periods=0, n_lines=7168, iters=1, reps=2, warm=1, seed=1):
    """C source for one or more barrier periods (0 = none) of the run-ahead sweep.

    Each 64 B line is NOPs and a jump to the next line in `fdip_order` (7168 lines exceed L2). Every P-th
    line ends in a `ret` the RAS mispredicts, so FDIP overlaps at most min(P, FTQ) line misses."""
    periods = [periods] if isinstance(periods, int) else list(periods)
    order = fdip_order(n_lines, seed)
    nxt = {order[i]: order[i + 1] for i in range(len(order) - 1)}
    nxt[order[-1]] = None
    asm = [".text"]
    for v, period in enumerate(periods):
        barrier = set()
        if period:
            barrier = {order[i] for i in range(period - 1, len(order), period)}
        base = f"fd{v}_lines"
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
            asm += [f".org {base} + {line * 64}", f"fd{v}_l{line}:"]
            if line in barrier:
                asm += [f"lla ra, {tgt}"] + ["nop"] * (w - 2) + ["ret"]
            else:
                asm += ["nop"] * w + [f"j {tgt}"]
        asm += [
            f".org {base} + {n_lines * 64}",
            f"fd{v}_tail:",
            "addi a0, a0, -1",
            f"bnez a0, fd{v}_l{order[0]}",
            "ld ra, 8(sp)",
            "addi sp, sp, 16",
            "ret",
        ]
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


# -------------------------------------- P3.4 direction-predictor history length


def _history_fn(name, n, variant):
    """One loop: LFSR step; branch A on bit 0; N taken filler jumps; branch B.
    variant: corr (B on bit 0, same as A), rand (B on bit 1), const (B on 0)."""
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


def history_source(ns, iters=1000, warm=1000, reps=2, variants=("corr", "rand", "const"), seed=0):
    """C source for one or more N (filler counts), each with the three B variants.

    Branch B repeats random branch A after N taken fillers, so it is predictable only while the global
    history reaches back N+1 branches; `rand` gives the unpredictable cost. `seed` picks the LFSR start."""
    ns = [ns] if isinstance(ns, int) else list(ns)
    asm = [".text"]
    for n in ns:
        for v in variants:
            asm += [f".globl hist_{v}_{n}"] + _history_fn(f"hist_{v}_{n}", n, v)
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


# --------------------------------------------- P3.5 indirect-predictor capacity


def indirect_source(smax, sizes, targets=1, rounds=8, warm=4, reps=2):
    """C source for the indirect-capacity sweep (all S in one binary).

    The last S of `smax` jalr sites each call a leaf rotated over `targets` addresses, so cost per call
    steps up once S*T exceeds capacity. Every branch is taken, so the direction history stays constant."""
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
