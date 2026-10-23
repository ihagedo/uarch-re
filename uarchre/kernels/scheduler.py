"""P1.5 scheduler-queue, P1.6 ALU-mix and P1.7 retire-bandwidth probes (RISC-V and AArch64, raw asm).

A kernel's loop body is `params["asm"]` for `params["isa"]`, lowered by that ISA's `AsmEmitter` through
the shared C harness (%0/%1 chase chains, %2 loop counter, %3 128 B scratch; RISC-V t0 = 1, t1 = 2,
ft8-ft11 = 2, 2, 1, 2; AArch64 as isa/aarch64.py plus address helpers x21/x22). P1.5: per chain a
miss, then N fillers that depend on it; the B miss overlaps A's only while A's fillers fit in their
queue, so the knee reads that queue. P1.7 (Handley): D x fsqrt blocks the ROB head, K nops complete,
then E adds need more int registers than the headroom, so cycles/iter grow with slope 1 / retire width.
"""

from . import FILLERS, Kernel, Op
from ..backends.isa.aarch64 import AArch64, INT_ROT, FP_ROT, OFFSETS
from ..backends.isa.riscv64 import RISCV64
from ..backends.isa_base import emit_c

INT_DEST = ["t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4"]
FP_DEST = [f"ft{i}" for i in range(8)]
HELPER = ["a5", "a6"]  # per-group address helper for memory fillers

# filler class -> (gem5 op class, template); {r} rotating dest, {g} miss
# operand (%0 / %1), {h} helper register, {o} scratch offset
CLASSES = {
    "add": ("IntAlu", "add {r}, {g}, t0"),
    "mul": ("IntMult", "mul {r}, {g}, t0"),
    "fp": ("FloatCvt", "fmv.d.x {f}, {g}"),
    "load": ("MemRead", "ld {r}, {o}({h})"),
    "store": ("MemWrite", "sd t0, {o}({h})"),
}
MEMORY = ("load", "store")

A64_HELPER = ["x21", "x22"]
A64_CLASSES = {  # gem5 ARM op classes: scalar FP is Float*, not Simd*
    "add": ("IntAlu", "add {r}, {g}, x6"),
    "mul": ("IntMult", "mul {r}, {g}, x7"),
    "fp": ("FloatMisc", "fmov {f}, {g}"),
    "fadd": ("FloatAdd", "fadd {f}, {fh}, d30"),
    "load": ("MemRead", "ldr {r}, [{h}, #{o}]"),
    "store": ("MemWrite", "str x6, [{h}, #{o}]"),
}

# per-ISA spellings; "addr" makes the scratch address depend on the miss (a chase pointer has bit 63 clear)
SPELL = {
    "riscv64": {
        "int": INT_DEST,
        "fp": FP_DEST,
        "helper": HELPER,
        "offsets": [8 * i for i in range(16)],
        "classes": CLASSES,
        "miss": "ld %{g}, 0(%{g})",
        "addr": ["andi {h}, %{g}, 0", "add {h}, {h}, %3"],
        "indep": {"add": "add {r}, t0, t1", "fp": "fadd.d {f}, ft10, ft11", "mul": "mul {r}, t0, t1"},
        "delay": "fsqrt.d ft8, ft8",
    },
    "aarch64": {
        "int": INT_ROT,
        "fp": FP_ROT,
        "helper": A64_HELPER,
        "offsets": OFFSETS,
        "classes": A64_CLASSES,
        "miss": "ldr %{g}, [%{g}]",
        "addr": ["lsr {h}, %{g}, #63", "add {h}, {h}, %3"],
        # fadd fillers' source: x6's bits (a normal double) moved over once per group, after the miss
        "fhelper": ["d24", "d25"],
        "fmov": ["add {x}, x6, %{g}, lsr #63", "fmov {fh}, {x}"],
        "xhelper": ["x19", "x20"],
        "indep": {
            "add": "add {r}, x6, x7",
            "fp": "fadd {f}, d30, d31",
            "mul": "mul {r}, x6, x7",
            "nop": "nop",
            "fabs": "fabs {f}, d30",
            "fcmp": "fcmp d30, d31",  # writes NZCV, not an FP register
        },
        "delay": "fsqrt d24, d24",
    },
}


class AsmEmitter(RISCV64):
    """RISCV64 harness whose loop body is the kernel's raw asm lines."""

    name = "riscv64-asm"

    def lower(self, k):
        return list(self.loop_head()) + list(k.params["asm"]) + list(self.loop_tail())


class A64AsmEmitter(AArch64):
    """AArch64 harness (M1 register claims, `counter` from COUNTERS) around the raw asm lines."""

    name = "aarch64-asm"
    clobbers = AArch64.clobbers + A64_HELPER

    def loop_head(self):
        head = super().loop_head()
        return head[:-2] + [f"mov {h}, x6" for h in A64_HELPER] + head[-2:]

    def lower(self, k):
        return list(self.loop_head()) + list(k.params["asm"]) + list(self.loop_tail())


EMITTER = AsmEmitter()
EMITTER_STD = RISCV64()
ASM_EMITTERS = {"riscv64": EMITTER, "aarch64": A64AsmEmitter()}
STD_EMITTERS = {"riscv64": EMITTER_STD, "aarch64": AArch64()}


def kernel_isa(kernels):
    isas = {k.params.get("isa", "riscv64") for k in kernels}
    if len(isas) != 1:
        raise ValueError(f"kernels mix ISAs: {sorted(isas)}")
    return isas.pop()


def asm_kernel(name, lines, depth=1, isa="riscv64", **params):
    return Kernel(
        name=name, body=[], n_chains=2, params={"asm": list(lines), "depth": depth, "isa": isa, **params}
    )


def c_source(kernel, em=None, **overrides):
    return emit_c(kernel, em or ASM_EMITTERS[kernel_isa([kernel])], **overrides)


def dep_op(cls, i, group, isa="riscv64"):
    """The i-th filler of class `cls` that depends on miss group 0 (A) or 1 (B)."""
    sp = SPELL[isa]
    _, tmpl = sp["classes"][cls]
    return tmpl.format(
        r=sp["int"][i % 8],
        f=sp["fp"][i % 8],
        g=f"%{group}",
        h=sp["helper"][group],
        o=sp["offsets"][i % len(sp["offsets"])],
        fh=sp["fhelper"][group] if "fhelper" in sp else None,
    )


def sched_probe(filler, n, pre=None, m=0, isa="riscv64"):
    """N fillers of class `filler` per miss group after m ops of class `pre`, all dependent on the miss."""
    sp = SPELL[isa]
    lines = []
    for g in (0, 1):
        lines.append(sp["miss"].format(g=g))
        if filler in MEMORY or pre in MEMORY:  # same overhead at every m of a sharing test
            lines += [ln.format(h=sp["helper"][g], g=g) for ln in sp["addr"]]
        if "fadd" in (filler, pre):
            lines += [ln.format(x=sp["xhelper"][g], g=g, fh=sp["fhelper"][g]) for ln in sp["fmov"]]
        lines += [dep_op(pre, i, g, isa) for i in range(m)] if pre else []
        lines += [dep_op(filler, m + i, g, isa) for i in range(n)]
    tag = f"_{pre}{m}" if pre and m else ""
    return asm_kernel(f"sched_{filler}{tag}_{n}", lines, isa=isa, filler=filler, n=n, pre=pre, m=m)


def dispatch_probe(sat, m, x="add", y="fp", isa="riscv64"):
    """Handley's dispatch-buffer probe without a delay: `sat` independent class-x ops (enough
    to fill x's queue), then m independent class-y ops, which are free if they use another queue."""
    sp = SPELL[isa]
    indep = sp["indep"]
    lines = [indep[x].format(r=sp["int"][i % 8], f=sp["fp"][i % 8]) for i in range(sat)]
    lines += [indep[y].format(r=sp["int"][i % 8], f=sp["fp"][i % 8]) for i in range(m)]
    return asm_kernel(f"disp_{x}{sat}_{y}_{m}", lines, depth=0, isa=isa, sat=sat, m=m, x=x, y=y)


def from_window(kernel, isa="riscv64"):
    """A window-probe Kernel (kernels/window.py) as a raw-asm kernel."""
    em = STD_EMITTERS[isa]
    lines = em.lower(kernel)
    body = lines[len(em.loop_head()) : len(lines) - len(em.loop_tail())]
    return asm_kernel(kernel.name, body, **{"depth": 1, **kernel.params, "isa": isa})


def retire_drain(k, links=12, exhaust=64, isa="riscv64"):
    """P1.7: FP delay chain, K NOPs, then `exhaust` int destinations."""
    sp = SPELL[isa]
    lines = [sp["delay"]] * links + ["nop"] * k
    lines += [sp["indep"]["add"].format(r=sp["int"][i % 8]) for i in range(exhaust)]
    return asm_kernel(
        f"retire_drain_l{links}_e{exhaust}_{k}", lines, depth=0, isa=isa, k=k, links=links, exhaust=exhaust
    )


def nop_stream(k, isa="riscv64"):
    """Pure throughput: K NOPs per iteration (narrowest pipeline stage)."""
    return asm_kernel(f"nop_stream_{k}", ["nop"] * k, depth=0, isa=isa, k=k)


def alu_mix(k, filler, adds=80, fill=16, isa="riscv64"):
    """P1.6: K blocks of `adds` independent adds, then `fill` window fillers (FILLERS names). The
    adds keep the ALUs the bottleneck, so cycles per block grow with every filler that needs one."""
    em = STD_EMITTERS[isa]
    lines = []
    for b in range(k):
        lines += [em.filler(Op.INT, i) for i in range(adds)]
        lines += [em.filler(FILLERS[filler], b * fill + i) for i in range(fill)]
    name = f"alu_mix_{filler}_a{adds}_f{fill}_{k}"
    return asm_kernel(name, lines, depth=0, isa=isa, k=k, mix=filler, adds=adds, fill=fill)


# --------------------------------------------------------------------------- multi-point harness
# One binary times every point of a sweep, so a curve costs one gem5 launch. Per point, `reps`
# back-to-back runs of `iters` iterations (rep 0 warms the code); the chase continues throughout.

MULTI_HEAD = r"""
/* generated by uarch-re: {name} ({npoints} points) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

{cycle_counter}

#define LINE {line}

static void *build_chain(size_t bytes, uint32_t seed) {{
    size_t n = bytes / LINE;
    char *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, bytes)) {{ perror("alloc"); exit(1); }}
    memset(buf, 0, bytes);
    size_t *perm = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    uint64_t s = seed | 1;
    for (size_t i = n - 1; i > 0; i--) {{
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t j = s % i;
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }}
    for (size_t i = 0; i < n; i++) {{
        size_t next = (i + 1 == n) ? 0 : i + 1;
        *(void **)(buf + perm[i] * LINE) = buf + perm[next] * LINE;
    }}
    free(perm);
    return buf;
}}
"""

MULTI_POINT = r"""
static __attribute__((noinline)) double point{idx}(void **pa, void **pb, uint64_t *scratch, uint64_t n) {{
    void *a = *pa, *b = *pb;
    uint64_t iters = n;
    uint64_t t0 = cycles();
    __asm__ __volatile__ (
{asm_body}
        : "+r"(a), "+r"(b), "+r"(n)
        : "r"(scratch)
        : "memory", "cc"{clobbers}
    );
    uint64_t t1 = cycles();
    *pa = a; *pb = b;
    return (double)(t1 - t0) / (double)iters;
}}
"""

MULTI_MAIN = r"""
typedef double (*point_fn)(void **, void **, uint64_t *, uint64_t);
static point_fn POINTS[] = {{ {names} }};

int main(int argc, char **argv) {{
    size_t chain_bytes = argc > 1 ? strtoull(argv[1], 0, 0) : (1ULL << 20);
    uint64_t iters      = argc > 2 ? strtoull(argv[2], 0, 0) : 60ULL;
    int reps            = argc > 3 ? atoi(argv[3]) : 3;
    uint32_t seed       = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 0) : 0;
    uint32_t salt = seed * 0x9E3779B9u;
    void *a = build_chain(chain_bytes, 0x1234 ^ salt);
    void *b = build_chain(chain_bytes, 0x5678 ^ salt);
    uint64_t scratch[16] = {{0}};
    double cpi[64];
    if (reps > 64) reps = 64;
    for (int p = 0; p < {npoints}; p++) {{
        for (int r = 0; r < reps; r++) cpi[r] = POINTS[p](&a, &b, scratch, iters);
        for (int r = 0; r < reps; r++) printf("point %d rep %d cycles_per_iter %.3f\n", p, r, cpi[r]);
    }}
    __asm__ __volatile__("" :: "r"(a), "r"(b));
    printf("cycles_per_iter done\n");
    return 0;
}}
"""


def _asm_lines(lines):
    return "\n".join(f'            "{line}\\n\\t"' for line in lines)


def multi_source(name, kernels, em=None):
    """C source timing every kernel of `kernels` (raw-asm kernels of one ISA) in turn.
    `em` overrides the emitter, e.g. A64AsmEmitter("mach") on macOS."""
    em = em or ASM_EMITTERS[kernel_isa(kernels)]
    clob = "".join(f', "{c}"' for c in em.clobbers)
    parts = [
        MULTI_HEAD.format(name=name, npoints=len(kernels), cycle_counter=em.cycle_counter_c, line=em.line)
    ]
    for i, k in enumerate(kernels):
        parts.append(MULTI_POINT.format(idx=i, asm_body=_asm_lines(em.lower(k)), clobbers=clob))
    parts.append(
        MULTI_MAIN.format(names=", ".join(f"point{i}" for i in range(len(kernels))), npoints=len(kernels))
    )
    return "".join(parts)


def chain_nodes_needed(kernels, iters, reps):
    """Chase nodes per chain a multi-point run visits (`depth`: chase loads per chain per iteration,
    0 for kernels without misses)."""
    return sum(k.params.get("depth", 1) for k in kernels) * iters * reps


def parse_multi(out, npoints):
    """{point index: [cycles_per_iter per rep]} from a multi-point run."""
    import re

    res = {i: [] for i in range(npoints)}
    for p, _r, v in re.findall(r"^point (\d+) rep (\d+) cycles_per_iter ([\d.]+)", out, re.M):
        res[int(p)].append(float(v))
    return res
