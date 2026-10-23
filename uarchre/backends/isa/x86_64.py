"""x86-64 emitter (AT&T syntax, GCC inline asm).

Fillers are independent of each other and of the misses, so throughput never limits before the
window does; the branch filler is cmp + jne (never taken, macro-fused). The asm block takes 4 of the
16 GPRs: int fillers rotate over r10-r15 and rcx/rdx carry the imul delay chains (3 c, pipelined, so
~3x the links of sqrtsd). The movimm filler is mov r64, imm32. The fp filler needs AVX; rdtscp counts
TSC ticks, not core cycles.
"""

from ..isa_base import Emitter
from ...kernels import Op

INT_ROT = ["%%r10", "%%r11", "%%r12", "%%r13", "%%r14", "%%r15"]
FP_ROT = [f"%%xmm{i}" for i in range(8)]
LOAD_ROT = INT_ROT


class X86_64(Emitter):
    name = "x86_64"
    clobbers = [
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rcx",
        "rdx",
        "xmm0",
        "xmm1",
        "xmm2",
        "xmm3",
        "xmm4",
        "xmm5",
        "xmm6",
        "xmm7",
        "xmm8",
        "xmm9",
        "xmm14",
        "xmm15",
    ]
    cycle_counter_c = r"""
#include <x86intrin.h>
static inline uint64_t cycles(void) {
    unsigned aux; __asm__ __volatile__("lfence" ::: "memory");
    return __rdtscp(&aux);
}
"""

    def loop_head(self):
        # constant sources for the fillers
        return [
            "movq $1, %%r8",
            "movq $2, %%r9",
            "vxorpd %%xmm14, %%xmm14, %%xmm14",
            "vxorpd %%xmm15, %%xmm15, %%xmm15",
            "vcvtsi2sd %%r9, %%xmm8, %%xmm8",
            "vcvtsi2sd %%r9, %%xmm9, %%xmm9",
            "movq $7, %%rcx",
            "movq $7, %%rdx",
            ".p2align 6",
            "1:",
        ]

    def loop_tail(self):
        return ["decq %2", "jnz 1b"]

    def probe_add(self):
        # x += n: reg-reg (Golden Cove folds add-immediate chains at rename), never a constant value
        return "addq %1, %0"

    def probe_tail(self):
        return ["decq %1", "jnz 1b"]

    def miss(self, chain):
        return f"movq (%{chain}), %{chain}"

    def filler(self, op, idx):
        if op == Op.NOP:
            return "nop"
        if op == Op.INT:
            return f"leaq (%%r8,%%r9), {INT_ROT[idx % len(INT_ROT)]}"
        if op == Op.FP:
            return f"vaddsd %%xmm14, %%xmm15, {FP_ROT[idx % 8]}"
        if op == Op.LOAD:
            return f"movq (%3), {LOAD_ROT[idx % len(LOAD_ROT)]}"
        if op == Op.STORE:
            return "movq %%r8, (%3)"
        if op == Op.BRANCH:
            return "cmpq %%r8, %%r8\\n\\tjne 3f\\n\\t3:"
        if op == Op.FLAGS:
            return "cmpq %%r8, %%r9"
        if op == Op.MOV_IMM:
            return f"movq ${1 + 37 * idx % 255}, {INT_ROT[idx % len(INT_ROT)]}"
        raise ValueError(op)

    def delay(self, op, chain):
        if op == Op.DELAY_FP:
            return f"sqrtsd %%xmm{8 + chain}, %%xmm{8 + chain}"
        return f"imulq %%r8, {['%%rcx', '%%rdx'][chain]}"
