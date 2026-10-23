"""RISC-V RV64GC emitter (GCC inline asm).

Fillers read constant sources (t0, t1, ft10, ft11) and rotate destinations. Delays are fsqrt.d (gem5
FloatSqrt, 24 c) or divu by t0 = 1 (gem5 IntDiv, 20 c), both unpipelined with 2 units. No flags
register, so no flags filler; the movimm filler is li, i.e. addi rd, zero, imm, an ordinary IntAlu
add. rdcycle counts core cycles (clock probe ~1.0); the probe is short because every add costs
simulation time.
"""

from ..isa_base import Emitter
from ...kernels import Op

INT_ROT = ["t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4"]
FP_ROT = [f"ft{i}" for i in range(8)]


class RISCV64(Emitter):
    name = "riscv64"
    default_probe_adds = 1 << 12
    clobbers = [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "ft0",
        "ft1",
        "ft2",
        "ft3",
        "ft4",
        "ft5",
        "ft6",
        "ft7",
        "ft8",
        "ft9",
        "ft10",
        "ft11",
    ]
    cycle_counter_c = r"""
static inline uint64_t cycles(void) {
    uint64_t c; __asm__ __volatile__("rdcycle %0" : "=r"(c)); return c;
}
"""

    def loop_head(self):
        return [
            "li t0, 1",
            "li t1, 2",
            "fcvt.d.w ft10, t0",
            "fcvt.d.w ft11, t1",
            "fcvt.d.w ft8, t1",
            "fcvt.d.w ft9, t1",
            "li a5, 7",
            "li a6, 7",
            ".p2align 6",
            "1:",
        ]

    def loop_tail(self):
        return ["addi %2, %2, -1", "bnez %2, 1b"]

    def probe_add(self):
        return "add %0, %0, %1"  # x += n: never a constant value

    def probe_tail(self):
        return ["addi %1, %1, -1", "bnez %1, 1b"]

    def miss(self, chain):
        return f"ld %{chain}, 0(%{chain})"

    def filler(self, op, idx):
        if op == Op.NOP:
            return "nop"
        if op == Op.INT:
            return f"add {INT_ROT[idx % 8]}, t0, t1"
        if op == Op.FP:
            return f"fadd.d {FP_ROT[idx % 8]}, ft10, ft11"
        if op == Op.LOAD:
            return f"ld {INT_ROT[idx % 8]}, 0(%3)"
        if op == Op.STORE:
            return "sd t0, 0(%3)"
        if op == Op.BRANCH:
            return "bne zero, zero, 3f\\n\\t3:"
        if op == Op.MOV_IMM:
            return f"addi {INT_ROT[idx % 8]}, zero, {1 + 37 * idx % 255}"
        raise ValueError(f"riscv64 has no {op} filler")

    def delay(self, op, chain):
        if op == Op.DELAY_FP:
            return f"fsqrt.d ft{8 + chain}, ft{8 + chain}"
        return f"divu a{5 + chain}, a{5 + chain}, t0"
