"""Abstract, ISA-independent kernel definitions.

A kernel is a list of abstract ops that form one loop iteration. Backends
lower the ops to assembly (native / gem5) or execute them directly (toy model).
"""

from dataclasses import dataclass, field
from enum import Enum


class Op(Enum):
    MISS_LOAD = "miss_load"  # pointer-chase load on chain `arg` (cache miss)
    NOP = "nop"  # takes a window slot, no destination
    INT = "int"  # integer ALU op with an integer destination
    FP = "fp"  # FP op with an FP destination
    LOAD = "load"  # L1-hit load with an integer destination
    STORE = "store"  # L1-hit store, no destination
    DELAY_FP = "delay_fp"  # one link of a dependent long-latency FP chain (fsqrt) on chain `arg`
    DELAY_INT = "delay_int"  # one link of a dependent long-latency int chain (div/imul) on chain `arg`
    BRANCH = "branch"  # never-taken conditional branch, no destination
    FLAGS = "flags"  # writes the flags register only (cmp); not on RISC-V
    MOV_IMM = "mov_imm"  # move of a small nonzero immediate into an integer register


FILLERS = {
    "nop": Op.NOP,
    "int": Op.INT,
    "fp": Op.FP,
    "load": Op.LOAD,
    "store": Op.STORE,
    "branch": Op.BRANCH,
    "flags": Op.FLAGS,
    "movimm": Op.MOV_IMM,
}
# "mix" alternates INT and FP fillers: it exhausts neither register pool first,
# so its knee lands on the shared limit (the ROB on gem5; P1.4)
MIXES = {"mix": (Op.INT, Op.FP)}
DELAYS = {"fp": Op.DELAY_FP, "int": Op.DELAY_INT}  # window_probe(delay=...) besides "miss"


@dataclass
class Instr:
    op: Op
    arg: int = 0  # chain id for MISS_LOAD, rotation index for fillers


@dataclass
class Kernel:
    """One loop iteration. The loop branch is implicit (added by the backend)."""

    name: str
    body: list[Instr] = field(default_factory=list)
    n_chains: int = 2
    params: dict = field(default_factory=dict)

    def count(self, op: Op) -> int:
        return sum(1 for i in self.body if i.op == op)
