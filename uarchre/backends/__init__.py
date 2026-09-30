"""ISA emitters (`isa/`) and platform drivers (`platform/`)."""

from dataclasses import dataclass, field
import statistics


@dataclass
class RunResult:
    """One process launch of a kernel.

    `cycles` is cycles_per_iter per inner rep in counter units (TSC ticks on x86, core cycles on gem5
    and the toy); `clock_ratio_*` is counter ticks per dependent add before/after the reps, or None.
    """

    cycles: list[float] = field(default_factory=list)
    clock_ratio_pre: float | None = None
    clock_ratio_post: float | None = None

    @property
    def best(self) -> float:
        return min(self.cycles)

    @property
    def median(self) -> float:
        return float(statistics.median(self.cycles))
