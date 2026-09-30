"""The out-of-order window probe kernel (Henry Wong's method).

One iteration: d x delay(A), N x filler, d x delay(B), N x filler, loop branch. While both groups and
the fillers between them fit in the window an iteration costs ~d delay latencies, past it ~2d; the last
N that fits, plus the offset, is the size of whatever the filler fills. Depth d > 1 keeps slow fillers
from binding first. `delay` is "miss", "fp" (fsqrt chain) or "int" (div/mul chain): use the pool the
filler does not. Miss and ALU knees disagree when a structure is allocated at issue (rule A5).
"""

from . import Kernel, Instr, Op, FILLERS, MIXES, DELAYS


def window_probe(filler: str, n: int, rotation: int = 8, depth: int = 1, delay: str = "miss") -> Kernel:
    ops = MIXES.get(filler) or (FILLERS[filler],)
    dop = Op.MISS_LOAD if delay == "miss" else DELAYS[delay]
    body = [Instr(dop, 0) for _ in range(depth)]
    body += [Instr(ops[i % len(ops)], i % rotation) for i in range(n)]
    body += [Instr(dop, 1) for _ in range(depth)]
    body += [Instr(ops[(n + i) % len(ops)], (n + i) % rotation) for i in range(n)]
    tag = "" if delay == "miss" else f"_{delay}"
    return Kernel(
        name=f"window_{filler}_d{depth}{tag}_{n}",
        body=body,
        n_chains=2,
        # the entries the delay ops hold are not 2 * depth: see analysis/offsets.py
        params={"filler": filler, "n": n, "depth": depth, "delay": delay},
    )
