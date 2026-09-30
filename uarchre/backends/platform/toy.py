"""Toy cycle-level out-of-order core with configurable structure sizes: known answers in seconds.

R10K-style: in-order dispatch of `width` ops/cycle, each taking a ROB entry plus, by type, an int/FP
register, LQ or SQ entry; out-of-order issue; in-order retirement frees all of it (R10K frees the old
mapping instead: same headroom). Chain ops wait for their chain's previous op; branch and flags take a
ROB entry only. PRF headroom = prf_size - arch_regs.

  run_toy(window_probe("int", 40), ToyConfig(rob=64))  # -> cycles/iter
"""

from dataclasses import dataclass, field
from collections import deque
import random
from ...kernels import Kernel, Op


@dataclass
class ToyConfig:
    """Structure sizes and latencies. The non-classic knobs, all off by default, exist so every anomaly
    rule can be shown to fire on a known mechanism: the toy is a fixture generator, not a real core."""

    rob: int = 128
    int_prf: int = 96  # total physical int regs
    fp_prf: int = 96
    lq: int = 32
    sq: int = 32
    width: int = 4
    miss_latency: int = 200
    load_latency: int = 4
    alu_latency: int = 1
    fp_latency: int = 3
    arch_int: int = 32
    arch_fp: int = 32
    alu_delay_latency: int = 24
    miss_latency_jitter: float = 0.0  # each miss x (1 + jitter * N(0, 1)), seeded; 0 = deterministic
    seed: int = 0
    # non-classic mechanisms
    retire_group: int = 1  # A4: `rob` counts retire rows of up to retire_group ops
    failable_per_group: int | None = None  # a row also closes after this many loads/stores/branches
    lsq_alloc: str = "dispatch"  # "dispatch" | "issue" (A5: a memory op waits in the LS scheduler)
    ls_sched: int = 0  # LS scheduler entries (issue mode), 0 = unlimited
    lq_release_early: bool = False  # A5: a done load/store frees its slot once no older op can squash it
    sq_release_early: bool = False
    sq_levels: tuple[int, int] | None = None  # A1: (fast, slow) two-level SQ, replaces sq
    sq_slow_penalty: int = 100  # extra cycles for a load with more than `fast` older stores in flight
    filler_throughput: dict[str, int] = field(default_factory=dict)  # A7, A10: allocations/cycle per Op
    load_addr_predict: bool = False  # P4.3: stride address prediction (stride and constant chains)
    load_value_predict: bool = False  # last-value prediction (constant chains)
    predict_train: int = 4  # loads of a chain before prediction starts; random chains never predict

    def rows_mode(self):
        return self.retire_group > 1 or self.failable_per_group is not None

    def sq_total(self):
        return sum(self.sq_levels) if self.sq_levels else self.sq

    def headroom(self):
        """Ground truth keyed like attribute() estimates; PRF totals kept separately."""
        h = {
            "rob": self.rob,
            "int_prf_headroom": self.int_prf - self.arch_int,
            "fp_prf_headroom": self.fp_prf - self.arch_fp,
            "lq": self.lq,
            "sq": self.sq_total(),
            "int_prf": self.int_prf,
            "fp_prf": self.fp_prf,
        }
        if self.rows_mode():
            h.update(
                retire_rows=self.rob,
                retire_group=self.retire_group,
                failable_per_group=self.failable_per_group,
            )
        if self.sq_levels:
            h.update(sq_fast=self.sq_levels[0], sq_slow=self.sq_levels[1])
        if self.lsq_alloc != "dispatch":
            h.update(lsq_alloc=self.lsq_alloc, ls_sched=self.ls_sched)
        return h


@dataclass
class _Entry:
    op: Op
    chain: int
    dispatch: int
    ready_at: int = -1  # cycle when sources are ready
    done_at: int = -1  # cycle when result is available
    issued: bool = False


FAILABLE = (Op.LOAD, Op.STORE, Op.MISS_LOAD, Op.BRANCH, None)  # None = loop branch
LOADS = (Op.LOAD, Op.MISS_LOAD)


def run_toy(kernel: Kernel, cfg: ToyConfig, iters: int = 60, warm: int = 20) -> float:
    """Return cycles per iteration in steady state."""
    body = list(kernel.body) + [None]  # None = loop branch
    stream_len = len(body)
    total = (iters + warm) * stream_len

    rob = deque()  # _Entry, in program order
    unissued = []  # entries not yet issued, program order (the issue scan)
    pending = []  # chain ops whose predecessor has not issued yet
    used = {"int_prf": 0, "fp_prf": 0, "lq": 0, "sq": 0, "ls_sched": 0}
    h = cfg.headroom()  # pools below are the rename headroom, not PRF totals
    cap = {
        "rob": h["rob"],
        "int_prf": h["int_prf_headroom"],
        "fp_prf": h["fp_prf_headroom"],
        "lq": h["lq"],
        "sq": h["sq"],
        "ls_sched": cfg.ls_sched or 10**9,
    }
    last_miss = {c: None for c in range(kernel.n_chains)}  # last MISS entry per chain
    pos = 0  # next op index in the stream
    cycle = 0
    iter_marks = []  # cycle at which each iteration's branch retires

    CHAIN = (Op.MISS_LOAD, Op.DELAY_FP, Op.DELAY_INT)
    at_issue = cfg.lsq_alloc == "issue"
    if cfg.lsq_alloc not in ("dispatch", "issue"):
        raise ValueError(f"lsq_alloc must be 'dispatch' or 'issue', not {cfg.lsq_alloc!r}")
    rows = cfg.rows_mode()
    G = cfg.retire_group
    F = cfg.failable_per_group
    early = cfg.lq_release_early or cfg.sq_release_early
    fast_sq = cfg.sq_levels[0] if cfg.sq_levels else None
    tput = {Op(k): v for k, v in cfg.filler_throughput.items()}
    pattern = kernel.params.get("chain", "random")
    predictable = (pattern == "constant" and (cfg.load_value_predict or cfg.load_addr_predict)) or (
        pattern == "stride" and cfg.load_addr_predict
    )
    chain_seen = {c: 0 for c in range(kernel.n_chains)}

    def needs(op):
        """Pools an op takes at dispatch."""
        if op in (Op.INT, Op.LOAD, Op.MISS_LOAD, Op.DELAY_INT):
            r = ["int_prf"]
        elif op in (Op.FP, Op.DELAY_FP):
            r = ["fp_prf"]
        else:
            r = []
        if op == Op.LOAD or op == Op.MISS_LOAD:
            r.append("ls_sched" if at_issue else "lq")
        if op == Op.STORE:
            r.append("ls_sched" if at_issue else "sq")
        return r

    rng = random.Random(cfg.seed)

    def latency(e):
        op = e.op
        if op == Op.MISS_LOAD and cfg.miss_latency_jitter:
            lat = max(1, round(cfg.miss_latency * (1 + cfg.miss_latency_jitter * rng.gauss(0, 1))))
        else:
            lat = {
                Op.MISS_LOAD: cfg.miss_latency,
                Op.LOAD: cfg.load_latency,
                Op.INT: cfg.alu_latency,
                Op.FP: cfg.fp_latency,
                Op.DELAY_FP: cfg.alu_delay_latency,
                Op.DELAY_INT: cfg.alu_delay_latency,
            }.get(op, 1)
        if fast_sq is not None and op in LOADS and e.stores_before - n_st[1] > fast_sq:
            lat += cfg.sq_slow_penalty  # its older stores spill into the slow level
        return lat

    # retire-row state (rows mode only)
    open_row = None  # id of the youngest row while it can still take ops
    open_count = open_fail = 0
    rows_used = 0
    next_row = 0
    fail_q = deque()  # failable entries in program order (early release only)
    n_st = [0, 0]  # stores dispatched, stores retired (two-level SQ only)

    while len(iter_marks) < iters + warm:
        # ---- retire (in order) ----
        if not rows:
            retired = 0
            while rob and retired < cfg.width and rob[0].done_at >= 0 and rob[0].done_at <= cycle:
                e = rob.popleft()
                for r in e.held:
                    used[r] -= 1
                e.retired = True
                if e.op is None:
                    iter_marks.append(cycle)
                elif e.op == Op.STORE:
                    n_st[1] += 1
                retired += 1
        else:
            retired = 0  # rows this cycle
            while rob and retired < cfg.width:
                r0 = rob[0].row
                j, n_rob, complete = 0, len(rob), True
                while j < n_rob and rob[j].row == r0:
                    if not (0 <= rob[j].done_at <= cycle):
                        complete = False
                        break
                    j += 1
                if not complete:
                    break
                for _ in range(j):
                    e = rob.popleft()
                    for r in e.held:
                        used[r] -= 1
                    e.retired = True
                    if e.op is None:
                        iter_marks.append(cycle)
                    elif e.op == Op.STORE:
                        n_st[1] += 1
                rows_used -= 1
                if open_row == r0:
                    open_row = None
                retired += 1

        # ---- issue (out of order) ----
        issued = 0
        keep = []
        for i, e in enumerate(unissued):
            if issued >= cfg.width:
                keep.extend(unissued[i:])
                break
            if e.ready_at > cycle:
                keep.append(e)
                continue
            if at_issue and e.op in (Op.LOAD, Op.MISS_LOAD, Op.STORE):
                q = "sq" if e.op == Op.STORE else "lq"
                if used[q] >= cap[q]:
                    keep.append(e)  # no LQ/SQ slot: wait in the LS scheduler
                    continue
                used[q] += 1
                used["ls_sched"] -= 1
                e.held.remove("ls_sched")
                e.held.append(q)
            e.issued = True
            e.done_at = cycle + latency(e)
            issued += 1
        unissued = keep

        # ---- early LQ / SQ release (non-speculative completed memory ops) ----
        if early:
            while fail_q and fail_q[0].retired:
                fail_q.popleft()
            inc_fail = inc_load = unissued_store = False
            for e in fail_q:
                done = 0 <= e.done_at <= cycle
                if (
                    done
                    and e.op in LOADS
                    and "lq" in e.held
                    and cfg.lq_release_early
                    and not inc_load
                    and not unissued_store
                ):
                    e.held.remove("lq")
                    used["lq"] -= 1
                elif done and e.op == Op.STORE and "sq" in e.held and cfg.sq_release_early and not inc_fail:
                    e.held.remove("sq")
                    used["sq"] -= 1
                if not done:
                    inc_fail = True
                    if e.op in LOADS:
                        inc_load = True
                if e.op == Op.STORE and not e.issued:
                    unissued_store = True
                if (inc_load or unissued_store or not cfg.lq_release_early) and (
                    inc_fail or not cfg.sq_release_early
                ):
                    break  # nothing younger can be released this cycle

        # ---- dispatch (in order) ----
        n_kind = {}
        for _ in range(cfg.width):
            if pos >= total:
                break
            if not rows and len(rob) >= cap["rob"]:
                break
            ins = body[pos % stream_len]
            op = ins.op if ins is not None else None
            chain = ins.arg if ins is not None else 0
            if tput and op in tput:
                if n_kind.get(op, 0) >= tput[op]:
                    break
                n_kind[op] = n_kind.get(op, 0) + 1
            if rows:
                fail = op in FAILABLE
                join = open_row is not None and open_count < G and (not fail or F is None or open_fail < F)
                if not join and rows_used >= cap["rob"]:
                    break
            need = needs(op)
            if any(used[r] >= cap[r] for r in need):
                break
            for r in need:
                used[r] += 1
            e = _Entry(op=op, chain=chain, dispatch=cycle)
            e.held = need
            e.retired = False
            e.stores_before = n_st[0]
            if op == Op.STORE:
                n_st[0] += 1
            if rows:
                if join:
                    open_count += 1
                    open_fail += fail
                else:
                    open_row, next_row = next_row, next_row + 1
                    open_count, open_fail = 1, int(fail)
                    rows_used += 1
                e.row = open_row
                if open_count >= G or (fail and F is not None and open_fail >= F):
                    open_row = None
            if early and op in FAILABLE:
                fail_q.append(e)
            # dependencies: a chain op waits for the previous one on its chain
            if op in CHAIN:
                prev = last_miss[chain]
                if op == Op.MISS_LOAD and predictable:
                    chain_seen[chain] += 1
                    if chain_seen[chain] > cfg.predict_train:
                        prev = None  # address / value predicted: no wait
                e.ready_at = max(cycle + 1, prev.done_at if prev and prev.done_at >= 0 else 10**9)
                if prev and prev.done_at < 0:
                    e.ready_at = 10**9  # resolved when prev issues (patched below)
                last_miss[chain] = e
                e._prev = prev
                if e.ready_at >= 10**9:
                    pending.append(e)
            else:
                e.ready_at = cycle + 1
            rob.append(e)
            unissued.append(e)
            pos += 1

        # patch chain-op readiness once predecessors have a done_at
        if pending:
            still = []
            for e in pending:
                p = e._prev
                if p is None:
                    e.ready_at = cycle + 1
                elif p.done_at >= 0:
                    e.ready_at = max(cycle + 1, p.done_at)
                else:
                    still.append(e)
            pending = still
        cycle += 1
        if cycle > 10**7:
            raise RuntimeError("toy model did not converge")

    steady = iter_marks[warm:]
    return (steady[-1] - steady[0]) / (len(steady) - 1)
