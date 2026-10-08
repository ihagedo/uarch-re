"""Event-trace export for the toy model, and the spec -> ToyConfig resolver.

`trace_toy()` is `run_toy()`'s loop plus event logging (classic fields only), so cycles/iter match
exactly; keep it in step with toy.py and explainer/factory/engine.js. Events are JSON Lines in
simulation order, per cycle: complete (by seq), retire, issue, dispatch, at most one stall, e.g.
  {"c":12,"ev":"dispatch","seq":61,"op":"miss_load","it":0,"res":["rob","int_prf","lq"],"chain":1}
  {"c":12,"ev":"stall","stage":"dispatch","why":"rob_full","seq":62}
A stall is logged only when (seq, why) changes, so a long stall is one event.
"""

from dataclasses import dataclass, field, fields
from typing import Optional
import random

from ...kernels import Kernel, Op
from .toy import ToyConfig

CHAIN = (Op.MISS_LOAD, Op.DELAY_FP, Op.DELAY_INT)
INF = 10**9
STALL_WHY = {
    "rob": "rob_full",
    "int_prf": "int_prf_empty",
    "fp_prf": "fp_prf_empty",
    "lq": "lq_full",
    "sq": "sq_full",
}


def op_name(op):
    return "loop" if op is None else op.value


def needs(op):
    """Pools besides the ROB an op holds from dispatch to retire (toy.run_toy.needs)."""
    if op in (Op.INT, Op.LOAD, Op.MISS_LOAD, Op.DELAY_INT):
        r = ["int_prf"]
    elif op in (Op.FP, Op.DELAY_FP):
        r = ["fp_prf"]
    else:
        r = []
    if op == Op.LOAD or op == Op.MISS_LOAD:
        r.append("lq")
    if op == Op.STORE:
        r.append("sq")
    return r


@dataclass
class _E:
    seq: int
    op: Op | None
    chain: int
    dispatch: int
    ready_at: int = -1
    done_at: int = -1
    issued: bool = False
    prev: Optional["_E"] = None


@dataclass
class Trace:
    cycles_per_iter: float
    iter_marks: list[int]
    final_cycle: int  # last cycle simulated (the last iteration mark)
    dispatched: int
    events: list[dict] = field(default_factory=list)


# ToyConfig fields of the classic model: what trace_toy and the JS engine implement
CLASSIC_FIELDS = [
    "rob",
    "int_prf",
    "fp_prf",
    "lq",
    "sq",
    "width",
    "miss_latency",
    "load_latency",
    "alu_latency",
    "fp_latency",
    "arch_int",
    "arch_fp",
    "alu_delay_latency",
    "miss_latency_jitter",
    "seed",
]


def check_classic(cfg: ToyConfig):
    """Raise if cfg turns on a toy mechanism the trace (and the JS engine) do not model."""
    default = ToyConfig()
    on = [
        f.name
        for f in fields(ToyConfig)
        if f.name not in CLASSIC_FIELDS and getattr(cfg, f.name) != getattr(default, f.name)
    ]
    if on:
        raise ValueError(f"trace_toy models the classic toy only; non-classic field(s) set: {on}")


def trace_toy(kernel: Kernel, cfg: ToyConfig, iters: int = 60, warm: int = 20, events: bool = True) -> Trace:
    """run_toy() with an event log. Same arguments, same cycles per iteration."""
    check_classic(cfg)
    body = list(kernel.body) + [None]  # None = loop branch
    stream_len = len(body)
    total = (iters + warm) * stream_len

    rob = []  # _E in program order; rob[head:] is live
    head = 0
    used = {"int_prf": 0, "fp_prf": 0, "lq": 0, "sq": 0}
    h = cfg.headroom()
    cap = {
        "rob": h["rob"],
        "int_prf": h["int_prf_headroom"],
        "fp_prf": h["fp_prf_headroom"],
        "lq": h["lq"],
        "sq": h["sq"],
    }
    last_miss = {c: None for c in range(kernel.n_chains)}
    pos = 0
    cycle = 0
    iter_marks = []
    ev = []
    completing = {}  # done_at -> [seq]
    last_stall = None  # (seq, why) of the previous cycle's stall

    rng = random.Random(cfg.seed)

    def latency(op):
        if op == Op.MISS_LOAD and cfg.miss_latency_jitter:
            return max(1, round(cfg.miss_latency * (1 + cfg.miss_latency_jitter * rng.gauss(0, 1))))
        return {
            Op.MISS_LOAD: cfg.miss_latency,
            Op.LOAD: cfg.load_latency,
            Op.INT: cfg.alu_latency,
            Op.FP: cfg.fp_latency,
            Op.DELAY_FP: cfg.alu_delay_latency,
            Op.DELAY_INT: cfg.alu_delay_latency,
        }.get(op, 1)

    while len(iter_marks) < iters + warm:
        if events:
            for s in sorted(completing.pop(cycle, ())):
                ev.append({"c": cycle, "ev": "complete", "seq": s})

        # ---- retire (in order) ----
        retired = 0
        while head < len(rob) and retired < cfg.width and 0 <= rob[head].done_at <= cycle:
            e = rob[head]
            head += 1
            for r in needs(e.op):
                used[r] -= 1
            if e.op is None:
                iter_marks.append(cycle)
            if events:
                ev.append({"c": cycle, "ev": "retire", "seq": e.seq})
            retired += 1
        if head > 4096:
            del rob[:head]
            head = 0

        # ---- issue (out of order) ----
        issued = 0
        for i in range(head, len(rob)):
            if issued >= cfg.width:
                break
            e = rob[i]
            if not e.issued and e.ready_at <= cycle:
                e.issued = True
                e.done_at = cycle + latency(e.op)
                issued += 1
                if events:
                    ev.append({"c": cycle, "ev": "issue", "seq": e.seq})
                    completing.setdefault(e.done_at, []).append(e.seq)

        # ---- dispatch (in order) ----
        stall = None
        for _ in range(cfg.width):
            if pos >= total:
                break
            if len(rob) - head >= cap["rob"]:
                stall = (pos, STALL_WHY["rob"])
                break
            ins = body[pos % stream_len]
            op = ins.op if ins is not None else None
            chain = ins.arg if ins is not None else 0
            full = [r for r in needs(op) if used[r] >= cap[r]]
            if full:
                stall = (pos, STALL_WHY[full[0]])
                break
            for r in needs(op):
                used[r] += 1
            e = _E(seq=pos, op=op, chain=chain, dispatch=cycle)
            if op in CHAIN:
                prev = last_miss[chain]
                e.ready_at = max(cycle + 1, prev.done_at if prev and prev.done_at >= 0 else INF)
                if prev and prev.done_at < 0:
                    e.ready_at = INF
                last_miss[chain] = e
                e.prev = prev
            else:
                e.ready_at = cycle + 1
            rob.append(e)
            if events:
                d = {
                    "c": cycle,
                    "ev": "dispatch",
                    "seq": pos,
                    "op": op_name(op),
                    "it": pos // stream_len,
                    "res": ["rob"] + needs(op),
                }
                if op in CHAIN:
                    d["chain"] = chain
                ev.append(d)
            pos += 1
        if events and stall is not None and stall != last_stall:
            ev.append({"c": cycle, "ev": "stall", "stage": "dispatch", "why": stall[1], "seq": stall[0]})
        last_stall = stall

        # patch chain-op readiness once predecessors have a done_at
        for i in range(head, len(rob)):
            e = rob[i]
            if e.op in CHAIN and e.ready_at >= INF:
                p = e.prev
                if p is None:
                    e.ready_at = cycle + 1
                elif p.done_at >= 0:
                    e.ready_at = max(cycle + 1, p.done_at)
        cycle += 1
        if cycle > 10**7:
            raise RuntimeError("toy model did not converge")

    steady = iter_marks[warm:]
    return Trace(
        cycles_per_iter=(steady[-1] - steady[0]) / (len(steady) - 1),
        iter_marks=iter_marks,
        final_cycle=cycle - 1,
        dispatched=pos,
        events=ev,
    )


# ---------------------------------------------------------------- resolver ---


class UnsupportedSpec(ValueError):
    """The spec needs a mechanism the toy semantics lack."""


TOY_FIELDS = CLASSIC_FIELDS  # the resolver only sets classic fields
_DEFAULT = ToyConfig()
UNBOUNDED = 1 << 20  # a null capacity: large enough never to bind at toy scale


def _val(node, path):
    """Value at a dotted path of a spec dict, or None if absent."""
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def resolve_spec(spec: dict, overrides: dict | None = None) -> dict:
    """Map a core spec (parsed YAML) onto ToyConfig fields: {"cfg", "prov", "assumptions"}.

    Must match explainer/factory/resolve.js. rob = active_list (other retire models raise); LQ/SQ fall
    back to a shared address queue, else unbounded; width = rename_width; miss_latency = dram_latency."""
    overrides = dict(overrides or {})
    bad = set(overrides) - set(TOY_FIELDS)
    if bad:
        raise KeyError(f"unknown ToyConfig field(s): {sorted(bad)}")
    be = spec.get("backend", {})
    cfg, prov, assume = {}, {}, []

    def take(field_, path, node=None, default=None, note=None):
        n = node if node is not None else _val(spec, path)
        v = n.get("value") if isinstance(n, dict) else n
        if v is None:
            return default
        cfg[field_] = int(v)
        prov[field_] = {"from": path, "provenance": n.get("provenance") if isinstance(n, dict) else None}
        if note:
            prov[field_]["note"] = note
        return int(v)

    model = _val(be, "retire.model") or "active_list"
    if model != "active_list":
        raise UnsupportedSpec(
            f"backend.retire.model is {model!r}: the engine has toy semantics, a classic "
            f"in-order ledger only (coalesced retire groups are not modeled)"
        )
    if take("rob", "backend.retire.active_list") is None and take("rob", "backend.rob_entries") is None:
        raise UnsupportedSpec("spec has no backend.retire.active_list (or rob_entries)")

    for pool, arch in (("int_prf", "arch_int"), ("fp_prf", "arch_fp")):
        phys = take(pool, f"backend.{pool}")
        head = _val(spec, f"backend.{pool}_headroom.value")
        if phys is None:
            cfg[pool], cfg[arch] = UNBOUNDED, 0
            prov[pool] = {"from": "null -> unbounded", "provenance": None}
            prov[arch] = {"from": "null -> unbounded", "provenance": None}
            assume.append(f"backend.{pool} is null: modelled as unbounded")
        elif head is None:
            raise UnsupportedSpec(f"backend.{pool}_headroom is missing: the architectural count is unknown")
        else:
            cfg[arch] = phys - int(head)
            prov[arch] = {
                "from": f"backend.{pool} - backend.{pool}_headroom",
                "provenance": _val(spec, f"backend.{pool}_headroom.provenance"),
            }

    for q, key in (("lq", "load_queue"), ("sq", "store_queue")):
        if take(q, f"backend.{key}") is None:
            aq = _val(be, "queues.address_queue")
            if isinstance(aq, dict) and aq.get("value") is not None:
                take(
                    q,
                    "backend.queues.address_queue",
                    note="shared by loads and stores; modelled as a separate pool of this size each",
                )
            else:
                cfg[q] = UNBOUNDED
                prov[q] = {"from": "absent -> unbounded", "provenance": None}
                assume.append(f"backend.{key} absent: modelled as unbounded")
    if "lq" in prov and prov["lq"]["from"] == "backend.queues.address_queue":
        assume.append(
            "the address queue is shared by loads and stores; the toy models a separate LQ and SQ "
            "of that size each, so only single-filler probes are faithful"
        )

    if take("width", "backend.rename_width") is None:
        raise UnsupportedSpec("spec has no backend.rename_width")
    assume.append(
        "one width for dispatch, issue and retire (toy semantics); issue ports and "
        "scheduler queues are not modelled"
    )
    l1 = _val(spec, "memory.l1d.latency_cycles")
    if l1 is not None:
        cfg["load_latency"] = int(l1)
        prov["load_latency"] = {
            "from": "memory.l1d.latency_cycles",
            "provenance": _val(spec, "memory.l1d.provenance"),
        }
    take("miss_latency", "memory.dram_latency")

    for k, v in overrides.items():
        cfg[k] = int(v) if not isinstance(v, float) else v
        prov[k] = {"from": "override", "provenance": None}
    if "miss_latency" not in cfg:
        raise UnsupportedSpec("spec has no memory.dram_latency: pass miss_latency as an override")
    for k in TOY_FIELDS:
        if k not in cfg:
            cfg[k] = getattr(_DEFAULT, k)
            prov[k] = {"from": "toy default", "provenance": None}
    if cfg["miss_latency_jitter"]:
        raise UnsupportedSpec("miss_latency_jitter is not deterministic across engines")
    cfg = {k: cfg[k] for k in TOY_FIELDS}
    return {"cfg": cfg, "prov": {k: prov[k] for k in TOY_FIELDS}, "assumptions": assume}


def toy_config_from_spec(spec: dict, overrides: dict | None = None) -> ToyConfig:
    return ToyConfig(**resolve_spec(spec, overrides)["cfg"])


def toy_arg(cfg: dict) -> str:
    """The ./sweep.py toy --toy string for a resolved config (non-default fields only)."""
    return ",".join(f"{k}={cfg[k]}" for k in TOY_FIELDS if cfg[k] != getattr(_DEFAULT, k))
