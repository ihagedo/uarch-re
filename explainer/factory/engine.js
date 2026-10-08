// Factory engine: cycle-level port of run_toy (uarchre/backends/platform/toy.py).
// It follows run_toy decision for decision, so cycles/iter and the event trace
// match golden/ exactly. Change semantics in Python first, then here.
// Per cycle: complete -> retire -> issue -> dispatch -> patch chain readiness.
// State is integer-only; cycles/iter is one IEEE division at the end, as in Python.

import { OP, LOOP } from "./kernels.js";

export const TOY_DEFAULTS = Object.freeze({
  rob: 128, int_prf: 96, fp_prf: 96, lq: 32, sq: 32, width: 4,
  miss_latency: 200, load_latency: 4, alu_latency: 1, fp_latency: 3,
  arch_int: 32, arch_fp: 32, alu_delay_latency: 24, miss_latency_jitter: 0, seed: 0,
});
export const TOY_FIELDS = Object.freeze(Object.keys(TOY_DEFAULTS));

const CHAIN = new Set([OP.MISS_LOAD, OP.DELAY_FP, OP.DELAY_INT]);
const INF = 1e9; // toy.py's 10**9 sentinel ("not ready until patched")
export const STALL_WHY = Object.freeze({
  rob: "rob_full", int_prf: "int_prf_empty", fp_prf: "fp_prf_empty", lq: "lq_full", sq: "sq_full",
});

const NEEDS = new Map();
/** Pools besides the ROB an op holds from dispatch to retire. */
export function needs(op) {
  let r = NEEDS.get(op);
  if (r) return r;
  r = [];
  if (op === OP.INT || op === OP.LOAD || op === OP.MISS_LOAD || op === OP.DELAY_INT) r.push("int_prf");
  else if (op === OP.FP || op === OP.DELAY_FP) r.push("fp_prf");
  if (op === OP.LOAD || op === OP.MISS_LOAD) r.push("lq");
  if (op === OP.STORE) r.push("sq");
  NEEDS.set(op, r);
  return r;
}

/** Capacities the engine enforces: the ROB and the rename headroom pools. */
export function capacities(cfg) {
  return {
    rob: cfg.rob, int_prf: cfg.int_prf - cfg.arch_int, fp_prf: cfg.fp_prf - cfg.arch_fp,
    lq: cfg.lq, sq: cfg.sq,
  };
}

function latencyOf(cfg, op) {
  switch (op) {
    case OP.MISS_LOAD: return cfg.miss_latency;
    case OP.LOAD: return cfg.load_latency;
    case OP.INT: return cfg.alu_latency;
    case OP.FP: return cfg.fp_latency;
    case OP.DELAY_FP: case OP.DELAY_INT: return cfg.alu_delay_latency;
    default: return 1; // nop, store, branch, flags, loop branch
  }
}

/** One kernel on one config; step() advances a cycle and the viewer reads the fields in between. */
export class Machine {
  /** cfg: ToyConfig fields, missing ones from TOY_DEFAULTS. */
  constructor(kernel, cfg, { iters = 60, warm = 20, trace = false, onEvent = null } = {}) {
    this.cfg = { ...TOY_DEFAULTS, ...cfg };
    if (this.cfg.miss_latency_jitter) throw new Error("miss_latency_jitter is not supported (not deterministic across engines)");
    for (const k of TOY_FIELDS) {
      if (k !== "miss_latency_jitter" && !Number.isInteger(this.cfg[k])) throw new Error(`cfg.${k} must be an integer`);
    }
    this.kernel = kernel;
    this.body = kernel.body.map(([op, arg]) => ({ op, arg })).concat([{ op: LOOP, arg: 0 }]);
    this.streamLen = this.body.length;
    this.iters = iters;
    this.warm = warm;
    this.total = (iters + warm) * this.streamLen;
    this.cap = capacities(this.cfg);
    this.used = { int_prf: 0, fp_prf: 0, lq: 0, sq: 0 };
    this.rob = [];          // entries in program order; rob[head..] are live
    this.head = 0;
    this.lastMiss = new Array(kernel.nChains).fill(null);
    this.pos = 0;           // next op index in the dynamic stream
    this.cycle = 0;
    this.iterMarks = [];    // cycle at which each iteration's loop branch retired
    this.completing = new Map(); // done_at -> [seq]
    this.lastStall = null;  // {seq, why} of the previous cycle's stall
    this.stall = null;      // this cycle's stall, for the viewer
    this.trace = trace;
    this.events = [];
    this.onEvent = onEvent;
    this.cycleEvents = [];  // last stepped cycle's events, kept even without trace (viewer)
  }

  get done() { return this.iterMarks.length >= this.iters + this.warm; }
  get occupancy() { return this.rob.length - this.head; }
  /** Live entries, oldest first. */
  live() { return this.rob.slice(this.head); }

  _emit(e) {
    this.cycleEvents.push(e);
    if (this.trace) this.events.push(e);
    if (this.onEvent) this.onEvent(e);
  }

  /** Advance one cycle. Mirrors one pass of run_toy's while loop. */
  step() {
    const cfg = this.cfg, cap = this.cap, used = this.used, rob = this.rob;
    const cycle = this.cycle;
    this.cycleEvents = [];

    const due = this.completing.get(cycle);
    if (due) {
      due.sort((a, b) => a - b);
      for (const s of due) this._emit({ c: cycle, ev: "complete", seq: s });
      this.completing.delete(cycle);
    }

    // retire, in order
    let retired = 0;
    while (this.head < rob.length && retired < cfg.width) {
      const e = rob[this.head];
      if (!(e.doneAt >= 0 && e.doneAt <= cycle)) break;
      this.head++;
      for (const r of needs(e.op)) used[r]--;
      e.retireAt = cycle;
      if (e.op === LOOP) this.iterMarks.push(cycle);
      this._emit({ c: cycle, ev: "retire", seq: e.seq });
      retired++;
    }
    if (this.head > 4096) { rob.splice(0, this.head); this.head = 0; }

    // issue, out of order, oldest first
    let issued = 0;
    for (let i = this.head; i < rob.length; i++) {
      if (issued >= cfg.width) break;
      const e = rob[i];
      if (!e.issued && e.readyAt <= cycle) {
        e.issued = true;
        e.issueAt = cycle;
        e.doneAt = cycle + latencyOf(cfg, e.op);
        issued++;
        this._emit({ c: cycle, ev: "issue", seq: e.seq });
        const q = this.completing.get(e.doneAt);
        if (q) q.push(e.seq); else this.completing.set(e.doneAt, [e.seq]);
      }
    }

    // dispatch, in order
    let stall = null;
    for (let w = 0; w < cfg.width; w++) {
      if (this.pos >= this.total) break;
      if (rob.length - this.head >= cap.rob) { stall = { seq: this.pos, why: STALL_WHY.rob }; break; }
      const ins = this.body[this.pos % this.streamLen];
      const op = ins.op;
      const nd = needs(op);
      let full = null;
      for (const r of nd) if (used[r] >= cap[r]) { full = r; break; }
      if (full) { stall = { seq: this.pos, why: STALL_WHY[full] }; break; }
      for (const r of nd) used[r]++;
      const e = {
        seq: this.pos, op, chain: op === LOOP ? 0 : ins.arg, it: Math.floor(this.pos / this.streamLen),
        dispatch: cycle, readyAt: -1, doneAt: -1, issued: false, issueAt: -1, retireAt: -1, prev: null,
      };
      if (CHAIN.has(op)) {
        const prev = this.lastMiss[e.chain];
        e.readyAt = Math.max(cycle + 1, prev && prev.doneAt >= 0 ? prev.doneAt : INF);
        if (prev && prev.doneAt < 0) e.readyAt = INF;
        this.lastMiss[e.chain] = e;
        e.prev = prev;
      } else {
        e.readyAt = cycle + 1;
      }
      rob.push(e);
      const d = { c: cycle, ev: "dispatch", seq: this.pos, op, it: e.it, res: ["rob", ...nd] };
      if (CHAIN.has(op)) d.chain = e.chain;
      this._emit(d);
      this.pos++;
    }
    const last = this.lastStall;
    if (stall && !(last && last.seq === stall.seq && last.why === stall.why)) {
      this._emit({ c: cycle, ev: "stall", stage: "dispatch", why: stall.why, seq: stall.seq });
    }
    this.lastStall = stall;
    this.stall = stall;

    // patch chain-op readiness once the predecessor has a done_at
    for (let i = this.head; i < rob.length; i++) {
      const e = rob[i];
      if (CHAIN.has(e.op) && e.readyAt >= INF) {
        const p = e.prev;
        if (p === null) e.readyAt = cycle + 1;
        else if (p.doneAt >= 0) e.readyAt = Math.max(cycle + 1, p.doneAt);
      }
    }
    this.cycle = cycle + 1;
    if (this.cycle > 1e7) throw new Error("toy model did not converge");
  }

  /** Cycles per steady-state iteration (run_toy's return value). */
  cyclesPerIter() {
    const steady = this.iterMarks.slice(this.warm);
    return (steady[steady.length - 1] - steady[0]) / (steady.length - 1);
  }

  result() {
    return {
      cyclesPerIter: this.cyclesPerIter(), iterMarks: this.iterMarks.slice(),
      finalCycle: this.cycle - 1, dispatched: this.pos, events: this.events,
    };
  }
}

/** run_toy(kernel, cfg, iters, warm) with an optional event trace. */
export function runToy(kernel, cfg, { iters = 60, warm = 20, trace = false } = {}) {
  const m = new Machine(kernel, cfg, { iters, warm, trace });
  while (!m.done) m.step();
  return m.result();
}

/** One-line JSON of an event, keys in emit order (the golden JSONL format). */
export function eventLine(e) { return JSON.stringify(e); }
