// Abstract kernels: port of uarchre/kernels (window.py). Instructions are
// [op, arg] pairs as in Python, arg being the chain id for chain ops and the
// rotation index for fillers. The engine appends the loop branch.

/** @typedef {{name: string, body: [string, number][], nChains: number, params: object}} Kernel */

export const OP = Object.freeze({
  MISS_LOAD: "miss_load", // pointer-chase load on chain `arg` (cache miss)
  NOP: "nop",             // takes a window slot, no destination
  INT: "int",             // integer ALU op with an integer destination
  FP: "fp",               // FP op with an FP destination
  LOAD: "load",           // L1-hit load with an integer destination
  STORE: "store",         // L1-hit store, no destination
  DELAY_FP: "delay_fp",   // one link of a dependent long-latency FP chain (fsqrt)
  DELAY_INT: "delay_int", // one link of a dependent long-latency int chain (div/imul)
  BRANCH: "branch",       // never-taken conditional branch, no destination
  FLAGS: "flags",         // writes the flags register only (cmp)
});
export const LOOP = "loop"; // the implicit loop branch (None in Python)

export const FILLERS = Object.freeze({
  nop: OP.NOP, int: OP.INT, fp: OP.FP, load: OP.LOAD, store: OP.STORE,
  branch: OP.BRANCH, flags: OP.FLAGS,
});
export const MIXES = Object.freeze({ mix: [OP.INT, OP.FP] }); // INT and FP fillers alternate (P1.4)
export const DELAYS = Object.freeze({ fp: OP.DELAY_FP, int: OP.DELAY_INT });

/** Window probe: `depth` delay ops on chain 0, N fillers, `depth` on chain 1, N fillers. */
export function windowProbe(filler, n, { rotation = 8, depth = 1, delay = "miss" } = {}) {
  const ops = MIXES[filler] || (filler in FILLERS ? [FILLERS[filler]] : null);
  if (!ops) throw new Error(`unknown filler ${filler}`);
  const dop = delay === "miss" ? OP.MISS_LOAD : DELAYS[delay];
  if (!dop) throw new Error(`unknown delay ${delay}`);
  const body = [];
  for (let i = 0; i < depth; i++) body.push([dop, 0]);
  for (let i = 0; i < n; i++) body.push([ops[i % ops.length], i % rotation]);
  for (let i = 0; i < depth; i++) body.push([dop, 1]);
  for (let i = 0; i < n; i++) body.push([ops[(n + i) % ops.length], (n + i) % rotation]);
  const tag = delay === "miss" ? "" : `_${delay}`;
  return {
    name: `window_${filler}_d${depth}${tag}_${n}`,
    body,
    nChains: 2,
    params: { filler, n, depth, delay },
  };
}
