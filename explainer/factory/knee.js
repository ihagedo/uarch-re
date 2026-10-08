// Knee detection: port of find_knee and the core of attribute() in
// uarchre/analysis/knee.py. Conformance requires knee, interval, anomaly and
// status strings to match Python exactly, jump ratio and fit quality to 1e-9.

const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const median = (a) => {
  const s = [...a].sort((p, q) => p - q), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Line fits through the first m points, m = 0..n: [sse[], val[]] (knee.py _prefix_fits, one row). */
function prefixFits(x, y) {
  const n = x.length, x0 = mean(x), y0 = mean(y);
  const xc = x.map((v) => v - x0), yc = y.map((v) => v - y0);
  const cs = (f) => { const o = [0]; let s = 0; for (let i = 0; i < n; i++) { s += f(i); o.push(s); } return o; };
  const sx = cs((i) => xc[i]), sxx = cs((i) => xc[i] * xc[i]);
  const sy = cs((i) => yc[i]), syy = cs((i) => yc[i] * yc[i]), sxy = cs((i) => yc[i] * xc[i]);
  const sse = new Array(n + 1), val = new Array(n + 1);
  for (let m = 0; m <= n; m++) {
    const atm = m >= 1 && m <= n - 1 ? (xc[m - 1] + xc[m]) / 2 : NaN;
    const cxx = sxx[m] - (sx[m] * sx[m]) / m;
    const cxy = sxy[m] - (sx[m] * sy[m]) / m;
    const cyy = syy[m] - (sy[m] * sy[m]) / m;
    const slope = cxx > 0 ? cxy / cxx : 0;
    const s = m < 2 ? 0 : cyy - slope * cxy;
    sse[m] = Number.isNaN(s) ? NaN : Math.max(s, 0);
    val[m] = y0 + sy[m] / m + slope * (atm - sx[m] / m);
  }
  return [sse, val];
}

/** Best two-segment split, left = first k points, k in [2, n-2], ties to the smallest k. */
function split(x, y) {
  const n = x.length;
  const [pre, preVal] = prefixFits(x, y);
  const [sufR, sufValR] = prefixFits([...x].reverse(), [...y].reverse());
  const suf = sufR.slice().reverse(), sufVal = sufValR.slice().reverse(); // [k] = points k..n-1
  let k = 2, best = Infinity;
  for (let j = 2; j <= n - 2; j++) {
    const two = pre[j] + suf[j];
    if (two < best) { best = two; k = j; }
  }
  const sse2 = pre[k] + suf[k], sse1 = pre[n];
  const lo = preVal[k], hi = sufVal[k];
  const c = [0]; for (let i = 0; i < n; i++) c.push(c[i] + y[i]);
  const a = Math.max(k - 3, 0), b = Math.min(k + 3, n);
  const lo3 = (c[k] - c[a]) / (k - a), hi3 = (c[b] - c[k]) / (b - k);
  const quality = sse1 > 0 ? 1 - sse2 / sse1 : 0;
  const jump = Math.max(lo > 0 ? hi / lo : 1, lo3 > 0 ? hi3 / lo3 : 1);
  return { k, quality, jump };
}

/** Move a split back over a staircase's intermediate sub-steps (knee.py _onset). */
function onset(y, k, expectedSteps) {
  const lo = median(y.slice(Math.max(0, k - 5), k));
  const hi = median(y.slice(k, k + 5));
  const thr = lo + ((hi - lo) * 0.5) / expectedSteps;
  while (k > 1 && y[k - 1] > thr) k--;
  return k;
}

const fmt2 = (v) => v.toFixed(2);

/** @returns {{knee: number|null, ci: number[]|null, jump_ratio: number, quality: number, anomaly: string|null, flags: string[]}} */
export function findKnee(ns, ys, minJump = 1.3, expectedSteps = 1) {
  const order = ns.map((_, i) => i).sort((p, q) => ns[p] - ns[q]);
  const x = order.map((i) => +ns[i]), y = order.map((i) => +ys[i]);
  const n = x.length;
  if (n < 4) return { knee: null, ci: null, jump_ratio: 1.0, quality: 0.0, anomaly: "too few points", flags: [] };
  const { k: ks, quality, jump } = split(x, y);
  let k = expectedSteps > 1 ? onset(y, ks, expectedSteps) : ks;
  if (jump < minJump) {
    return { knee: null, ci: null, jump_ratio: jump, quality,
             anomaly: `no step: max ratio ${fmt2(jump)} < ${minJump}`, flags: [] };
  }
  const x0 = x[k - 1], x1 = x[k];
  let anomaly = null;
  const flags = [];
  if (ks + 4 < n) {
    const tail = y.slice(ks);
    let extra = false;
    for (let i = 1; i < tail.length; i++) if ((tail[i] - tail[i - 1]) / tail[i - 1] > 0.4) extra = true;
    if (extra) { anomaly = "second step after the knee"; flags.push("shape.extra_step"); }
  }
  if (quality < 0.6) {
    anomaly = (anomaly ? anomaly + "; " : "") + `poor two-segment fit (${fmt2(quality)})`;
    flags.push("shape.poor_fit");
  }
  return { knee: x0, ci: [x0, x1 - 1], jump_ratio: jump, quality, anomaly, flags };
}

export const STRUCT = Object.freeze({
  nop: "rob", int: "int_prf_headroom", fp: "fp_prf_headroom", load: "lq", store: "sq",
  branch: "rob_branch", flags: "flags_prf_headroom", mix: "shared_window",
});

/** attribute() for one context: size = knee + offset, tie.rob within `tol` of the ROB. Maps are keyed by filler. */
export function attribute(knees, offsets, tol = 0.08) {
  const out = {};
  const rob = knees.nop;
  const robSize = rob && rob.knee !== null && offsets.nop !== undefined ? rob.knee + offsets.nop : null;
  for (const [filler, kn] of Object.entries(knees)) {
    const struct = STRUCT[filler];
    const e = { filler, knee: kn.knee, value: null, status: "unmeasured", anomalies: [...kn.flags], offset: offsets[filler] ?? null };
    if (kn.knee === null) { e.status = "no knee: " + (kn.anomaly || ""); out[struct] = e; continue; }
    const size = kn.knee + (offsets[filler] ?? 0);
    e.value = Math.round(size * 10) / 10;
    e.status = kn.anomaly ? "anomaly: " + kn.anomaly : "measured";
    if (filler !== "nop" && robSize !== null && size >= robSize * (1 - tol)) {
      e.anomalies.push("tie.rob");
      e.status = `ambiguous: knee ${size.toFixed(0)} is within ${Math.round(tol * 100)}% of ROB ` +
        `${robSize.toFixed(0)}; may be ROB-limited, report as >= ${size.toFixed(0)}`;
    }
    out[struct] = e;
  }
  return out;
}

/** Offsets for one context from specs/offsets.json (a mirror of uarchre/analysis/offsets.yaml). */
export function offsetsFor(table, { emitter = "abstract", platform = "toy", depth = 1, delay = "miss" } = {}) {
  const out = {};
  for (const e of table) {
    if (e.emitter === emitter && e.platform === platform && e.depth === depth && e.delay === delay) out[e.filler] = e.offset;
  }
  return out;
}
