// The factory floor as SVG strings: pure functions of engine state with no DOM,
// so Node can render them in tests. Colours are CSS custom properties from index.html.

import { LOOP, OP } from "./kernels.js";

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/** Colour token and one-letter tag for an op. Chain A / B get their own colours. */
export function opStyle(e) {
  if (e.op === OP.MISS_LOAD || e.op === OP.DELAY_FP || e.op === OP.DELAY_INT) {
    return e.chain === 0 ? { fill: "var(--chain-a)", tag: "A" } : { fill: "var(--chain-b)", tag: "B" };
  }
  if (e.op === LOOP) return { fill: "var(--loop)", tag: "↺" };
  const tags = { nop: "n", int: "i", fp: "f", load: "l", store: "s", branch: "b", flags: "c" };
  return { fill: "var(--filler)", tag: tags[e.op] ?? "?" };
}

/** waiting (dispatched, not issued) / executing / done (waiting to retire). */
export function opState(e, cycle) {
  if (!e.issued) return "waiting";
  return e.doneAt <= cycle ? "done" : "executing";
}

const STATE_OPACITY = { waiting: 0.18, executing: 0.55, done: 1 };

function cellRow(x, y, n, used, size, gap, cls, title) {
  let s = "";
  for (let i = 0; i < n; i++) {
    const cx = x + (i % 16) * (size + gap), cy = y + Math.floor(i / 16) * (size + gap);
    s += `<rect class="${i < used ? cls + " on" : cls}" x="${cx}" y="${cy}" width="${size}" height="${size}" rx="2"><title>${esc(title)}</title></rect>`;
  }
  return s;
}

/** The whole floor for one Machine state; prov is resolveSpec()'s provenance per config field. */
export function renderFloor(m, { prov = {} } = {}) {
  const cfg = m.cfg, cap = m.cap, live = m.live(), cycle = m.cycle;
  const W = 720;
  let s = "";
  const label = (x, y, t, cls = "lbl") => `<text class="${cls}" x="${x}" y="${y}">${esc(t)}</text>`;
  const provTag = (field) => (prov[field] && prov[field].provenance ? ` · ${prov[field].provenance}` : "");

  // order ledger: columns of 8 lines, oldest first (R10K: 4 parallel 8-deep FIFOs)
  const cols = Math.ceil(cap.rob / 8), cw = 46, ch = 22, lx = 10, ly = 46;
  s += label(lx, 16, `Order ledger ${live.length} / ${cap.rob}`, "hd");
  s += label(lx, 32, `active list / ROB${provTag("rob")}`, "sm");
  for (let i = 0; i < cols * 8; i++) {
    const x = lx + Math.floor(i / 8) * (cw + 4), y = ly + (i % 8) * (ch + 3);
    if (i >= cap.rob) continue;
    const e = live[i];
    if (!e) { s += `<rect class="slot" x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3"/>`; continue; }
    const st = opStyle(e), state = opState(e, cycle);
    const left = state === "executing" ? e.doneAt - cycle : null;
    s += `<g><title>#${e.seq} ${esc(e.op)} (iteration ${e.it}) ${state}${left !== null ? `, ${left} cycles left` : ""}</title>` +
      `<rect class="slot" x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3"/>` +
      `<rect x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3" fill="${st.fill}" fill-opacity="${STATE_OPACITY[state]}" stroke="${st.fill}" stroke-width="1.5"/>` +
      `<text class="cell${state === "done" ? " inv" : ""}" x="${x + 5}" y="${y + 15}">${st.tag}${e.seq % 1000}</text></g>`;
  }
  const ledgerW = cols * (cw + 4);

  // tote racks and dock ledgers
  const rx = Math.max(lx + ledgerW + 28, 230), sz = 13, gap = 3;
  let ry = 16;
  const pools = [
    ["int_prf", "Int tote rack (rename headroom)", "tote-int", `${cfg.int_prf} physical − ${cfg.arch_int} architectural`],
    ["fp_prf", "FP tote rack (rename headroom)", "tote-fp", `${cfg.fp_prf} physical − ${cfg.arch_fp} architectural`],
    ["lq", "Inbound dock ledger (LQ)", "dock", "load queue entries"],
    ["sq", "Outbound dock ledger (SQ)", "dock", "store queue entries"],
  ];
  for (const [k, name, cls, why] of pools) {
    const n = cap[k];
    if (n > 256) { s += label(rx, ry, `${name}: unbounded (spec gap)`, "hd"); ry += 26; continue; }
    s += label(rx, ry, `${name} ${m.used[k]} / ${n}${provTag(k)}`, "hd");
    s += cellRow(rx, ry + 8, n, m.used[k], sz, gap, cls, `${name}: ${m.used[k]} of ${n} in use (${why})`);
    ry += 8 + Math.ceil(n / 16) * (sz + gap) + 22;
  }

  // supplier trucks: chain ops in flight
  const trucks = live.filter((e) => (e.op === OP.MISS_LOAD || e.op === OP.DELAY_FP || e.op === OP.DELAY_INT) && e.issued && e.doneAt > cycle);
  const lat = trucks.length && trucks[0].op === OP.MISS_LOAD ? cfg.miss_latency : cfg.alu_delay_latency;
  s += label(rx, ry, `Supplier trucks (chain ops in flight): ${trucks.length}`, "hd");
  ry += 8;
  for (const e of trucks.slice(0, 6)) {
    const left = e.doneAt - cycle, frac = Math.max(0, Math.min(1, 1 - left / lat));
    const st = opStyle(e);
    s += `<rect class="slot" x="${rx}" y="${ry}" width="240" height="12" rx="2"/>` +
      `<rect x="${rx}" y="${ry}" width="${(240 * frac).toFixed(1)}" height="12" rx="2" fill="${st.fill}"/>` +
      label(rx + 248, ry + 10, `${st.tag}${e.seq % 1000}: ${left} cycles`, "sm");
    ry += 17;
  }
  const H = Math.max(ly + 8 * (ch + 3) + 10, ry + 10);
  const VW = Math.max(W, rx + 330);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${VW}" viewBox="0 0 ${VW} ${H}" role="img" aria-label="factory floor">${s}</svg>`;
}

/** Plain-words stall readout for the current cycle. */
export function stallText(m) {
  if (m.done) return "Run complete.";
  if (!m.stall) return "Line moving.";
  const words = {
    rob_full: "order ledger full (active list)", int_prf_empty: "int tote rack empty (integer rename registers)",
    fp_prf_empty: "FP tote rack empty (FP rename registers)", lq_full: "inbound dock ledger full (LQ)",
    sq_full: "outbound dock ledger full (SQ)",
  };
  return `Line stopped: ${words[m.stall.why] ?? m.stall.why}; order #${m.stall.seq} waits to dispatch.`;
}

/**
 * cycles/iter against N as an SVG line chart with the knee marked.
 * @param {{ns: number[], ys: number[], knee: {knee: number|null}, label?: string, estimate?: string}} sw
 */
export function renderSweep(sw, { width = 720, height = 260, current = null } = {}) {
  const { ns, ys } = sw;
  const pad = { l: 56, r: 16, t: 26, b: 40 };
  const x0 = Math.min(...ns), x1 = Math.max(...ns), y1 = Math.max(...ys) * 1.08;
  const X = (n) => pad.l + ((n - x0) / Math.max(1, x1 - x0)) * (width - pad.l - pad.r);
  const Y = (y) => height - pad.b - (y / y1) * (height - pad.t - pad.b);
  let s = "";
  for (let t = 0; t <= 4; t++) {
    const v = (y1 * t) / 4;
    s += `<line class="grid" x1="${pad.l}" x2="${width - pad.r}" y1="${Y(v)}" y2="${Y(v)}"/>` +
      `<text class="tick" x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
  }
  const step = Math.max(1, Math.ceil((x1 - x0) / 10 / 5) * 5);
  for (let n = Math.ceil(x0 / step) * step; n <= x1; n += step) {
    s += `<text class="tick" x="${X(n)}" y="${height - pad.b + 16}" text-anchor="middle">${n}</text>`;
  }
  s += `<text class="axis" x="${(pad.l + width - pad.r) / 2}" y="${height - 6}" text-anchor="middle">filler count N</text>`;
  s += `<text class="axis" x="14" y="${(pad.t + height - pad.b) / 2}" transform="rotate(-90 14 ${(pad.t + height - pad.b) / 2})" text-anchor="middle">cycles / iteration</text>`;
  if (sw.knee && sw.knee.knee !== null) {
    const k = sw.knee.knee;
    s += `<line class="knee" x1="${X(k)}" x2="${X(k)}" y1="${pad.t}" y2="${height - pad.b}"/>` +
      `<text class="kneelbl" x="${X(k) + 6}" y="${pad.t + 12}">knee N = ${k}${sw.estimate ? " → " + esc(sw.estimate) : ""}</text>`;
  }
  if (current !== null && current >= x0 && current <= x1) {
    s += `<line class="cur" x1="${X(current)}" x2="${X(current)}" y1="${pad.t}" y2="${height - pad.b}"/>`;
  }
  s += `<polyline class="series" points="${ns.map((n, i) => `${X(n).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" ")}"/>`;
  s += ns.map((n, i) => `<circle class="pt" cx="${X(n).toFixed(1)}" cy="${Y(ys[i]).toFixed(1)}" r="3"><title>N = ${n}: ${ys[i].toFixed(1)} cycles/iter</title></circle>`).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(sw.label || "sweep")}">${s}</svg>`;
}
