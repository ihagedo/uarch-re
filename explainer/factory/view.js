// DOM glue for index.html: controls -> engine -> render.js. Every number shown
// comes from the engine running a spec in specs/.

import { windowProbe, FILLERS, MIXES } from "./kernels.js";
import { Machine, runToy } from "./engine.js";
import { resolveSpec } from "./resolve.js";
import { findKnee, attribute, offsetsFor, STRUCT } from "./knee.js";
import { renderFloor, renderSweep, stallText } from "./render.js";

const $ = (id) => document.getElementById(id);
const SPECS = ["r10000", "firestorm"];
const st = { specs: {}, offsets: [], resolved: null, m: null, running: false, sweep: null };

const getJSON = async (p) => (await fetch(p)).json();

function controls() {
  return {
    spec: $("spec").value, filler: $("filler").value, n: Number($("n").value),
    depth: Number($("depth").value), delay: $("delay").value, miss: Number($("miss").value),
  };
}

function specTable(r) {
  const rows = Object.entries(r.cfg).filter(([k]) => k !== "miss_latency_jitter" && k !== "seed").map(([k, v]) => {
    const p = r.prov[k];
    const cls = p.provenance ? `prov-${p.provenance}` : p.from === "override" ? "prov-override" : "prov-default";
    return `<tr><td>${k}</td><td class="num">${v}</td><td><span class="badge ${cls}">${p.provenance || (p.from === "override" ? "override" : "toy default")}</span></td><td class="from">${p.from}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>engine field</th><th>value</th><th>provenance</th><th>from</th></tr></thead><tbody>${rows}</tbody></table>` +
    (r.assumptions.length ? `<ul class="assume">${r.assumptions.map((a) => `<li>${a}</li>`).join("")}</ul>` : "");
}

function reset() {
  stop();
  const c = controls();
  $("nval").textContent = c.n;
  try {
    st.resolved = resolveSpec(st.specs[c.spec], { miss_latency: c.miss });
  } catch (e) {
    st.resolved = null; st.m = null;
    $("floor").innerHTML = `<div class="gap"><strong>This spec does not run in the engine.</strong><br>${e.message}</div>`;
    $("specinfo").innerHTML = "";
    $("readout").textContent = "";
    $("stats").textContent = "";
    for (const id of ["step", "run", "sweepbtn"]) $(id).disabled = true;
    return;
  }
  for (const id of ["step", "run", "sweepbtn"]) $(id).disabled = false;
  st.m = new Machine(windowProbe(c.filler, c.n, { depth: c.depth, delay: c.delay }), st.resolved.cfg, { iters: 60, warm: 20 });
  $("specinfo").innerHTML = specTable(st.resolved);
  draw();
}

function draw() {
  const m = st.m;
  if (!m) return;
  $("floor").innerHTML = renderFloor(m, { prov: st.resolved.prov });
  $("readout").textContent = stallText(m);
  $("readout").className = m.stall ? "readout stopped" : "readout";
  const marks = m.iterMarks;
  const cpi = marks.length >= 2 ? ((marks[marks.length - 1] - marks[0]) / (marks.length - 1)).toFixed(1) : "–";
  $("stats").textContent = `cycle ${m.cycle} · iterations retired ${marks.length} / ${m.iters + m.warm} · ` +
    `cycles per iteration so far ${cpi} · in flight ${m.occupancy}`;
  $("events").textContent = m.cycleEvents.map((e) => JSON.stringify(e)).join("\n") || "(no events this cycle)";
  if (st.sweep) $("sweep").innerHTML = renderSweep(st.sweep, { current: Number($("n").value) });
}

function step(k = 1) {
  const m = st.m;
  if (!m) return;
  for (let i = 0; i < k && !m.done; i++) {
    m.step();
    if ($("skip").checked) while (!m.done && m.cycleEvents.length === 0) m.step(); // fast-forward truck countdowns
  }
  draw();
  if (m.done) stop();
}

function loop() {
  if (!st.running) return;
  step(Number($("speed").value));
  requestAnimationFrame(loop);
}
function stop() { st.running = false; $("run").textContent = "Run"; }

function sweep() {
  const c = controls();
  if (!st.resolved) return;
  const cfg = st.resolved.cfg;
  const hi = Math.min(160, Math.max(48, 2 * cfg.rob));
  const ns = []; for (let n = 2; n <= hi; n++) ns.push(n);
  const mj = c.delay === "miss" ? 1 + 0.4 / c.depth : 1.3, steps = c.depth === 1 ? 1 : 2;
  const curve = (f) => ns.map((n) => runToy(windowProbe(f, n, { depth: c.depth, delay: c.delay }), cfg).cyclesPerIter);
  const knees = {}, ys = curve(c.filler);
  knees[c.filler] = findKnee(ns, ys, mj, steps);
  if (c.filler !== "nop") knees.nop = findKnee(ns, curve("nop"), mj, steps);
  const est = attribute(knees, offsetsFor(st.offsets, { depth: c.depth, delay: c.delay }))[STRUCT[c.filler]];
  const estimate = est && est.value !== null ? `${STRUCT[c.filler]} = ${est.value}` : "";
  st.sweep = { ns, ys, knee: knees[c.filler], estimate, label: `${c.filler} filler sweep` };
  $("sweepnote").textContent = est ? `${c.filler} filler, depth ${c.depth}, ${c.delay} delay: knee N = ${knees[c.filler].knee}` +
    (est.offset !== null && est.value !== null ? ` + offset ${est.offset} = ${est.value} (${STRUCT[c.filler]}); ${est.status}` : `; ${est.status}`) : "";
  draw();
}

async function init() {
  for (const s of SPECS) st.specs[s] = await getJSON(`specs/${s}.json`);
  st.offsets = await getJSON("specs/offsets.json");
  $("spec").innerHTML = SPECS.map((s) => `<option value="${s}">${st.specs[s].core}</option>`).join("");
  $("filler").innerHTML = [...Object.keys(FILLERS), ...Object.keys(MIXES)].map((f) => `<option>${f}</option>`).join("");
  for (const id of ["spec", "filler", "n", "depth", "delay", "miss"]) $(id).addEventListener("change", () => { st.sweep = null; $("sweep").innerHTML = ""; $("sweepnote").textContent = ""; reset(); });
  $("n").addEventListener("input", () => { $("nval").textContent = $("n").value; });
  $("step").addEventListener("click", () => { stop(); step(1); });
  $("run").addEventListener("click", () => {
    if (st.running) { stop(); return; }
    if (st.m && st.m.done) reset();
    st.running = true; $("run").textContent = "Pause"; requestAnimationFrame(loop);
  });
  $("reset").addEventListener("click", reset);
  $("sweepbtn").addEventListener("click", sweep);
  // optional URL parameters for demos and screenshots: ?filler=int&n=30&cycles=300
  const q = new URLSearchParams(location.search);
  for (const k of ["spec", "filler", "n", "depth", "delay", "miss"]) if (q.has(k)) $(k).value = q.get(k);
  reset();
  sweep();
  if (q.has("cycles") && st.m) { for (let i = 0; i < Number(q.get("cycles")) && !st.m.done; i++) st.m.step(); draw(); }
}

init().catch((e) => { $("floor").innerHTML = `<div class="gap">Could not load the specs: ${e.message}. Serve this folder over HTTP (see the footer).</div>`; });
