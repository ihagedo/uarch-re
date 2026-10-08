// Conformance: the JS engine against the Python golden suite. `load(relPath)`
// returns a file under explainer/factory/ as text (cli.mjs reads it from disk,
// test.html fetches it). Checks golden traces event for event, the resolver,
// the R10K sweep from specs/r10000.json (cycles/iter, knees, attribute; NOP knee
// 30, int filler ambiguous) and find_knee on stored curves.

import { windowProbe } from "./kernels.js";
import { runToy, eventLine } from "./engine.js";
import { resolveSpec, UnsupportedSpec } from "./resolve.js";
import { findKnee, attribute, offsetsFor } from "./knee.js";

const json = async (load, p) => JSON.parse(await load(p));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => a === b || Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

function sameKnee(js, py) {
  const why = [];
  if (js.knee !== py.knee) why.push(`knee ${js.knee} != ${py.knee}`);
  if (!same(js.ci, py.ci)) why.push(`ci ${JSON.stringify(js.ci)} != ${JSON.stringify(py.ci)}`);
  if (!close(js.jump_ratio, py.jump_ratio)) why.push(`jump ${js.jump_ratio} != ${py.jump_ratio}`);
  if (!close(js.quality, py.quality)) why.push(`quality ${js.quality} != ${py.quality}`);
  if (js.anomaly !== py.anomaly) why.push(`anomaly ${js.anomaly} != ${py.anomaly}`);
  if (!same(js.flags, py.flags)) why.push(`flags ${js.flags} != ${py.flags}`);
  return why;
}

/** Compare one golden JSONL case with the engine. */
export function checkCase(text) {
  const lines = text.split("\n").filter((l) => l.length);
  const head = JSON.parse(lines[0]);
  const gold = lines.slice(1);
  const k = head.kernel;
  const kernel = windowProbe(k.filler, k.n, { rotation: k.rotation, depth: k.depth, delay: k.delay });
  const out = { case: head.case, events: gold.length, identical: 0, problems: [] };
  if (!same(kernel.body, head.body)) out.problems.push("kernel body differs from Python's window_probe");
  const r = runToy(kernel, head.cfg, { iters: head.iters, warm: head.warm, trace: true });
  const mine = r.events.map(eventLine);
  for (let i = 0; i < gold.length; i++) {
    if (mine[i] === gold[i]) out.identical++;
    else if (!out.firstMismatch) out.firstMismatch = { index: i, golden: gold[i], engine: mine[i] ?? null };
  }
  if (mine.length !== gold.length) out.problems.push(`engine emitted ${mine.length} events, golden has ${gold.length}`);
  if (r.cyclesPerIter !== head.cycles_per_iter) out.problems.push(`cycles/iter ${r.cyclesPerIter} != ${head.cycles_per_iter}`);
  if (!same(r.iterMarks, head.iter_marks)) out.problems.push("iteration marks differ");
  if (r.finalCycle !== head.final_cycle) out.problems.push(`final cycle ${r.finalCycle} != ${head.final_cycle}`);
  if (r.dispatched !== head.dispatched) out.problems.push(`dispatched ${r.dispatched} != ${head.dispatched}`);
  if (out.firstMismatch) out.problems.push(`first event mismatch at #${out.firstMismatch.index}`);
  out.pass = out.problems.length === 0 && out.identical === out.events;
  out.cyclesPerIter = r.cyclesPerIter;
  return out;
}

export async function runConformance(load, { log = () => {} } = {}) {
  const index = await json(load, "golden/index.json");
  const summary = { cases: [], problems: [] };

  // 1. golden traces
  for (const c of index.cases) {
    const res = checkCase(await load(`golden/${c.file}`));
    summary.cases.push(res);
    log(`${res.pass ? "ok  " : "FAIL"} ${res.case}: ${res.identical} / ${res.events} events identical` +
      (res.problems.length ? ` (${res.problems.join("; ")})` : ""));
  }
  const ev = summary.cases.reduce((a, c) => a + c.events, 0);
  const evOk = summary.cases.reduce((a, c) => a + c.identical, 0);
  summary.events = { total: ev, identical: evOk, kernels: summary.cases.length,
                     casesPassing: summary.cases.filter((c) => c.pass).length };
  if (ev !== index.events_total) summary.problems.push(`index says ${index.events_total} events, files have ${ev}`);

  // 2. resolver
  const specs = {};
  const resolveCases = await json(load, `golden/${index.resolve}`);
  let resOk = 0;
  for (const rc of resolveCases) {
    specs[rc.spec] ??= await json(load, `specs/${rc.spec}.json`);
    let got;
    try { got = { result: resolveSpec(specs[rc.spec], rc.overrides) }; }
    catch (e) { if (!(e instanceof UnsupportedSpec)) throw e; got = { error: e.message }; }
    const want = rc.error !== undefined ? { error: rc.error } : { result: rc.result };
    // compare values, not formatting: Python writes 0.0 where JS has 0
    const ok = same(JSON.parse(JSON.stringify(got)), JSON.parse(JSON.stringify(want)));
    if (ok) resOk++; else summary.problems.push(`resolver differs for ${rc.spec} ${JSON.stringify(rc.overrides)}`);
  }
  summary.resolver = { cases: resolveCases.length, identical: resOk };
  log(`${resOk === resolveCases.length ? "ok  " : "FAIL"} resolver: ${resOk} / ${resolveCases.length} cases identical`);

  // 3. R10K sweep from the spec
  const sw = await json(load, `golden/${index.sweep}`);
  const spec = specs.r10000 ?? await json(load, "specs/r10000.json");
  const { cfg } = resolveSpec(spec, sw.overrides);
  const table = await json(load, "specs/offsets.json");
  const sweep = { points: 0, identical: 0, knees: {}, attribute: {} };
  if (!same(cfg, JSON.parse(JSON.stringify(sw.cfg)))) summary.problems.push("R10K config resolved in JS differs from Python's");
  const kneesByDepth = {};
  for (const [name, s] of Object.entries(sw.sweeps)) {
    const ys = s.ns.map((n) => runToy(windowProbe(s.filler, n, { depth: s.depth }), cfg,
                                      { iters: sw.iters, warm: sw.warm }).cyclesPerIter);
    let same_ = 0;
    ys.forEach((y, i) => { if (y === s.ys[i]) same_++; });
    sweep.points += ys.length; sweep.identical += same_;
    const kn = findKnee(s.ns, ys, s.min_jump, s.expected_steps);
    const why = sameKnee(kn, s.knee);
    sweep.knees[name] = { js: kn.knee, python: s.knee.knee, identical: why.length === 0, ys };
    if (why.length) summary.problems.push(`knee ${name}: ${why.join(", ")}`);
    if (same_ !== ys.length) summary.problems.push(`sweep ${name}: ${ys.length - same_} points differ`);
    (kneesByDepth[s.depth] ??= {})[s.filler] = kn;
    log(`${same_ === ys.length && !why.length ? "ok  " : "FAIL"} R10K sweep ${name}: ${same_} / ${ys.length} points identical, knee JS ${kn.knee} / Python ${s.knee.knee}`);
  }
  for (const [depth, knees] of Object.entries(kneesByDepth)) {
    const js = attribute(knees, offsetsFor(table, { depth: +depth }));
    const py = sw[`attribute-d${depth}`];
    for (const [struct, p] of Object.entries(py)) {
      const j = js[struct];
      const ok = j && j.value === p.value && j.status === p.status && same(j.anomalies, p.anomalies);
      sweep.attribute[`${struct}-d${depth}`] = { value: j?.value, status: j?.status, identical: !!ok };
      if (!ok) summary.problems.push(`attribute ${struct} d${depth}: JS ${JSON.stringify(j)} != Python ${JSON.stringify(p)}`);
    }
  }
  summary.sweep = sweep;

  // milestone: R10K NOP knee 30 -> active list 32, int filler ambiguous
  const nop = sweep.knees["nop-d1"], rob = sweep.attribute["rob-d1"], intr = sweep.attribute["int_prf_headroom-d1"];
  summary.milestone = {
    nopKnee: nop?.js ?? null, activeList: rob?.value ?? null,
    intAmbiguous: !!intr && intr.status.startsWith("ambiguous"),
  };
  summary.milestone.pass = summary.milestone.nopKnee === 30 && summary.milestone.activeList === 32 &&
    summary.milestone.intAmbiguous;
  if (!summary.milestone.pass) summary.problems.push(`milestone: ${JSON.stringify(summary.milestone)}`);
  log(`${summary.milestone.pass ? "ok  " : "FAIL"} milestone: R10K NOP knee N = ${summary.milestone.nopKnee} ` +
    `-> active list ${summary.milestone.activeList}; int filler ambiguous: ${summary.milestone.intAmbiguous}`);

  // 4. find_knee on stored curves
  const kc = await json(load, `golden/${index.knee}`);
  let kOk = 0;
  for (const c of kc) {
    const why = sameKnee(findKnee(c.ns, c.ys, c.min_jump, c.expected_steps), c.python);
    if (why.length) summary.problems.push(`find_knee ${c.name}: ${why.join(", ")}`); else kOk++;
  }
  summary.knee = { cases: kc.length, identical: kOk };
  log(`${kOk === kc.length ? "ok  " : "FAIL"} find_knee: ${kOk} / ${kc.length} curves identical`);

  for (const c of summary.cases) if (!c.pass) summary.problems.push(`case ${c.case}: ${c.problems.join("; ")}`);
  summary.pass = summary.problems.length === 0;
  return summary;
}
