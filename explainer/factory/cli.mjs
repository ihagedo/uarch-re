#!/usr/bin/env node
// Headless entry points for the factory engine (Node >= 18, no dependencies).
//   node cli.mjs conformance [--json]   engine vs golden/; exit 1 on any difference
//   node cli.mjs sweep ...              cycles/iter over --n lo:hi:step plus the knee, as JSON
//   node cli.mjs trace ...              the event trace as JSON Lines (golden/*.jsonl format)
//   node cli.mjs render ... --cycles C  the viewer's floor SVG and stall readout at cycle C
// ... = --spec r10000 --set miss_latency=200 --filler nop --n 30 [--depth 1 --delay miss --iters --warm]

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { runConformance } from "./conformance.js";
import { windowProbe } from "./kernels.js";
import { runToy, eventLine, Machine } from "./engine.js";
import { resolveSpec } from "./resolve.js";
import { findKnee } from "./knee.js";
import { renderFloor, renderSweep, stallText } from "./render.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const load = (rel) => readFile(path.join(HERE, rel), "utf8");

function args(argv) {
  const o = { _: [], set: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { o._.push(a); continue; }
    const k = a.slice(2);
    const v = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
    if (k === "set") o.set.push(v); else o[k] = v;
  }
  return o;
}

async function configFrom(o) {
  const spec = JSON.parse(await load(`specs/${o.spec || "r10000"}.json`));
  const ov = {};
  for (const kv of o.set) { const [k, v] = kv.split("="); ov[k] = Number(v); }
  return resolveSpec(spec, ov);
}

const range = (s) => {
  const [a, b, step = 1] = String(s).split(":").map(Number);
  if (b === undefined || Number.isNaN(b)) return [a];
  const out = []; for (let n = a; n <= b; n += step) out.push(n); return out;
};

async function main() {
  const o = args(process.argv.slice(2));
  const cmd = o._[0];
  if (cmd === "conformance") {
    const s = await runConformance(load, { log: o.json ? () => {} : (m) => console.log(m) });
    if (o.json) {
      for (const c of s.cases) delete c.firstMismatch?.engine;
      for (const k of Object.values(s.sweep.knees)) delete k.ys;
      console.log(JSON.stringify(s));
    } else {
      console.log(`\n${s.events.identical.toLocaleString("en-US")} / ${s.events.total.toLocaleString("en-US")} ` +
        `events identical across ${s.events.kernels} kernels; ${s.sweep.identical} / ${s.sweep.points} sweep points; ` +
        `${s.knee.identical} / ${s.knee.cases} knee curves`);
      console.log(s.pass ? "PASS" : "FAIL\n  " + s.problems.join("\n  "));
    }
    process.exit(s.pass ? 0 : 1);
  } else if (cmd === "render") {
    const r = await configFrom(o);
    const m = new Machine(windowProbe(o.filler || "nop", Number(o.n || 30), { depth: Number(o.depth || 1), delay: o.delay || "miss" }), r.cfg);
    for (let i = 0; i < Number(o.cycles || 0) && !m.done; i++) m.step();
    const sw = { ns: [1, 2, 3, 4], ys: [1, 1, 2, 2], knee: { knee: 2 } };
    console.log(JSON.stringify({ cycle: m.cycle, readout: stallText(m), floor: renderFloor(m, { prov: r.prov }),
                                 sweep: renderSweep(sw) }));
  } else if (cmd === "sweep" || cmd === "trace") {
    const { cfg, assumptions } = await configFrom(o);
    const depth = Number(o.depth || 1), delay = o.delay || "miss";
    const iters = Number(o.iters || (cmd === "trace" ? 3 : 60)), warm = Number(o.warm || (cmd === "trace" ? 1 : 20));
    if (cmd === "trace") {
      const r = runToy(windowProbe(o.filler || "nop", Number(o.n), { depth, delay }), cfg, { iters, warm, trace: true });
      process.stdout.write(r.events.map(eventLine).join("\n") + "\n");
      return;
    }
    const ns = range(o.n || "2:60:1");
    const t0 = Date.now();
    const ys = ns.map((n) => runToy(windowProbe(o.filler || "nop", n, { depth, delay }), cfg, { iters, warm }).cyclesPerIter);
    const minJump = Number(o["min-jump"] || (1 + 0.4 / depth)), steps = depth === 1 ? 1 : 2;
    const knee = findKnee(ns, ys, minJump, steps);
    console.log(JSON.stringify({ spec: o.spec || "r10000", cfg, assumptions, filler: o.filler || "nop", depth, delay,
                                 iters, warm, ns, ys, knee, elapsed_ms: Date.now() - t0 }));
  } else {
    console.error("usage: cli.mjs conformance [--json] | sweep ... | trace ...  (see the header of cli.mjs)");
    process.exit(2);
  }
}

main().catch((e) => { console.error(e.stack || String(e)); process.exit(2); });
