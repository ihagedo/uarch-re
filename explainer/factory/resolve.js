// Spec -> engine config with provenance: port of resolve_spec in
// uarchre/backends/platform/toy_trace.py, error and assumption strings included
// (conformance compares the two on golden/resolve_cases.json).

import { TOY_DEFAULTS, TOY_FIELDS } from "./engine.js";

export const UNBOUNDED = 1 << 20;

export class UnsupportedSpec extends Error {}

function at(node, path) {
  for (const p of path.split(".")) {
    if (node === null || typeof node !== "object" || !(p in node)) return null;
    node = node[p];
  }
  return node === undefined ? null : node;
}

/**
 * @param {object} spec  a parsed core spec (specs/*.json)
 * @returns {{cfg: object, prov: object, assumptions: string[]}}
 */
export function resolveSpec(spec, overrides = {}) {
  for (const k of Object.keys(overrides)) {
    if (!TOY_FIELDS.includes(k)) throw new Error(`unknown ToyConfig field(s): ${k}`);
  }
  const cfg = {}, prov = {}, assume = [];
  const take = (field, path, note) => {
    const n = at(spec, path);
    const v = n !== null && typeof n === "object" ? n.value : n;
    if (v === null || v === undefined) return null;
    cfg[field] = Math.trunc(v);
    prov[field] = { from: path, provenance: n !== null && typeof n === "object" ? (n.provenance ?? null) : null };
    if (note) prov[field].note = note;
    return cfg[field];
  };

  const model = at(spec, "backend.retire.model") || "active_list";
  if (model !== "active_list") {
    throw new UnsupportedSpec(`backend.retire.model is '${model}': the engine has toy semantics, a classic ` +
      "in-order ledger only (coalesced retire groups are not modeled)");
  }
  if (take("rob", "backend.retire.active_list") === null && take("rob", "backend.rob_entries") === null) {
    throw new UnsupportedSpec("spec has no backend.retire.active_list (or rob_entries)");
  }

  for (const [pool, arch] of [["int_prf", "arch_int"], ["fp_prf", "arch_fp"]]) {
    const phys = take(pool, `backend.${pool}`);
    const head = at(spec, `backend.${pool}_headroom.value`);
    if (phys === null) {
      cfg[pool] = UNBOUNDED; cfg[arch] = 0;
      prov[pool] = { from: "null -> unbounded", provenance: null };
      prov[arch] = { from: "null -> unbounded", provenance: null };
      assume.push(`backend.${pool} is null: modelled as unbounded`);
    } else if (head === null) {
      throw new UnsupportedSpec(`backend.${pool}_headroom is missing: the architectural count is unknown`);
    } else {
      cfg[arch] = phys - Math.trunc(head);
      prov[arch] = { from: `backend.${pool} - backend.${pool}_headroom`,
                     provenance: at(spec, `backend.${pool}_headroom.provenance`) };
    }
  }

  for (const [q, key] of [["lq", "load_queue"], ["sq", "store_queue"]]) {
    if (take(q, `backend.${key}`) === null) {
      const aq = at(spec, "backend.queues.address_queue");
      if (aq !== null && typeof aq === "object" && aq.value !== null && aq.value !== undefined) {
        take(q, "backend.queues.address_queue", "shared by loads and stores; modelled as a separate pool of this size each");
      } else {
        cfg[q] = UNBOUNDED;
        prov[q] = { from: "absent -> unbounded", provenance: null };
        assume.push(`backend.${key} absent: modelled as unbounded`);
      }
    }
  }
  if (prov.lq && prov.lq.from === "backend.queues.address_queue") {
    assume.push("the address queue is shared by loads and stores; the toy models a separate LQ and SQ " +
      "of that size each, so only single-filler probes are faithful");
  }

  if (take("width", "backend.rename_width") === null) throw new UnsupportedSpec("spec has no backend.rename_width");
  assume.push("one width for dispatch, issue and retire (toy semantics); issue ports and " +
    "scheduler queues are not modelled");
  const l1 = at(spec, "memory.l1d.latency_cycles");
  if (l1 !== null) {
    cfg.load_latency = Math.trunc(l1);
    prov.load_latency = { from: "memory.l1d.latency_cycles", provenance: at(spec, "memory.l1d.provenance") };
  }
  take("miss_latency", "memory.dram_latency");

  for (const [k, v] of Object.entries(overrides)) {
    cfg[k] = v;
    prov[k] = { from: "override", provenance: null };
  }
  if (!("miss_latency" in cfg)) {
    throw new UnsupportedSpec("spec has no memory.dram_latency: pass miss_latency as an override");
  }
  for (const k of TOY_FIELDS) {
    if (!(k in cfg)) { cfg[k] = TOY_DEFAULTS[k]; prov[k] = { from: "toy default", provenance: null }; }
  }
  if (cfg.miss_latency_jitter) throw new UnsupportedSpec("miss_latency_jitter is not deterministic across engines");
  const ordered = {}, oprov = {};
  for (const k of TOY_FIELDS) { ordered[k] = cfg[k]; oprov[k] = prov[k]; }
  return { cfg: ordered, prov: oprov, assumptions: assume };
}
