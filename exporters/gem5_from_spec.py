#!/usr/bin/env python3
"""Spec YAML -> gem5 RiscvO3CPU parameter overrides for the backend fields in FIELDS (P6.1).

A null field is refused unless --allow-default names it, and so is a non-classic `model`
(gem5 O3 has no such structure).

  exporters/gem5_from_spec.py results/gem5/<tag>/spec.yaml            # prints k=v,...
  exporters/gem5_from_spec.py spec.yaml --json
"""

import argparse
import json
import sys

import yaml

FIELDS = {
    "rob_entries": "numROBEntries",
    "int_prf_headroom": "numPhysIntRegs",
    "fp_prf_headroom": "numPhysFloatRegs",
    "load_queue": "LQEntries",
    "store_queue": "SQEntries",
}
ARCH = {"numPhysIntRegs": 32, "numPhysFloatRegs": 32}  # RISC-V architectural registers


class ExportError(ValueError):
    pass


def export(spec, allow_default=()):
    be = spec.get("backend", {})
    if any(isinstance(v, dict) and v.get("model") for v in be.values()):
        raise ExportError("spec declares a non-classic model; gem5 O3 cannot express it")
    out = {}
    for field, param in FIELDS.items():
        f = be.get(field)
        v = f.get("value") if isinstance(f, dict) else None
        if v is None:
            if field in allow_default:
                continue
            raise ExportError(
                f"backend.{field} has no value ({(f or {}).get('note') or 'unmeasured'}); "
                f"pass --allow-default {field} to leave gem5's default"
            )
        out[param] = int(round(v)) + ARCH.get(param, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--allow-default", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    try:
        params = export(spec, args.allow_default)
    except ExportError as err:
        sys.exit(f"refused: {err}")
    print(json.dumps(params) if args.json else ",".join(f"{k}={v}" for k, v in params.items()))


if __name__ == "__main__":
    main()
