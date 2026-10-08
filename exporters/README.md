# exporters

Spec → model generators.

- `gem5_from_spec.py`: spec YAML → gem5 O3 parameter overrides. The P6.1
  round trip (`validation/gem5/p61_roundtrip.py`) measures, emits a spec,
  regenerates the config from it and measures again.

Exporters honour `model` fields (e.g. `coalesced_retire_groups`) rather than
forcing every core into a classic ROB shape, and refuse `null` fields rather
than fill them in.
