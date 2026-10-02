# spec

- `schema/core.schema.json`: one core description with sections `frontend`,
  `decoupling`, `backend`, `memory`. Every field has a `provenance` of
  measured / documented / inferred / speculative. Measured fields need
  `evidence` (and `ci` when they carry a value); documented and inferred fields
  must cite ids from `sources.yaml`. `null` is a gap, never a guess.
- `cores/`: hand-maintained descriptions. `firestorm.yaml` (flagship, mostly
  `inferred` from published reverse engineering), `r10000.yaml` (teaching
  baseline, `documented` from Yeager 1996).
- Specs emitted by sweeps land in `results/<platform>/<tag>/spec.yaml` and
  use the same schema.

Validate everything with `tools/validate_specs.py`.
