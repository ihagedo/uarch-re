"""Write a core spec YAML from analysis estimates.

Every field carries provenance (measured / documented / inferred / speculative); measured fields add
a CI, `ci_method` and the evidence path. A measured field with an ambiguous or anomaly finding gets
`value: null`, a `bound` and the rule ids in `anomalies`; info findings keep the value. Unknown = null.
"""

import yaml

from .analysis.anomaly import SEVERITY

FIELD_MAP = {
    "rob": ("backend", "rob_entries"),
    "int_prf_headroom": ("backend", "int_prf_headroom"),
    "fp_prf_headroom": ("backend", "fp_prf_headroom"),
    "lq": ("backend", "load_queue"),
    "sq": ("backend", "store_queue"),
}


def field(
    value=None,
    provenance="speculative",
    ci=None,
    evidence=None,
    note=None,
    *,
    ci_method=None,
    ci_level=None,
    n_reps=None,
    resolution=None,
    offset=None,
    bound=None,
    anomalies=None,
):
    d = {"value": value, "provenance": provenance}
    if ci is not None:
        d["ci"] = ci
    for k, v in (
        ("ci_method", ci_method),
        ("ci_level", ci_level),
        ("n_reps", n_reps),
        ("resolution", resolution),
        ("offset", offset),
        ("bound", bound),
        ("anomalies", anomalies),
    ):
        if v:
            d[k] = v
    if evidence:
        d["evidence"] = evidence
    if note:
        d["note"] = note
    return d


def spec_from_estimates(core, isa, estimates, evidence, documented=None):
    spec = {"core": core, "isa": isa, "frontend": {}, "decoupling": {}, "backend": {}, "memory": {}}
    for key, (sect, name) in FIELD_MAP.items():
        e = estimates.get(key)
        if e is None or e["value"] is None:
            spec[sect][name] = field()
            continue
        common = dict(
            ci_method=e.get("ci_method", "bracket"),
            ci_level=e.get("ci_level"),
            n_reps=e.get("n_reps"),
            resolution=e.get("resolution"),
            offset=e.get("offset"),
            anomalies=e.get("anomalies") or None,
        )
        flagged = [a for a in e.get("anomalies", []) if SEVERITY.get(a, "anomaly") != "info"]
        if not flagged:
            spec[sect][name] = field(e["value"], "measured", e["ci"], evidence, **common)
            continue
        # the probe's number survives only as a bound (the rule's own bound when it set one)
        bound = e.get("bound") or (
            {"lower": e["value"]} if flagged == ["tie.rob"] else {"lower": e["ci"][0], "upper": e["ci"][1]}
        )
        bound = {
            k: float(v)
            for k, v in (bound.items() if isinstance(bound, dict) else zip(("lower", "upper"), bound))
        }
        spec[sect][name] = field(
            None,
            "measured",
            evidence=evidence,
            note=e["status"],
            bound=bound,
            **{**common, "ci_method": None},
        )
    for key, v in (documented or {}).items():
        sect, name = FIELD_MAP.get(key, ("backend", key))
        spec[sect].setdefault(name, field())  # unmeasured here: null gap, truth alongside
        spec[sect][name]["documented_value"] = v
    return spec


def write_spec(path, spec):
    with open(path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
