#!/usr/bin/env python3
"""Validate core spec YAML files against spec/schema/core.schema.json and spec/sources.yaml.

A measured value without `ci_method` is read as `bracket` with a warning; --strict makes it an error.

  tools/validate_specs.py                 # spec/cores/*.yaml and results/**/spec.yaml
  tools/validate_specs.py path/a.yaml ... # just these files
  tools/validate_specs.py --strict ...    # no migration of old specs
"""

import glob
import json
import os
import sys

import jsonschema
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA = os.path.join(ROOT, "spec", "schema", "core.schema.json")
SOURCES = os.path.join(ROOT, "spec", "sources.yaml")


def cited_sources(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "sources" and isinstance(v, list):
                yield from v
            else:
                yield from cited_sources(v)
    elif isinstance(node, list):
        for v in node:
            yield from cited_sources(v)


def migrate(node, path=()):
    """Fill `ci_method: bracket` into measured values that lack it; yield the paths changed."""
    if isinstance(node, dict):
        if (
            node.get("provenance") == "measured"
            and isinstance(node.get("value"), (int, float))
            and "ci" in node
            and "ci_method" not in node
        ):
            node["ci_method"] = "bracket"
            yield "/".join(path)
        for k, v in node.items():
            yield from migrate(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from migrate(v, path + (str(i),))


def validate(path, validator, known, strict=False):
    """Return (errors, warnings)."""
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    warnings = [] if strict else [f"{p}: no ci_method, read as bracket" for p in migrate(spec)]
    errors = [
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(spec)
    ]
    errors += [f"unknown source id '{s}'" for s in sorted(set(cited_sources(spec))) if s not in known]
    return errors, warnings


def main(argv):
    strict = "--strict" in argv
    argv = [a for a in argv if a != "--strict"]
    paths = argv or sorted(
        glob.glob(os.path.join(ROOT, "spec", "cores", "*.yaml"))
        + glob.glob(os.path.join(ROOT, "results", "**", "spec.yaml"), recursive=True)
    )
    with open(SCHEMA) as fh:
        validator = jsonschema.Draft202012Validator(json.load(fh))
    with open(SOURCES) as fh:
        known = set(yaml.safe_load(fh))

    failed = 0
    for p in paths:
        errors, warnings = validate(p, validator, known, strict)
        rel = os.path.relpath(p, ROOT)
        if errors:
            failed += 1
            print(f"FAIL {rel}")
            for e in errors:
                print(f"     {e}")
        else:
            print(f"{'warn' if warnings else 'ok  '} {rel}")
        for w in warnings:
            print(f"     warning: {w}")
    print(f"{len(paths) - failed}/{len(paths)} valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
