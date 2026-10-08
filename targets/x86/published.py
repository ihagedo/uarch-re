"""Published structure sizes for the host's two core types, the truth for native x86 runs.

Keys match attribute()'s structure names. Register-file values are rename headroom
(physical minus the architectural registers the source assumed), which is what the
int/fp fillers probe. DOC = Intel stated it; MEAS = Chips and Cheese measured it with
a Henry Wong-style window probe, the same method as ours.
"""

ORM = "Intel 64 and IA-32 Architectures Optimization Reference Manual"
AD21 = "Intel Architecture Day 2021 presentation"
CNC_GLC = "Chips and Cheese, Popping the Hood on Golden Cove"
CNC_VRF = "Chips and Cheese, Golden Cove's Lopsided Vector Register File"
CNC_GRT = "Chips and Cheese, Gracemont: Revenge of the Atom Cores"

TRUTH = {
    "p": {"rob": 512, "lq": 192, "sq": 114, "int_prf_headroom": 248, "fp_prf_headroom": 300, "bob": 128},
    "e": {"rob": 256, "lq": 80, "sq": 50, "int_prf_headroom": 198, "fp_prf_headroom": 191, "bob": 126},
}
SOURCES = {
    "p": {
        "rob": ("DOC", f"{AD21}: '512-entry Reorder-Buffer'"),
        "lq": ("MEAS", f"{CNC_GLC} (Intel: 'deeper load buffer', no number)"),
        "sq": ("MEAS", f"{CNC_GLC} (Intel: 'deeper store buffer', no number)"),
        "int_prf_headroom": ("MEAS", f"{CNC_GLC}: int PRF 280 = ~248 renames + 32 assumed architectural"),
        "fp_prf_headroom": (
            "MEAS",
            f"{CNC_GLC}: FP PRF 332 = 300 renames + 32; {CNC_VRF} revises to ~295 renames for 256-bit writes",
        ),
        "bob": ("MEAS", f"{CNC_GLC}: branch order buffer 128"),
        "alloc_width": ("DOC", f"{ORM}: allocation 5 -> 6 uops per cycle"),
        "retire_width": ("DOC", f"{ORM}: 4 -> 8 wide retirement"),
    },
    "e": {
        "rob": ("DOC", f"{ORM}: '256-entry retirement buffer'; {AD21}"),
        "lq": ("MEAS", f"{CNC_GRT} (Intel: 'larger load and store buffers', no number)"),
        "sq": ("MEAS", CNC_GRT),
        "int_prf_headroom": ("MEAS", f"{CNC_GRT}: int PRF 214 = 198 renames + 16"),
        "fp_prf_headroom": ("MEAS", f"{CNC_GRT}: vector PRF 207 = 191 renames + 16 (128-bit entries)"),
        "bob": ("MEAS", f"{CNC_GRT}: 116 taken / 126 not-taken branches in flight"),
        "alloc_width": ("DOC", f"{ORM}: up to five uops per cycle"),
        "retire_width": ("DOC", f"{ORM}: up to eight instructions per cycle"),
    },
}


def truth(core):
    return dict(TRUTH.get(core, {}))


def documented(core, key):
    return SOURCES.get(core, {}).get(key, ("", ""))[0] == "DOC"
