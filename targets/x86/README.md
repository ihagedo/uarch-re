# targets/x86

Checks on an x86 host against Intel's published sizes (Raptor Cove P-core,
Gracemont E-core).

| file | does |
| --- | --- |
| `window_campaign.py` | window probe with launch hygiene: pinning, SMT-sibling and drift checks |
| `micro_campaign.py` | instruction latency and throughput, rename width, fusion |
| `mem_native.py` | memory probes on one pinned P-core |
| `footprint.py` | whether the probe's chains miss to DRAM |
| `hygiene.py` | preemption, context-switch and level-drift checks |
| `published.py` | the published sizes |
| `runs/` | the campaigns as run on an i5-14600 (CPU numbers are that host's) |
