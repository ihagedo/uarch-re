# Experiments

Each PoC is one measurement, checked on cores where the answer is known (toy
model, gem5, BOOM RTL, x86 with published sizes) before it is used on the M1.
Pass means within max(2 entries, 2%) of the configured value unless noted.

| PoC | Measures | Checked on | Status |
| --- | --- | --- | --- |
| P0.1 | knee-to-size offsets | toy, gem5 | pass |
| P0.2 | knee confidence intervals | synthetic curves | pass |
| P0.3 | anomaly rules A1-A12 | synthetic, toy, gem5 | pass |
| P0.4 | measurement hygiene on a hybrid CPU | x86 | partial |
| P0.5 | window probe on RTL | BOOM | pass |
| P0.6 | AArch64 emitter and macOS driver | gem5 ARM | partial: driver untested on a Mac |
| P1.1 | ROB | gem5, BOOM | pass |
| P1.2 | int, FP and flags register headroom | gem5 RISC-V, gem5 ARM | pass |
| P1.3 | load and store queues | gem5 | pass |
| P1.4 | near-ties and shared pools | gem5 | pass |
| P1.5 | scheduler queues and queue sharing | gem5 | pass |
| P1.6 | rename width, move elimination, zero idioms | x86 | pass |
| P1.7 | retire width | gem5 | pass |
| P1.8 | instruction latency and throughput | gem5, x86 | pass |
| P2.1 | cache capacity, line size, associativity, latency | gem5 | pass |
| P2.2 | TLB sizes and page-walk cost | gem5 full-system | pass |
| P2.3 | memory-level parallelism (MSHRs) | gem5 | pass |
| P2.4 | prefetcher detection | gem5, x86 | pass |
| P2.5 | store-to-load forwarding, store sets | gem5 | pass |
| P3.1 | fetch and decode width | gem5 | pass |
| P3.2 | L1i capacity and latency | gem5 | pass |
| P3.3 | BTB capacity, taken-branch bubble | gem5 | pass |
| P3.4 | branch history length | gem5 | pass |
| P3.5 | indirect predictor capacity | gem5 | partial: intervals too narrow |
| P3.6 | return address stack depth | gem5 | pass |
| P3.7 | mispredict penalty | gem5 | partial |
| P3.8 | decoupled fetch and fetch-directed prefetch | gem5 | control |
| P3.9 | fetch queue depth | gem5 | pass |
| P3.10 | loop buffer | | blocked: gem5 has none |
| P4.1 | coalesced retire groups (rule A4) | toy, gem5 | pass |
| P4.2 | LSQ allocated at issue (rule A5) | toy, gem5 | pass |
| P4.3 | load address and value prediction | toy | pass |
| P4.4 | two-level store queue (rule A1) | toy | pass |
| P4.5 | fusion and elimination vs retire structures | x86 | pass |
| P5.1 | window structures vs published sizes | x86 | partial: x86 offsets uncalibrated |
| P5.2 | published Firestorm numbers | M1 | blocked: needs an M1 |
| P5.3 | new Firestorm measurements | M1 | blocked: needs an M1 |
| P5.4 | Icestorm (E-core) | M1 | blocked: needs an M1 |
| P5.5 | later Apple cores | M2+ | blocked |
| P6.1 | measure, export a gem5 config, measure again | gem5 | pass |
| P6.2 | llvm-mca export | | not started |
| P6.3 | factory model matches the toy model | JS engine | partial |
| P6.4 | register renaming lab | toy | pass |
| P6.5 | settle published disagreements | M1 | blocked: needs an M1 |
| P6.6 | harness others can run on their Macs | M1+ | blocked |
