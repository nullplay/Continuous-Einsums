# `table_ceinsum` benchmark

Generated 2026-07-13 by `benchmarks/bench_table_ceinsum.py` — torch 2.9.0, arm64 CPU, 8 threads, torch.float32, median of 5 repeats (1 warmup). Table sizes above 2e+08 elements are skipped.

`table` times include the flatten to a COO piece list, so both columns produce the same kind of object. `max rel Δ` is the pointwise value disagreement between the two implementations at probe points.

## `i,i->i` (interval × interval)

| n | table shape | table MB | table (ms) | ceinsum (ms) | ratio | max rel Δ |
|---:|---|---:|---:|---:|---:|---:|
| 25 | 99×25×25 | 0.2 | 0.32 | 0.24 | 1.3× | 0.0e+00 |
| 50 | 199×50×50 | 2.0 | 0.62 | 0.23 | 2.7× | 0.0e+00 |
| 100 | 399×100×100 | 16.0 | 2.44 | 0.25 | 9.7× | 0.0e+00 |
| 200 | 799×200×200 | 127.8 | 21.92 | 0.34 | 64.6× | 0.0e+00 |
| 400 | 1599×400×400 | 1,023 | skipped (> guard) | 0.46 | — | — |
| 800 | 3198×800×800 | 8,187 | skipped (> guard) | 0.65 | — | — |

## `ij,j->i` (interval-i, pinpoint-j)

| n | table shape | table MB | table (ms) | ceinsum (ms) | ratio | max rel Δ |
|---:|---|---:|---:|---:|---:|---:|
| 25 | 49×25×25 | 0.1 | 0.12 | 0.43 | 0.3× | 1.3e-07 |
| 50 | 99×50×50 | 1.0 | 0.42 | 0.80 | 0.5× | 1.6e-07 |
| 100 | 199×100×100 | 8.0 | 2.18 | 1.49 | 1.5× | 1.9e-07 |
| 200 | 399×200×200 | 63.8 | 10.70 | 2.63 | 4.1× | 2.0e-07 |
| 400 | 799×400×400 | 511.4 | 70.30 | 5.12 | 13.7× | 1.9e-07 |
| 800 | 1599×800×800 | 4,093 | skipped (> guard) | 9.46 | — | — |

## `ik,kj->ij` (A: P,I × B: I,I — integrated k)

| n | table shape | table MB | table (ms) | ceinsum (ms) | ratio | max rel Δ |
|---:|---|---:|---:|---:|---:|---:|
| 25 | 5×49×25×25 | 0.6 | 0.42 | 0.46 | 0.9× | 3.0e-08 |
| 50 | 7×99×50×50 | 6.9 | 1.95 | 0.47 | 4.1× | 1.5e-08 |
| 100 | 10×199×100×100 | 79.6 | 17.77 | 0.64 | 28.0× | 3.7e-09 |
| 200 | 14×399×200×200 | 894 | skipped (> guard) | 0.71 | — | — |
| 400 | 20×799×400×400 | 10,227 | skipped (> guard) | 1.21 | — | — |
| 800 | 28×1599×800×800 | 114,616 | skipped (> guard) | 2.16 | — | — |

## Observations

- The table is dense: its element count is (candidate counts) × (product of operand piece counts), so time and memory grow with the *product* of the piece counts — the size guard trips exactly where that product-scaling wall predicts. This is the expected cost of the specification-level implementation; the piece-join pipeline (`continuous_einsum.ceinsum`) exists to avoid it.
- At small piece counts the dense table can be *faster* than the optimized pipeline: a handful of broadcast comparisons plus one einsum has less fixed overhead than the multi-step join. The product-scaling cost only dominates once the piece counts grow.
- The two implementations share integral semantics (a contracted all-interval variable is weighted by its overlap length) and agree pointwise to float32 precision on every case (`max rel Δ` column).
