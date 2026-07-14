# Continuous Einsums

Experiments on mapping pipelines for continuous (interval/pinpoint) einsum-style
operations, comparing a brute-force boolean-table backend against an optimized
`searchsorted`-based backend and a Polars backend.

## Layout

```
src/      mapping builders + data synthesis
  mask_dense.py          brute-force N-D boolean table backend
  mask_binary_search.py  optimized searchsorted backend
  table_ceinsum.py       dense interaction-table einsum (thesis-chapter
                         reference; integrates all-interval contractions)
  synth_dataset.py       non-overlapping ND box generator for tests
tests/    pytest suite
  test_mapping.py        correctness + benchmark cases
  conftest.py            CLI options & fixtures
docs/     experiment write-ups (experiment_cpu.md, experiment_gpu.md)
```

## Running

```bash
pytest                      # correctness tests
pytest --mapping-bench      # include the timing benchmark
python benchmarks/bench_table_ceinsum.py   # table-einsum benchmark
                                           # -> docs/table_ceinsum_bench.md
```

Useful options (see `tests/conftest.py`): `--mapping-n`, `--mapping-skew`,
`--device-mode {gpu,cpu-single,cpu-multi}`.
