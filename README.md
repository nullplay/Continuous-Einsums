# Continuous Einsums

Experiments on mapping pipelines for continuous (interval/pinpoint) einsum-style
operations, comparing a brute-force boolean-table backend against an optimized
`searchsorted`-based backend and a Polars backend.

## Layout

```
src/      mapping builders + data synthesis
  table_mapping.py       brute-force N-D boolean table backend
  table_opt_mapping.py   optimized searchsorted backend
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
```

Useful options (see `tests/conftest.py`): `--mapping-n`, `--mapping-skew`,
`--device-mode {gpu,cpu-single,cpu-multi}`.
