# Continuous Einsums

Einsums over piecewise-constant *continuous* tensors: COO pieces indexed by
real-valued pinpoints and intervals, where shared indices combine when their
coordinates **intersect** rather than being equal, and contracting an
all-interval index computes an integral (each contribution weighted by its
overlap length).

The evaluation follows the format-aware **mask → product → merge** strategy of
the thesis chapter:

1. **Mask** — a sparse join over the operands' piece positions; entry
   `(a, b, ...)` exists iff the pieces intersect on every shared index, and
   carries the intersection lengths of the reduced interval indices.
2. **Product** — per mask entry, the candidate value (operand values × mask
   measure) and its output coordinates (max of starts / min of ends gathered
   through the mask positions).
3. **Merge** — one function resolves every collision: candidates are mapped
   to integer rank space (pinpoint = one cell, interval = a run of cells),
   identical coordinates are summed by packed-key grouping, and partial
   interval overlaps are cut via a sweep-line (1-D) or per-dimension
   cutting (N-D) before a final grouped sum.

Mask creation is pluggable, with the chapter's three strategies implemented:
dense broadcast comparison, binary search via `torch.searchsorted` (the
default), and a Polars database join.

## Layout

```
src/
  ctensor.py             ContinuousTensor: COO pieces + per-dim property codes
  continuous_einsum.py   public API: ceinsum(equation, *operands)
  ceinsum_equation.py    einsum-string parsing
  ceinsum_mask.py        mask creation: conditions + integral measure (MV)
  ceinsum_product.py     per-tuple candidate values and output coordinates
  ceinsum_merge.py       unified rank-space merge (dedup, sweep, N-D cut)
  mask_dense.py          mask builder: brute-force N-D boolean table
  mask_binary_search.py  mask builder: searchsorted lead + post-filter
  mask_db_join.py        mask builder: polars database join (optional dep)
  table_ceinsum.py       dense interaction-table einsum (thesis-chapter
                         specification reference; same integral semantics)
  synth_dataset.py       non-overlapping ND box generator for tests
tests/
  test_ceinsum.py        hand-checked pipeline cases (incl. the manuscript's
                         worked examples)
  test_merge.py          merge unit tests (sweep, N-D boxes, pinpoint groups)
  test_table_ceinsum.py  table reference + ceinsum agreement
  test_mapping.py        mask-builder correctness + benchmark cases (Polars)
  conftest.py            CLI options & fixtures
benchmarks/
  bench_table_ceinsum.py table vs pipeline timing -> docs/table_ceinsum_bench.md
gui/      interactive Dash visualizer (deployed via wsgi.py / Procfile)
docs/     write-ups: walkthrough.md (theory), experiment_*.md, bench results
```

## Running

```bash
pytest                      # correctness tests
pytest --mapping-bench      # include the timing benchmark
python benchmarks/bench_table_ceinsum.py   # table-einsum benchmark
                                           # -> docs/table_ceinsum_bench.md
python gui/app.py           # interactive visualizer at 127.0.0.1:8050
```

Useful options (see `tests/conftest.py`): `--mapping-n`, `--mapping-skew`,
`--device-mode {gpu,cpu-single,cpu-multi}`. Tests need `pytest`;
`test_mapping.py` additionally needs `polars` (skipped when absent).
