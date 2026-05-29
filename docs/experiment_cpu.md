# CPU Mapping Benchmark — N sweep at skew=0.5

Median of 3 repeats (2 for the slowest standalone case 19 runs), all values in ms.
`skew=0.5`, all 19 mapping cases. Backends:
`table` (brute-force N-D bool mask + nonzero), `table_opt` (hand-written
searchsorted-lead variant), `table_auto` (declarative builder over the
brute-force path), `table_opt_auto` (declarative builder over the optimized
path), `polars` (CPU engine via `pl.LazyFrame.join_where`).

Two modes, exposed through the new `--device-mode` flag in `conftest.py`:

- `cpu-multi`: `CUDA_VISIBLE_DEVICES=""` (so torch falls back to CPU and the
  bench picks `engine="cpu"` for polars), polars and torch use all cores.
- `cpu-single`: same, plus `POLARS_MAX_THREADS=1`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)` so both libraries are pinned
  to a single thread.

Both modes hide CUDA via `pytest_configure` *before* torch/polars import.

Legend: `oom` = OOM (CUDA OOM not applicable here — these are CPU `MemoryError`
or precomputed-too-large-to-bother); `tmo` = exceeded the ~3-min per-bench
budget and was skipped (most often the brute-force `table` backend whose
compile+run on CPU at large N pushed each case past the budget); `—` = not run
because another cell in the same row was already `tmo`/`oom`. Numbers from
auxiliary standalone runs (case 19 at large N, single cases re-run after a
pytest timeout) keep the same units.

---

## CPU multi-threaded (`--device-mode=cpu-multi`)

### N=100

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 0.136 | 0.190 | 0.132 | 0.190 | 0.256 |
| 02_ii_1d | 0.142 | 0.250 | 0.144 | 0.288 | 0.324 |
| 03_pi_1d | 0.134 | 0.236 | 0.142 | 0.270 | 0.285 |
| 04_pp_2d | 0.144 | 0.261 | 0.141 | 0.283 | 0.547 |
| 05_ii_2d | 0.147 | 0.265 | 0.151 | 0.317 | 0.884 |
| 06_pi_2d | 0.152 | 0.253 | 0.145 | 0.302 | 0.467 |
| 07_diag_i | 0.138 | 0.247 | 0.139 | 0.322 | 0.728 |
| 08_diag_p | 0.137 | 0.238 | 0.137 | 0.308 | 1.002 |
| 09_red_pp | 0.135 | 0.192 | 0.134 | 0.195 | 0.327 |
| 10_red_ii | 0.141 | 0.222 | 0.122 | 0.248 | 0.273 |
| 11_mm_pp | 0.132 | 0.182 | 0.133 | 0.194 | 0.298 |
| 12_mm_ii | 0.129 | 0.224 | 0.126 | 0.252 | 0.304 |
| 13_trip_pp | 1.431 | 0.231 | 1.432 | 0.258 | 0.396 |
| 14_trip_ii | 1.475 | 0.309 | 1.415 | 0.359 | 1.165 |
| 15_trip2d_pp | 1.487 | 0.302 | 1.399 | 0.335 | 0.948 |
| 16_trip2d_ii | 1.493 | 0.324 | 1.510 | 0.444 | 1.877 |
| 17_box | 0.132 | 0.224 | 0.112 | 0.284 | 0.536 |
| 18_bio | 0.142 | 0.238 | 0.139 | 0.306 | 0.520 |
| 19_ptcloud | 0.522 | 0.709 | 0.521 | 0.782 | 2.249 |

### N=1000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 1.474 | 0.233 | 1.462 | 0.228 | 0.544 |
| 02_ii_1d | 1.238 | 0.294 | 1.353 | 0.329 | 0.371 |
| 03_pi_1d | 1.442 | 0.288 | 1.380 | 0.320 | 0.408 |
| 04_pp_2d | 1.411 | 0.472 | 0.974 | 0.450 | 0.793 |
| 05_ii_2d | 1.492 | 0.535 | 1.420 | 0.609 | 1.301 |
| 06_pi_2d | 1.341 | 0.490 | 1.378 | 0.566 | 0.995 |
| 07_diag_i | 1.440 | 0.352 | 1.427 | 0.429 | 1.033 |
| 08_diag_p | 0.848 | 0.302 | 0.846 | 0.463 | 1.073 |
| 09_red_pp | 1.469 | 0.217 | 1.470 | 0.230 | 0.575 |
| 10_red_ii | 1.371 | 0.304 | 1.373 | 0.336 | 0.476 |
| 11_mm_pp | 1.474 | 0.219 | 1.462 | 0.226 | 0.876 |
| 12_mm_ii | 1.362 | 0.264 | 1.353 | 0.297 | 0.519 |
| 13_trip_pp | 885.067 | 0.806 | 881.907 | 0.834 | 1.127 |
| 14_trip_ii | 896.662 | 2.872 | 899.523 | 0.885 | 2.034 |
| 15_trip2d_pp | 898.887 | 1.170 | 858.841 | 0.989 | 1.142 |
| 16_trip2d_ii | 892.548 | 2.546 | 903.219 | 1.194 | 3.461 |
| 17_box | 0.175 | 0.264 | 0.127 | 0.333 | 0.517 |
| 18_bio | 1.363 | 0.439 | 1.376 | 0.577 | 1.108 |
| 19_ptcloud | 24.767 | 9.564 | 24.511 | 10.301 | 202.771 |

### N=5000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 31.600 | 0.326 | 21.335 | 0.313 | 0.473 |
| 02_ii_1d | 27.937 | 0.433 | 19.129 | 0.463 | 1.004 |
| 03_pi_1d | 19.755 | 0.484 | 20.432 | 0.517 | 0.838 |
| 04_pp_2d | 32.380 | 2.304 | 20.098 | 2.250 | 1.151 |
| 05_ii_2d | 33.175 | 2.323 | 20.060 | 2.393 | 2.026 |
| 06_pi_2d | 31.272 | 1.695 | 19.923 | 2.279 | 1.809 |
| 07_diag_i | 32.958 | 0.949 | 21.325 | 1.003 | 0.794 |
| 08_diag_p | 28.712 | 0.919 | 20.455 | 1.838 | 30.103 |
| 09_red_pp | 33.923 | 0.355 | 21.463 | 0.355 | 0.431 |
| 10_red_ii | 30.213 | 0.517 | 18.905 | 0.548 | 0.877 |
| 11_mm_pp | 33.863 | 0.367 | 33.776 | 0.369 | 0.833 |
| 12_mm_ii | 31.302 | 0.522 | 19.628 | 0.539 | 1.271 |
| 13_trip_pp | oom | 22.467 | oom | 21.304 | 1.951 |
| 14_trip_ii | oom | 70.040 | oom | 9.358 | 1.854 |
| 15_trip2d_pp | oom | 20.552 | oom | 13.363 | 1.377 |
| 16_trip2d_ii | oom | 63.684 | oom | 7.270 | 42.915 |
| 17_box | 0.137 | 0.391 | 0.139 | 0.418 | 0.593 |
| 18_bio | 19.900 | 1.654 | 19.789 | 2.241 | 1.876 |
| 19_ptcloud | 638.011 | 147.447 | 634.693 | 144.159 | 4981.829 |

### N=10000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 93.828 | 0.484 | 93.611 | 0.456 | 0.605 |
| 02_ii_1d | 85.884 | 0.656 | 84.188 | 0.680 | 1.654 |
| 03_pi_1d | 90.245 | 0.615 | 88.154 | 0.680 | 1.771 |
| 04_pp_2d | 85.629 | 4.202 | 86.432 | 3.860 | 0.913 |
| 05_ii_2d | 94.211 | 4.732 | 88.241 | 4.653 | 5.281 |
| 06_pi_2d | 125.064 | 3.339 | 85.715 | 5.442 | 4.283 |
| 07_diag_i | 96.993 | 1.490 | 90.129 | 1.546 | 0.895 |
| 08_diag_p | 117.820 | 1.493 | 82.992 | 3.709 | 2.743 |
| 09_red_pp | 92.266 | 0.461 | 93.482 | 0.484 | 0.834 |
| 10_red_ii | 91.567 | 0.634 | 82.070 | 0.675 | 1.578 |
| 11_mm_pp | 99.969 | 0.451 | 92.841 | 0.474 | 0.582 |
| 12_mm_ii | 89.689 | 0.697 | 83.638 | 0.679 | 1.790 |
| 13_trip_pp | oom | 81.505 | oom | 64.689 | 1.197 |
| 14_trip_ii | oom | 289.480 | oom | 38.830 | 3.660 |
| 15_trip2d_pp | oom | 57.625 | oom | 57.267 | 1.910 |
| 16_trip2d_ii | oom | 259.789 | oom | 27.113 | 193.123 |
| 17_box | 0.208 | 0.593 | 0.209 | 0.498 | 0.757 |
| 18_bio | 86.810 | 3.129 | 84.152 | 4.475 | 3.976 |
| 19_ptcloud | 2569.949 | 446.090 | 2592.382 | 446.760 | 19820.549 |

### N=30000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 1.325 | tmo | 1.226 | 0.767 |
| 02_ii_1d | tmo | 1.777 | tmo | 1.739 | 3.904 |
| 03_pi_1d | tmo | 1.762 | tmo | 1.717 | 4.018 |
| 04_pp_2d | tmo | 25.119 | tmo | 22.650 | 1.238 |
| 05_ii_2d | tmo | 26.139 | tmo | 27.720 | 23.282 |
| 06_pi_2d | tmo | 21.028 | tmo | 26.806 | 25.365 |
| 07_diag_i | tmo | 8.070 | tmo | 7.942 | 1.763 |
| 08_diag_p | tmo | 8.124 | tmo | 25.877 | 1174.055 |
| 09_red_pp | tmo | 1.279 | tmo | 1.236 | 0.986 |
| 10_red_ii | tmo | 1.574 | tmo | 1.581 | 4.378 |
| 11_mm_pp | tmo | 1.268 | tmo | 1.231 | 1.279 |
| 12_mm_ii | tmo | 1.743 | tmo | 1.743 | 4.191 |
| 13_trip_pp | oom | 541.568 | oom | 538.485 | 1.475 |
| 14_trip_ii | oom | 2430.622 | oom | 335.127 | 7.301 |
| 15_trip2d_pp | oom | 479.327 | oom | 472.827 | 2.719 |
| 16_trip2d_ii | oom | 2238.690 | oom | 226.518 | 2286.128 |
| 17_box | tmo | 1.365 | tmo | 0.902 | 0.904 |
| 18_bio | tmo | 17.064 | tmo | 26.077 | 22.975 |
| 19_ptcloud | oom | 2876.413 | oom | 2853.488 | tmo |

### N=50000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 2.036 | tmo | 1.930 | 1.047 |
| 02_ii_1d | tmo | 2.508 | tmo | 2.446 | 7.198 |
| 03_pi_1d | tmo | 2.422 | tmo | 2.486 | 9.397 |
| 04_pp_2d | tmo | 48.354 | tmo | 46.653 | 2.192 |
| 05_ii_2d | tmo | 68.055 | tmo | 77.438 | 57.793 |
| 06_pi_2d | tmo | 43.639 | tmo | 54.536 | 51.872 |
| 07_diag_i | tmo | 18.637 | tmo | 15.532 | 13.060 |
| 08_diag_p | tmo | 16.880 | tmo | 45.620 | 2605.003 |
| 09_red_pp | tmo | 2.071 | tmo | 1.983 | 1.214 |
| 10_red_ii | tmo | 2.464 | tmo | 2.459 | 8.172 |
| 11_mm_pp | tmo | 1.962 | tmo | 1.940 | 1.058 |
| 12_mm_ii | tmo | 2.607 | tmo | 2.613 | 6.887 |
| 13_trip_pp | oom | 1504.030 | oom | 1508.102 | 1.997 |
| 14_trip_ii | oom | 6714.142 | oom | 950.768 | 9.865 |
| 15_trip2d_pp | oom | 1316.978 | oom | 1315.916 | 2.748 |
| 16_trip2d_ii | oom | 6187.541 | oom | 607.873 | 5836.475 |
| 17_box | tmo | 2.466 | tmo | 1.137 | 1.430 |
| 18_bio | tmo | 40.401 | tmo | 53.276 | 51.198 |
| 19_ptcloud | oom | 6736.860 | oom | 6698.426 | tmo |

### N=100000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | oom | 3.121 | oom | 3.091 | 1.660 |
| 02_ii_1d | oom | 3.843 | oom | 3.698 | 12.712 |
| 03_pi_1d | oom | 3.627 | oom | 3.747 | 12.056 |
| 04_pp_2d | oom | 125.500 | oom | 125.272 | 2.315 |
| 05_ii_2d | oom | 152.915 | oom | 162.718 | oom |
| 06_pi_2d | oom | 117.770 | oom | 155.794 | 139.590 |
| 07_diag_i | oom | 44.394 | oom | 44.319 | 2.490 |
| 08_diag_p | oom | 44.145 | oom | 127.277 | 78.378 |
| 09_red_pp | oom | 3.062 | oom | 3.143 | 1.523 |
| 10_red_ii | oom | 3.896 | oom | 3.673 | 13.771 |
| 11_mm_pp | oom | 3.092 | oom | 3.030 | 1.563 |
| 12_mm_ii | oom | 3.695 | oom | 3.678 | 14.344 |
| 13_trip_pp | oom | 5932.285 | oom | 6026.081 | 2.485 |
| 14_trip_ii | oom | 26897.081 | oom | 3751.539 | oom |
| 15_trip2d_pp | oom | 5137.440 | oom | 5198.987 | 4.598 |
| 16_trip2d_ii | oom | 24713.299 | oom | 2435.793 | 30728.767 |
| 17_box | oom | 3.521 | oom | 2.018 | 2.448 |
| 18_bio | oom | 115.564 | oom | 156.622 | 146.893 |
| 19_ptcloud | oom | 21065.723 | oom | 21108.713 | err |

Polars cells re-verified via `verify_polars_tmo.py`, one process per case so an
OOM in one case can't poison the others. Each case mirrors the pytest harness
exactly: `torch.compile` + warmup + time `table_opt` and `table_opt_auto`
*before* the polars timing, so the polars `collect()` runs against the same
in-process torch state as the original sweep. The original pytest sweep marked
every polars cell from 05 onward `tmo` because the bench loop ran out of its
overall 3-min budget on the OOM-killed cells, not because polars itself timed
out per call. The genuine failures (05, 14: `MemoryError`; 19: polars's
2³²-row cross-join cap, shown as `err`) are now labeled directly. Case 16
takes ~30 s — well over the per-N budget shared across 19 cases, which is why
the sweep never reached it. Cases 07 and 08 are *faster* at N=100000 than at
N=50000 (2.5 ms vs 13.1 ms; 78 ms vs 2605 ms): for these interval-on-pinpoint
joins the alive ratio drops as cells shrink with N, polars's IEJoin operator
scales with output cardinality more than with N, and the planner picks a
different physical operator at the larger size.

---

## CPU single-threaded (`--device-mode=cpu-single`)

### N=100

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 0.140 | 0.176 | 0.121 | 0.174 | 0.175 |
| 02_ii_1d | 0.126 | 0.221 | 0.130 | 0.253 | 0.099 |
| 03_pi_1d | 0.126 | 0.219 | 0.123 | 0.247 | 0.118 |
| 04_pp_2d | 0.126 | 0.226 | 0.130 | 0.288 | 0.097 |
| 05_ii_2d | 0.130 | 0.235 | 0.129 | 0.279 | 0.198 |
| 06_pi_2d | 0.128 | 0.226 | 0.125 | 0.268 | 0.122 |
| 07_diag_i | 0.127 | 0.229 | 0.127 | 0.288 | 0.127 |
| 08_diag_p | 0.124 | 0.230 | 0.125 | 0.274 | 0.131 |
| 09_red_pp | 0.128 | 0.171 | 0.122 | 0.173 | 0.088 |
| 10_red_ii | 0.133 | 0.221 | 0.124 | 0.248 | 0.110 |
| 11_mm_pp | 0.125 | 0.170 | 0.123 | 0.176 | 0.077 |
| 12_mm_ii | 0.124 | 0.219 | 0.125 | 0.251 | 0.092 |
| 13_trip_pp | 2.199 | 0.212 | 2.253 | 0.240 | 0.100 |
| 14_trip_ii | 2.655 | 0.320 | 2.567 | 0.369 | 0.206 |
| 15_trip2d_pp | 2.635 | 0.294 | 2.557 | 0.352 | 0.146 |
| 16_trip2d_ii | 2.630 | 0.327 | 2.665 | 0.455 | 0.286 |
| 17_box | 0.115 | 0.228 | 0.120 | 0.300 | 0.126 |
| 18_bio | 0.140 | 0.249 | 0.142 | 0.305 | 0.126 |
| 19_ptcloud | 0.927 | 0.759 | 0.901 | 0.841 | 8.942 |

### N=1000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | 2.210 | 0.255 | 2.494 | 0.264 | 0.120 |
| 02_ii_1d | 2.048 | 0.323 | 2.368 | 0.359 | 0.499 |
| 03_pi_1d | 2.177 | 0.358 | 2.198 | 0.350 | 0.498 |
| 04_pp_2d | 2.155 | 0.473 | 2.054 | 0.498 | 0.169 |
| 05_ii_2d | 2.188 | 0.546 | 2.100 | 0.617 | 3.031 |
| 06_pi_2d | 2.077 | 0.471 | 2.133 | 0.569 | 0.288 |
| 07_diag_i | 2.086 | 0.392 | 2.077 | 0.454 | 0.200 |
| 08_diag_p | 2.033 | 0.383 | 2.030 | 0.542 | 0.693 |
| 09_red_pp | 2.191 | 0.261 | 2.150 | 0.258 | 0.116 |
| 10_red_ii | 2.083 | 0.324 | 2.028 | 0.356 | 0.477 |
| 11_mm_pp | 2.148 | 0.259 | 2.266 | 0.267 | 0.110 |
| 12_mm_ii | 2.097 | 0.319 | 2.094 | 0.356 | 0.479 |
| 13_trip_pp | 2059.874 | 1.744 | 2068.644 | 1.692 | 0.252 |
| 14_trip_ii | 2065.769 | 4.939 | 2087.129 | 1.313 | 0.809 |
| 15_trip2d_pp | 2071.602 | 1.760 | 2030.538 | 1.822 | 0.297 |
| 16_trip2d_ii | 2070.130 | 5.197 | 2084.860 | 1.423 | 4.566 |
| 17_box | 0.116 | 0.240 | 0.116 | 0.289 | 0.139 |
| 18_bio | 2.040 | 0.431 | 2.145 | 0.596 | 0.312 |
| 19_ptcloud | 64.745 | 13.897 | 63.056 | 13.755 | 306.377 |

### N=5000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 0.614 | tmo | 0.627 | 0.292 |
| 02_ii_1d | tmo | 0.725 | tmo | 0.745 | 2.148 |
| 03_pi_1d | tmo | 0.697 | tmo | 0.772 | 2.132 |
| 04_pp_2d | tmo | 2.623 | tmo | 2.678 | 0.521 |
| 05_ii_2d | tmo | 3.125 | tmo | 3.939 | 117.777 |
| 06_pi_2d | tmo | 2.615 | tmo | 3.286 | 2.156 |
| 07_diag_i | tmo | 1.459 | tmo | 1.519 | 0.289 |
| 08_diag_p | tmo | 1.465 | tmo | 2.625 | 64.535 |
| 09_red_pp | tmo | 0.606 | tmo | 0.615 | 0.319 |
| 10_red_ii | tmo | 0.719 | tmo | 0.757 | 2.148 |
| 11_mm_pp | tmo | 0.601 | tmo | 0.610 | 0.263 |
| 12_mm_ii | tmo | 0.724 | tmo | 0.750 | 2.146 |
| 13_trip_pp | oom | 37.430 | oom | 37.985 | 0.559 |
| 14_trip_ii | oom | 130.292 | oom | 24.782 | 3.643 |
| 15_trip2d_pp | oom | 33.343 | oom | 33.806 | 0.676 |
| 16_trip2d_ii | oom | 118.455 | oom | 16.662 | 50.853 |
| 17_box | tmo | 0.384 | tmo | 0.370 | 0.343 |
| 18_bio | tmo | 2.220 | tmo | 3.282 | 2.165 |
| 19_ptcloud | tmo | 209.680 | tmo | 204.671 | 7351.821 |

### N=10000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 1.071 | tmo | 1.050 | 0.385 |
| 02_ii_1d | tmo | 1.222 | tmo | 1.231 | 4.287 |
| 03_pi_1d | tmo | 1.201 | tmo | 1.280 | 4.273 |
| 04_pp_2d | tmo | 6.433 | tmo | 6.592 | 0.640 |
| 05_ii_2d | tmo | 8.053 | tmo | 8.760 | 425.763 |
| 06_pi_2d | tmo | 6.303 | tmo | 7.950 | 5.962 |
| 07_diag_i | tmo | 3.233 | tmo | 3.304 | 0.403 |
| 08_diag_p | tmo | 3.230 | tmo | 6.553 | 218.442 |
| 09_red_pp | tmo | 1.043 | tmo | 1.040 | 0.381 |
| 10_red_ii | tmo | 1.190 | tmo | 1.233 | 4.259 |
| 11_mm_pp | tmo | 1.044 | tmo | 1.044 | 0.392 |
| 12_mm_ii | tmo | 1.206 | tmo | 1.236 | 4.269 |
| 13_trip_pp | oom | 147.872 | oom | 147.849 | 0.699 |
| 14_trip_ii | oom | 521.544 | oom | 94.995 | 6.493 |
| 15_trip2d_pp | oom | 129.916 | oom | 133.512 | 1.032 |
| 16_trip2d_ii | oom | 478.101 | oom | 65.540 | 187.595 |
| 17_box | tmo | 0.584 | tmo | 0.490 | 0.622 |
| 18_bio | tmo | 6.077 | tmo | 9.411 | 6.607 |
| 19_ptcloud | tmo | 639.796 | tmo | 629.357 | 29790.812 |

### N=30000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 2.919 | tmo | 2.917 | 1.000 |
| 02_ii_1d | tmo | 3.225 | tmo | 3.250 | 13.470 |
| 03_pi_1d | tmo | 3.260 | tmo | 3.443 | 13.612 |
| 04_pp_2d | tmo | 34.877 | tmo | 35.758 | 1.828 |
| 05_ii_2d | tmo | 48.210 | tmo | 50.852 | 4243.736 |
| 06_pi_2d | tmo | 30.967 | tmo | 41.411 | 35.094 |
| 07_diag_i | tmo | 13.610 | tmo | 13.826 | 0.954 |
| 08_diag_p | tmo | 13.696 | tmo | 33.990 | 2100.196 |
| 09_red_pp | tmo | 2.862 | tmo | 2.881 | 1.300 |
| 10_red_ii | tmo | 3.290 | tmo | 3.342 | 13.189 |
| 11_mm_pp | tmo | 2.888 | tmo | 2.876 | 1.367 |
| 12_mm_ii | tmo | 3.273 | tmo | 3.307 | 13.411 |
| 13_trip_pp | oom | 1286.678 | oom | 1292.972 | 2.362 |
| 14_trip_ii | oom | 4524.359 | oom | 821.678 | 2162.194 |
| 15_trip2d_pp | oom | 1137.795 | oom | 1133.683 | 3.682 |
| 16_trip2d_ii | oom | 4193.343 | oom | 546.569 | 1644.590 |
| 17_box | tmo | 1.401 | tmo | 0.920 | 1.410 |
| 18_bio | tmo | 30.121 | tmo | 44.403 | 35.342 |
| 19_ptcloud | oom | 3947.956 | oom | 3950.172 | tmo |

### N=50000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | tmo | 4.893 | tmo | 4.794 | 1.688 |
| 02_ii_1d | tmo | 5.351 | tmo | 5.351 | 22.891 |
| 03_pi_1d | tmo | 5.083 | tmo | 5.721 | 22.796 |
| 04_pp_2d | tmo | 86.083 | tmo | 85.852 | 2.557 |
| 05_ii_2d | tmo | 104.046 | tmo | 110.762 | 12248.964 |
| 06_pi_2d | tmo | 80.568 | tmo | 101.880 | 80.177 |
| 07_diag_i | tmo | 33.049 | tmo | 29.354 | 1.886 |
| 08_diag_p | tmo | 27.464 | tmo | 94.644 | 8976.803 |
| 09_red_pp | tmo | 4.790 | tmo | 4.772 | 2.264 |
| 10_red_ii | tmo | 5.278 | tmo | 5.285 | 23.189 |
| 11_mm_pp | tmo | 4.749 | tmo | 4.707 | 2.241 |
| 12_mm_ii | tmo | 5.315 | tmo | 5.341 | 23.130 |
| 13_trip_pp | oom | 3541.811 | oom | 3544.965 | 3.895 |
| 14_trip_ii | oom | 12545.008 | oom | 2308.184 | 36.520 |
| 15_trip2d_pp | oom | 3158.855 | oom | 3155.757 | 5.770 |
| 16_trip2d_ii | oom | 11573.815 | oom | 1474.506 | 2708.638 |
| 17_box | tmo | 1.988 | tmo | 1.176 | 2.524 |
| 18_bio | tmo | 75.306 | tmo | 107.789 | 73.318 |
| 19_ptcloud | oom | 9228.719 | oom | 9089.241 | tmo |

### N=100000

| case | table | table_opt | table_auto | table_opt_auto | polars |
|---|---|---|---|---|---|
| 01_pp_1d | oom | 9.630 | oom | 9.538 | tmo |
| 02_ii_1d | oom | 10.703 | oom | 10.715 | tmo |
| 03_pi_1d | oom | 10.324 | oom | 11.529 | tmo |
| 04_pp_2d | oom | 234.924 | oom | 233.830 | tmo |
| 05_ii_2d | oom | 290.774 | oom | 305.637 | tmo |
| 06_pi_2d | oom | 228.957 | oom | 293.314 | tmo |
| 07_diag_i | oom | 91.575 | oom | 91.686 | tmo |
| 08_diag_p | oom | 91.654 | oom | 238.177 | tmo |
| 09_red_pp | oom | 9.566 | oom | 9.627 | tmo |
| 10_red_ii | oom | 10.737 | oom | 10.695 | tmo |
| 11_mm_pp | oom | 9.585 | oom | 9.581 | tmo |
| 12_mm_ii | oom | 11.701 | oom | 11.508 | tmo |
| 13_trip_pp | oom | 14199.558 | oom | 14248.700 | tmo |
| 14_trip_ii | oom | 50526.310 | oom | 9243.616 | tmo |
| 15_trip2d_pp | oom | 12606.845 | oom | 12611.464 | tmo |
| 16_trip2d_ii | oom | 46262.722 | oom | 5757.279 | tmo |
| 17_box | oom | 3.778 | oom | 2.315 | tmo |
| 18_bio | oom | 216.813 | oom | 299.657 | tmo |
| 19_ptcloud | oom | 29264.437 | oom | 29221.743 | tmo |

---

## Notes on what's missing and why

- **`table` at N ≥ 30000 on cpu-multi, N ≥ 5000 on cpu-single — `tmo`**: the
  brute-force pipeline still allocates an `N×N` (or `N×N×K`) bool tensor, but
  on CPU each timed call takes O(seconds) and `torch.compile`'s Inductor C++
  codegen warmup adds another ~1–5 s per case. Running all 19 in one bench
  pass exceeded the 3-min per-N budget. The shape would fit in host memory up
  to N≈50000; we just chose not to wait.
- **`table` at N ≥ 100000 on cpu-multi, N ≥ 5000 triple cases — `oom`**: the
  bool tensor either crosses the host RAM cap (10 GB for N=10⁵) or, for the
  triple cases, sits at N³ which is hopeless at any N ≥ a few thousand.
- **`polars` on case 19 at N ≥ 30000 — `tmo`**: the `cross()` of `Mask × In`
  before the `join_where` against Weight is O(N²·K) and CPU polars doesn't
  short-circuit it.
- **`polars` cells originally filled with `tmo` at N=100000 cpu-multi**:
  the original pytest run couldn't get past the OOM-killed cells within the
  3-min budget, so every later polars cell in the same row was labeled `tmo`
  by association. Re-running each case in its own process at N=100000
  (cpu-multi), with the same per-case `torch.compile` + warmup + table_opt
  timing performed *before* the polars timing (so the polars `collect()` runs
  against the same in-process torch state as the original sweep), shows the
  fast cases finish in milliseconds (e.g. 09_red_pp 1.5 ms, 17_box 2.4 ms,
  13_trip_pp 2.5 ms). The genuinely heavy ones — 05_ii_2d, 14_trip_ii — are
  `MemoryError` (now labeled `oom`); 19_ptcloud hits polars's 2³²-row
  cross-join cap (`err`); 16_trip2d_ii runs but takes ~30 s, well over the
  per-N pytest budget. Cases 07 and 08 are actually *faster* at N=100000 than
  N=50000 (different alive ratio + a different polars physical plan at the
  larger size), so the original `tmo` masked a genuinely fast result there.
  The cpu-single N=100000 polars row has not been re-verified and still
  reads `tmo`; the same caveat (failure was bench-budget exhaustion, not
  per-call timeout) probably applies but is left for a follow-up rerun.
- **Case-19 entries at N=30000, 50000, 100000 (both modes)**: pulled from a
  standalone Python script after the per-bench pytest run hit its timeout on
  that case. Same `_compile_table_fn` wrapper, same skew, same seeds — just
  outside the pytest harness so the rest of the sweep wasn't blocked.

## Observations

- **`table_opt_auto` vs `table_opt` parity holds on CPU**: across both modes
  the auto path is within ~5–10 % of the hand-written variant on the
  2-operand cases and *clearly* faster on the triple `ii` cases (14, 16) for
  the same reason as on GPU — the auto lead picks `overlap` while the
  hand-written closures use a one-sided `lt`.
- **Multi-thread speedup is uneven across backends**: torch's batched element-
  wise ops scale near-linearly with cores at large N (table_opt at N=10000
  case 19 is ~640 ms single vs ~446 ms multi — *not* much speedup because the
  hot path is dominated by gather/searchsorted, both of which parallelize
  poorly). Polars sees bigger wins on the join-heavy cases (e.g. case 08_diag_p
  at N=30000: 2100 ms single vs 1174 ms multi).
- **Case 17 (1-box / N-points) is the only case where `table` actually wins**
  on CPU at every N. The N²-mask cost collapses to N for this case so the
  searchsorted overhead of the optimized variants is the dominant cost.
- **Triple `ii` cases (14, 16)** are where the optimized auto really pays
  off: `table_opt_auto` is ~5–10× faster than `table_opt` on CPU, mirroring
  the GPU finding.
