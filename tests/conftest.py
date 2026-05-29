"""Pytest CLI options for ``test_mapping.py``.

* ``--mapping-n``: number of pieces per operand (default: ``300``)
* ``--mapping-skew``: comma-separated skew levels in ``[0, 1]``
  (default: ``0.0,0.5,1.0``). Replaces the prior ``--mapping-intersect``
  ``low/med/high`` axis: ``0`` is uniform cell placement, ``1`` clusters all
  cells toward the origin corner (so two operands sharing the same grid end
  up with many shared cells and a high alive ratio).
* ``--mapping-bench``: opt in to the timing smoke test
* ``--mapping-bench-repeats``: timed repeats for the benchmark (default: ``3``)
* ``--no-mapping-bench-polars``: skip the polars backend in the benchmark
* ``--no-mapping-bench-table``: skip the brute-force table backend
* ``--device-mode``: ``gpu`` (default, uses CUDA if available), ``cpu-single``
  (CPU with 1 thread for torch and polars), ``cpu-multi`` (CPU with all cores).
  Configured early via ``pytest_configure`` so torch and polars see the right
  thread / device env vars before they're imported.
"""

from __future__ import annotations

import os

import pytest

DEFAULT_SKEWS: tuple[float, ...] = (0.0, 0.5, 1.0)


def pytest_addoption(parser):
    group = parser.getgroup("mapping")
    group.addoption(
        "--mapping-n",
        type=int,
        default=300,
        help="number of pieces per operand (default: 300)",
    )
    group.addoption(
        "--mapping-skew",
        default=None,
        help=(
            "comma-separated skew values in [0, 1] "
            "(default: 0.0,0.5,1.0). 0 is uniform, 1 is fully clustered."
        ),
    )
    group.addoption(
        "--mapping-bench",
        action="store_true",
        help="run the opt-in timing smoke test",
    )
    group.addoption(
        "--mapping-bench-repeats",
        type=int,
        default=3,
        help="timed repeats for the benchmark (default: 3)",
    )
    group.addoption(
        "--no-mapping-bench-polars",
        dest="mapping_bench_polars",
        action="store_false",
        default=True,
        help="skip the polars backend in the benchmark",
    )
    group.addoption(
        "--no-mapping-bench-table",
        dest="mapping_bench_table",
        action="store_false",
        default=True,
        help="skip the brute-force table_mapping/table_auto (OOM-prone at large N)",
    )
    group.addoption(
        "--device-mode",
        choices=("gpu", "cpu-single", "cpu-multi"),
        default="gpu",
        help=(
            "Execution mode. ``gpu`` uses CUDA if available (default). "
            "``cpu-single`` hides CUDA and pins torch + polars to 1 thread. "
            "``cpu-multi`` hides CUDA and lets torch + polars use all cores."
        ),
    )


def pytest_configure(config):
    """Apply device-mode env vars *before* torch/polars import.

    pytest_configure runs after conftest.py is parsed and CLI args are read,
    but *before* test modules are collected (so before ``test_mapping.py``
    imports torch and polars). That's the right hook for ``CUDA_VISIBLE_DEVICES``
    and ``POLARS_MAX_THREADS``, both of which are read at module init.
    """
    mode = config.getoption("--device-mode")
    if mode in ("cpu-single", "cpu-multi"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if mode == "cpu-single":
        os.environ["POLARS_MAX_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["RAYON_NUM_THREADS"] = "1"
    # torch.set_num_threads applies at runtime, so import is safe after the
    # env vars above are set.
    import torch
    if mode == "cpu-single":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already initialized — fine; ``OMP_NUM_THREADS=1`` from above
            # already constrains the interop pool to 1.
            pass


def _parse_skews(raw):
    if raw is None:
        return DEFAULT_SKEWS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    values: list[float] = []
    for part in parts:
        v = float(part)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"--mapping-skew values must be in [0, 1], got {v}")
        values.append(v)
    if not values:
        raise ValueError("--mapping-skew must contain at least one value")
    return tuple(values)


def _skew_id(value: float) -> str:
    return f"skew{value:g}"


def pytest_generate_tests(metafunc):
    if "skew" in metafunc.fixturenames:
        skews = _parse_skews(metafunc.config.getoption("--mapping-skew"))
        metafunc.parametrize("skew", skews, ids=[_skew_id(s) for s in skews])


@pytest.fixture
def mapping_n(request):
    return request.config.getoption("--mapping-n")


@pytest.fixture
def mapping_skews(request):
    return _parse_skews(request.config.getoption("--mapping-skew"))


@pytest.fixture
def mapping_bench(request):
    return request.config.getoption("--mapping-bench")


@pytest.fixture
def mapping_bench_repeats(request):
    return request.config.getoption("--mapping-bench-repeats")


@pytest.fixture
def mapping_bench_polars(request):
    return request.config.getoption("mapping_bench_polars")


@pytest.fixture
def mapping_bench_table(request):
    return request.config.getoption("mapping_bench_table")
