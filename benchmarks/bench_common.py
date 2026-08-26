"""Shared utilities for the thesis benchmark runners (exp1/exp2/exp3).

This module imports only the standard library at the top level so runners can
do::

    import bench_common
    device = bench_common.apply_device_mode(mode)   # sets env BEFORE torch
    import torch                                     # now safe

Everything torch-flavored inside is imported lazily for the same reason.

Device modes mirror ``tests/conftest.py::pytest_configure``:

* ``gpu``        — CUDA if available (error if not).
* ``cpu-single`` — CUDA hidden, torch + polars pinned to 1 thread.
* ``cpu-multi``  — CUDA hidden, all cores.

The env vars must be set before torch/polars import and
``torch.set_num_interop_threads`` fails once torch has initialized, so a
runner that wants several modes must re-exec itself once per mode
(:func:`run_all_modes`).
"""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

DEVICE_MODES = ("gpu", "cpu-single", "cpu-multi")

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5
DEFAULT_BUDGET_S = 120.0

STATUS_OK = "ok"
STATUS_OOM = "oom"
STATUS_BUDGET = "budget"
STATUS_GUARD = "skipped_guard"
STATUS_PRIOR = "skipped_prior_fail"
STATUS_ERROR = "error"
# A cell writes ``started`` before running and its real row after; a cell
# whose *last* row is still ``started`` on resume was killed hard (e.g. the
# kernel OOM killer, which Python cannot catch) and is recorded as ``killed``.
STATUS_STARTED = "started"
STATUS_KILLED = "killed"

FAILURE_STATUSES = (STATUS_OOM, STATUS_BUDGET, STATUS_ERROR, STATUS_KILLED)


def add_src_to_path() -> None:
    sys.path.insert(0, str(ROOT / "src"))


def add_tests_to_path() -> None:
    sys.path.insert(0, str(ROOT / "tests"))


def apply_device_mode(mode: str):
    """Set thread/device env vars, import torch, return the torch device.

    Must be called before torch or polars is imported anywhere in the
    process. Mirrors ``tests/conftest.py::pytest_configure``.
    """
    if mode not in DEVICE_MODES:
        raise ValueError(f"unknown device mode {mode!r}; expected {DEVICE_MODES}")
    if mode in ("cpu-single", "cpu-multi"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Cap the address space so runaway allocations raise (MemoryError /
        # torch RuntimeError) instead of tripping the kernel OOM killer,
        # which SIGKILLs the process before Python can record the cell.
        # CUDA modes are exempt: unified addressing reserves huge VM ranges.
        try:
            import resource

            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            cap = int(total * 0.85)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard == resource.RLIM_INFINITY or cap < hard:
                resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        except (ImportError, ValueError, OSError):
            pass
    if mode == "cpu-single":
        os.environ["POLARS_MAX_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["RAYON_NUM_THREADS"] = "1"

    import torch

    if mode == "cpu-single":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already initialized — OMP_NUM_THREADS=1 above still constrains it.
            pass
    if mode == "gpu" and not torch.cuda.is_available():
        raise SystemExit(
            "--device-mode gpu requested but torch.cuda.is_available() is False"
        )
    return torch.device("cuda" if mode == "gpu" else "cpu")


def run_all_modes(script_path: str, passthrough: list[str]) -> int:
    """Re-exec ``script_path`` once per device mode, forwarding ``passthrough``.

    Returns the worst exit code. Fresh processes are required: env vars and
    interop-thread counts are read once at library init.
    """
    worst = 0
    for mode in DEVICE_MODES:
        cmd = [sys.executable, script_path, "--device-mode", mode, *passthrough]
        print(f"\n=== {' '.join(cmd)} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        worst = max(worst, rc)
    return worst


def thread_info(device) -> str:
    """Observed thread/device state, recorded into every CSV row."""
    import torch

    parts = [f"torch_threads={torch.get_num_threads()}"]
    try:
        import polars as pl

        parts.append(f"polars_threads={pl.thread_pool_size()}")
    except ImportError:
        pass
    parts.append(f"cuda={torch.cuda.is_available()}")
    return ";".join(parts)


def timed_cell(
    fn: Callable[[], object],
    *,
    repeats: int,
    device,
    warmup: int = DEFAULT_WARMUP,
    budget_s: float = DEFAULT_BUDGET_S,
    probe: Callable[[object], None] | None = None,
) -> dict:
    """Warm up and time ``fn``; CUDA-synchronized, OOM/budget-guarded.

    Returns ``{"status", "median_ms", "all_ms", "note"}``. ``all_ms`` is a
    list of the timed repeats in ms (empty unless status is ok/budget).
    The budget applies to the *first* run (warmup included): a cell whose
    single run exceeds it is reported once and not repeated. ``probe`` is
    called with the first run's output (before deletion) so callers can
    record result sizes without an extra run.
    """
    import torch

    cuda = device.type == "cuda"
    probed = False

    def _sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def _one() -> float:
        nonlocal probed
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        dt = time.perf_counter() - t0
        if probe is not None and not probed:
            probe(out)
            probed = True
        del out
        if cuda:
            torch.cuda.empty_cache()
        return dt

    try:
        first = _one()  # first warmup: absorbs torch.compile / autotune
        if first > budget_s:
            return {
                "status": STATUS_BUDGET,
                "median_ms": first * 1e3,
                "all_ms": [first * 1e3],
                "note": f"first run {first:.1f}s > budget {budget_s:.0f}s",
            }
        for _ in range(warmup - 1):
            _one()
        times = [_one() * 1e3 for _ in range(repeats)]
        return {
            "status": STATUS_OK,
            "median_ms": statistics.median(times),
            "all_ms": times,
            "note": "",
        }
    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
        if cuda:
            torch.cuda.empty_cache()
        return {"status": STATUS_OOM, "median_ms": None, "all_ms": [],
                "note": type(e).__name__}
    except Exception as e:  # noqa: BLE001 — a broken cell must not kill the sweep
        if cuda:
            torch.cuda.empty_cache()
        # torch's CPU allocator fails with a RuntimeError, not MemoryError.
        status = STATUS_OOM if "memory" in str(e).lower() else STATUS_ERROR
        return {"status": status, "median_ms": None, "all_ms": [],
                "note": f"{type(e).__name__}: {str(e)[:200]}"}


def fmt_ms(ms: float | None) -> str:
    return "" if ms is None else f"{ms:.3f}"


def fmt_all(all_ms: list[float]) -> str:
    return ";".join(f"{t:.3f}" for t in all_ms)


class FailureLadder:
    """Escalation skip: once a key fails at some n, skip all larger n."""

    def __init__(self) -> None:
        self._failed_at: dict[object, int] = {}

    def should_skip(self, key: object, n: int) -> bool:
        return key in self._failed_at and n >= self._failed_at[key]

    def record(self, key: object, n: int, status: str) -> None:
        if status in FAILURE_STATUSES:
            prev = self._failed_at.get(key)
            if prev is None or n < prev:
                self._failed_at[key] = n


def load_last_status(path: Path, key_cols: tuple[str, ...]) -> dict[tuple, str]:
    """Last-written status per measurement key, for ``--resume``.

    Keys are tuples of the CSV's string values for ``key_cols``; callers must
    stringify their own key parts the same way (``str(value)``).
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[tuple(row[c] for c in key_cols)] = row["status"]
    return out


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class CsvWriter:
    """Append-mode CSV writer stamping provenance columns on every row."""

    PROVENANCE = (
        "git_rev", "torch_version", "polars_version",
        "device_name", "thread_info", "timestamp",
    )

    def __init__(self, path: Path, fieldnames: list[str], device) -> None:
        import torch

        self.path = Path(path)
        self.fieldnames = list(fieldnames) + list(self.PROVENANCE)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import polars as pl

            polars_version = pl.__version__
        except ImportError:
            polars_version = ""
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = f"cpu ({os.cpu_count()} cores)"
        self._prov = {
            "git_rev": _git_rev(),
            "torch_version": torch.__version__,
            "polars_version": polars_version,
            "device_name": device_name,
            "thread_info": thread_info(device),
        }
        self._write_header_if_new()

    def _write_header_if_new(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def write(self, row: dict) -> None:
        full = dict(row)
        full.update(self._prov)
        full["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(full)


def values_for(n: int, seed: int, dtype=None):
    """Strictly positive value array (sums never cancel to exact zero)."""
    import torch

    if dtype is None:
        dtype = torch.float32
    gen = torch.Generator().manual_seed(seed)
    return 0.5 + torch.rand(n, generator=gen, dtype=dtype)


def parse_int_list(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(s) for s in raw.split(",") if s.strip()]
