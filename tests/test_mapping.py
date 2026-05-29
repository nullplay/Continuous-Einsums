"""Raw mapping-only tests and benchmark cases.

This file intentionally does not call ``continuous_einsum``, ``compile_einsum``,
or any v2 mapping wrapper. Each case defines its table and Polars mapping
programs together:

    def case_...(...):
        def table_mapping():
            ...
            return torch.nonzero(mask, as_tuple=False)

        def polars_plan():
            ...
            return plan.select(piece_cols)   # returns a LazyFrame, no collect

The benchmark builds each Polars ``LazyFrame`` once outside the timed region
(DataFrame construction, GPU→CPU copies, and ``join_where`` AST building are
input-prep / plan-build work, analogous to ``torch.compile`` warmup for the
table backend). Only ``plan.collect(engine=...)`` is timed.

Canonical sorting and table-vs-Polars correctness checks also happen outside
the timed sections.

Per-operand data now comes from :func:`synth_dataset.create_nd_pieces`, which
samples ``n`` non-overlapping ND boxes from a grid covering ``∏[0, max_d)``.
The ``skew`` knob in ``[0, 1]`` replaces the prior ``low/med/high`` intersect
levels: ``0`` is uniform cell selection, ``1`` clusters cells at the origin
corner (so two operands with the same grid share more cells and the join's
alive ratio grows).

Command-line options (registered in ``conftest.py``):

* ``--mapping-n``: number of pieces per input operand (default: ``300``),
  except cases that explicitly fix one operand size, such as Box=1 or Weight=27.
* ``--mapping-skew``: comma-separated skew values in ``[0, 1]``
  (default: ``0.0,0.5,1.0``).
* ``--mapping-bench``: run the opt-in timing smoke test.
* ``--mapping-bench-repeats``: timed repeats for the benchmark (default: ``3``).
* ``--no-mapping-bench-polars``: skip the polars backend in the benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import polars as pl
import pytest
import torch

from synth_dataset import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    INTERVAL,
    PINPOINT,
    create_nd_pieces,
)
from table_mapping import build_table_mapping
from table_opt_mapping import build_table_opt_mapping


DTYPE = DEFAULT_DTYPE
DEVICE = DEFAULT_DEVICE
POLARS_ENGINE = "gpu" if torch.cuda.is_available() else "cpu"

# All axes use the same range so cross-operand joins on equality / overlap are
# on comparable scales. Cell sizes scale as ``SPACE_MAX / cells_per_dim``; the
# absolute value doesn't matter for correctness, only relative geometry.
SPACE_MAX = 64.0

# Stable per-operand seeds; ``create_nd_pieces`` is deterministic for a given
# (n, dim_kinds, dim_maxes, skew, seed) so cases are reproducible.
SEED_A = 11
SEED_B = 22
SEED_C = 33
SEED_BOX = 44
SEED_POINTS = 55
SEED_DATA = 66
SEED_QUERY = 77
SEED_MASK = 88
SEED_IN = 99
SEED_WEIGHT = 110

try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    # Each case pre-builds a fresh closure from build_table_*_mapping. They
    # share the same code object so Dynamo dedupes by code and respecializes
    # per closure cell — easily blows past the default 8-entry limit when we
    # run all cases in one benchmark.
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
except AttributeError:  # pragma: no cover - old torch fallback
    pass


@dataclass(frozen=True)
class MappingCase:
    label: str
    skew: float
    table_mapping: Callable[[], torch.Tensor]
    polars_plan: Callable[[], pl.LazyFrame]
    polars_piece_cols: tuple[str, ...]
    total_candidates: int
    table_auto: Callable[[], torch.Tensor]
    table_opt_auto: Callable[[], tuple[torch.Tensor, ...]]


@dataclass(frozen=True)
class MappingCaseSpec:
    label: str
    build: Callable[[int, float], MappingCase]


# ---------------------------------------------------------------------------
# Synthesis helpers — thin wrappers around create_nd_pieces.
# ---------------------------------------------------------------------------


def _pp_1d(n: int, skew: float, seed: int) -> torch.Tensor:
    """1D pinpoint coord, length-n tensor."""
    (coord,), = create_nd_pieces(
        n, (PINPOINT,), (SPACE_MAX,), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return coord


def _ii_1d(n: int, skew: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """1D interval (start, end), each length-n."""
    (s, e), = create_nd_pieces(
        n, (INTERVAL,), (SPACE_MAX,), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return s, e


def _pp_2d(n: int, skew: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """2D pinpoint coords (x, y)."""
    (x,), (y,) = create_nd_pieces(
        n, (PINPOINT, PINPOINT), (SPACE_MAX, SPACE_MAX), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return x, y


def _ii_2d(n: int, skew: float, seed: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """2D intervals: (x_s, x_e, y_s, y_e)."""
    (xs, xe), (ys, ye) = create_nd_pieces(
        n, (INTERVAL, INTERVAL), (SPACE_MAX, SPACE_MAX), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return xs, xe, ys, ye


def _pi_pp_then_ii(n: int, skew: float, seed: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Axis 0 = pinpoint (returns coord), axis 1 = interval (returns s, e)."""
    (p,), (s, e) = create_nd_pieces(
        n, (PINPOINT, INTERVAL), (SPACE_MAX, SPACE_MAX), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return p, s, e


def _ip_ii_then_pp(n: int, skew: float, seed: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Axis 0 = interval (returns s, e), axis 1 = pinpoint (returns coord)."""
    (s, e), (c,) = create_nd_pieces(
        n, (INTERVAL, PINPOINT), (SPACE_MAX, SPACE_MAX), skew,
        seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return s, e, c


def _case(
    label: str,
    skew: float,
    n: int,
    table_mapping: Callable[[], torch.Tensor],
    polars_plan: Callable[[], pl.LazyFrame],
    piece_cols: tuple[str, ...],
    table_auto: Callable[[], torch.Tensor],
    table_opt_auto: Callable[[], tuple[torch.Tensor, ...]],
    total_candidates: int | None = None,
) -> MappingCase:
    return MappingCase(
        label=f"{label}_skew{skew:g}_n{n}",
        skew=skew,
        table_mapping=table_mapping,
        polars_plan=polars_plan,
        polars_piece_cols=piece_cols,
        total_candidates=(
            total_candidates if total_candidates is not None else n ** len(piece_cols)
        ),
        table_auto=table_auto,
        table_opt_auto=table_opt_auto,
    )


def _auto_builders(
    op: dict[str, torch.Tensor],
    output: tuple[str, ...],
    eqs: list[str],
) -> tuple[Callable[[], torch.Tensor], Callable[[], tuple[torch.Tensor, ...]]]:
    """Build the brute-force and optimized auto mappings from one DSL spec."""
    return (
        build_table_mapping(op, output, eqs),
        build_table_opt_mapping(op, output, eqs),
    )


# ---------------------------------------------------------------------------
# Individual cases: table_mapping and polars_plan live together.
# ---------------------------------------------------------------------------


def case_01_pointwise_1d_pp(n: int, skew: float) -> MappingCase:
    a = _pp_1d(n, skew, SEED_A)
    b = _pp_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = a[:, None] == b[None, :]
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(b_lf, pl.col("A_i") == pl.col("B_i"))
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_x": a, "B_x": b}
    auto_output = ("A", "B")
    auto_eqs = ["A_x[A] == B_x[B]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "01_pointwise_1d_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_02_pointwise_1d_ii(n: int, skew: float) -> MappingCase:
    a_s, a_e = _ii_1d(n, skew, SEED_A)
    b_s, b_e = _ii_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (a_s[:, None] < b_e[None, :]) & (b_s[None, :] < a_e[:, None])
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i_s": a_s.cpu().contiguous(),
                "A_i_e": a_e.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_s.cpu().contiguous(),
                "B_i_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i_s") < pl.col("B_i_e"),
            pl.col("B_i_s") < pl.col("A_i_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_s": a_s, "A_e": a_e, "B_s": b_s, "B_e": b_e}
    auto_output = ("A", "B")
    auto_eqs = ["A_s[A] < B_e[B]", "B_s[B] < A_e[A]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "02_pointwise_1d_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_03_pointwise_1d_pi(n: int, skew: float) -> MappingCase:
    a = _pp_1d(n, skew, SEED_A)
    b_s, b_e = _ii_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (a[:, None] >= b_s[None, :]) & (a[:, None] < b_e[None, :])
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_s.cpu().contiguous(),
                "B_i_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i") >= pl.col("B_i_s"),
            pl.col("A_i") < pl.col("B_i_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_x": a, "B_s": b_s, "B_e": b_e}
    auto_output = ("A", "B")
    auto_eqs = ["A_x[A] >= B_s[B]", "A_x[A] < B_e[B]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "03_pointwise_1d_pi", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_04_pointwise_2d_pp(n: int, skew: float) -> MappingCase:
    a_i, a_j = _pp_2d(n, skew, SEED_A)
    b_i, b_j = _pp_2d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (a_i[:, None] == b_i[None, :]) & (a_j[:, None] == b_j[None, :])
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a_i.cpu().contiguous(),
                "A_j": a_j.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b_i.cpu().contiguous(),
                "B_j": b_j.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i") == pl.col("B_i"),
            pl.col("A_j") == pl.col("B_j"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_i": a_i, "A_j": a_j, "B_i": b_i, "B_j": b_j}
    auto_output = ("A", "B")
    auto_eqs = ["A_i[A] == B_i[B]", "A_j[A] == B_j[B]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "04_pointwise_2d_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_05_pointwise_2d_ii(n: int, skew: float) -> MappingCase:
    a_is, a_ie, a_js, a_je = _ii_2d(n, skew, SEED_A)
    b_is, b_ie, b_js, b_je = _ii_2d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (
            (a_is[:, None] < b_ie[None, :])
            & (b_is[None, :] < a_ie[:, None])
            & (a_js[:, None] < b_je[None, :])
            & (b_js[None, :] < a_je[:, None])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i_s": a_is.cpu().contiguous(),
                "A_i_e": a_ie.cpu().contiguous(),
                "A_j_s": a_js.cpu().contiguous(),
                "A_j_e": a_je.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_is.cpu().contiguous(),
                "B_i_e": b_ie.cpu().contiguous(),
                "B_j_s": b_js.cpu().contiguous(),
                "B_j_e": b_je.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i_s") < pl.col("B_i_e"),
            pl.col("B_i_s") < pl.col("A_i_e"),
            pl.col("A_j_s") < pl.col("B_j_e"),
            pl.col("B_j_s") < pl.col("A_j_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {
        "A_i_s": a_is, "A_i_e": a_ie, "A_j_s": a_js, "A_j_e": a_je,
        "B_i_s": b_is, "B_i_e": b_ie, "B_j_s": b_js, "B_j_e": b_je,
    }
    auto_output = ("A", "B")
    auto_eqs = [
        "A_i_s[A] < B_i_e[B]", "B_i_s[B] < A_i_e[A]",
        "A_j_s[A] < B_j_e[B]", "B_j_s[B] < A_j_e[A]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "05_pointwise_2d_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_06_pointwise_2d_pi(n: int, skew: float) -> MappingCase:
    # Axis 0 = pinpoint (i), axis 1 = interval (j).
    a_i, a_js, a_je = _pi_pp_then_ii(n, skew, SEED_A)
    b_i, b_js, b_je = _pi_pp_then_ii(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (
            (a_i[:, None] == b_i[None, :])
            & (a_js[:, None] < b_je[None, :])
            & (b_js[None, :] < a_je[:, None])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a_i.cpu().contiguous(),
                "A_j_s": a_js.cpu().contiguous(),
                "A_j_e": a_je.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b_i.cpu().contiguous(),
                "B_j_s": b_js.cpu().contiguous(),
                "B_j_e": b_je.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i") == pl.col("B_i"),
            pl.col("A_j_s") < pl.col("B_j_e"),
            pl.col("B_j_s") < pl.col("A_j_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {
        "A_i": a_i, "A_j_s": a_js, "A_j_e": a_je,
        "B_i": b_i, "B_j_s": b_js, "B_j_e": b_je,
    }
    auto_output = ("A", "B")
    auto_eqs = ["A_i[A] == B_i[B]", "A_j_s[A] < B_j_e[B]", "B_j_s[B] < A_j_e[A]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "06_pointwise_2d_pi", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_07_diagonal_ii_i(n: int, skew: float) -> MappingCase:
    # A has two interval axes (i0, i1), both sampled from a single 2D
    # ``create_nd_pieces`` call so the two axes share a coordinate space
    # ([0, SPACE_MAX)) but live in independent cells on a 2D grid. A.i0 and
    # A.i1 overlap only when their 2D cell lies on the diagonal (c0 == c1) —
    # the diagonal mapping logic still has to assemble all conditions.
    (a0_s, a0_e), (a1_s, a1_e) = create_nd_pieces(
        n, (INTERVAL, INTERVAL), (SPACE_MAX, SPACE_MAX), skew,
        seed=SEED_A, device=DEVICE, dtype=DTYPE,
    )
    b_s, b_e = _ii_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (
            (a0_s[:, None] < a1_e[:, None])
            & (a1_s[:, None] < a0_e[:, None])
            & (a0_s[:, None] < b_e[None, :])
            & (b_s[None, :] < a0_e[:, None])
            & (a1_s[:, None] < b_e[None, :])
            & (b_s[None, :] < a1_e[:, None])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i0_s": a0_s.cpu().contiguous(),
                "A_i0_e": a0_e.cpu().contiguous(),
                "A_i1_s": a1_s.cpu().contiguous(),
                "A_i1_e": a1_e.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_s.cpu().contiguous(),
                "B_i_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_i0_s") < pl.col("A_i1_e"),
            pl.col("A_i1_s") < pl.col("A_i0_e"),
            pl.col("A_i0_s") < pl.col("B_i_e"),
            pl.col("B_i_s") < pl.col("A_i0_e"),
            pl.col("A_i1_s") < pl.col("B_i_e"),
            pl.col("B_i_s") < pl.col("A_i1_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {
        "A_i0_s": a0_s, "A_i0_e": a0_e, "A_i1_s": a1_s, "A_i1_e": a1_e,
        "B_s": b_s, "B_e": b_e,
    }
    auto_output = ("A", "B")
    auto_eqs = [
        "A_i0_s[A] < A_i1_e[A]", "A_i1_s[A] < A_i0_e[A]",
        "A_i0_s[A] < B_e[B]",    "B_s[B] < A_i0_e[A]",
        "A_i1_s[A] < B_e[B]",    "B_s[B] < A_i1_e[A]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "07_diagonal_ii_i", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_08_diagonal_ii_p(n: int, skew: float) -> MappingCase:
    # Like case_07 but B is a 1D pinpoint instead of an interval.
    (a0_s, a0_e), (a1_s, a1_e) = create_nd_pieces(
        n, (INTERVAL, INTERVAL), (SPACE_MAX, SPACE_MAX), skew,
        seed=SEED_A, device=DEVICE, dtype=DTYPE,
    )
    b = _pp_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        point = b[None, :]
        table = (
            (point >= a0_s[:, None])
            & (point < a0_e[:, None])
            & (point >= a1_s[:, None])
            & (point < a1_e[:, None])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i0_s": a0_s.cpu().contiguous(),
                "A_i0_e": a0_e.cpu().contiguous(),
                "A_i1_s": a1_s.cpu().contiguous(),
                "A_i1_e": a1_e.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("B_i") >= pl.col("A_i0_s"),
            pl.col("B_i") < pl.col("A_i0_e"),
            pl.col("B_i") >= pl.col("A_i1_s"),
            pl.col("B_i") < pl.col("A_i1_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {
        "A_i0_s": a0_s, "A_i0_e": a0_e, "A_i1_s": a1_s, "A_i1_e": a1_e,
        "B_x": b,
    }
    auto_output = ("A", "B")
    auto_eqs = [
        "B_x[B] >= A_i0_s[A]", "B_x[B] < A_i0_e[A]",
        "B_x[B] >= A_i1_s[A]", "B_x[B] < A_i1_e[A]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "08_diagonal_ii_p", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_09_reduce_1d_pp(n: int, skew: float) -> MappingCase:
    a_k = _pp_1d(n, skew, SEED_A)
    b_k = _pp_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = a_k[:, None] == b_k[None, :]
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_k": a_k.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_k": b_k.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(b_lf, pl.col("A_k") == pl.col("B_k"))
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_k": a_k, "B_k": b_k}
    auto_output = ("A", "B")
    auto_eqs = ["A_k[A] == B_k[B]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "09_reduce_1d_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_10_reduce_1d_ii(n: int, skew: float) -> MappingCase:
    a_ks, a_ke = _ii_1d(n, skew, SEED_A)
    b_s, b_e = _ii_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (a_ks[:, None] < b_e[None, :]) & (b_s[None, :] < a_ke[:, None])
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_k_s": a_ks.cpu().contiguous(),
                "A_k_e": a_ke.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_k_s": b_s.cpu().contiguous(),
                "B_k_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_k_s") < pl.col("B_k_e"),
            pl.col("B_k_s") < pl.col("A_k_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_k_s": a_ks, "A_k_e": a_ke, "B_k_s": b_s, "B_k_e": b_e}
    auto_output = ("A", "B")
    auto_eqs = ["A_k_s[A] < B_k_e[B]", "B_k_s[B] < A_k_e[A]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "10_reduce_1d_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_11_matmul_pp(n: int, skew: float) -> MappingCase:
    a_k = _pp_1d(n, skew, SEED_A)
    b_k = _pp_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = a_k[:, None] == b_k[None, :]
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_k": a_k.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_k": b_k.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(b_lf, pl.col("A_k") == pl.col("B_k"))
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_k": a_k, "B_k": b_k}
    auto_output = ("A", "B")
    auto_eqs = ["A_k[A] == B_k[B]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "11_matmul_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_12_matmul_ii(n: int, skew: float) -> MappingCase:
    a_ks, a_ke = _ii_1d(n, skew, SEED_A)
    b_s, b_e = _ii_1d(n, skew, SEED_B)

    def table_mapping() -> torch.Tensor:
        table = (a_ks[:, None] < b_e[None, :]) & (b_s[None, :] < a_ke[:, None])
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_k_s": a_ks.cpu().contiguous(),
                "A_k_e": a_ke.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_k_s": b_s.cpu().contiguous(),
                "B_k_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        plan = a_lf.join_where(
            b_lf,
            pl.col("A_k_s") < pl.col("B_k_e"),
            pl.col("B_k_s") < pl.col("A_k_e"),
        )
        return plan.select(("A_piece", "B_piece"))

    auto_op = {"A_k_s": a_ks, "A_k_e": a_ke, "B_k_s": b_s, "B_k_e": b_e}
    auto_output = ("A", "B")
    auto_eqs = ["A_k_s[A] < B_k_e[B]", "B_k_s[B] < A_k_e[A]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "12_matmul_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_13_triple_1d_pp(n: int, skew: float) -> MappingCase:
    a = _pp_1d(n, skew, SEED_A)
    b = _pp_1d(n, skew, SEED_B)
    c = _pp_1d(n, skew, SEED_C)

    def table_mapping() -> torch.Tensor:
        mask = torch.ones((n, n, n), dtype=torch.bool, device=a.device)
        mask &= a[:, None, None] == b[None, :, None]
        mask &= a[:, None, None] == c[None, None, :]
        return torch.nonzero(mask, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b.cpu().contiguous(),
            }
        ).lazy()
        c_lf = pl.DataFrame(
            {
                "C_piece": torch.arange(n, dtype=torch.long),
                "C_i": c.cpu().contiguous(),
            }
        ).lazy()
        plan = (
            a_lf.join_where(b_lf, pl.col("A_i") == pl.col("B_i"))
            .join_where(c_lf, pl.col("A_i") == pl.col("C_i"))
        )
        return plan.select(("A_piece", "B_piece", "C_piece"))

    auto_op = {"A_x": a, "B_x": b, "C_x": c}
    auto_output = ("A", "B", "C")
    auto_eqs = ["A_x[A] == B_x[B]", "A_x[A] == C_x[C]"]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "13_triple_1d_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece", "C_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_14_triple_1d_ii(n: int, skew: float) -> MappingCase:
    a_s, a_e = _ii_1d(n, skew, SEED_A)
    b_s, b_e = _ii_1d(n, skew, SEED_B)
    c_s, c_e = _ii_1d(n, skew, SEED_C)

    def table_mapping() -> torch.Tensor:
        mask = torch.ones((n, n, n), dtype=torch.bool, device=a_s.device)
        mask &= a_s[:, None, None] < b_e[None, :, None]
        mask &= b_s[None, :, None] < a_e[:, None, None]
        mask &= a_s[:, None, None] < c_e[None, None, :]
        mask &= c_s[None, None, :] < a_e[:, None, None]
        mask &= b_s[None, :, None] < c_e[None, None, :]
        mask &= c_s[None, None, :] < b_e[None, :, None]
        return torch.nonzero(mask, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i_s": a_s.cpu().contiguous(),
                "A_i_e": a_e.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_s.cpu().contiguous(),
                "B_i_e": b_e.cpu().contiguous(),
            }
        ).lazy()
        c_lf = pl.DataFrame(
            {
                "C_piece": torch.arange(n, dtype=torch.long),
                "C_i_s": c_s.cpu().contiguous(),
                "C_i_e": c_e.cpu().contiguous(),
            }
        ).lazy()
        plan = (
            a_lf.join_where(
                b_lf,
                pl.col("A_i_s") < pl.col("B_i_e"),
                pl.col("B_i_s") < pl.col("A_i_e"),
            )
            .join_where(
                c_lf,
                pl.col("A_i_s") < pl.col("C_i_e"),
                pl.col("C_i_s") < pl.col("A_i_e"),
                pl.col("B_i_s") < pl.col("C_i_e"),
                pl.col("C_i_s") < pl.col("B_i_e"),
            )
        )
        return plan.select(("A_piece", "B_piece", "C_piece"))

    auto_op = {
        "A_s": a_s, "A_e": a_e,
        "B_s": b_s, "B_e": b_e,
        "C_s": c_s, "C_e": c_e,
    }
    auto_output = ("A", "B", "C")
    auto_eqs = [
        "A_s[A] < B_e[B]", "B_s[B] < A_e[A]",
        "A_s[A] < C_e[C]", "C_s[C] < A_e[A]",
        "B_s[B] < C_e[C]", "C_s[C] < B_e[B]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "14_triple_1d_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece", "C_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_15_triple_2d_pp(n: int, skew: float) -> MappingCase:
    a_i, a_j = _pp_2d(n, skew, SEED_A)
    b_i, b_j = _pp_2d(n, skew, SEED_B)
    c_i, c_j = _pp_2d(n, skew, SEED_C)

    def table_mapping() -> torch.Tensor:
        mask = torch.ones((n, n, n), dtype=torch.bool, device=a_i.device)
        mask &= a_i[:, None, None] == b_i[None, :, None]
        mask &= a_i[:, None, None] == c_i[None, None, :]
        mask &= a_j[:, None, None] == b_j[None, :, None]
        mask &= a_j[:, None, None] == c_j[None, None, :]
        return torch.nonzero(mask, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i": a_i.cpu().contiguous(),
                "A_j": a_j.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i": b_i.cpu().contiguous(),
                "B_j": b_j.cpu().contiguous(),
            }
        ).lazy()
        c_lf = pl.DataFrame(
            {
                "C_piece": torch.arange(n, dtype=torch.long),
                "C_i": c_i.cpu().contiguous(),
                "C_j": c_j.cpu().contiguous(),
            }
        ).lazy()
        plan = (
            a_lf.join_where(
                b_lf,
                pl.col("A_i") == pl.col("B_i"),
                pl.col("A_j") == pl.col("B_j"),
            )
            .join_where(
                c_lf,
                pl.col("A_i") == pl.col("C_i"),
                pl.col("A_j") == pl.col("C_j"),
            )
        )
        return plan.select(("A_piece", "B_piece", "C_piece"))

    auto_op = {
        "A_i": a_i, "A_j": a_j,
        "B_i": b_i, "B_j": b_j,
        "C_i": c_i, "C_j": c_j,
    }
    auto_output = ("A", "B", "C")
    auto_eqs = [
        "A_i[A] == B_i[B]", "A_j[A] == B_j[B]",
        "A_i[A] == C_i[C]", "A_j[A] == C_j[C]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "15_triple_2d_pp", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece", "C_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_16_triple_2d_ii(n: int, skew: float) -> MappingCase:
    a_is, a_ie, a_js, a_je = _ii_2d(n, skew, SEED_A)
    b_is, b_ie, b_js, b_je = _ii_2d(n, skew, SEED_B)
    c_is, c_ie, c_js, c_je = _ii_2d(n, skew, SEED_C)

    def table_mapping() -> torch.Tensor:
        mask = torch.ones((n, n, n), dtype=torch.bool, device=a_is.device)
        mask &= a_is[:, None, None] < b_ie[None, :, None]
        mask &= b_is[None, :, None] < a_ie[:, None, None]
        mask &= a_is[:, None, None] < c_ie[None, None, :]
        mask &= c_is[None, None, :] < a_ie[:, None, None]
        mask &= b_is[None, :, None] < c_ie[None, None, :]
        mask &= c_is[None, None, :] < b_ie[None, :, None]
        mask &= a_js[:, None, None] < b_je[None, :, None]
        mask &= b_js[None, :, None] < a_je[:, None, None]
        mask &= a_js[:, None, None] < c_je[None, None, :]
        mask &= c_js[None, None, :] < a_je[:, None, None]
        mask &= b_js[None, :, None] < c_je[None, None, :]
        mask &= c_js[None, None, :] < b_je[None, :, None]
        return torch.nonzero(mask, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        a_lf = pl.DataFrame(
            {
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_i_s": a_is.cpu().contiguous(),
                "A_i_e": a_ie.cpu().contiguous(),
                "A_j_s": a_js.cpu().contiguous(),
                "A_j_e": a_je.cpu().contiguous(),
            }
        ).lazy()
        b_lf = pl.DataFrame(
            {
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_i_s": b_is.cpu().contiguous(),
                "B_i_e": b_ie.cpu().contiguous(),
                "B_j_s": b_js.cpu().contiguous(),
                "B_j_e": b_je.cpu().contiguous(),
            }
        ).lazy()
        c_lf = pl.DataFrame(
            {
                "C_piece": torch.arange(n, dtype=torch.long),
                "C_i_s": c_is.cpu().contiguous(),
                "C_i_e": c_ie.cpu().contiguous(),
                "C_j_s": c_js.cpu().contiguous(),
                "C_j_e": c_je.cpu().contiguous(),
            }
        ).lazy()
        plan = (
            a_lf.join_where(
                b_lf,
                pl.col("A_i_s") < pl.col("B_i_e"),
                pl.col("B_i_s") < pl.col("A_i_e"),
                pl.col("A_j_s") < pl.col("B_j_e"),
                pl.col("B_j_s") < pl.col("A_j_e"),
            )
            .join_where(
                c_lf,
                pl.col("A_i_s") < pl.col("C_i_e"),
                pl.col("C_i_s") < pl.col("A_i_e"),
                pl.col("B_i_s") < pl.col("C_i_e"),
                pl.col("C_i_s") < pl.col("B_i_e"),
                pl.col("A_j_s") < pl.col("C_j_e"),
                pl.col("C_j_s") < pl.col("A_j_e"),
                pl.col("B_j_s") < pl.col("C_j_e"),
                pl.col("C_j_s") < pl.col("B_j_e"),
            )
        )
        return plan.select(("A_piece", "B_piece", "C_piece"))

    auto_op = {
        "A_i_s": a_is, "A_i_e": a_ie, "A_j_s": a_js, "A_j_e": a_je,
        "B_i_s": b_is, "B_i_e": b_ie, "B_j_s": b_js, "B_j_e": b_je,
        "C_i_s": c_is, "C_i_e": c_ie, "C_j_s": c_js, "C_j_e": c_je,
    }
    auto_output = ("A", "B", "C")
    auto_eqs = [
        "A_i_s[A] < B_i_e[B]", "B_i_s[B] < A_i_e[A]",
        "A_j_s[A] < B_j_e[B]", "B_j_s[B] < A_j_e[A]",
        "A_i_s[A] < C_i_e[C]", "C_i_s[C] < A_i_e[A]",
        "A_j_s[A] < C_j_e[C]", "C_j_s[C] < A_j_e[A]",
        "B_i_s[B] < C_i_e[C]", "C_i_s[C] < B_i_e[B]",
        "B_j_s[B] < C_j_e[C]", "C_j_s[C] < B_j_e[B]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "16_triple_2d_ii", skew, n,
        table_mapping, polars_plan,
        ("A_piece", "B_piece", "C_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_17_box_search(n: int, skew: float) -> MappingCase:
    # One fixed query box; ``n`` random pinpoint candidates. The hit count is
    # roughly ``n * (box_area / SPACE_MAX²)``, with ``skew`` shifting the point
    # cloud's center (skew=1 packs everything into the box's corner, so the
    # alive fraction goes up).
    box_x_s = torch.tensor([0.0], dtype=DTYPE, device=DEVICE)
    box_x_e = torch.tensor([SPACE_MAX / 4], dtype=DTYPE, device=DEVICE)
    box_y_s = torch.tensor([0.0], dtype=DTYPE, device=DEVICE)
    box_y_e = torch.tensor([SPACE_MAX / 4], dtype=DTYPE, device=DEVICE)
    point_x, point_y = _pp_2d(n, skew, SEED_POINTS)

    def table_mapping() -> torch.Tensor:
        table = (
            (point_x[None, :] >= box_x_s[:, None])
            & (point_x[None, :] < box_x_e[:, None])
            & (point_y[None, :] >= box_y_s[:, None])
            & (point_y[None, :] < box_y_e[:, None])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        box_lf = pl.DataFrame(
            {
                "Box_piece": torch.arange(1, dtype=torch.long),
                "Box_x_s": box_x_s.cpu().contiguous(),
                "Box_x_e": box_x_e.cpu().contiguous(),
                "Box_y_s": box_y_s.cpu().contiguous(),
                "Box_y_e": box_y_e.cpu().contiguous(),
            }
        ).lazy()
        points_lf = pl.DataFrame(
            {
                "Points_piece": torch.arange(n, dtype=torch.long),
                "Points_x": point_x.cpu().contiguous(),
                "Points_y": point_y.cpu().contiguous(),
            }
        ).lazy()
        plan = box_lf.join_where(
            points_lf,
            pl.col("Points_x") >= pl.col("Box_x_s"),
            pl.col("Points_x") < pl.col("Box_x_e"),
            pl.col("Points_y") >= pl.col("Box_y_s"),
            pl.col("Points_y") < pl.col("Box_y_e"),
        )
        return plan.select(("Box_piece", "Points_piece"))

    auto_op = {
        "Box_x_s": box_x_s, "Box_x_e": box_x_e,
        "Box_y_s": box_y_s, "Box_y_e": box_y_e,
        "Points_x": point_x, "Points_y": point_y,
    }
    auto_output = ("Box", "Points")
    auto_eqs = [
        "Points_x[Points] >= Box_x_s[Box]", "Points_x[Points] < Box_x_e[Box]",
        "Points_y[Points] >= Box_y_s[Box]", "Points_y[Points] < Box_y_e[Box]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "17_box_search", skew, n,
        table_mapping, polars_plan,
        ("Box_piece", "Points_piece"),
        total_candidates=n,
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_18_bio_intersect(n: int, skew: float) -> MappingCase:
    # Each operand has an x-interval (axis 0) and a categorical pinpoint
    # (axis 1). The join needs both x-overlap and c-equality.
    data_x_s, data_x_e, data_c = _ip_ii_then_pp(n, skew, SEED_DATA)
    query_x_s, query_x_e, query_c = _ip_ii_then_pp(n, skew, SEED_QUERY)

    def table_mapping() -> torch.Tensor:
        table = (
            (data_x_s[:, None] < query_x_e[None, :])
            & (query_x_s[None, :] < data_x_e[:, None])
            & (data_c[:, None] == query_c[None, :])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        data_lf = pl.DataFrame(
            {
                "Data_piece": torch.arange(n, dtype=torch.long),
                "Data_x_s": data_x_s.cpu().contiguous(),
                "Data_x_e": data_x_e.cpu().contiguous(),
                "Data_c": data_c.cpu().contiguous(),
            }
        ).lazy()
        query_lf = pl.DataFrame(
            {
                "Query_piece": torch.arange(n, dtype=torch.long),
                "Query_x_s": query_x_s.cpu().contiguous(),
                "Query_x_e": query_x_e.cpu().contiguous(),
                "Query_c": query_c.cpu().contiguous(),
            }
        ).lazy()
        plan = data_lf.join_where(
            query_lf,
            pl.col("Data_x_s") < pl.col("Query_x_e"),
            pl.col("Query_x_s") < pl.col("Data_x_e"),
            pl.col("Data_c") == pl.col("Query_c"),
        )
        return plan.select(("Data_piece", "Query_piece"))

    auto_op = {
        "Data_x_s": data_x_s,  "Data_x_e": data_x_e,  "Data_c": data_c,
        "Query_x_s": query_x_s, "Query_x_e": query_x_e, "Query_c": query_c,
    }
    auto_output = ("Data", "Query")
    auto_eqs = [
        "Data_x_s[Data] < Query_x_e[Query]",
        "Query_x_s[Query] < Data_x_e[Data]",
        "Data_c[Data] == Query_c[Query]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "18_bio_intersect", skew, n,
        table_mapping, polars_plan,
        ("Data_piece", "Query_piece"),
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def case_19_point_cloud(n: int, skew: float) -> MappingCase:
    # 3-operand convolution-flavored case: a 3D pinpoint Mask, 3D pinpoint In,
    # and a kernel-shaped Weight (27 entries, each three intervals around the
    # offset ``In - Mask``). Mask and In share the same 3D grid so they pick
    # the same cells when ``skew`` is high; ``matched_weight_count`` (driven
    # by skew) controls how many of the 27 weights cover the matching offset.
    mask_x, mask_y, mask_z = _pp_3d(n, skew, SEED_MASK)
    input_x, input_y, input_z = _pp_3d(n, skew, SEED_IN)

    # ``matched_weight_count`` ramps from 1 at skew=0 to 5 at skew=1 (replaces
    # the prior {low:1, med:2, high:5} table).
    matched_weight_count = max(1, min(27, int(round(1 + skew * 4))))
    weight_piece = torch.arange(27, dtype=torch.long, device=DEVICE)
    near_weight = weight_piece < matched_weight_count
    # ``near`` weights span a full cell (so they accept any same-cell offset),
    # ``far`` weights sit at large offsets that no real ``In - Mask`` pair
    # will reach.
    cell_size = SPACE_MAX / max(2, int((2 * n) ** (1 / 3)))
    near_lo = torch.full((27,), -cell_size, dtype=DTYPE, device=DEVICE)
    near_hi = torch.full((27,), cell_size, dtype=DTYPE, device=DEVICE)
    far_base = 1000.0 + weight_piece.to(dtype=DTYPE) * cell_size
    weight_r_s = torch.where(near_weight, near_lo, far_base)
    weight_r_e = torch.where(near_weight, near_hi, far_base + 0.5)
    weight_s_s = torch.where(near_weight, near_lo, far_base + cell_size)
    weight_s_e = torch.where(near_weight, near_hi, far_base + cell_size + 0.5)
    weight_t_s = torch.where(near_weight, near_lo, far_base + cell_size * 2.0)
    weight_t_e = torch.where(near_weight, near_hi, far_base + cell_size * 2.0 + 0.5)

    def table_mapping() -> torch.Tensor:
        table = (
            (input_x[None, :, None] >= mask_x[:, None, None] + weight_r_s[None, None, :])
            & (input_x[None, :, None] <  mask_x[:, None, None] + weight_r_e[None, None, :])
            & (input_y[None, :, None] >= mask_y[:, None, None] + weight_s_s[None, None, :])
            & (input_y[None, :, None] <  mask_y[:, None, None] + weight_s_e[None, None, :])
            & (input_z[None, :, None] >= mask_z[:, None, None] + weight_t_s[None, None, :])
            & (input_z[None, :, None] <  mask_z[:, None, None] + weight_t_e[None, None, :])
        )
        return torch.nonzero(table, as_tuple=False).to(torch.long)

    def polars_plan() -> pl.LazyFrame:
        mask_lf = pl.DataFrame(
            {
                "Mask_piece": torch.arange(n, dtype=torch.long),
                "Mask_x": mask_x.cpu().contiguous(),
                "Mask_y": mask_y.cpu().contiguous(),
                "Mask_z": mask_z.cpu().contiguous(),
            }
        ).lazy()
        input_lf = pl.DataFrame(
            {
                "In_piece": torch.arange(n, dtype=torch.long),
                "In_x": input_x.cpu().contiguous(),
                "In_y": input_y.cpu().contiguous(),
                "In_z": input_z.cpu().contiguous(),
            }
        ).lazy()
        weight_lf = pl.DataFrame(
            {
                "Weight_piece": weight_piece.cpu().contiguous(),
                "Weight_r_s": weight_r_s.cpu().contiguous(),
                "Weight_r_e": weight_r_e.cpu().contiguous(),
                "Weight_s_s": weight_s_s.cpu().contiguous(),
                "Weight_s_e": weight_s_e.cpu().contiguous(),
                "Weight_t_s": weight_t_s.cpu().contiguous(),
                "Weight_t_e": weight_t_e.cpu().contiguous(),
            }
        ).lazy()
        plan = mask_lf.join(input_lf, how="cross").join_where(
            weight_lf,
            pl.col("In_x") >= pl.col("Mask_x") + pl.col("Weight_r_s"),
            pl.col("In_x") <  pl.col("Mask_x") + pl.col("Weight_r_e"),
            pl.col("In_y") >= pl.col("Mask_y") + pl.col("Weight_s_s"),
            pl.col("In_y") <  pl.col("Mask_y") + pl.col("Weight_s_e"),
            pl.col("In_z") >= pl.col("Mask_z") + pl.col("Weight_t_s"),
            pl.col("In_z") <  pl.col("Mask_z") + pl.col("Weight_t_e"),
        )
        return plan.select(("Mask_piece", "In_piece", "Weight_piece"))

    auto_op = {
        "Mask_x": mask_x, "Mask_y": mask_y, "Mask_z": mask_z,
        "In_x": input_x, "In_y": input_y, "In_z": input_z,
        "Weight_r_s": weight_r_s, "Weight_r_e": weight_r_e,
        "Weight_s_s": weight_s_s, "Weight_s_e": weight_s_e,
        "Weight_t_s": weight_t_s, "Weight_t_e": weight_t_e,
    }
    auto_output = ("Mask", "In", "Weight")
    auto_eqs = [
        "In_x[In] >= Mask_x[Mask] + Weight_r_s[Weight]",
        "In_x[In] < Mask_x[Mask] + Weight_r_e[Weight]",
        "In_y[In] >= Mask_y[Mask] + Weight_s_s[Weight]",
        "In_y[In] < Mask_y[Mask] + Weight_s_e[Weight]",
        "In_z[In] >= Mask_z[Mask] + Weight_t_s[Weight]",
        "In_z[In] < Mask_z[Mask] + Weight_t_e[Weight]",
    ]
    table_auto, table_opt_auto = _auto_builders(auto_op, auto_output, auto_eqs)

    return _case(
        "19_point_cloud", skew, n,
        table_mapping, polars_plan,
        ("Mask_piece", "In_piece", "Weight_piece"),
        total_candidates=n * n * 27,
        table_auto=table_auto, table_opt_auto=table_opt_auto,
    )


def _pp_3d(n: int, skew: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (x,), (y,), (z,) = create_nd_pieces(
        n, (PINPOINT, PINPOINT, PINPOINT),
        (SPACE_MAX, SPACE_MAX, SPACE_MAX),
        skew, seed=seed, device=DEVICE, dtype=DTYPE,
    )
    return x, y, z


CASE_SPECS: tuple[MappingCaseSpec, ...] = (
    MappingCaseSpec("01_pointwise_1d_pp", case_01_pointwise_1d_pp),
    MappingCaseSpec("02_pointwise_1d_ii", case_02_pointwise_1d_ii),
    MappingCaseSpec("03_pointwise_1d_pi", case_03_pointwise_1d_pi),
    MappingCaseSpec("04_pointwise_2d_pp", case_04_pointwise_2d_pp),
    MappingCaseSpec("05_pointwise_2d_ii", case_05_pointwise_2d_ii),
    MappingCaseSpec("06_pointwise_2d_pi", case_06_pointwise_2d_pi),
    MappingCaseSpec("07_diagonal_ii_i", case_07_diagonal_ii_i),
    MappingCaseSpec("08_diagonal_ii_p", case_08_diagonal_ii_p),
    MappingCaseSpec("09_reduce_1d_pp", case_09_reduce_1d_pp),
    MappingCaseSpec("10_reduce_1d_ii", case_10_reduce_1d_ii),
    MappingCaseSpec("11_matmul_pp", case_11_matmul_pp),
    MappingCaseSpec("12_matmul_ii", case_12_matmul_ii),
    MappingCaseSpec("13_triple_1d_pp", case_13_triple_1d_pp),
    MappingCaseSpec("14_triple_1d_ii", case_14_triple_1d_ii),
    MappingCaseSpec("15_triple_2d_pp", case_15_triple_2d_pp),
    MappingCaseSpec("16_triple_2d_ii", case_16_triple_2d_ii),
    MappingCaseSpec("17_box_search", case_17_box_search),
    MappingCaseSpec("18_bio_intersect", case_18_bio_intersect),
    MappingCaseSpec("19_point_cloud", case_19_point_cloud),
)


def make_mapping_cases(n: int, skew: float) -> list[MappingCase]:
    return [spec.build(n, skew) for spec in CASE_SPECS]


def _canonical_rows(rows: torch.Tensor) -> torch.Tensor:
    rows = rows.detach().cpu().to(torch.long)
    if rows.numel() == 0:
        return rows.reshape(0, rows.shape[1] if rows.ndim == 2 else 0)
    order = torch.arange(rows.shape[0])
    for col in range(rows.shape[1] - 1, -1, -1):
        order = order[torch.argsort(rows[order, col], stable=True)]
    return rows[order]


def _canonical_cols(cols: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Canonicalize the tuple output of ``table_opt_auto``.

    The stacking happens here (outside any timed region) so the optimized
    mapping itself can return a tuple of 1-D index columns. Columns are
    moved to CPU before the stack so very large outputs don't need a
    contiguous GPU buffer of size ``(P, len(cols))``.
    """
    if cols[0].numel() == 0:
        return torch.empty((0, len(cols)), dtype=torch.long)
    return _canonical_rows(
        torch.stack([c.detach().cpu().to(torch.long) for c in cols], dim=1)
    )


def _polars_rows(df: pl.DataFrame, piece_cols: tuple[str, ...]) -> torch.Tensor:
    if df.height == 0:
        return torch.empty((0, len(piece_cols)), dtype=torch.long)
    return df.select(piece_cols).to_torch().to(torch.long)


def _assert_table_and_polars_agree(case: MappingCase) -> int:
    table_rows = _canonical_rows(case.table_mapping())
    opt_rows = _canonical_cols(case.table_opt_auto())
    polars_df = case.polars_plan().collect(engine=POLARS_ENGINE)
    polars_rows = _canonical_rows(_polars_rows(polars_df, case.polars_piece_cols))
    assert table_rows.shape == polars_rows.shape, (
        case.label,
        table_rows.shape,
        polars_rows.shape,
    )
    assert torch.equal(table_rows, polars_rows), case.label
    assert opt_rows.shape == table_rows.shape, (
        case.label,
        opt_rows.shape,
        table_rows.shape,
    )
    assert torch.equal(opt_rows, table_rows), case.label
    return int(table_rows.shape[0])


def _compile_table_fn(fn: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    return torch.compile(fn, dynamic=True, fullgraph=False)


@pytest.mark.parametrize("spec", CASE_SPECS, ids=lambda spec: spec.label)
def test_table_and_polars_mapping_piece_tuples(
    spec: MappingCaseSpec, skew: float, mapping_n: int
) -> None:
    case = spec.build(mapping_n, skew)
    rows = _assert_table_and_polars_agree(case)
    assert rows > 0


@pytest.mark.parametrize("spec", CASE_SPECS, ids=lambda spec: spec.label)
def test_skew_parameter_changes_mapping_rows(
    spec: MappingCaseSpec, mapping_n: int
) -> None:
    """Alive count should grow when boxes cluster harder at the origin corner.

    Both operands sample from the same grid; higher skew → more shared cells
    → more candidate joins → more alive tuples. ``skew=1`` strictly above
    ``skew=0`` is the meaningful check; the middle value is a sanity bound.
    """
    counts: list[int] = []
    for sk in (0.0, 0.5, 1.0):
        case = spec.build(mapping_n, sk)
        counts.append(int(case.table_mapping().shape[0]))
    assert counts[0] <= counts[2], (spec.label, counts)
    if mapping_n > 1:
        assert counts[0] < counts[2], (spec.label, counts)


def _time_callable(fn: Callable[[], object] | None, repeats: int) -> float | None:
    """Run ``fn`` ``repeats`` times with CUDA synchronization around each call
    and return the median wall-clock time in milliseconds (or ``None`` if
    ``fn`` is ``None``)."""
    if fn is None:
        return None
    times: list[float] = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del out
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return torch.tensor(times).median().item() * 1e3


def test_mapping_backend_benchmark(
    mapping_n: int,
    mapping_skews: tuple[float, ...],
    mapping_bench: bool,
    mapping_bench_repeats: int,
    mapping_bench_polars: bool,
    mapping_bench_table: bool,
) -> None:
    if not mapping_bench:
        pytest.skip("pass --mapping-bench to run mapping timings")

    n = mapping_n
    repeats = mapping_bench_repeats
    run_polars = mapping_bench_polars
    run_table = mapping_bench_table
    rows: list[str] = []

    def _fmt(ms: float | None) -> str:
        return "skipped" if ms is None else f"{ms:.3f}"

    for skew in mapping_skews:
        for case in make_mapping_cases(n, skew):
            try:
                # Compile every available variant under the same torch.compile
                # settings so the timings are apples-to-apples.
                table_mapping = (
                    _compile_table_fn(case.table_mapping) if run_table else None
                )
                table_auto = _compile_table_fn(case.table_auto) if run_table else None
                table_opt_auto = _compile_table_fn(case.table_opt_auto)

                polars_plan_obj = case.polars_plan() if run_polars else None

                # ---- Warmup + correctness ----
                # table_opt_auto always runs and serves as the reference when
                # the brute-force paths are skipped via --no-mapping-bench-table.
                opt_auto_out = table_opt_auto()
                opt_auto_rows = _canonical_cols(opt_auto_out)
                del opt_auto_out
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                table_rows = None
                if table_mapping is not None:
                    try:
                        table_out = table_mapping()
                        table_rows = _canonical_rows(table_out)
                        del table_out
                    except Exception:
                        table_mapping = None
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                reference_rows = table_rows if table_rows is not None else opt_auto_rows
                assert torch.equal(opt_auto_rows, reference_rows), (
                    f"{case.label} table_opt_auto mismatch"
                )

                if table_auto is not None:
                    try:
                        auto_out = table_auto()
                        auto_rows = _canonical_rows(auto_out)
                        del auto_out
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                        assert torch.equal(auto_rows, reference_rows), (
                            f"{case.label} table_auto mismatch"
                        )
                    except AssertionError:
                        raise
                    except Exception:
                        table_auto = None
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()

                if run_polars:
                    polars_df = polars_plan_obj.collect(engine=POLARS_ENGINE)
                    polars_rows = _canonical_rows(
                        _polars_rows(polars_df, case.polars_piece_cols)
                    )
                    del polars_df
                    assert torch.equal(reference_rows, polars_rows), case.label

                # ---- Timing ----
                table_ms = _time_callable(table_mapping, repeats)
                auto_ms = _time_callable(table_auto, repeats)
                opt_auto_ms = _time_callable(table_opt_auto, repeats)
                polars_ms: float | None = None
                if run_polars:
                    polars_times: list[float] = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        polars_df = polars_plan_obj.collect(engine=POLARS_ENGINE)
                        polars_times.append(time.perf_counter() - start)
                        del polars_df
                    polars_ms = torch.tensor(polars_times).median().item() * 1e3

                satisfied = int(reference_rows.shape[0])
                alive_pct = 100.0 * satisfied / case.total_candidates

                rows.append(
                    f"{case.label}: satisfied={satisfied} "
                    f"total={case.total_candidates} alive_pct={alive_pct:.6f}% "
                    f"skew={case.skew:g} "
                    f"table_ms={_fmt(table_ms)} "
                    f"table_auto_ms={_fmt(auto_ms)} table_opt_auto_ms={_fmt(opt_auto_ms)} "
                    f"polars_ms={_fmt(polars_ms)}"
                )
            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                rows.append(f"{case.label}: OOM ({type(e).__name__}: skipping)")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            except AssertionError as e:
                rows.append(f"{case.label}: ASSERT_FAIL ({e})")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                rows.append(f"{case.label}: BACKEND_ERR ({type(e).__name__}: {e})")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            print(rows[-1], flush=True)
