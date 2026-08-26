"""Equation-level benchmark cases for the end-to-end ``ceinsum`` experiments.

The mask-level suite in ``tests/test_mapping.py`` carries raw coordinate
tensors and DSL conditions but no :class:`ContinuousTensor` operands or einsum
equation, so it can't drive ``ceinsum()``. These cases mirror its geometry —
same ``create_nd_pieces`` grid, ``SPACE_MAX``, and per-operand seeds — at the
equation level.

Import order matters: this module imports torch and ``src/`` modules at the
top level, so runners must call ``bench_common.apply_device_mode()`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ctensor import ContinuousTensor, continuous_tensor
from synth_dataset import (
    INTERVAL,
    PINPOINT,
    cells_per_dim_for,
    create_nd_pieces,
    sample_cell_coords,
)

from bench_common import values_for

DTYPE = torch.float32

# Mirror tests/test_mapping.py: shared coordinate range and per-operand seeds.
SPACE_MAX = 64.0
SEED_A = 11
SEED_B = 22
SEED_C = 33
SEED_POINTS = 55

# Offset separating coordinate seeds from value seeds.
_VALUE_SEED = 500


@dataclass(frozen=True)
class CeinsumCase:
    label: str
    equation: str
    has_reduction: bool
    make_operands: Callable[[int, float, torch.device], tuple[ContinuousTensor, ...]]


def _interval_ct(
    n: int, skew: float, seed: int, ndim: int, device: torch.device
) -> ContinuousTensor:
    dims = create_nd_pieces(
        n, (INTERVAL,) * ndim, (SPACE_MAX,) * ndim, skew,
        seed=seed, device=device, dtype=DTYPE,
    )
    vals = values_for(n, seed + _VALUE_SEED, dtype=DTYPE).to(device)
    return continuous_tensor(dims, vals, ["[)"] * ndim)


def _make_pointwise_1d(n, skew, device):
    return (
        _interval_ct(n, skew, SEED_A, 1, device),
        _interval_ct(n, skew, SEED_B, 1, device),
    )


def _make_pointwise_2d(n, skew, device):
    return (
        _interval_ct(n, skew, SEED_A, 2, device),
        _interval_ct(n, skew, SEED_B, 2, device),
    )


def _make_diagonal(n, skew, device):
    # As in mapping case 07: A's two interval axes come from one 2D
    # create_nd_pieces call — they share [0, SPACE_MAX) but live in
    # independent cells, so the ``ii`` diagonal only survives on-diagonal.
    A = _interval_ct(n, skew, SEED_A, 2, device)
    B = _interval_ct(n, skew, SEED_B, 1, device)
    return (A, B)


def _make_matvec(n, skew, device):
    return (
        _interval_ct(n, skew, SEED_A, 2, device),
        _interval_ct(n, skew, SEED_B, 1, device),
    )


def _aligned_interval_ct(
    n: int, skew: float, seed: int, ndim: int, device: torch.device
) -> ContinuousTensor:
    """Interval pieces spanning exactly their grid cell (shared endpoints).

    With free sub-cell endpoints (``create_nd_pieces``), overlapping output
    candidates get cut at ~2n distinct breakpoints per axis, so a
    contraction's canonical output covers most of a (2n)^D grid — matmul's
    true answer is ~1e10 pieces at n=1e5, OOM on any hardware. Cell-aligned
    pieces share endpoints, so candidates either coincide exactly (merged by
    dedup) or are disjoint, bounding the output by the covered cell pairs.
    """
    cpd = cells_per_dim_for(n, ndim)
    cells = sample_cell_coords(
        n, ndim, cpd, skew, seed=seed, device=device, dtype=DTYPE
    )
    cell_size = SPACE_MAX / cpd
    dims = []
    for ci in cells:
        lo = ci.to(DTYPE) * cell_size
        dims.append((lo, lo + cell_size))
    vals = values_for(n, seed + _VALUE_SEED, dtype=DTYPE).to(device)
    return continuous_tensor(dims, vals, ["[)"] * ndim)


def _pinpoint_ct(
    n: int, skew: float, seed: int, ndim: int, device: torch.device
) -> ContinuousTensor:
    specs = create_nd_pieces(
        n, (PINPOINT,) * ndim, (SPACE_MAX,) * ndim, skew,
        seed=seed, device=device, dtype=DTYPE,
    )
    vals = values_for(n, seed + _VALUE_SEED, dtype=DTYPE).to(device)
    return continuous_tensor(specs, vals, ["P"] * ndim)


def _make_matmul(n, skew, device):
    # Cell-aligned operands so the contraction's output stays bounded (see
    # _aligned_interval_ct) and n=100k is feasible.
    return (
        _aligned_interval_ct(n, skew, SEED_A, 2, device),
        _aligned_interval_ct(n, skew, SEED_B, 2, device),
    )


def _make_triple_contract(n, skew, device):
    # D^PP_xy = A^PP_xy · B^II_xr · C^II_ry — the pinpoint operand pins the
    # output coordinates, so merge dedups instead of grid-cutting.
    return (
        _pinpoint_ct(n, skew, SEED_A, 2, device),
        _interval_ct(n, skew, SEED_B, 2, device),
        _interval_ct(n, skew, SEED_C, 2, device),
    )


def _make_box_search(n, skew, device):
    # As in mapping case 17: one fixed query box, n pinpoint candidates.
    box = continuous_tensor(
        [
            (torch.tensor([0.0], dtype=DTYPE, device=device),
             torch.tensor([SPACE_MAX / 4], dtype=DTYPE, device=device)),
            (torch.tensor([0.0], dtype=DTYPE, device=device),
             torch.tensor([SPACE_MAX / 4], dtype=DTYPE, device=device)),
        ],
        torch.ones(1, dtype=DTYPE, device=device),
        ["[)", "[)"],
    )
    (px,), (py,) = create_nd_pieces(
        n, (PINPOINT, PINPOINT), (SPACE_MAX, SPACE_MAX), skew,
        seed=SEED_POINTS, device=device, dtype=DTYPE,
    )
    points = continuous_tensor(
        [(px,), (py,)],
        values_for(n, SEED_POINTS + _VALUE_SEED, dtype=DTYPE).to(device),
        ["P", "P"],
    )
    return (box, points)


CEINSUM_CASES: tuple[CeinsumCase, ...] = (
    CeinsumCase("pointwise_1d_ii", "i,i->i", False, _make_pointwise_1d),
    CeinsumCase("pointwise_2d_ii", "ij,ij->ij", False, _make_pointwise_2d),
    CeinsumCase("diagonal_ii_i", "ii,i->i", False, _make_diagonal),
    CeinsumCase("matvec_ii", "ik,k->i", True, _make_matvec),
    CeinsumCase("matmul_ii", "ik,kj->ij", True, _make_matmul),
    CeinsumCase("triple_contract_pii", "ij,ik,kj->ij", True, _make_triple_contract),
    CeinsumCase("box_search", "xy,xy->xy", False, _make_box_search),
)

CASE_BY_LABEL = {c.label: c for c in CEINSUM_CASES}
