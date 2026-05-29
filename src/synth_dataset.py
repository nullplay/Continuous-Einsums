"""Non-overlapping ND box synthesis for mapping tests.

:func:`create_nd_pieces` is the main entry point: it returns ``n`` pieces in
``D`` dimensions where every piece occupies a distinct cell of a regular grid
over ``∏_d [0, max_d)``. Two pieces therefore differ in at least one cell
index and are disjoint along that axis, regardless of whether the axis is
declared ``'pinpoint'`` or ``'interval'``. Pinpoint coordinates land on cell
centers (deterministic per cell) so two operands that pick the same cell
produce equal pinpoint values — the equality joins in :mod:`test_mapping`
need that.

The ``skew`` knob in ``[0, 1]`` replaces the old ``low/med/high`` intersect
levels:

* ``skew == 0`` — cells are sampled uniformly from the grid (Efraimidis-Spirakis
  reservoir with equal weights).
* ``skew == 1`` — cells are heavily clustered toward the origin corner; two
  operands generated with different seeds end up sharing far more cells, which
  in turn drives a higher alive ratio in the mapping joins.

For the diagonal-condition cases the public helpers :func:`sample_cell_coords`
and :func:`fill_pieces_from_cells` let callers reuse a single cell selection
for several independent sub-interval draws (so e.g. ``A.i0`` and ``A.i1`` can
both live inside the same cell and are guaranteed to overlap).

The implementation is vectorized end-to-end (one ``topk`` for cell selection,
elementwise tensor ops per dim) and contains no Python-side per-piece loop.
"""

from __future__ import annotations

from math import ceil
from typing import Sequence

import torch

PINPOINT = "pinpoint"
INTERVAL = "interval"

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The cell grid is sized to hold ``DENSITY_FACTOR × n`` cells. Two operands
# sampled with the same grid but different seeds therefore share ~``n /
# DENSITY_FACTOR`` cells in expectation at ``skew=0``. Higher skew shrinks the
# effectively used region and pushes more cells into the shared set.
DENSITY_FACTOR = 2

# Skew=1 should make the weight ratio between the origin cell and the
# diagonally opposite cell large enough to dominate the random tiebreaker
# inside top-n cell selection. ``exp(8) ≈ 3000`` is comfortably above the
# typical ``-log(U)`` magnitude (median ~0.37).
_SKEW_ALPHA = 8.0


def cells_per_dim_for(n: int, D: int) -> int:
    """Number of cells per axis that ``create_nd_pieces`` will use."""
    target = max(n * DENSITY_FACTOR, 2)
    return max(2, int(ceil(target ** (1.0 / D))))


def _decompose(flat: torch.Tensor, base: int, D: int) -> tuple[torch.Tensor, ...]:
    """Convert a flat index in ``[0, base**D)`` to its ``D`` per-axis indices."""
    coords = []
    rem = flat
    for _ in range(D):
        coords.append(rem % base)
        rem = rem // base
    return tuple(coords)


def _make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    gen_dev = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=gen_dev).manual_seed(int(seed))


def sample_cell_coords(
    n: int,
    D: int,
    cells_per_dim: int,
    skew: float = 0.0,
    *,
    seed: int | None = None,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, ...]:
    """Sample ``n`` distinct cell indices from a ``cells_per_dim^D`` grid.

    Returns a tuple of ``D`` long tensors of shape ``(n,)``, each entry being
    the per-axis integer cell index of one of the ``n`` picked cells. Cells
    are picked via Efraimidis-Spirakis weighted reservoir with weights
    ``exp(-alpha · L1_distance_to_origin)`` and ``alpha = SKEW_ALPHA · skew``.
    """
    total_cells = cells_per_dim ** D
    if n > total_cells:
        raise RuntimeError(
            f"cell grid {cells_per_dim}^{D}={total_cells} cannot hold n={n}"
        )

    gen = _make_generator(seed, device)
    flat = torch.arange(total_cells, device=device)
    cell_coords = torch.stack(_decompose(flat, cells_per_dim, D), dim=1).to(dtype)
    denom = float(cells_per_dim - 1) if cells_per_dim > 1 else 1.0
    dist = (cell_coords / denom).mean(dim=1)

    u = torch.rand(total_cells, device=device, dtype=dtype, generator=gen)
    alpha = _SKEW_ALPHA * float(skew)
    key = torch.log(-torch.log(u.clamp(min=1e-30))) + alpha * dist
    _, picked = torch.topk(key, n, largest=False, sorted=False)
    picked, _ = torch.sort(picked)
    return _decompose(picked, cells_per_dim, D)


def fill_pieces_from_cells(
    cell_idx: Sequence[torch.Tensor],
    dim_kinds: Sequence[str],
    dim_maxes: Sequence[float],
    cells_per_dim: int,
    *,
    seed: int | None = None,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> list[tuple[torch.Tensor, ...]]:
    """Fill in pinpoint / interval values for an existing cell-index selection.

    ``cell_idx`` is a sequence of ``D`` ``(n,)`` long tensors as returned by
    :func:`sample_cell_coords`. For each axis:

    * ``'pinpoint'`` → ``(coord,)`` with ``coord`` placed at the cell center.
    * ``'interval'`` → ``(start, end)`` drawn uniformly inside the cell.

    Sub-interval RNG is driven by ``seed``, so the same ``cell_idx`` with
    different ``seed`` values yields independent draws inside the same cells —
    that's how the diagonal cases get two distinct intervals per cell.
    """
    if len(cell_idx) != len(dim_kinds) or len(dim_kinds) != len(dim_maxes):
        raise ValueError("cell_idx / dim_kinds / dim_maxes length mismatch")
    gen = _make_generator(seed, device)
    out: list[tuple[torch.Tensor, ...]] = []
    for ci, kind, max_d in zip(cell_idx, dim_kinds, dim_maxes):
        n = int(ci.shape[0])
        cell = ci.to(dtype)
        cell_size = float(max_d) / cells_per_dim
        cell_lo = cell * cell_size
        if kind == PINPOINT:
            out.append((cell_lo + cell_size * 0.5,))
        elif kind == INTERVAL:
            u1 = torch.rand(n, device=device, dtype=dtype, generator=gen)
            u2 = torch.rand(n, device=device, dtype=dtype, generator=gen)
            lo = torch.minimum(u1, u2)
            hi = torch.maximum(u1, u2)
            out.append((cell_lo + lo * cell_size, cell_lo + hi * cell_size))
        else:
            raise ValueError(f"unknown dim kind: {kind!r}")
    return out


def create_nd_pieces(
    n: int,
    dim_kinds: Sequence[str],
    dim_maxes: Sequence[float],
    skew: float = 0.0,
    *,
    seed: int | None = None,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> list[tuple[torch.Tensor, ...]]:
    """Sample ``n`` non-overlapping ND boxes from a grid over ``∏[0, max_d)``.

    Thin composition of :func:`sample_cell_coords` and
    :func:`fill_pieces_from_cells`; see those for the per-step contract. The
    same ``seed`` drives both cell selection and sub-interval draws.
    """
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    if len(dim_kinds) != len(dim_maxes):
        raise ValueError("dim_kinds / dim_maxes length mismatch")
    if not 0.0 <= float(skew) <= 1.0:
        raise ValueError(f"skew must be in [0, 1], got {skew}")
    for k in dim_kinds:
        if k not in (PINPOINT, INTERVAL):
            raise ValueError(f"unknown dim kind: {k!r}")

    D = len(dim_kinds)
    if n == 0:
        empty = torch.zeros(0, dtype=dtype, device=device)
        return [
            (empty.clone(),) if k == PINPOINT else (empty.clone(), empty.clone())
            for k in dim_kinds
        ]

    cpd = cells_per_dim_for(n, D)
    cells = sample_cell_coords(
        n, D, cpd, skew, seed=seed, device=device, dtype=dtype,
    )
    return fill_pieces_from_cells(
        cells, dim_kinds, dim_maxes, cpd,
        seed=seed, device=device, dtype=dtype,
    )


def create_2d_boxes(
    n: int,
    max_x: float,
    max_y: float,
    skew: float = 0.0,
    *,
    seed: int | None = None,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """``n`` non-overlapping 2D rectangles → ``(n, 4)`` tensor of ``[x_s, x_e, y_s, y_e]``.

    Thin wrapper over :func:`create_nd_pieces` with both axes set to
    ``'interval'``; matches the ``2dbox(N, max)`` sketch in the design note.
    """
    pieces = create_nd_pieces(
        n,
        (INTERVAL, INTERVAL),
        (max_x, max_y),
        skew,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    (xs, xe), (ys, ye) = pieces
    return torch.stack((xs, xe, ys, ye), dim=1)
