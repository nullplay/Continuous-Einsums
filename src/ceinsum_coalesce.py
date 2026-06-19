"""Step 6 — coalesce: rewrite overlapping output pieces into a non-overlapping
set, summing values where pieces overlap.

Contracting an index that is interleaved with an output index (e.g. a pinpoint
``j`` in ``ij,i->i``, stored as several COO pieces each with its own ``i``
interval) can leave the output with pieces that overlap *along an output
dimension*. Such a tensor still evaluates correctly — a point's value is the
sum of every piece covering it — but it is not in minimal non-overlapping form.
This step canonicalizes it: it lays a grid over every distinct boundary
coordinate per output dim and sums the covering pieces in each grid cell, so
``ij,i->i`` with overlapping ``i`` intervals yields disjoint pieces whose values
are the per-region sums.

Boundary handling: containment is ``start <= cell`` and ``cell_end <= end``,
which is exact for half-open ``[)`` / ``(]`` (and pinpoint) dimensions. The
measure-zero boundary contributions of fully-closed ``[]`` or fully-open ``()``
dimensions are not separately represented — the COO model carries one property
per dimension and cannot mix point and interval pieces on a single axis. Pieces
that merely touch (zero-measure overlap) are left untouched.
"""

from __future__ import annotations

import torch

from ctensor import ContinuousTensor, is_pinpoint


def _has_interior_overlap(ct: ContinuousTensor) -> bool:
    """True if any two distinct pieces overlap with positive measure.

    Interval dims overlap when ``max(starts) < min(ends)`` (strict, so merely
    touching boundaries do not count); pinpoint dims overlap on equal coords.
    Two pieces overlap as regions iff they overlap along *every* dimension.
    """
    n = ct.nnz
    if n < 2 or ct.ndim == 0:
        return False
    mask = torch.ones((n, n), dtype=torch.bool, device=ct.device)
    for d in range(ct.ndim):
        spec = ct.dims[d]
        if is_pinpoint(ct.property[d]):
            c = spec[0]
            dim_overlap = c.unsqueeze(0) == c.unsqueeze(1)
        else:
            s, e = spec[0], spec[1]
            lo = torch.maximum(s.unsqueeze(0), s.unsqueeze(1))
            hi = torch.minimum(e.unsqueeze(0), e.unsqueeze(1))
            dim_overlap = lo < hi
        mask &= dim_overlap
    mask.fill_diagonal_(False)
    return bool(mask.any())


def coalesce(ct: ContinuousTensor) -> ContinuousTensor:
    """Return an equivalent tensor whose pieces do not overlap.

    A no-op (returns ``ct`` unchanged) when the pieces are already disjoint, so
    it never reshuffles an already-canonical output.
    """
    if not _has_interior_overlap(ct):
        return ct

    ndim = ct.ndim
    dtype, device = ct.dtype, ct.device

    # Per output dim: the elementary cells the dim is partitioned into.
    #   interval -> consecutive breakpoint pairs (lo, hi)
    #   pinpoint -> the distinct coordinates
    per_dim: list[tuple] = []
    counts: list[int] = []
    for d in range(ndim):
        spec = ct.dims[d]
        if is_pinpoint(ct.property[d]):
            coords = torch.unique(spec[0])                  # sorted
            per_dim.append(("P", coords))
            counts.append(int(coords.shape[0]))
        else:
            bps = torch.unique(torch.cat([spec[0], spec[1]]))  # sorted breakpoints
            per_dim.append(("I", bps[:-1], bps[1:]))
            counts.append(int(bps.shape[0]) - 1)

    # Cartesian product of per-dim cell indices -> one row per grid cell.
    ranges = [torch.arange(c, device=device) for c in counts]
    if any(c == 0 for c in counts):
        grid = torch.empty((0, ndim), dtype=torch.long, device=device)
    elif ndim == 1:
        grid = ranges[0].unsqueeze(1)
    else:
        grid = torch.cartesian_prod(*ranges).reshape(-1, ndim)
    num_cells = int(grid.shape[0])

    # For each cell, which pieces cover it? AND containment across dims.
    contain = torch.ones((num_cells, ct.nnz), dtype=torch.bool, device=device)
    cell_dims: list[tuple] = []
    for d in range(ndim):
        sel = grid[:, d]
        spec = ct.dims[d]
        info = per_dim[d]
        if info[0] == "P":
            cell_c = info[1][sel]                                   # (num_cells,)
            contain &= cell_c.unsqueeze(1) == spec[0].unsqueeze(0)
            cell_dims.append(("P", cell_c))
        else:
            cell_lo, cell_hi = info[1][sel], info[2][sel]           # (num_cells,)
            ps, pe = spec[0], spec[1]
            contain &= (ps.unsqueeze(0) <= cell_lo.unsqueeze(1)) & (
                cell_hi.unsqueeze(1) <= pe.unsqueeze(0))
            cell_dims.append(("I", cell_lo, cell_hi))

    values = contain.to(dtype) @ ct.values                          # (num_cells,)
    keep = (values != 0).nonzero(as_tuple=True)[0]                  # drop empty regions

    out_dims: list[tuple[torch.Tensor, ...]] = []
    for cd in cell_dims:
        if cd[0] == "P":
            out_dims.append((cd[1][keep],))
        else:
            out_dims.append((cd[1][keep], cd[2][keep]))
    return ContinuousTensor(tuple(out_dims), values[keep], ct.property)
