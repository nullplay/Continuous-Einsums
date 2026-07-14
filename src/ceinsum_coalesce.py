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

import itertools

import torch

from ctensor import ContinuousTensor, is_pinpoint


def coalesce_dense(piece, value, *, return_A=False, static_shapes=False):
    """
    Dense bipartite coalescing over all dimensions.

    piece: (N, D, 2)   piece[i, d, 0] / [i, d, 1] = lower / upper endpoint
    value: (N,) or (N, C...)

    Returns:
        out_piece: (*cell_shape, D, 2)
        out_value: (*cell_shape,) or (*cell_shape, C...)
        A:         (M, N) if return_A else None
        mask:      (*cell_shape,)  -- True iff the cell is covered by >= 1
                   input piece (and has positive width, in static mode)

    Instead of materialising the (M, N, D) containment tensor and doing an
    (M, N) @ (N, C) matmul, each input piece is located on the grid with
    searchsorted, giving per-dim cell ranges [s, e). Its value is scattered
    with inclusion-exclusion signs onto the 2^D corners of that range, and
    a cumsum along each dim recovers out_value:  O(2^D * N * C + M * C)
    work instead of O(M * N * (D + C)).

    static_shapes=True keeps duplicate endpoints (sort instead of unique),
    so every shape is a data-independent function of N (2N-1 cells per
    dim). This avoids torch.unique's device->host sync (needed to learn
    the output size) and gives torch.compile fully static shapes instead
    of unbacked symints. Degenerate zero-width cells are excluded by
    `mask`. Cost: more cells when endpoints repeat heavily.
    """
    if piece.ndim != 3 or piece.shape[-1] != 2:
        raise ValueError("`piece` must have shape (N, D, 2)")
    if value.shape[0] != piece.shape[0]:
        raise ValueError("`value.shape[0]` must match `piece.shape[0]`")

    N, D, _ = piece.shape
    tail_shape = tuple(value.shape[1:])

    endpoints = piece.transpose(0, 1).reshape(D, 2 * N)  # (D, 2N)

    # ------------------------------------------------------------
    # 1. Grid of cell boundaries per dimension.
    # ------------------------------------------------------------
    if static_shapes:
        # One batched sort, duplicates kept: fixed (D, 2N) shape.
        grids = endpoints.sort(dim=-1).values
        grid_list = list(grids.unbind(0))
    else:
        # Compact grid, but torch.unique has data-dependent output
        # shapes (graph breaks / recompiles under torch.compile).
        grid_list = [torch.unique(endpoints[d]) for d in range(D)]

    cell_shape = tuple(g.numel() - 1 for g in grid_list)

    # ------------------------------------------------------------
    # 2. Output pieces (broadcast, no meshgrid materialisation
    #    beyond the final stack).
    # ------------------------------------------------------------
    lows, highs = [], []
    for d, g in enumerate(grid_list):
        shape = [1] * D
        shape[d] = cell_shape[d]
        lows.append(g[:-1].reshape(shape).expand(cell_shape))
        highs.append(g[1:].reshape(shape).expand(cell_shape))

    out_lo = torch.stack(lows, dim=-1)                 # (*cell_shape, D)
    out_hi = torch.stack(highs, dim=-1)                # (*cell_shape, D)
    out_piece = torch.stack([out_lo, out_hi], dim=-1)  # (*cell_shape, D, 2)

    # ------------------------------------------------------------
    # 3. Per-piece cell ranges: piece i covers cells [s, e) along
    #    each dim. Endpoints are grid values, so searchsorted
    #    (side='left') hits them exactly, duplicates included.
    # ------------------------------------------------------------
    if static_shapes:
        lo = piece[:, :, 0].T.contiguous()             # (D, N)
        hi = piece[:, :, 1].T.contiguous()
        s = torch.searchsorted(grids, lo).T            # (N, D)
        e = torch.searchsorted(grids, hi).T
    else:
        lo = piece[:, :, 0].T.contiguous()             # (D, N)
        hi = piece[:, :, 1].T.contiguous()
        s = torch.stack(
            [torch.searchsorted(g, lo[d]) for d, g in enumerate(grid_list)], dim=-1
        )
        e = torch.stack(
            [torch.searchsorted(g, hi[d]) for d, g in enumerate(grid_list)], dim=-1
        )

    # ------------------------------------------------------------
    # 4. out_value via a D-dimensional difference array:
    #    scatter signed values onto the 2^D corners of each piece's
    #    cell range, then cumsum along every dim. A ones-channel is
    #    appended so the coverage count (-> mask) comes for free.
    # ------------------------------------------------------------
    value_2d = value.reshape(N, -1)
    C = value_2d.shape[1]
    payload = torch.cat([value_2d, value_2d.new_ones(N, 1)], dim=1)  # (N, C+1)

    diff = payload.new_zeros(tuple(c + 1 for c in cell_shape) + (C + 1,))
    for corner in itertools.product((0, 1), repeat=D):
        idx = tuple((e if bit else s)[:, d] for d, bit in enumerate(corner))
        signed = -payload if (sum(corner) % 2) else payload
        diff.index_put_(idx, signed, accumulate=True)

    for d in range(D):
        diff = diff.cumsum(dim=d)
    diff = diff[tuple(slice(0, c) for c in cell_shape)]  # trim the +1 pad

    out_value = diff[..., :C].reshape(cell_shape + tail_shape)
    mask = diff[..., C] > 0.5                            # coverage count > 0

    if static_shapes:
        mask = mask & (out_hi > out_lo).all(dim=-1)      # drop degenerate cells

    # ------------------------------------------------------------
    # 5. Optional dense A, built separably: one (cells_d, N)
    #    boolean per dim, combined by broadcasting -- never a
    #    (M, N, D) intermediate.
    # ------------------------------------------------------------
    A = None
    if return_A:
        per_dim = []
        for d in range(D):
            c = torch.arange(cell_shape[d], device=piece.device)
            per_dim.append((c[:, None] >= s[None, :, d]) & (c[:, None] < e[None, :, d]))
        A = per_dim[0]
        for d in range(1, D):
            A = A.unsqueeze(-2) & per_dim[d]             # (..., cells_d, N)
        A = A.reshape(-1, N)

    return out_piece, out_value, A, mask


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

    The interval dimensions are coalesced with the dense grid engine
    (:func:`coalesce_dense`). Pinpoint dimensions cannot be represented on that
    interval grid, so pieces are first grouped by their exact pinpoint-coordinate
    tuple — two pieces only overlap if their pinpoints coincide — and each group
    is coalesced independently over its interval dims.
    """
    if not _has_interior_overlap(ct):
        return ct

    ndim, nnz = ct.ndim, ct.nnz
    device = ct.device
    pin = [d for d in range(ndim) if is_pinpoint(ct.property[d])]
    itv = [d for d in range(ndim) if not is_pinpoint(ct.property[d])]

    # Group piece indices by their exact pinpoint-coordinate tuple. Pieces in
    # different groups can never overlap (a pinpoint dim overlaps only on equal
    # coords), so each group coalesces independently.
    if pin:
        pin_coords = torch.stack([ct.dims[d][0] for d in pin], dim=1)  # (nnz, |pin|)
        uniq, inv = torch.unique(pin_coords, dim=0, return_inverse=True)
        groups = uniq.shape[0]
    else:
        uniq = None
        inv = torch.zeros(nnz, dtype=torch.long, device=device)
        groups = 1

    # Per output dim, accumulate the kept cells across groups.
    out_cols: dict[int, list] = {d: [] for d in range(ndim)}
    out_vals: list[torch.Tensor] = []

    for g in range(groups):
        rows = (inv == g).nonzero(as_tuple=True)[0]
        if itv:
            piece = torch.stack(
                [
                    torch.stack([ct.dims[d][0][rows], ct.dims[d][1][rows]], dim=-1)
                    for d in itv
                ],
                dim=1,
            )  # (|rows|, |itv|, 2)
            out_piece, out_value, _, _ = coalesce_dense(piece, ct.values[rows])
            out_piece = out_piece.reshape(-1, len(itv), 2)
            out_value = out_value.reshape(-1)
            keep = (out_value != 0).nonzero(as_tuple=True)[0]  # drop empty regions
            out_piece, out_value = out_piece[keep], out_value[keep]
            m = int(out_value.shape[0])
            for k, d in enumerate(itv):
                out_cols[d].append((out_piece[:, k, 0], out_piece[:, k, 1]))
        else:
            # All-pinpoint output: coalescing is just summing coincident pieces.
            out_value = ct.values[rows].sum().reshape(1)
            m = 1
        out_vals.append(out_value)
        for j, d in enumerate(pin):
            out_cols[d].append(uniq[g, j].expand(m))

    values = torch.cat(out_vals) if out_vals else ct.values[:0]
    out_dims: list[tuple[torch.Tensor, ...]] = []
    for d in range(ndim):
        parts = out_cols[d]
        if is_pinpoint(ct.property[d]):
            out_dims.append((torch.cat(parts),))
        else:
            los = torch.cat([lo for lo, _ in parts])
            his = torch.cat([hi for _, hi in parts])
            out_dims.append((los, his))
    return ContinuousTensor(tuple(out_dims), values, ct.property)
