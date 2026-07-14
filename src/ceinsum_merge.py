"""Step 3 — merge: reduce the candidates into the output by coordinates.

The discrete half (:func:`merge_discrete`) sums the candidates that share the
*same* output coordinates: a ``torch.unique`` over the product step's grouping
key assigns each join tuple an output piece id, and an ``index_add_`` performs
the accumulating scatter. This produces a valid COO output whose pieces may
still *partially* overlap along interval dimensions when the einsum has a
reduction; :func:`coalesce` (the continuous half) rewrites those into disjoint
pieces.

Coalesce is a vectorized sweep-line. In 1-D the candidates' endpoints are the
breakpoints: each candidate deposits ``+v`` at its start and ``-v`` at its
end, and a running sum over the sorted breakpoints yields each segment's
value, while the same trick with ``±1`` counts coverage so uncovered gaps are
dropped. With several interval dimensions the sum may not happen per dimension
(two candidates can overlap in one dim while disjoint in another), so the
candidates are first *cut* one dimension at a time at that dimension's
breakpoints; afterwards any two candidates are per-dim identical or disjoint,
and the discrete merge (unique + scatter-add) finishes the job. Pinpoint
dimensions partition the candidates into independent groups (pieces only
overlap at equal pinpoint coordinates).

Exact-zero regions are dropped: explicit zeros do not survive coalescing.
Intervals are half-open ``[s, e)``, so segments between breakpoints tile the
line exactly — no boundary is double-counted and pieces that merely touch are
never summed.
"""

from __future__ import annotations

import torch

from ceinsum_product import Product
from ctensor import ContinuousTensor, is_pinpoint


def merge_discrete(product: Product, out_property: tuple[str, ...]) -> ContinuousTensor:
    """Sum candidates with identical output coordinates into one piece each.

    Tuples are grouped by the *provider piece columns* rather than the float
    coordinates: tuples agreeing on every output-providing operand necessarily
    share their output coordinates, and integer keys are robust.
    """
    P = int(product.values.shape[0])
    device = product.values.device

    if product.key_cols:
        key = torch.stack(product.key_cols, dim=1)  # (P, |providers|)
        _uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        num_out = int(_uniq.shape[0])
    else:
        # Scalar output (full contraction): all tuples collapse to one piece.
        inv = torch.zeros(P, dtype=torch.long, device=device)
        num_out = 1 if P > 0 else 0

    # One representative tuple per output piece supplies the coordinates
    # (every tuple in a group shares them).
    rep = torch.empty(num_out, dtype=torch.long, device=device)
    rep[inv] = torch.arange(P, dtype=torch.long, device=device)

    values = torch.zeros(num_out, dtype=product.values.dtype, device=device)
    values.index_add_(0, inv, product.values)

    dims = tuple(tuple(col[rep] for col in spec) for spec in product.coords)
    return ContinuousTensor(dims, values, out_property)


def _has_interior_overlap(ct: ContinuousTensor) -> bool:
    """True if any two distinct pieces overlap with positive measure.

    Interval dims overlap when ``max(starts) < min(ends)`` (strict, so merely
    touching boundaries do not count); pinpoint dims overlap on equal coords.
    Two pieces overlap as regions iff they overlap along *every* dimension.

    Quadratic in the piece count; used only to keep the N-D cut a no-op on
    already-disjoint outputs (cutting could otherwise fragment them).
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


def _group_breakpoints(
    g: torch.Tensor, s: torch.Tensor, e: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group unique breakpoints of one interval dimension.

    Builds ``(group, boundary)`` rows from all starts and ends and uniques
    them lexicographically, so each group's boundaries form one contiguous,
    sorted run. Returns ``(U, s_idx, e_idx)``: the unique rows and, per
    candidate, the row of its own start / end. Because every candidate's
    endpoints are themselves breakpoints of its group, the candidate covers
    exactly the segments ``[s_idx, e_idx)``.

    The composite key is float64: exact for boundaries already ≤ 64-bit floats
    and for group ids below 2^53.
    """
    gf = g.to(torch.float64)
    rows = torch.stack(
        [torch.cat([gf, gf]), torch.cat([s, e]).to(torch.float64)], dim=1
    )
    U, inv = torch.unique(rows, dim=0, return_inverse=True)
    P = s.shape[0]
    return U, inv[:P], inv[P:]


def _sweep_1d(
    g: torch.Tensor,
    s: torch.Tensor,
    e: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped 1-D sweep-line: deposit ``±v`` at the endpoints, running-sum.

    Returns ``(group, start, end, value)`` of the kept segments. All groups
    sweep in one pass: each candidate's deposits cancel within its group's
    row range, so the running sums return to zero at every group's last
    boundary and any segment spanning two groups has zero coverage and is
    dropped, along with genuine gaps. Zero-width candidates self-cancel.
    """
    U, s_idx, e_idx = _group_breakpoints(g, s, e)
    P = values.shape[0]
    device = values.device

    dval = torch.zeros(U.shape[0], dtype=values.dtype, device=device)
    dval.scatter_add_(0, s_idx, values)
    dval.scatter_add_(0, e_idx, -values)
    ones = torch.ones(P, dtype=torch.long, device=device)
    dcnt = torch.zeros(U.shape[0], dtype=torch.long, device=device)
    dcnt.scatter_add_(0, s_idx, ones)
    dcnt.scatter_add_(0, e_idx, -ones)

    seg_val = dval.cumsum(0)[:-1]                  # value on segment [U[o], U[o+1])
    seg_cnt = dcnt.cumsum(0)[:-1]                  # candidates covering it
    keep = (seg_cnt > 0) & (seg_val != 0)

    return (
        U[:-1, 0][keep].to(torch.long),
        U[:-1, 1][keep].to(s.dtype),
        U[1:, 1][keep].to(e.dtype),
        seg_val[keep],
    )


def _cut_then_merge(
    g: torch.Tensor,
    itv_specs: list[tuple[torch.Tensor, torch.Tensor]],
    values: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """N-D coalesce: cut per dimension, then merge discretely.

    Cuts every candidate at its group's breakpoints one interval dimension at
    a time — the sum must wait until candidates are per-dim identical or
    disjoint, since overlap in one dim does not imply overlap as regions.
    Each cut replicates a candidate into one piece per spanned segment
    (``repeat_interleave``), carrying all other columns along. Values are not
    rescaled: piece values are densities over their region. The final merge
    is the discrete one, a ``unique`` over the all-integer key (group,
    segment ids) plus ``index_add_``.

    Returns ``(group, [(start, end) per interval dim], value)``.
    """
    device = values.device
    starts = [s for s, _ in itv_specs]
    ends = [e for _, e in itv_specs]
    dtypes = [(s.dtype, e.dtype) for s, e in itv_specs]
    seg_cols: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []

    for k in range(len(itv_specs)):
        U, s_idx, e_idx = _group_breakpoints(g, starts[k], ends[k])
        boundaries.append(U[:, 1])

        counts = e_idx - s_idx                     # segments spanned; 0 → dropped
        P = counts.shape[0]
        rep = torch.repeat_interleave(
            torch.arange(P, dtype=torch.long, device=device), counts
        )
        offs = counts.cumsum(0) - counts
        seg = s_idx[rep] + (
            torch.arange(int(counts.sum()), dtype=torch.long, device=device)
            - offs[rep]
        )

        g = g[rep]
        values = values[rep]
        seg_cols = [c[rep] for c in seg_cols] + [seg]
        starts = [t[rep] for t in starts]
        ends = [t[rep] for t in ends]

    # Discrete merge over the all-integer key (group, seg_0, seg_1, ...).
    key = torch.stack([g] + seg_cols, dim=1)
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    out_values = torch.zeros(uniq.shape[0], dtype=values.dtype, device=device)
    out_values.index_add_(0, inv, values)
    kept = out_values != 0
    uniq, out_values = uniq[kept], out_values[kept]

    out_specs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, B in enumerate(boundaries):
        seg = uniq[:, 1 + k]
        s_dtype, e_dtype = dtypes[k]
        out_specs.append((B[seg].to(s_dtype), B[seg + 1].to(e_dtype)))
    return uniq[:, 0], out_specs, out_values


def coalesce(ct: ContinuousTensor) -> ContinuousTensor:
    """Return an equivalent tensor whose pieces do not overlap.

    Dispatch: pinpoint dims partition the pieces into groups (pieces only
    overlap at equal pinpoint coordinates); one interval dim runs the grouped
    sweep unconditionally (log-linear, and a semantic no-op on disjoint
    input); several interval dims run cut-then-merge behind the exact
    pairwise-overlap guard, which keeps already-disjoint outputs untouched.
    """
    if ct.nnz < 2 or ct.ndim == 0:
        return ct

    ndim = ct.ndim
    device = ct.device
    pin = [d for d in range(ndim) if is_pinpoint(ct.property[d])]
    itv = [d for d in range(ndim) if not is_pinpoint(ct.property[d])]

    # Group pieces by their exact pinpoint-coordinate tuple.
    if pin:
        pin_coords = torch.stack([ct.dims[d][0] for d in pin], dim=1)  # (nnz, |pin|)
        pin_uniq, g = torch.unique(pin_coords, dim=0, return_inverse=True)
    else:
        pin_uniq = None
        g = torch.zeros(ct.nnz, dtype=torch.long, device=device)

    if not itv:
        # All-pinpoint: coalescing is summing coincident pieces.
        values = torch.zeros(
            pin_uniq.shape[0], dtype=ct.values.dtype, device=device
        )
        values.index_add_(0, g, ct.values)
        kept = values != 0
        values = values[kept]
        grp = torch.arange(pin_uniq.shape[0], device=device)[kept]
        out_specs: list[tuple[torch.Tensor, torch.Tensor]] = []
    elif len(itv) == 1:
        s, e = ct.dims[itv[0]]
        grp, out_s, out_e, values = _sweep_1d(g, s, e, ct.values)
        out_specs = [(out_s, out_e)]
    else:
        if not _has_interior_overlap(ct):
            return ct
        grp, out_specs, values = _cut_then_merge(
            g, [ct.dims[d] for d in itv], ct.values
        )

    # Reassemble dims in the original dimension order.
    dims: list[tuple[torch.Tensor, ...]] = [()] * ndim
    for j, d in enumerate(pin):
        dims[d] = (pin_uniq[grp, j],)
    for k, d in enumerate(itv):
        dims[d] = out_specs[k]
    return ContinuousTensor(tuple(dims), values, ct.property)
