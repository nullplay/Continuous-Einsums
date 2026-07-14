"""Step 3 — merge: reduce the candidates into the output by coordinates.

The product step emits one candidate per mask entry: a value plus output
coordinates — a pinpoint coordinate or a half-open interval per dimension.
When the einsum has a reduction, several candidates can claim the same
region, either exactly (identical coordinates — the terms of the contraction
sum) or partially (intervals that overlap without coinciding). :func:`merge`
resolves both at once: it returns an equivalent tensor whose pieces are
disjoint, with values summed per region and exact zeros dropped.

Pinpoints and intervals are handled uniformly by working in *rank space*:
each dimension's coordinates map to dense integer ranks (one 1-D
``torch.unique``), so a pinpoint becomes a single cell id and an interval
becomes a run of cells between its endpoint ranks. Merging is then integer
bookkeeping:

1. **Dedup** — candidates with identical coordinates have identical rank
   tuples; pack the rank columns into one int64 key (re-ranking between
   packs keeps ids bounded) and sum per key. This is the discrete half of
   the merge.
2. **Cut** — the pinpoint ranks pack into a *group* id: pieces in different
   groups can never overlap. With one interval dimension a sweep-line
   resolves partial overlaps without any expansion: sort the ``±value``
   deposits by (group, coordinate) and cumsum; each segment between
   consecutive endpoints reads its value off the running sum. Deposits are
   interleaved rather than pre-summed per breakpoint, so the running sum
   returns to *exact* zero between disjoint pieces — no floating-point
   residue pieces. With several interval dimensions the sum must wait (two
   candidates can overlap in one dim while disjoint in another), so
   candidates are first cut at their group's breakpoints one dimension at a
   time (``repeat_interleave`` expansion); afterwards every pair is per-dim
   identical or disjoint and a final dedup finishes the job.

Everything is built from fast 1-D primitives — ``sort``, 1-D ``unique``,
``cumsum``, ``index_add_``, ``repeat_interleave`` — never
``torch.unique(dim=0)``, whose row-comparison path is 20–50× slower on CPU
than 1-D ``unique`` over the same data.
"""

from __future__ import annotations

import torch

from ctensor import ContinuousTensor, is_pinpoint


def _pack(cols: list[torch.Tensor], sizes: list[int]) -> tuple[torch.Tensor, int]:
    """Dense ids in ``[0, n)`` for the integer-tuple key formed by ``cols``.

    Successively packs columns into one int64 key (``key * size + col``),
    re-ranking after each pack so intermediate ids stay below the row count —
    the running product of raw sizes could overflow int64, ``rows * size``
    cannot. A single column must already be dense (true for rank columns:
    every rank is used by construction). Returns ``(ids, n)``.
    """
    key = cols[0]
    n = sizes[0]
    for col, size in zip(cols[1:], sizes[1:]):
        uniq, key = torch.unique(key * size + col, return_inverse=True)
        n = int(uniq.shape[0])
    return key, n


def _sweep(
    g: torch.Tensor | None,
    s: torch.Tensor,
    e: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped sweep-line over one interval dimension.

    Sorts the ``2P`` endpoints by (group, coordinate) — one argsort, plus one
    stable argsort when groups exist — and cumsums the interleaved ``+v``/
    ``-v`` deposits and ``±1`` coverage counts in that order. The segment
    between consecutive sorted endpoints holds the running sums; kept iff it
    is covered, nonzero, and has positive width (duplicate endpoints make
    zero-width segments, which is how repeated breakpoints collapse without
    any ``unique``). Deposits within a group cancel to exactly zero by the
    group's last endpoint, so segments spanning two groups have zero coverage
    and drop out.

    Returns ``(group, start, end, value)`` per kept segment (group ``None``
    when ``g`` is ``None``).
    """
    P = values.shape[0]
    device = values.device
    x = torch.cat([s, e])
    dep = torch.cat([values, -values])
    one = torch.ones(P, dtype=torch.long, device=device)
    cov = torch.cat([one, -one])

    order = torch.argsort(x)
    gs = None
    if g is not None:
        g2 = torch.cat([g, g])
        order = order[torch.argsort(g2[order], stable=True)]
        gs = g2[order]
    xs = x[order]
    val = torch.cumsum(dep[order], 0)[:-1]
    cnt = torch.cumsum(cov[order], 0)[:-1]

    keep = (cnt > 0) & (val != 0) & (xs[1:] > xs[:-1])
    grp = gs[:-1][keep] if g is not None else None
    return grp, xs[:-1][keep], xs[1:][keep], val[keep]


def merge(
    coords: list[tuple[torch.Tensor, ...]],
    values: torch.Tensor,
    out_property: tuple[str, ...],
) -> ContinuousTensor:
    """Merge candidates into the canonical disjoint COO output.

    ``coords`` holds one spec per output dimension — ``(coord,)`` for a
    pinpoint dim, ``(start, end)`` for a half-open interval dim — with every
    column of length ``P``; ``values`` is the candidate value per row.
    Candidates covering the same region are summed; regions summing to exact
    zero are dropped. Overlapping interval candidates are cut at their
    group's breakpoints, so the output pieces of a group tile a grid over
    its candidates' endpoints.
    """
    out_property = tuple(out_property)
    P = int(values.shape[0])
    ndim = len(out_property)
    device = values.device

    if ndim == 0:
        # Scalar output: every candidate lands on the one (empty) cell.
        if P == 0:
            return ContinuousTensor((), values, ())
        return ContinuousTensor((), values.sum().reshape(1), ())
    if P == 0:
        return ContinuousTensor(tuple(tuple(spec) for spec in coords), values, out_property)

    pin = [d for d in range(ndim) if is_pinpoint(out_property[d])]
    itv = [d for d in range(ndim) if not is_pinpoint(out_property[d])]

    # ---- 1) Rank space: dense integer ranks per dimension. --------------
    pin_vals: dict[int, torch.Tensor] = {}
    pin_rank: dict[int, torch.Tensor] = {}
    for d in pin:
        u, r = torch.unique(coords[d][0], return_inverse=True)
        pin_vals[d], pin_rank[d] = u, r
    bounds: dict[int, torch.Tensor] = {}
    rs: dict[int, torch.Tensor] = {}
    re_: dict[int, torch.Tensor] = {}
    for d in itv:
        s, e = coords[d]
        u, r = torch.unique(torch.cat([s, e]), return_inverse=True)
        bounds[d], rs[d], re_[d] = u, r[:P], r[P:]

    # ---- 2) Dedup: sum candidates with identical coordinates. -----------
    cols = [pin_rank[d] for d in pin] + [c for d in itv for c in (rs[d], re_[d])]
    sizes = [int(pin_vals[d].shape[0]) for d in pin] + [
        x for d in itv for x in (int(bounds[d].shape[0]),) * 2
    ]
    inv, D = _pack(cols, sizes)
    vals = torch.zeros(D, dtype=values.dtype, device=device)
    vals.index_add_(0, inv, values)
    rep = torch.empty(D, dtype=torch.long, device=device)
    rep[inv] = torch.arange(P, device=device)
    pin_rank = {d: pin_rank[d][rep] for d in pin}
    rs = {d: rs[d][rep] for d in itv}
    re_ = {d: re_[d][rep] for d in itv}

    # Group id from the pinpoint ranks; pieces in different groups can never
    # overlap, so all interval work happens within a group.
    if pin:
        g, G = _pack(
            [pin_rank[d] for d in pin], [int(pin_vals[d].shape[0]) for d in pin]
        )
        pin_of_group = {}
        for d in pin:
            t = torch.empty(G, dtype=pin_vals[d].dtype, device=device)
            t[g] = pin_vals[d][pin_rank[d]]
            pin_of_group[d] = t
    else:
        g, G = torch.zeros(D, dtype=torch.long, device=device), 1
        pin_of_group = {}

    dims: list[tuple[torch.Tensor, ...]] = [()] * ndim

    if not itv:
        # All-pinpoint: the dedup was the whole merge.
        keep = vals != 0
        for d in pin:
            dims[d] = (pin_vals[d][pin_rank[d]][keep],)
        return ContinuousTensor(tuple(dims), vals[keep], out_property)

    if len(itv) == 1:
        d0 = itv[0]
        grp, out_s, out_e, out_v = _sweep(
            g if pin else None, bounds[d0][rs[d0]], bounds[d0][re_[d0]], vals
        )
        for d in pin:
            dims[d] = (pin_of_group[d][grp],)
        dims[d0] = (out_s, out_e)
        return ContinuousTensor(tuple(dims), out_v, out_property)

    # ---- 3) N-D cut: split candidates at their group's breakpoints, one
    # dimension at a time; then a final dedup over the all-integer cell key.
    seg_cols: list[torch.Tensor] = []
    seg_lookup: list[tuple[torch.Tensor, int]] = []  # (U rows, rank modulus)
    for d in itv:
        R = int(bounds[d].shape[0])
        Pd = int(vals.shape[0])
        # Rows (group, endpoint-rank) packed into int64; unique sorts them so
        # each group's breakpoints are one contiguous ascending run, and each
        # candidate covers exactly the rows [s_idx, e_idx) of its group.
        packed = torch.cat([g, g]) * R + torch.cat([rs[d], re_[d]])
        U, inv2 = torch.unique(packed, return_inverse=True)
        s_idx, e_idx = inv2[:Pd], inv2[Pd:]

        counts = e_idx - s_idx
        rep2 = torch.repeat_interleave(
            torch.arange(Pd, dtype=torch.long, device=device), counts
        )
        offs = counts.cumsum(0) - counts
        seg = s_idx[rep2] + (
            torch.arange(int(counts.sum()), dtype=torch.long, device=device)
            - offs[rep2]
        )

        g = g[rep2]
        vals = vals[rep2]
        seg_cols = [c[rep2] for c in seg_cols] + [seg]
        rs = {dd: rs[dd][rep2] for dd in itv}
        re_ = {dd: re_[dd][rep2] for dd in itv}
        seg_lookup.append((U, R))

    inv3, C = _pack(
        [g] + seg_cols, [G] + [int(U.shape[0]) for U, _ in seg_lookup]
    )
    cell_vals = torch.zeros(C, dtype=vals.dtype, device=device)
    cell_vals.index_add_(0, inv3, vals)
    rep3 = torch.empty(C, dtype=torch.long, device=device)
    rep3[inv3] = torch.arange(int(vals.shape[0]), device=device)

    keep = cell_vals != 0
    kept = rep3[keep]  # one representative replica per kept cell
    for d in pin:
        dims[d] = (pin_of_group[d][g[kept]],)
    for k, d in enumerate(itv):
        U, R = seg_lookup[k]
        seg = seg_cols[k][kept]
        # Row seg of U is the segment [bound(U[seg]), bound(U[seg+1])); row
        # seg+1 is in the same group because seg < the candidate's e_idx row.
        dims[d] = (bounds[d][U[seg] % R], bounds[d][U[seg + 1] % R])
    return ContinuousTensor(tuple(dims), cell_vals[keep], out_property)
