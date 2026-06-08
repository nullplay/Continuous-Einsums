"""Kernel-backed variant of the optimized table mapping pipeline.

This is the same pipeline as :mod:`table_opt_mapping` — pick a "lead" condition
on the first two output axes, enumerate candidate pairs, then post-filter the
remaining conditions — with one difference: instead of generating the lead's
candidate pairs with hand-written ``torch.searchsorted`` binary searches, it
calls the Triton range-intersection kernel
:func:`Intersection_Kernels.intersection_kernel.intersect`.

Every lead the kernel can serve reduces to the kernel's single predicate

    A[i] intersects B[j]   <=>   As[i] <= Be[j]  AND  Bs[j] <= Ae[i]

where a point is the zero-width interval ``[c, c]``:

* ``eq``               (``A[a] == B[b]``)        → point vs point.
* ``overlap``          (interval/interval)        → interval vs interval.
* ``point_in_interval``(point in ``[s, e)``)      → point vs interval.

The kernel uses *closed* intervals (``<=``), so for the interval leads it
returns a superset of the strict-``<`` predicates the cases actually use. That
is harmless here: exactly as in :mod:`table_opt_mapping`, the ``overlap`` and
``point_in_interval`` leads re-check *all* their predicates in the post-filter
(``consumed = set()``), so any boundary-touching extra pairs are dropped. The
``eq`` lead is exact (``p == p``) and drops its predicate just like the opt
builder.

Leads the 1-D kernel can't model — a bare ``ineq`` half-line and the 3-axis
batched band (point-cloud case) — fall back to the exact ``table_opt_mapping``
implementations so this builder accepts the same DSL surface.

Output: tuple of 1-D long index tensors matching the column order of
``output``, identical in shape/contents to :func:`build_table_opt_mapping`.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from Intersection_Kernels.intersection_kernel import intersect
from table_opt_mapping import (
    _Lead,
    _apply_cmp,
    _build_three_axis_band_closure,
    _detect_lead,
    _detect_three_axis_band_lead,
    _eval_expr_pairs,
    _eval_expr_pk,
    _make_composite_eq_lead,
    _parse_cond,
    _window_pairs,
)


def _lead_axis_tensors(lead, axis_a: str, axis_b: str) -> dict[str, set[str]]:
    """Tensor names the lead consumes on each output axis (one spatial dim)."""
    p = lead.payload
    if lead.kind in ("eq", "ineq"):
        return {axis_a: {p["a_tensor"]}, axis_b: {p["b_tensor"]}}
    if lead.kind == "overlap":
        return {axis_a: {p["a_lo"], p["a_hi"]}, axis_b: {p["b_lo"], p["b_hi"]}}
    if lead.kind == "point_in_interval":
        return {
            p["point_axis"]: {p["point_tensor"]},
            p["interval_axis"]: {p["interval_lo"], p["interval_hi"]},
        }
    raise AssertionError(lead.kind)


def _axis_self_overlaps(parsed, axis: str, lead_tensors: set[str]) -> bool:
    """Whether ``axis``'s operand may self-overlap along this single axis.

    Synthetic operands are non-overlapping ND boxes (see ``synth_dataset``), so
    a piece's projection onto one axis is internally disjoint *iff* that axis
    carries exactly one spatial dimension. An output axis is multi-dimensional
    — and thus its 1-D projection may self-overlap (intervals) or hold
    duplicates (pinpoints) — iff it is indexed by any tensor beyond the lead's
    own dimension for that axis. (Relies on the non-overlapping-pieces property
    of the data; arbitrary self-overlapping 1-D input would need ``True`` here.)
    """
    for c in parsed:
        for expr in (c.left, c.right):
            for t in expr.terms:
                if t.axis == axis and t.tensor is not None and t.tensor not in lead_tensors:
                    return True
    return False


def _maybe_compile(fn: Callable) -> Callable:
    """``torch.compile`` ``fn`` for kernel fusion, falling back to eager if the
    compiler is unavailable. ``intersect()`` itself can't be traced (it launches
    Triton eagerly), but the post-lead tail — the pair post-filter and the dense
    ``(P, K)`` third-axis expansion + ``nonzero`` — is pure PyTorch, and that
    tail is what dominates the 3-axis cases. Compiling it fuses those passes,
    matching the advantage the brute-force/opt builders get from being compiled
    by the benchmark harness."""
    try:
        return torch.compile(fn, dynamic=True, fullgraph=False)
    except Exception:  # pragma: no cover - compiler unavailable
        return fn


def build_table_kernel_mapping(
    op: Mapping[str, torch.Tensor],
    output: Sequence[str],
    cond: Sequence[str],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Build a closure that joins ``op``'s 1-D tensors into index columns.

    Parameters
    ----------
    op
        Mapping from tensor name to a 1-D tensor. A tensor's length is the
        size of whatever axis indexes it; two tensors on the same axis must
        share that length.
    output
        Ordered axis names, length 2 or 3. The builder auto-leads on the
        first two (``axis_a``, ``axis_b``); a third axis is expanded last.
        Every output axis must be referenced by some condition.
    cond
        Condition strings in the ``tensor[axis] OP tensor[axis]`` DSL (see
        :mod:`table_opt_mapping`). ``OP`` is one of ``== != < <= > >=``;
        sides may be linear combos like ``2*x[a] - 1``.

    Returns
    -------
    A zero-arg callable. Each call runs the lead kernel + post-filter and
    returns a tuple of 1-D ``long`` index tensors, one per ``output`` axis,
    in ``output`` order. Shapes/contents match
    :func:`table_opt_mapping.build_table_opt_mapping`.

    Examples
    --------
    Equality lead (2 axes) — join rows where ``A[i] == B[j]``. This is the
    exact ``eq`` path: point-vs-point in the kernel, predicate dropped::

        import torch
        from table_kernel_mapping import build_table_kernel_mapping

        op = {
            "A": torch.tensor([10, 20, 30, 20]),   # axis "i", len 4
            "B": torch.tensor([20, 99, 30]),       # axis "j", len 3
        }
        run = build_table_kernel_mapping(op, output=("i", "j"), cond=["A[i] == B[j]"])
        idx_i, idx_j = run()
        # idx_i / idx_j are paired indices: A[idx_i] == B[idx_j] elementwise.
        # Here pairs (i=1,j=0), (i=3,j=0) [value 20] and (i=2,j=2) [value 30].

    Interval-overlap lead (2 axes) — join boxes on axis ``a`` against boxes
    on axis ``b`` that overlap. Two conditions form the ``overlap`` lead; the
    kernel widens to closed intervals, so both predicates are re-checked in
    the post-filter::

        op = {
            "As": torch.tensor([0.0, 5.0, 10.0]),  # a-interval starts, len 3
            "Ae": torch.tensor([3.0, 8.0, 15.0]),  # a-interval ends
            "Bs": torch.tensor([2.0, 9.0]),        # b-interval starts, len 2
            "Be": torch.tensor([6.0, 20.0]),       # b-interval ends
        }
        run = build_table_kernel_mapping(
            op,
            output=("a", "b"),
            cond=["As[a] < Be[b]", "Bs[b] < Ae[a]"],   # standard overlap test
        )
        idx_a, idx_b = run()
        # Pairs of (a, b) whose half-open intervals [start, end) overlap.

    A third axis name in ``output`` adds a post-expansion column; a bare
    inequality lead falls back to searchsorted, and a 3-axis band lead falls
    back to the exact ``table_opt_mapping`` closure — all transparently.
    """
    output = tuple(output)
    cond = tuple(cond)
    if len(output) < 2:
        raise ValueError("at least 2 output axes required")
    if len(output) > 3:
        raise NotImplementedError("only 2- and 3-axis cases are supported")
    if not cond:
        raise ValueError("at least one condition is required")
    if len(set(output)) != len(output):
        raise ValueError(f"duplicate axis names in output: {output}")

    parsed = [_parse_cond(c) for c in cond]

    # Infer + validate axis sizes from tensor lengths (same as the opt builder).
    axis_size: dict[str, int] = {}
    for p in parsed:
        for expr in (p.left, p.right):
            for t in expr.terms:
                if t.tensor is None:
                    continue
                if t.tensor not in op:
                    raise ValueError(f"unknown tensor {t.tensor!r}")
                size = op[t.tensor].shape[0]
                if t.axis in axis_size:
                    if axis_size[t.axis] != size:
                        raise ValueError(
                            f"axis {t.axis!r} bound to inconsistent sizes "
                            f"{axis_size[t.axis]} and {size} (from {t.tensor!r})"
                        )
                else:
                    axis_size[t.axis] = size

    for a in output:
        if a not in axis_size:
            raise ValueError(f"output axis {a!r} not referenced in any condition")

    axis_a = output[0]
    axis_b = output[1]
    axis_c = output[2] if len(output) == 3 else None

    pair_conds = []
    c_conds = []
    for p in parsed:
        ax = p.axes()
        if axis_c is not None and axis_c in ax:
            c_conds.append(p)
        else:
            if not ax.issubset({axis_a, axis_b}):
                raise ValueError(
                    f"condition references axes {set(ax)} outside leading pair "
                    f"{{{axis_a!r}, {axis_b!r}}}; this builder only auto-leads on "
                    f"the first two output axes"
                )
            pair_conds.append(p)

    # A composite-equality lead packs ≥2 equalities into one exact key, so the
    # kernel's eq path produces ≈ the final candidate set instead of the
    # single-coordinate flood. ``comp_overlap`` carries the packed key's
    # per-side duplicate flags, which override the generic self-overlap probe.
    lead = None
    comp_overlap: tuple[bool, bool] | None = None
    composite = _make_composite_eq_lead(op, pair_conds, axis_a, axis_b) if pair_conds else None
    if composite is not None:
        lead, op, overlap_a_c, overlap_b_c = composite
        comp_overlap = (overlap_a_c, overlap_b_c)
    if lead is None:
        lead = _detect_lead(pair_conds, axis_a, axis_b) if pair_conds else None
    three_axis_lead = None
    if lead is None and axis_c is not None:
        three_axis_lead = _detect_three_axis_band_lead(c_conds, axis_a, axis_b, axis_c)
    if lead is None and three_axis_lead is None:
        raise NotImplementedError(
            "no auto-leadable pattern among the conditions; "
            "supply a hand-written optimized mapping for this case"
        )

    # The 3-axis batched band is not a 1-D range join, so it can't be served by
    # the intersection kernel — reuse the exact opt closure for it. It is pure
    # PyTorch (searchsorted), so compile it for parity with the opt builder.
    if three_axis_lead is not None:
        return _maybe_compile(
            _build_three_axis_band_closure(
                op, axis_a, axis_b, axis_c, axis_size,
                three_axis_lead, pair_conds, c_conds,
            )
        )

    # Same post-filter bookkeeping as the opt builder: the interval leads widen
    # to closed intervals (the kernel uses ``<=``), so their own predicates must
    # be re-checked; ``eq`` is exact and drops its predicate.
    if lead.kind in ("overlap", "point_in_interval"):
        consumed: set[int] = set()
    else:
        consumed = set(lead.consumed)
    pair_post = [c for i, c in enumerate(pair_conds) if i not in consumed]

    # Per-axis self-overlap flags drive the kernel's write-path dispatch: a
    # single-dimension (1-D) operand is internally disjoint, so it can use the
    # fast atomic paths (overlap=False); a multi-dimension operand may
    # self-overlap on this axis and needs the general scatter path.
    if comp_overlap is not None:
        # Packed key: duplicate flags came straight from the key tensors. The
        # consumed equality tensors live only inside the key, so the generic
        # probe (which would see them as "extra" dims on the axis) doesn't apply.
        overlap_a, overlap_b = comp_overlap
    else:
        lead_tensors = _lead_axis_tensors(lead, axis_a, axis_b)
        overlap_a = _axis_self_overlaps(parsed, axis_a, lead_tensors.get(axis_a, set()))
        overlap_b = _axis_self_overlaps(parsed, axis_b, lead_tensors.get(axis_b, set()))

    def _run_lead() -> tuple[torch.Tensor, torch.Tensor]:
        # ---- 1) Run lead via the intersection kernel (or searchsorted for ineq). ----
        # Eager: ``intersect`` launches Triton and can't be traced by torch.compile.
        if lead.kind == "eq":
            # A[a] == B[b]  ⇔  point a intersects point b  (zero-width [v, v]).
            a = op[lead.payload["a_tensor"]]
            b = op[lead.payload["b_tensor"]]
            out_a, out_b = intersect(
                {"crd": a}, {"crd": b}, False, False, overlap_a, overlap_b
            )
            idx_a, idx_b = out_a.long(), out_b.long()

        elif lead.kind == "overlap":
            # a interval intersects b interval.
            pl = lead.payload
            A = {"start": op[pl["a_lo"]], "end": op[pl["a_hi"]]}
            B = {"start": op[pl["b_lo"]], "end": op[pl["b_hi"]]}
            out_a, out_b = intersect(A, B, True, True, overlap_a, overlap_b)
            idx_a, idx_b = out_a.long(), out_b.long()

        elif lead.kind == "point_in_interval":
            # point intersects interval. A = point side, B = interval side; the
            # overlap flags follow the point/interval orientation.
            pl = lead.payload
            point_is_a = pl["point_axis"] == axis_a
            ov_point = overlap_a if point_is_a else overlap_b
            ov_interval = overlap_b if point_is_a else overlap_a
            A = {"crd": op[pl["point_tensor"]]}
            B = {"start": op[pl["interval_lo"]], "end": op[pl["interval_hi"]]}
            out_a, out_b = intersect(A, B, False, True, ov_point, ov_interval)
            if point_is_a:
                idx_a, idx_b = out_a.long(), out_b.long()
            else:
                idx_a, idx_b = out_b.long(), out_a.long()

        elif lead.kind == "ineq":
            # A bare open half-line isn't a range-intersection; keep the exact
            # searchsorted enumeration from the opt builder.
            op_str = lead.payload["op"]
            a = op[lead.payload["a_tensor"]]
            b = op[lead.payload["b_tensor"]]
            perm = torch.argsort(b)
            b_sorted = b[perm].contiguous()
            J = b.shape[0]
            I_ = a.shape[0]
            if op_str == "<":
                lo = torch.searchsorted(b_sorted, a, right=True)
                hi = torch.full((I_,), J, dtype=lo.dtype, device=lo.device)
            elif op_str == "<=":
                lo = torch.searchsorted(b_sorted, a, right=False)
                hi = torch.full((I_,), J, dtype=lo.dtype, device=lo.device)
            elif op_str == ">":
                hi = torch.searchsorted(b_sorted, a, right=False)
                lo = None
            else:  # ">="
                hi = torch.searchsorted(b_sorted, a, right=True)
                lo = None
            idx_a, idx_b = _window_pairs(perm, lo, hi)

        else:
            raise AssertionError(lead.kind)
        return idx_a, idx_b

    def _tail(idx_a: torch.Tensor, idx_b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Pure PyTorch — this is the part torch.compile can fuse.
        # ---- 2) Apply remaining pair conds as a fused pair-level filter ----
        idx_pair = {axis_a: idx_a, axis_b: idx_b}
        keep_mask: torch.Tensor | None = None
        for cond in pair_post:
            left = _eval_expr_pairs(cond.left, op, idx_pair)
            right = _eval_expr_pairs(cond.right, op, idx_pair)
            m = _apply_cmp(left, cond.op, right)
            keep_mask = m if keep_mask is None else keep_mask & m
        if keep_mask is not None:
            idx_a = idx_a[keep_mask]
            idx_b = idx_b[keep_mask]
            idx_pair = {axis_a: idx_a, axis_b: idx_b}

        if axis_c is None:
            return (idx_a, idx_b)

        # ---- 3) Third axis: build (P, K) mask, then nonzero ----
        pk_mask: torch.Tensor | None = None
        for cond in c_conds:
            left = _eval_expr_pk(cond.left, op, idx_pair, axis_c)
            right = _eval_expr_pk(cond.right, op, idx_pair, axis_c)
            m = _apply_cmp(left, cond.op, right)
            pk_mask = m if pk_mask is None else pk_mask & m
        pk = torch.nonzero(pk_mask, as_tuple=False)
        p_idx, k_idx = pk[:, 0], pk[:, 1]
        return (idx_a[p_idx], idx_b[p_idx], k_idx)

    # Compile the tail when it does substantial work: a 3-axis case (dense
    # ``(P, K)`` expansion), or a multi-dimensional operand (``overlap`` lead →
    # large/scattered candidate set whose pair post-filter is heavy). When both
    # operands are 1-D and disjoint the candidate set is small and the filter is
    # cheap, so keep that fast path eager (compiling it only adds dispatch
    # overhead).
    compile_tail = axis_c is not None or overlap_a or overlap_b
    tail_fn = _maybe_compile(_tail) if compile_tail else _tail

    def kernel_mapping() -> tuple[torch.Tensor, ...]:
        idx_a, idx_b = _run_lead()
        return tail_fn(idx_a, idx_b)

    return kernel_mapping
