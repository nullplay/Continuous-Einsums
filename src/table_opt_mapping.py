"""Declarative builder for the optimized table mapping pipeline.

Uses the same DSL as :mod:`table_mapping` (a ``op`` dict of 1-D tensors, an
ordered ``output`` of axis names, and a list of ``tensor[axis]`` expression
strings). Where the brute-force builder materializes the full N-D boolean
table, this builder picks a "lead" condition, enumerates candidate pairs via
``searchsorted``, then post-filters the rest.

Heuristics, picking among conditions that touch only the first two output
axes ``axis_a``, ``axis_b``:

1. Interval overlap pair ``s[a] < e[b]`` AND ``s[b] < e[a]`` (or with ``<=``)
   → band lead on midpoints with radius ``a_half + b_half.max()``.
2. Point-in-interval pair ``p[a] >= s[b]`` AND ``p[a] < e[b]`` (or the
   symmetric variant with sides swapped) → band lead with per-row radius
   ``b_half``.
3. Single equality ``A[a] == B[b]`` → equality lead.
4. Single inequality with both sides single indexed-tensor refs → inequality
   lead (``<``, ``<=``, ``>``, ``>=``).

Each lead emits candidate pairs ``(idx_a, idx_b)``. Remaining
``(axis_a, axis_b)``-only conditions are applied as a 1-D boolean mask over
the pairs. For band leads (``overlap`` and ``point_in_interval``) the lead's
own two predicates are included in this post-filter, so the result stays
correct even when interval widths vary across rows. Equality and inequality
leads are already exact and drop their predicate.

A third output axis ``axis_c`` is then handled by broadcasting surviving
pairs against ``axis_c``'s values to form a ``(P, K)`` mask and calling
``torch.nonzero`` for the final column.

Output: tuple of 1-D long index tensors matching the column order of
``output`` passed to :func:`build_table_opt_mapping`.

Limitations: cases whose only feasible lead involves three axes at once
(for example the point-cloud case where the band radius depends on the third
axis's values) are not auto-derived — :func:`build_table_opt_mapping` raises
a ``NotImplementedError`` and the case must keep its hand-written form.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch


# ---------------------------------------------------------------------------
# Parsing: condition string → structured linear form.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Term:
    """``coef * tensor[axis]``. ``tensor`` / ``axis`` are ``None`` for pure
    constants, in which case ``coef`` is the constant value."""

    coef: float
    tensor: str | None
    axis: str | None


@dataclass(frozen=True)
class _Expr:
    """Sum of terms (linear combination of indexed-tensor refs and constants)."""

    terms: tuple[_Term, ...]

    def axes(self) -> frozenset[str]:
        return frozenset(t.axis for t in self.terms if t.axis is not None)


@dataclass(frozen=True)
class _Cond:
    left: _Expr
    op: str  # one of "==", "!=", "<", "<=", ">", ">="
    right: _Expr

    def axes(self) -> frozenset[str]:
        return self.left.axes() | self.right.axes()


_CMP = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _parse_expr(node) -> _Expr:
    if isinstance(node, ast.Expression):
        return _parse_expr(node.body)
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise ValueError("only `tensor[axis]` references are supported")
        if not isinstance(node.slice, ast.Name):
            raise ValueError("only single-name axis subscripts are supported")
        return _Expr((_Term(1.0, node.value.id, node.slice.id),))
    if isinstance(node, ast.Constant):
        return _Expr((_Term(float(node.value), None, None),))
    if isinstance(node, ast.UnaryOp):
        inner = _parse_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return _Expr(tuple(_Term(-t.coef, t.tensor, t.axis) for t in inner.terms))
        if isinstance(node.op, ast.UAdd):
            return inner
        raise ValueError(f"unsupported unary op {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _parse_expr(node.left)
        right = _parse_expr(node.right)
        if isinstance(node.op, ast.Add):
            return _Expr(left.terms + right.terms)
        if isinstance(node.op, ast.Sub):
            return _Expr(
                left.terms + tuple(_Term(-t.coef, t.tensor, t.axis) for t in right.terms)
            )
        if isinstance(node.op, ast.Mult):
            l_const = all(t.axis is None for t in left.terms)
            r_const = all(t.axis is None for t in right.terms)
            if l_const:
                k = sum(t.coef for t in left.terms)
                return _Expr(tuple(_Term(k * t.coef, t.tensor, t.axis) for t in right.terms))
            if r_const:
                k = sum(t.coef for t in right.terms)
                return _Expr(tuple(_Term(k * t.coef, t.tensor, t.axis) for t in left.terms))
            raise ValueError("only constant * indexed-tensor multiplication is supported")
        raise ValueError(f"unsupported binary op {type(node.op).__name__}")
    raise ValueError(f"unsupported AST node {type(node).__name__}")


def _parse_cond(s: str) -> _Cond:
    tree = ast.parse(s, mode="eval").body
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
        raise ValueError(f"condition must be a single comparison: {s!r}")
    return _Cond(
        left=_parse_expr(tree.left),
        op=_CMP[type(tree.ops[0])],
        right=_parse_expr(tree.comparators[0]),
    )


# ---------------------------------------------------------------------------
# Searchsorted primitives.
# ---------------------------------------------------------------------------


def _window_pairs(
    perm: torch.Tensor,
    lo: torch.Tensor | None,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row ``i``, enumerate ``(i, j)`` such that ``j_sort ∈ [lo[i], hi[i])``
    and ``j = perm[j_sort]`` indexes the original (unsorted) array.

    ``lo=None`` is shorthand for an implicit zero-vector (windows start at 0).
    That avoids one P-sized gather on the ``gt``/``ge`` ineq leads.
    """
    counts = hi if lo is None else hi - lo
    I = counts.shape[0]
    i_pairs = torch.repeat_interleave(
        torch.arange(I, dtype=torch.long, device=hi.device), counts
    )
    row_starts = counts.cumsum(0) - counts
    offsets = -row_starts if lo is None else lo - row_starts
    j_sorted = offsets[i_pairs] + torch.arange(
        i_pairs.shape[0], dtype=torch.long, device=hi.device
    )
    return i_pairs, perm[j_sorted]


# ---------------------------------------------------------------------------
# Lead detection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lead:
    kind: str  # "eq" | "ineq" | "overlap" | "point_in_interval" | "three_axis_band"
    consumed: tuple[int, ...]  # indices into the cond list this lead was detected over
    payload: dict


_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def _expr_is_indexed_tensor(expr: _Expr) -> tuple[str, str] | None:
    """Return (tensor, axis) if expr is a bare ``tensor[axis]``."""
    if len(expr.terms) != 1:
        return None
    t = expr.terms[0]
    if t.axis is None or t.coef != 1.0:
        return None
    return (t.tensor, t.axis)


def _normalize_to_a_op_b(c: _Cond, axis_a: str, axis_b: str) -> tuple[str, str, str] | None:
    """If cond is ``t1[axis_a] OP t2[axis_b]`` or its swap, return
    ``(a_tensor, op, b_tensor)`` with LHS normalized to the ``axis_a`` side.
    None if cond isn't a pure binary tensor cond between axes ``axis_a`` and
    ``axis_b``."""
    lhs = _expr_is_indexed_tensor(c.left)
    rhs = _expr_is_indexed_tensor(c.right)
    if lhs is None or rhs is None:
        return None
    lhs_tensor, lhs_axis = lhs
    rhs_tensor, rhs_axis = rhs
    if lhs_axis == axis_a and rhs_axis == axis_b:
        return (lhs_tensor, c.op, rhs_tensor)
    if lhs_axis == axis_b and rhs_axis == axis_a:
        return (rhs_tensor, _FLIP[c.op], lhs_tensor)
    return None


def _try_overlap(c1: _Cond, c2: _Cond, axis_a: str, axis_b: str) -> dict | None:
    """Detect `s[a] < e[b] AND s[b] < e[a]` (in any order, with <= permitted)."""
    n1 = _normalize_to_a_op_b(c1, axis_a, axis_b)
    n2 = _normalize_to_a_op_b(c2, axis_a, axis_b)
    if n1 is None or n2 is None:
        return None

    def split(n):
        a_tensor, op, b_tensor = n
        if op in ("<", "<="):
            return ("a<b", a_tensor, b_tensor, op)
        if op in (">", ">="):
            return ("b<a", a_tensor, b_tensor, op)
        return None

    s1, s2 = split(n1), split(n2)
    if s1 is None or s2 is None:
        return None
    if s1[0] == s2[0]:
        return None
    # Identify which is a_lo < b_hi and which is b_lo < a_hi.
    if s1[0] == "a<b":
        a_lo, b_hi = s1[1], s1[2]
        a_hi, b_lo = s2[1], s2[2]
    else:
        a_lo, b_hi = s2[1], s2[2]
        a_hi, b_lo = s1[1], s1[2]
    if a_lo == a_hi or b_lo == b_hi:
        return None
    return {"a_lo": a_lo, "a_hi": a_hi, "b_lo": b_lo, "b_hi": b_hi}


def _try_point_in_interval(
    c1: _Cond, c2: _Cond, axis_a: str, axis_b: str
) -> dict | None:
    """Detect ``p[a] >= s[b] AND p[a] < e[b]`` (or with ``b`` as point and
    ``a`` as interval). Returns dict describing point side and interval
    side."""
    for point_axis, interval_axis in ((axis_a, axis_b), (axis_b, axis_a)):
        n1 = _normalize_to_a_op_b(c1, point_axis, interval_axis)
        n2 = _normalize_to_a_op_b(c2, point_axis, interval_axis)
        if n1 is None or n2 is None:
            continue

        def split(n):
            point_tensor, op, interval_tensor = n
            if op in (">=", ">"):
                return ("ge", point_tensor, interval_tensor)
            if op in ("<", "<="):
                return ("lt", point_tensor, interval_tensor)
            return None

        s1, s2 = split(n1), split(n2)
        if s1 is None or s2 is None:
            continue
        if s1[0] == s2[0]:
            continue
        if s1[1] != s2[1]:  # point tensor must match across the two conds
            continue
        ge = s1 if s1[0] == "ge" else s2
        lt = s1 if s1[0] == "lt" else s2
        return {
            "point_axis": point_axis,
            "interval_axis": interval_axis,
            "point_tensor": ge[1],
            "interval_lo": ge[2],
            "interval_hi": lt[2],
        }
    return None


def _try_eq(c: _Cond, axis_a: str, axis_b: str) -> dict | None:
    if c.op != "==":
        return None
    n = _normalize_to_a_op_b(c, axis_a, axis_b)
    if n is None:
        return None
    return {"a_tensor": n[0], "b_tensor": n[2]}


def _try_ineq(c: _Cond, axis_a: str, axis_b: str) -> dict | None:
    if c.op not in ("<", "<=", ">", ">="):
        return None
    n = _normalize_to_a_op_b(c, axis_a, axis_b)
    if n is None:
        return None
    return {"a_tensor": n[0], "op": n[1], "b_tensor": n[2]}


def _detect_lead(pair_conds: list[_Cond], axis_a: str, axis_b: str) -> _Lead | None:
    n = len(pair_conds)
    # 1) interval overlap pair
    for i in range(n):
        for j in range(i + 1, n):
            d = _try_overlap(pair_conds[i], pair_conds[j], axis_a, axis_b)
            if d is not None:
                return _Lead("overlap", (i, j), d)
    # 2) point-in-interval pair
    for i in range(n):
        for j in range(i + 1, n):
            d = _try_point_in_interval(pair_conds[i], pair_conds[j], axis_a, axis_b)
            if d is not None:
                return _Lead("point_in_interval", (i, j), d)
    # 3) single equality
    for i, c in enumerate(pair_conds):
        d = _try_eq(c, axis_a, axis_b)
        if d is not None:
            return _Lead("eq", (i,), d)
    # 4) single inequality
    for i, c in enumerate(pair_conds):
        d = _try_ineq(c, axis_a, axis_b)
        if d is not None:
            return _Lead("ineq", (i,), d)
    return None


# ---------------------------------------------------------------------------
# 3-axis batched band lead: ``r[b] OP a_tensor[a] + c_tensor[c]`` paired with
# the symmetric upper bound. Used by point-cloud-style cases where the band's
# endpoints combine an axis-a tensor (e.g. mask center) with an axis-c tensor
# (e.g. kernel radius), and we want to enumerate ``(a, c)`` candidates per
# axis-b value. ``_window_pairs`` already handles the ``I = M·K`` case
# directly — we just build the ``(M·K,)`` lower/upper vectors via outer-sum.
# ---------------------------------------------------------------------------


def _expr_is_two_axis_sum(expr: _Expr) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Return ``((tensor_x, axis_x), (tensor_y, axis_y))`` if expr is exactly
    ``X[axis_x] + Y[axis_y]`` with both coefficients ``1.0``, no constant, and
    the two axes distinct. Term order in the returned tuple matches the order
    in the expression."""
    non_const = [t for t in expr.terms if t.axis is not None]
    const = sum(t.coef for t in expr.terms if t.axis is None)
    if const != 0.0 or len(non_const) != 2:
        return None
    if any(t.coef != 1.0 for t in non_const):
        return None
    if non_const[0].axis == non_const[1].axis:
        return None
    return (
        (non_const[0].tensor, non_const[0].axis),
        (non_const[1].tensor, non_const[1].axis),
    )


def _try_three_axis_band_endpoint(
    c: _Cond, axis_a: str, axis_b: str, axis_c: str
) -> tuple[str, str, str, str, bool] | None:
    """Detect a single endpoint of a 3-axis batched band.

    Returns ``(kind, b_tensor, a_tensor, c_tensor, right_arg)`` if the cond is
    one of:

      * ``B[b] >= A[a] + C[c]`` →  kind="lower", right_arg=False
      * ``B[b] >  A[a] + C[c]`` →  kind="lower", right_arg=True
      * ``B[b] <  A[a] + C[c]`` →  kind="upper", right_arg=False
      * ``B[b] <= A[a] + C[c]`` →  kind="upper", right_arg=True

    or the symmetric forms with the comparison flipped (``A[a] + C[c] <= B[b]``
    etc.). ``right_arg`` is the value to pass to ``searchsorted`` so that the
    returned index satisfies the predicate exactly without re-checking.
    """
    # Form 1: B[b] OP (A[a] + C[c])
    lhs = _expr_is_indexed_tensor(c.left)
    if lhs is not None and lhs[1] == axis_b:
        rhs = _expr_is_two_axis_sum(c.right)
        if rhs is not None and {rhs[0][1], rhs[1][1]} == {axis_a, axis_c}:
            a_term = rhs[0] if rhs[0][1] == axis_a else rhs[1]
            c_term = rhs[1] if rhs[0][1] == axis_a else rhs[0]
            b_tensor = lhs[0]
            if c.op == ">=":
                return ("lower", b_tensor, a_term[0], c_term[0], False)
            if c.op == ">":
                return ("lower", b_tensor, a_term[0], c_term[0], True)
            if c.op == "<":
                return ("upper", b_tensor, a_term[0], c_term[0], False)
            if c.op == "<=":
                return ("upper", b_tensor, a_term[0], c_term[0], True)
    # Form 2: (A[a] + C[c]) OP B[b] — flip op to normalize B on the left.
    rhs2 = _expr_is_indexed_tensor(c.right)
    if rhs2 is not None and rhs2[1] == axis_b:
        lhs2 = _expr_is_two_axis_sum(c.left)
        if lhs2 is not None and {lhs2[0][1], lhs2[1][1]} == {axis_a, axis_c}:
            a_term = lhs2[0] if lhs2[0][1] == axis_a else lhs2[1]
            c_term = lhs2[1] if lhs2[0][1] == axis_a else lhs2[0]
            b_tensor = rhs2[0]
            # A+C OP B  ⇔  B reverse_op A+C
            if c.op == "<=":   # A+C ≤ B  ⇔  B ≥ A+C
                return ("lower", b_tensor, a_term[0], c_term[0], False)
            if c.op == "<":    # A+C < B  ⇔  B > A+C
                return ("lower", b_tensor, a_term[0], c_term[0], True)
            if c.op == ">":    # A+C > B  ⇔  B < A+C
                return ("upper", b_tensor, a_term[0], c_term[0], False)
            if c.op == ">=":   # A+C ≥ B  ⇔  B ≤ A+C
                return ("upper", b_tensor, a_term[0], c_term[0], True)
    return None


def _detect_three_axis_band_lead(
    c_conds: list[_Cond], axis_a: str, axis_b: str, axis_c: str
) -> _Lead | None:
    """Find a (lower, upper) pair of 3-axis band endpoints sharing the same
    b-tensor and a-tensor. ``c_conds`` is the bucket of conds touching all
    three axes; ``consumed`` in the returned ``_Lead`` indexes into it."""
    endpoints: list[tuple[int, tuple[str, str, str, str, bool]]] = []
    for i, c in enumerate(c_conds):
        ep = _try_three_axis_band_endpoint(c, axis_a, axis_b, axis_c)
        if ep is not None:
            endpoints.append((i, ep))
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            idx_a, ep_a = endpoints[i]
            idx_b, ep_b = endpoints[j]
            if ep_a[0] == ep_b[0]:
                continue  # both lower or both upper
            if ep_a[1] != ep_b[1] or ep_a[2] != ep_b[2]:
                continue  # need same b_tensor and same a_tensor
            lower = ep_a if ep_a[0] == "lower" else ep_b
            upper = ep_a if ep_a[0] == "upper" else ep_b
            return _Lead(
                "three_axis_band",
                consumed=(idx_a, idx_b),
                payload={
                    "b_tensor": ep_a[1],
                    "a_tensor": ep_a[2],
                    "c_lower_tensor": lower[3],
                    "c_upper_tensor": upper[3],
                    "right_lo": lower[4],
                    "right_hi": upper[4],
                },
            )
    return None


# ---------------------------------------------------------------------------
# Codegen helpers: evaluate a parsed expression against gathered/broadcast data.
# ---------------------------------------------------------------------------


def _eval_expr_pairs(
    expr: _Expr,
    op: Mapping[str, torch.Tensor],
    idx_by_axis: Mapping[str, torch.Tensor],
) -> torch.Tensor | float:
    """Evaluate ``expr`` over candidate pairs. ``idx_by_axis[name]`` is a 1-D
    index tensor of length ``P`` for each axis referenced. Returns a 1-D tensor
    of length ``P`` (or a Python float if expr has no tensor refs)."""
    out: torch.Tensor | float | None = None
    for t in expr.terms:
        if t.axis is None:
            term = t.coef
        else:
            gathered = op[t.tensor][idx_by_axis[t.axis]]
            term = gathered if t.coef == 1.0 else t.coef * gathered
        out = term if out is None else out + term
    return 0.0 if out is None else out


def _eval_expr_pk(
    expr: _Expr,
    op: Mapping[str, torch.Tensor],
    pair_idx_by_axis: Mapping[str, torch.Tensor],
    c_axis: str,
) -> torch.Tensor | float:
    """Evaluate ``expr`` as a ``(P, K)`` tensor. Pair-axis tensors are gathered
    to shape ``(P, 1)``; the third-axis tensor broadcasts as ``(1, K)``."""
    out: torch.Tensor | float | None = None
    for t in expr.terms:
        if t.axis is None:
            term = t.coef
        elif t.axis == c_axis:
            v = op[t.tensor].unsqueeze(0)
            term = v if t.coef == 1.0 else t.coef * v
        else:
            v = op[t.tensor][pair_idx_by_axis[t.axis]].unsqueeze(1)
            term = v if t.coef == 1.0 else t.coef * v
        out = term if out is None else out + term
    return 0.0 if out is None else out


def _eval_expr_cached(
    expr: _Expr, cache: Mapping[tuple[str, str], torch.Tensor]
) -> torch.Tensor | float:
    """Evaluate ``expr`` using a pre-populated cache of gathered tensors.
    Reads each ``tensor[axis]`` from ``cache[(tensor, axis)]`` instead of
    regathering. The caller is responsible for populating the cache."""
    out: torch.Tensor | float | None = None
    for t in expr.terms:
        if t.axis is None:
            term = t.coef
        else:
            gathered = cache[(t.tensor, t.axis)]
            term = gathered if t.coef == 1.0 else t.coef * gathered
        out = term if out is None else out + term
    return 0.0 if out is None else out


_APPLY_CMP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _apply_cmp(left, op, right):
    return _APPLY_CMP[op](left, right)


# ---------------------------------------------------------------------------
# Closure builders.
# ---------------------------------------------------------------------------


def _build_three_axis_band_closure(
    op: Mapping[str, torch.Tensor],
    axis_a: str, axis_b: str, axis_c: str,
    axis_size: Mapping[str, int],
    lead: _Lead,
    pair_conds: list[_Cond],
    c_conds: list[_Cond],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Closure for the 3-axis batched-band lead.

    The lead enumerates ``(a, c)`` Cartesian pairs (``M·K`` of them) and for
    each finds ``b`` values inside ``[A[a] + C_lo[c], A[a] + C_hi[c])`` via one
    searchsorted on sorted ``B[b]``. The searchsorted ``right=`` flags are set
    so the lead is *exact* — no re-check is needed for the two consumed
    predicates. All other conds (whether they touch c-axis or not) post-filter
    on the surviving ``(idx_a, idx_b, idx_c)`` triples.
    """
    p = lead.payload
    consumed_c = set(lead.consumed)
    post_conds: list[_Cond] = [c for i, c in enumerate(c_conds) if i not in consumed_c]
    # Pure pair conds (if any) join the same post-filter.
    post_conds.extend(pair_conds)
    M_static = axis_size[axis_a]
    K_static = axis_size[axis_c]
    MK_static = M_static * K_static

    def opt_mapping() -> tuple[torch.Tensor, ...]:
        b_vals = op[p["b_tensor"]]
        a_vals = op[p["a_tensor"]]
        c_lo_vals = op[p["c_lower_tensor"]]
        c_hi_vals = op[p["c_upper_tensor"]]

        perm = torch.argsort(b_vals)
        b_sorted = b_vals[perm].contiguous()
        # (M, K) outer-sum → (M·K,) for searchsorted. mk_p will index into
        # this flattened space; recover (a, c) via integer division by K.
        lower = (a_vals.unsqueeze(1) + c_lo_vals.unsqueeze(0)).reshape(-1).contiguous()
        upper = (a_vals.unsqueeze(1) + c_hi_vals.unsqueeze(0)).reshape(-1).contiguous()
        lo = torch.searchsorted(b_sorted, lower, right=p["right_lo"])
        hi = torch.searchsorted(b_sorted, upper, right=p["right_hi"])
        counts = hi - lo
        mk_arange = torch.arange(MK_static, dtype=torch.long, device=lower.device)
        mk_p = torch.repeat_interleave(mk_arange, counts)
        row_starts = counts.cumsum(0) - counts
        offsets = lo - row_starts
        sorted_pos = offsets[mk_p] + torch.arange(
            mk_p.shape[0], dtype=torch.long, device=lower.device
        )
        idx_b = perm[sorted_pos]
        idx_a = mk_p // K_static
        idx_c = mk_p - idx_a * K_static

        idx_all = {axis_a: idx_a, axis_b: idx_b, axis_c: idx_c}
        # Pre-gather every (tensor, axis) the post-filter references, so a
        # tensor used in multiple conds (e.g. ``In_y`` in both the ≥ and <
        # endpoints) is gathered once.
        cache: dict[tuple[str, str], torch.Tensor] = {}
        for cond in post_conds:
            for expr in (cond.left, cond.right):
                for t in expr.terms:
                    if t.tensor is not None:
                        key = (t.tensor, t.axis)
                        if key not in cache:
                            cache[key] = op[t.tensor][idx_all[t.axis]]

        keep_mask: torch.Tensor | None = None
        for cond in post_conds:
            left = _eval_expr_cached(cond.left, cache)
            right = _eval_expr_cached(cond.right, cache)
            m = _apply_cmp(left, cond.op, right)
            keep_mask = m if keep_mask is None else keep_mask & m
        if keep_mask is not None:
            return (idx_a[keep_mask], idx_b[keep_mask], idx_c[keep_mask])
        return (idx_a, idx_b, idx_c)

    return opt_mapping


# ---------------------------------------------------------------------------
# Public builder.
# ---------------------------------------------------------------------------


def build_table_opt_mapping(
    op: Mapping[str, torch.Tensor],
    output: Sequence[str],
    cond: Sequence[str],
) -> Callable[[], tuple[torch.Tensor, ...]]:
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

    # Infer + validate axis sizes from tensor lengths.
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
            raise ValueError(
                f"output axis {a!r} not referenced in any condition"
            )

    axis_a = output[0]
    axis_b = output[1]
    axis_c = output[2] if len(output) == 3 else None

    pair_conds: list[_Cond] = []
    c_conds: list[_Cond] = []
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

    # Priority: 2-axis lead on (axis_a, axis_b) first (tighter windows). Only
    # fall back to a 3-axis batched band when no 2-axis lead is feasible —
    # that's the case when every condition couples all three axes at once
    # (point-cloud-style cases).
    lead = _detect_lead(pair_conds, axis_a, axis_b) if pair_conds else None
    three_axis_lead: _Lead | None = None
    if lead is None and axis_c is not None:
        three_axis_lead = _detect_three_axis_band_lead(c_conds, axis_a, axis_b, axis_c)
    if lead is None and three_axis_lead is None:
        raise NotImplementedError(
            "no auto-leadable pattern among the conditions; "
            "supply a hand-written optimized mapping for this case"
        )

    if three_axis_lead is not None:
        return _build_three_axis_band_closure(
            op, axis_a, axis_b, axis_c, axis_size,
            three_axis_lead, pair_conds, c_conds,
        )

    # Apply pair conds NOT consumed by the lead as a post-filter. Equality and
    # inequality leads are exact and can drop their own predicate. Band leads
    # (overlap, point_in_interval) widen the candidate window by ``W = max
    # width`` to handle the worst case, so the actual predicates must be
    # re-checked per pair to stay correct under non-uniform widths.
    if lead.kind in ("overlap", "point_in_interval"):
        consumed: set[int] = set()
    else:
        consumed = set(lead.consumed)
    pair_post = [c for i, c in enumerate(pair_conds) if i not in consumed]

    def opt_mapping() -> tuple[torch.Tensor, ...]:
        # ---- 1) Run lead: every kind reduces to (perm, lo, hi) over a sorted b-side. ----
        if lead.kind == "eq":
            # A[i] == B[j]   ⇔   B_sorted[j_sort] ∈ [A[i], A[i]]   (inclusive both ends)
            a = op[lead.payload["a_tensor"]]
            b = op[lead.payload["b_tensor"]]
            perm = torch.argsort(b)
            b_sorted = b[perm].contiguous()
            lo = torch.searchsorted(b_sorted, a, right=False)
            hi = torch.searchsorted(b_sorted, a, right=True)
            idx_a, idx_b = _window_pairs(perm, lo, hi)

        elif lead.kind == "ineq":
            # A[i] OP B[j], rewritten as a half-line on sorted B.
            op_str = lead.payload["op"]
            a = op[lead.payload["a_tensor"]]
            b = op[lead.payload["b_tensor"]]
            perm = torch.argsort(b)
            b_sorted = b[perm].contiguous()
            J = b.shape[0]
            I_ = a.shape[0]
            if op_str == "<":   # B[j] >  A[i]  →  j_sort ∈ [searchsorted(>A), J)
                lo = torch.searchsorted(b_sorted, a, right=True)
                hi = torch.full((I_,), J, dtype=lo.dtype, device=lo.device)
            elif op_str == "<=":  # B[j] >= A[i]
                lo = torch.searchsorted(b_sorted, a, right=False)
                hi = torch.full((I_,), J, dtype=lo.dtype, device=lo.device)
            elif op_str == ">":   # B[j] <  A[i]  →  j_sort ∈ [0, searchsorted(>=A))
                hi = torch.searchsorted(b_sorted, a, right=False)
                lo = None
            else:                 # op_str == ">="    B[j] <= A[i]
                hi = torch.searchsorted(b_sorted, a, right=True)
                lo = None
            idx_a, idx_b = _window_pairs(perm, lo, hi)

        elif lead.kind == "overlap":
            # a_lo[i] < b_hi[j] ∧ b_lo[j] < a_hi[i]
            #   ⇔   b_lo[j] ∈ ( a_lo[i] - W,  a_hi[i] )   with W = max_j (b_hi - b_lo)
            pl = lead.payload
            a_lo = op[pl["a_lo"]]
            a_hi = op[pl["a_hi"]]
            b_lo = op[pl["b_lo"]]
            b_hi = op[pl["b_hi"]]
            W = (b_hi - b_lo).max()
            perm = torch.argsort(b_lo)
            b_lo_sorted = b_lo[perm].contiguous()
            lo = torch.searchsorted(b_lo_sorted, a_lo - W, right=False)
            hi = torch.searchsorted(b_lo_sorted, a_hi,     right=False)
            idx_a, idx_b = _window_pairs(perm, lo, hi)

        elif lead.kind == "point_in_interval":
            # point[i] ∈ [interval_lo[j], interval_hi[j])
            #   ⇔   interval_lo[j] ∈ ( point[i] - W,  point[i] ]
            #   with W = max_j (interval_hi - interval_lo)
            pl = lead.payload
            point_is_a = pl["point_axis"] == axis_a
            point = op[pl["point_tensor"]]
            is_ = op[pl["interval_lo"]]
            ie_ = op[pl["interval_hi"]]
            W = (ie_ - is_).max()
            perm = torch.argsort(is_)
            is_sorted = is_[perm].contiguous()
            lo = torch.searchsorted(is_sorted, point - W, right=True)
            hi = torch.searchsorted(is_sorted, point,     right=True)
            row_p, col_p = _window_pairs(perm, lo, hi)
            # row_p indexes the point side; col_p indexes the interval side.
            if point_is_a:
                idx_a, idx_b = row_p, col_p
            else:
                idx_a, idx_b = col_p, row_p

        else:
            raise AssertionError(lead.kind)

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
        # c_conds is guaranteed non-empty here: every output axis must be
        # referenced (validated above), so an output axis_c implies at least
        # one cond references it and got bucketed into c_conds.
        pk_mask: torch.Tensor | None = None
        for cond in c_conds:
            left = _eval_expr_pk(cond.left, op, idx_pair, axis_c)
            right = _eval_expr_pk(cond.right, op, idx_pair, axis_c)
            m = _apply_cmp(left, cond.op, right)
            pk_mask = m if pk_mask is None else pk_mask & m
        pk = torch.nonzero(pk_mask, as_tuple=False)
        p_idx, k_idx = pk[:, 0], pk[:, 1]
        return (idx_a[p_idx], idx_b[p_idx], k_idx)

    return opt_mapping
