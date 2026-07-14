"""Interaction-table continuous einsum — the thesis-chapter reference implementation.

Evaluates a continuous einsum in three steps (see the "Continuous Einsums via
Interaction Tables" section):

1. **Output coordinate construction** — one candidate array per distinct
   output variable, built from the input carriers alone: the unique points of
   a P-typed carrier if one exists (the output dimension becomes P), else the
   cells between the sorted-unique endpoints of *all* I-typed carriers (the
   output dimension becomes I).
2. **Table construction** — one boolean tensor with one axis per distinct
   output variable plus one axis per input operand. Every variable demands
   that all pairs of its carriers (output included) *meet*: point–point
   equality, point-in-interval containment, interval–interval overlap. Each
   condition is a single broadcast comparison.
3. **Value computation** — one ``torch.einsum`` of the table with the value
   arrays, plus one measure factor per contracted variable whose carriers are
   all I-typed (its reduction is an integral, so each tuple is weighted by
   the length of its common overlap). The result is a dense value grid over
   the candidate axes; :meth:`GridResult.to_coo` flattens it to a COO piece
   list.

Differences from :func:`continuous_einsum.ceinsum`: the two share integral
semantics (a contracted variable whose carriers are all intervals is
measure-weighted), but this implementation keeps explicit zero output pieces
(shapes depend only on the input coordinates) and supports only half-open
``"[)"`` intervals and ``"P"`` pinpoints. It materializes the full dense
table, so it is a specification-level reference, not an optimized path.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import torch

from ceinsum_equation import parse_equation
from ctensor import ContinuousTensor, continuous_tensor

PINPOINT = "P"
INTERVAL = "[)"

# A "carrier spec" is the coordinate arrays of one dimension of one tensor,
# tagged with the table axis it varies along: (spec, axis) where spec is
# (coord,) for a P-typed dimension and (start, end) for an I-typed one.


def _expand(t: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    """Reshape a 1-D array to broadcast along ``axis`` of the ``ndim``-axis table."""
    shape = [1] * ndim
    shape[axis] = t.shape[0]
    return t.reshape(shape)


def _meet(spec_a, axis_a, spec_b, axis_b, ndim) -> torch.Tensor:
    """Boolean broadcast tensor: coordinates of the two carriers intersect."""
    a_pin, b_pin = len(spec_a) == 1, len(spec_b) == 1
    if a_pin and b_pin:
        return _expand(spec_a[0], axis_a, ndim) == _expand(spec_b[0], axis_b, ndim)
    if a_pin:  # point in [s, e)
        c = _expand(spec_a[0], axis_a, ndim)
        s, e = (_expand(t, axis_b, ndim) for t in spec_b)
        return (s <= c) & (c < e)
    if b_pin:
        c = _expand(spec_b[0], axis_b, ndim)
        s, e = (_expand(t, axis_a, ndim) for t in spec_a)
        return (s <= c) & (c < e)
    s1, e1 = (_expand(t, axis_a, ndim) for t in spec_a)
    s2, e2 = (_expand(t, axis_b, ndim) for t in spec_b)
    return torch.maximum(s1, s2) < torch.minimum(e1, e2)


@dataclass
class GridResult:
    """Dense output of :func:`table_ceinsum`.

    ``grid`` has one axis per distinct output variable (``out_vars``, in order
    of first appearance on the output), indexed by that variable's candidate
    coordinates in ``cands``: ``(coords,)`` for a P-typed output dimension,
    ``(starts, ends)`` for an I-typed one.
    """

    grid: torch.Tensor
    cands: dict[str, tuple[torch.Tensor, ...]]
    out_vars: str
    out_indices: str

    def to_coo(self) -> ContinuousTensor:
        """Flatten the value grid into a COO piece list (explicit zeros kept).

        Output dimensions holding the same variable reuse that variable's
        candidate column.
        """
        if not self.out_vars:
            return continuous_tensor([], self.grid.reshape(1), [])
        ranges = [
            torch.arange(self.cands[v][0].shape[0], device=self.grid.device)
            for v in self.out_vars
        ]
        idx = torch.cartesian_prod(*ranges)
        if idx.ndim == 1:
            idx = idx[:, None]
        dims, props = [], []
        for ch in self.out_indices:
            spec = self.cands[ch]
            col = idx[:, self.out_vars.index(ch)]
            dims.append(tuple(t[col] for t in spec))
            props.append(PINPOINT if len(spec) == 1 else INTERVAL)
        # cartesian_prod orders rows last-axis-fastest, matching C-order reshape.
        return continuous_tensor(dims, self.grid.reshape(-1), props)


def _candidates(
    carriers: list[tuple[int, int]], operands: tuple[ContinuousTensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Step 1 — the candidate coordinate arrays of one output variable."""
    for n, d in carriers:  # a P-typed carrier's points already contain
        if operands[n].property[d] == PINPOINT:  # every realizable coordinate
            return (torch.unique(operands[n].dims[d][0]),)
    pts = torch.unique(
        torch.cat([t for n, d in carriers for t in operands[n].dims[d]])
    )
    return (pts[:-1], pts[1:])  # cells between consecutive endpoints


def table_ceinsum(equation: str, *operands: ContinuousTensor) -> GridResult:
    """Evaluate a continuous einsum via the dense interaction table."""
    for op in operands:
        for prop in op.property:
            if prop not in (PINPOINT, INTERVAL):
                raise NotImplementedError(
                    f"table_ceinsum supports only {INTERVAL!r} and {PINPOINT!r} "
                    f"dimensions, got {prop!r}"
                )

    in_indices, out_indices, carriers_of = parse_equation(equation, len(operands))
    out_vars = "".join(dict.fromkeys(out_indices))  # distinct, in output order
    for v in out_vars:
        if v not in carriers_of:
            raise ValueError(f"output index {v!r} appears in no input operand")

    num_in = len(operands)
    g = len(out_vars)
    ndim = g + num_in
    if ndim > len(string.ascii_lowercase):
        raise NotImplementedError("too many table axes for einsum subscripts")

    # Step 1 — output coordinate construction.
    cands = {v: _candidates(carriers_of[v], operands) for v in out_vars}

    # Step 2 — table construction: axes [out_vars..., operands...]; for every
    # variable, all pairs of its carriers must meet.
    dtype = operands[0].dtype
    device = operands[0].device
    sizes = [cands[v][0].shape[0] for v in out_vars] + [op.nnz for op in operands]

    table = None
    for var, in_carriers in carriers_of.items():
        specs = [(operands[n].dims[d], g + n) for n, d in in_carriers]
        if var in out_vars:
            specs.append((cands[var], out_vars.index(var)))
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                cond = _meet(*specs[i], *specs[j], ndim)
                table = cond if table is None else table & cond
    if table is None:
        table = torch.ones(1, dtype=torch.bool, device=device).reshape([1] * ndim)
    table = table.expand(sizes)

    # Step 3 — value computation: one discrete einsum of the table, the value
    # arrays, and one measure factor per all-interval contracted variable.
    letters = string.ascii_lowercase
    einsum_operands: list[torch.Tensor] = [table.to(dtype)]
    subscripts: list[str] = [letters[:ndim]]

    for var, in_carriers in carriers_of.items():
        if var in out_vars:
            continue  # output variables are pointwise: no measure factor
        if any(operands[n].property[d] == PINPOINT for n, d in in_carriers):
            continue  # summation semantics: no factor
        lo = hi = None
        for n, d in in_carriers:
            s = _expand(operands[n].dims[d][0], g + n, ndim)
            e = _expand(operands[n].dims[d][1], g + n, ndim)
            lo = s if lo is None else torch.maximum(lo, s)
            hi = e if hi is None else torch.minimum(hi, e)
        w = (hi - lo).clamp(min=0)
        axes = sorted({g + n for n, _ in in_carriers})
        einsum_operands.append(w.reshape([sizes[a] for a in axes]))
        subscripts.append("".join(letters[a] for a in axes))

    for n, op in enumerate(operands):
        einsum_operands.append(op.values)
        subscripts.append(letters[g + n])

    grid = torch.einsum(
        f"{','.join(subscripts)}->{letters[:g]}", *einsum_operands
    )
    return GridResult(grid, cands, out_vars, out_indices)
