"""Step 1 — mask creation: join the operands' pieces under intersection
conditions.

The mask is a sparse COO tensor over the operands' piece positions, with one
axis per operand: entry ``(a, b, ...)`` exists iff piece ``a`` of operand 0,
piece ``b`` of operand 1, ... intersect on every shared index. Its coordinate
arrays are ``piece_idx`` (one column per operand); its value array carries the
integral measure.

Which coordinates must intersect is decided by the einsum expression: every
index shared between two operands contributes one intersection condition
(equality for pinpoints, overlap for intervals, containment for mixed). The
conditions are compiled to a small DSL and handed to a pluggable *mask
builder* — the manuscript's three strategies:

* :func:`mask_binary_search.build_binary_search_mask` (default) — sort one
  side and ``torch.searchsorted`` the other; log-linear.
* :func:`mask_dense.build_dense_mask` — the naive all-pair broadcast
  comparison; quadratic but branch-free and fast on small operands.
* a database join (e.g. polars ``join_where``) shares the same signature.

**Integral semantics.** When the einsum *reduces* over an index whose carriers
are all intervals, the sum over that index is an integral, and the mask entry
carries the *length* of the carriers' intersection on it (product over all
such indices) instead of a plain 0/1. Multiplying that weight into the product
step computes ``∫ A·B dk`` over each overlap region. A reduced index with any
pinpoint carrier keeps summation semantics (no factor), as do output indices.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import torch

from ctensor import ContinuousTensor, is_pinpoint, left_closed, right_closed
from mask_binary_search import build_binary_search_mask


class Mask(NamedTuple):
    """The sparse mask: COO coordinates plus the integral measure.

    * ``piece_idx`` — one 1-D long column per operand, all of length ``P``
      (the number of surviving join tuples); ``piece_idx[op][t]`` is the piece
      of operand ``op`` participating in tuple ``t``.
    * ``values`` — length ``P``, the product of intersection lengths over the
      reduced all-interval indices, or ``None`` when the mask is binary (no
      such index).
    """

    piece_idx: tuple[torch.Tensor, ...]
    values: torch.Tensor | None


# A "mask builder" turns (op, output, cond) into a 0-arg closure that runs
# the join. All strategies share this signature; their closures return either
# a tuple of 1-D index columns or a single 2-D ``(P, num_axes)`` tensor,
# which :func:`build_mask` normalizes.
MaskBuilder = Callable[..., Callable[[], object]]


# Tensor-name conventions shared by the `op` dict and the condition strings.
def _coord(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_coord"


def _start(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_start"


def _end(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_end"


def build_mapping_ops(operands: Sequence[ContinuousTensor]) -> dict[str, torch.Tensor]:
    """Collect every operand's coordinate tensors as the `op` dict.

    e.g. ``{"Op0_Dim0_start": ..., "Op0_Dim0_end": ..., "Op2_Dim0_coord": ...}``.
    """
    mapping_ops: dict[str, torch.Tensor] = {}
    for op_idx, op in enumerate(operands):
        for dim in range(op.ndim):
            if is_pinpoint(op.property[dim]):
                mapping_ops[_coord(op_idx, dim)] = op.dims[dim][0]
            else:
                mapping_ops[_start(op_idx, dim)] = op.dims[dim][0]
                mapping_ops[_end(op_idx, dim)] = op.dims[dim][1]
    return mapping_ops


def build_conditions(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
) -> list[str]:
    """Intersection conditions over the join axes.

    For each shared index, every pair of operand dims carrying it must
    intersect:

    * interval / interval  -> overlap (``sA <op> eB`` and ``sB <op> eA``)
    * pinpoint / interval  -> point-in-interval (``s <op> p`` and ``p <op> e``)
    * pinpoint / pinpoint  -> equality

    Bracket codes decide whether a touching boundary (``==``) counts: a
    comparison is non-strict (``<=``) only when *both* adjoining boundaries are
    closed, otherwise strict (``<``).
    """
    cond: list[str] = []
    for occ in index_to_operand_dims.values():
        for ai in range(len(occ)):
            for bi in range(ai + 1, len(occ)):
                opA, dimA = occ[ai]
                opB, dimB = occ[bi]
                pA = operands[opA].property[dimA]
                pB = operands[opB].property[dimB]
                # Axis names must be valid Python identifiers (the condition
                # DSL parses ``tensor[axis]`` and requires ``axis`` to be a Name,
                # not a numeric literal) — so "ax0", not "0".
                axA, axB = f"ax{opA}", f"ax{opB}"
                a_pin, b_pin = is_pinpoint(pA), is_pinpoint(pB)

                if a_pin and b_pin:
                    # cA == cB
                    cond.append(
                        f"{_coord(opA, dimA)}[{axA}] == {_coord(opB, dimB)}[{axB}]"
                    )
                elif a_pin:
                    # point A inside interval B = [sB, eB]:  sB <op> cA <op> eB
                    lo = "<=" if left_closed(pB) else "<"
                    hi = "<=" if right_closed(pB) else "<"
                    cond.append(
                        f"{_start(opB, dimB)}[{axB}] {lo} {_coord(opA, dimA)}[{axA}]"
                    )
                    cond.append(
                        f"{_coord(opA, dimA)}[{axA}] {hi} {_end(opB, dimB)}[{axB}]"
                    )
                elif b_pin:
                    # point B inside interval A = [sA, eA]:  sA <op> cB <op> eA
                    lo = "<=" if left_closed(pA) else "<"
                    hi = "<=" if right_closed(pA) else "<"
                    cond.append(
                        f"{_start(opA, dimA)}[{axA}] {lo} {_coord(opB, dimB)}[{axB}]"
                    )
                    cond.append(
                        f"{_coord(opB, dimB)}[{axB}] {hi} {_end(opA, dimA)}[{axA}]"
                    )
                else:
                    # interval A overlaps interval B:  sA <op1> eB  AND  sB <op2> eA
                    op1 = "<=" if (left_closed(pA) and right_closed(pB)) else "<"
                    op2 = "<=" if (left_closed(pB) and right_closed(pA)) else "<"
                    cond.append(
                        f"{_start(opA, dimA)}[{axA}] {op1} {_end(opB, dimB)}[{axB}]"
                    )
                    cond.append(
                        f"{_start(opB, dimB)}[{axB}] {op2} {_end(opA, dimA)}[{axA}]"
                    )
    return cond


def reduced_interval_indices(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
) -> list[list[tuple[int, int]]]:
    """Carrier lists of the reduced indices with integral semantics.

    A reduced (contracted) index integrates — its mask factor is the
    intersection length — exactly when *every* carrier dim is an interval.
    Any pinpoint carrier pins the coordinate to a point (a Dirac mass), and
    the contraction stays a plain sum with no factor.
    """
    return [
        occ
        for ix, occ in index_to_operand_dims.items()
        if ix not in out_indices
        and all(not is_pinpoint(operands[op].property[d]) for op, d in occ)
    ]


def compute_mask_values(
    operands: Sequence[ContinuousTensor],
    piece_idx: tuple[torch.Tensor, ...],
    reduced_occs: list[list[tuple[int, int]]],
) -> torch.Tensor | None:
    """The integral measure per join tuple: ∏ intersection lengths.

    For each reduced all-interval index, gather every carrier's interval
    through the mask positions and take ``min(ends) - max(starts)``. A single
    carrier (an index reduced within one operand) contributes its own piece
    length. Returns ``None`` when there is no such index (binary mask).
    """
    mv: torch.Tensor | None = None
    for occ in reduced_occs:
        lo: torch.Tensor | None = None
        hi: torch.Tensor | None = None
        for op, d in occ:
            s = operands[op].dims[d][0][piece_idx[op]]
            e = operands[op].dims[d][1][piece_idx[op]]
            lo = s if lo is None else torch.maximum(lo, s)
            hi = e if hi is None else torch.minimum(hi, e)
        # The mask conditions already guarantee a positive overlap for strict
        # (open) predicates; closed brackets can produce an exactly-zero
        # touching overlap, which correctly contributes nothing.
        w = (hi - lo).clamp(min=0)
        mv = w if mv is None else mv * w
    return mv


def build_mask(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
    builder: MaskBuilder | None = None,
) -> Mask:
    """Run the piece-join and return the sparse :class:`Mask`.

    ``builder`` selects the join strategy (see :data:`MaskBuilder`); the
    default is the binary-search builder. All strategies produce the same
    join; only their output layout differs, which this function normalizes.
    """
    if builder is None:
        builder = build_binary_search_mask
    num = len(operands)
    if num > 3:
        raise NotImplementedError(
            "ceinsum supports at most 3 operands "
            "(the join builder leads on 2-3 output axes)"
        )

    if num == 1:
        # No join: every piece of the lone operand survives as-is.
        nnz = operands[0].nnz
        piece_idx: tuple[torch.Tensor, ...] = (
            torch.arange(nnz, dtype=torch.long, device=operands[0].device),
        )
    else:
        mapping_ops = build_mapping_ops(operands)
        cond = build_conditions(operands, index_to_operand_dims)
        # One output axis per operand: piece index of operand k on axis "axk".
        mapping_output = tuple(f"ax{i}" for i in range(num))

        run = builder(mapping_ops, output=mapping_output, cond=cond)
        out = run()
        if isinstance(out, torch.Tensor):
            # Dense builder: a single (P, num_axes) tensor from nonzero.
            # Column k is operand k's piece index — split into columns.
            piece_idx = tuple(out[:, k] for k in range(num))
        else:
            # Binary-search builder: already a tuple of 1-D columns.
            piece_idx = out

    reduced_occs = reduced_interval_indices(
        operands, index_to_operand_dims, out_indices
    )
    values = compute_mask_values(operands, piece_idx, reduced_occs)
    return Mask(piece_idx, values)
