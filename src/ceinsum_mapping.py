"""Step 2 — mapping: join the operands' pieces under intersection conditions.

Builds the ``op``/``output``/``cond`` arguments for
:func:`table_opt_mapping.build_table_opt_mapping` from the operand properties
and the index → operand-dim map, runs the join, and returns the surviving
piece index per operand.

The join builder leads on the first two output axes and expands a third, so it
serves 2- and 3-operand equations. A single operand needs no join.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from ctensor import ContinuousTensor, is_pinpoint, left_closed, right_closed
from table_opt_mapping import build_table_opt_mapping

# A "mapping builder" turns (op, output, cond) into a 0-arg closure that runs
# the join. Two implementations share this signature:
#
# * ``table_opt_mapping.build_table_opt_mapping`` (default) — searchsorted lead
#   + post-filter; its closure returns a TUPLE of 1-D columns (one per axis).
# * ``table_mapping.build_table_mapping`` — naive all-pair boolean table +
#   ``nonzero``; its closure returns a single 2-D ``(num_matches, num_axes)``
#   tensor. ``run_mapping`` normalizes that into the same tuple-of-columns.
MappingBuilder = Callable[..., Callable[[], object]]


# Tensor-name conventions shared by the `op` dict and the condition strings.
def _coord(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_coord"


def _start(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_start"


def _end(op: int, dim: int) -> str:
    return f"Op{op}_Dim{dim}_end"


def build_mapping_ops(operands: Sequence[ContinuousTensor]) -> dict[str, torch.Tensor]:
    """Step 2-1 — collect every operand's coordinate tensors as the `op` dict.

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
    """Step 2-3 — intersection conditions over the join axes.

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


def run_mapping(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    builder: MappingBuilder | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the piece-join and return ``piece_idx``: one index column per operand.

    ``piece_idx[op]`` is a 1-D long tensor giving which piece of operand ``op``
    participates in each surviving join tuple. Every column has the same length
    P (the number of surviving tuples).

    ``builder`` selects the join implementation (see :data:`MappingBuilder`):
    the optimized searchsorted builder (default, when ``None``) or the naive
    all-pair ``table_mapping.build_table_mapping``. Both produce identical
    joins; only their output layout differs, which this function normalizes.
    """
    if builder is None:
        builder = build_table_opt_mapping
    num = len(operands)
    if num > 3:
        raise NotImplementedError(
            "ceinsum supports at most 3 operands "
            "(the join builder leads on 2-3 output axes)"
        )

    if num == 1:
        # No join: every piece of the lone operand survives as-is.
        nnz = operands[0].nnz
        return (torch.arange(nnz, dtype=torch.long, device=operands[0].device),)

    mapping_ops = build_mapping_ops(operands)
    cond = build_conditions(operands, index_to_operand_dims)
    # Step 2-2 — one output axis per operand: piece index of operand k on axis "axk".
    mapping_output = tuple(f"ax{i}" for i in range(num))

    run = builder(mapping_ops, output=mapping_output, cond=cond)
    out = run()
    if isinstance(out, torch.Tensor):
        # Naive builder: a single (num_matches, num_axes) tensor from nonzero.
        # Column k is operand k's piece index — split into the tuple of columns.
        return tuple(out[:, k] for k in range(num))
    # Optimized/kernel builder: already a tuple of 1-D columns in operand order.
    return out
