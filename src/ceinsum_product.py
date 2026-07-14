"""Step 2 — product: per-tuple candidate values and output coordinates.

For every mask entry (join tuple) this step materializes one *candidate*
output contribution:

* its value ``Prod_t = MV_t · ∏_op values_op[piece_idx_op[t]]`` — the plain
  product of the participating pieces' values, times the mask's integral
  measure (the intersection lengths of the reduced interval indices);
* its output coordinates — per output index, the *intersection* of the
  carriers' coordinates: ``max`` of the gathered starts and ``min`` of the
  gathered ends for intervals, or a plain copy from any pinpoint carrier
  (the mask's equality conditions guarantee all pinpoint carriers agree).

Everything is a gather through the mask's piece columns followed by
elementwise maximum/minimum/multiply — an indirect einsum. Candidates are not
deduplicated here; several may claim the same output region (the terms of the
contraction sum), which :func:`ceinsum_merge.merge` resolves.

This module also derives the output tensor's per-index property codes, which
decide the coordinate layout (pinpoint vs interval) used above.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch

from ceinsum_mask import Mask
from ctensor import INTERVAL, PINPOINT, ContinuousTensor, is_pinpoint


class Product(NamedTuple):
    """Per-tuple candidates, all of length ``P`` (the mask's entry count).

    * ``values`` — candidate value per join tuple.
    * ``coords`` — one coord spec per output index, in order: ``(coord,)``
      for a pinpoint index, ``(start, end)`` for an interval index.
    """

    values: torch.Tensor
    coords: list[tuple[torch.Tensor, ...]]


def compute_output_properties(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
) -> dict[str, str]:
    """Derive each *output* index's property from the operand dims that carry it.

    If **any** contributing dim is a pinpoint, the output index is a pinpoint
    (the point pins the coordinate regardless of the other intervals);
    otherwise it is a half-open interval — the intersection of half-open
    intervals is half-open.
    """
    total: dict[str, str] = {}
    for index in out_indices:
        op_dims = index_to_operand_dims[index]
        properties = [operands[op].property[dim] for (op, dim) in op_dims]
        total[index] = PINPOINT if PINPOINT in properties else INTERVAL
    return total


def compute_product(
    operands: Sequence[ContinuousTensor],
    mask: Mask,
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
    index_properties: dict[str, str],
) -> Product:
    """Build the per-tuple candidate values, coordinates, and grouping key."""
    piece_idx = mask.piece_idx

    # Value: gather every operand's values through its mask column, multiply,
    # and weight by the mask's integral measure.
    values = operands[0].values[piece_idx[0]]
    for op in range(1, len(operands)):
        values = values * operands[op].values[piece_idx[op]]
    if mask.values is not None:
        values = values * mask.values

    # Coordinates: intersect the carriers of each output index per tuple.
    coords: list[tuple[torch.Tensor, ...]] = []
    for oi in out_indices:
        occ = index_to_operand_dims[oi]
        if index_properties[oi] == PINPOINT:
            # Coordinate comes from any pinpoint carrier (equality conditions
            # guarantee all pinpoint carriers agree).
            coord = next(
                operands[op].dims[dim][0][piece_idx[op]]
                for (op, dim) in occ
                if is_pinpoint(operands[op].property[dim])
            )
            coords.append((coord,))
        else:
            starts = torch.stack(
                [operands[op].dims[dim][0][piece_idx[op]] for (op, dim) in occ], dim=0
            )
            ends = torch.stack(
                [operands[op].dims[dim][1][piece_idx[op]] for (op, dim) in occ], dim=0
            )
            coords.append((starts.amax(0), ends.amin(0)))

    return Product(values, coords)
