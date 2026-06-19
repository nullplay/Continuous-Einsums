"""Step 4 — output coordinates: intersect the contributing pieces per index.

For each output index, ``start = max`` of contributing starts and ``end = min``
of contributing ends; a pinpoint provider copies its coordinate.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ctensor import PINPOINT, ContinuousTensor, is_pinpoint


def compute_output_coordinates(
    operands: Sequence[ContinuousTensor],
    piece_idx: tuple[torch.Tensor, ...],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
    index_properties: dict[str, str],
    rep: torch.Tensor,
) -> list[tuple[torch.Tensor, ...]]:
    """Build the output ``dims`` (one coord spec per output index, in order).

    ``rep`` selects one representative join tuple per output piece (see
    :func:`ceinsum_output.build_output_pieces`).
    """
    out_dims: list[tuple[torch.Tensor, ...]] = []
    for oi in out_indices:
        occ = index_to_operand_dims[oi]
        if index_properties[oi] == PINPOINT:
            # Coordinate comes from any pinpoint provider (equality conditions
            # guarantee all providers agree there).
            coord = next(
                operands[op].dims[dim][0][piece_idx[op]]
                for (op, dim) in occ
                if is_pinpoint(operands[op].property[dim])
            )
            out_dims.append((coord[rep],))
        else:
            starts = torch.stack(
                [operands[op].dims[dim][0][piece_idx[op]] for (op, dim) in occ], dim=0
            )
            ends = torch.stack(
                [operands[op].dims[dim][1][piece_idx[op]] for (op, dim) in occ], dim=0
            )
            out_dims.append((starts.amax(0)[rep], ends.amin(0)[rep]))
    return out_dims
