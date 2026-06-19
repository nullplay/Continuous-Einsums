"""Step 5 — values: product across operands per join tuple, scatter-added.

Each join tuple contributes ``prod(operand.value[piece])``; tuples sharing an
output piece are summed (contraction over the eliminated indices).
"""

from __future__ import annotations

from typing import Sequence

import torch

from ctensor import ContinuousTensor


def compute_output_values(
    operands: Sequence[ContinuousTensor],
    piece_idx: tuple[torch.Tensor, ...],
    inv: torch.Tensor,
    num_out: int,
) -> torch.Tensor:
    """Gather → multiply → scatter-add the per-piece output values."""
    prod = operands[0].values[piece_idx[0]]
    for op in range(1, len(operands)):
        prod = prod * operands[op].values[piece_idx[op]]
    out_values = torch.zeros(num_out, dtype=prod.dtype, device=prod.device)
    out_values.index_add_(0, inv, prod)
    return out_values
