"""Step 3 — output construction: group join tuples into output pieces.

Two join tuples land in the same output piece iff they agree on every operand
that *provides* an output index — those operands fix the output coordinates.
"""

from __future__ import annotations

import torch


def build_output_pieces(
    piece_idx: tuple[torch.Tensor, ...],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Assign each join tuple to an output piece.

    Returns ``(inv, num_out, rep)`` where

    * ``inv`` — length ``P``, the output piece id of each join tuple (the
      scatter target for the values step);
    * ``num_out`` — number of distinct output pieces;
    * ``rep`` — length ``num_out``, one representative join tuple per output
      piece. Every tuple in a group shares the same provider pieces (hence the
      same coordinates), so any representative yields the correct coordinate.

    ``P`` (number of join tuples) and the device are read from ``piece_idx``.
    """
    P = int(piece_idx[0].shape[0])
    device = piece_idx[0].device
    provider_ops = sorted(
        {op for oi in out_indices for (op, _d) in index_to_operand_dims[oi]}
    )
    if provider_ops:
        key = torch.stack([piece_idx[op] for op in provider_ops], dim=1)  # (P, |prov|)
        _uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        num_out = int(_uniq.shape[0])
    else:
        # Scalar output (full contraction): all tuples collapse to one piece.
        inv = torch.zeros(P, dtype=torch.long, device=device)
        num_out = 1 if P > 0 else 0

    rep = torch.empty(num_out, dtype=torch.long, device=device)
    rep[inv] = torch.arange(P, dtype=torch.long, device=device)
    return inv, num_out, rep
