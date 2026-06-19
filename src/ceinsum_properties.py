"""Step 1 — derive the output tensor's per-index properties.

Intersecting pieces along a shared index combines their boundary kinds, which
this step resolves into a single property code per output index.
"""

from __future__ import annotations

from typing import Sequence

from ctensor import PINPOINT, ContinuousTensor, left_closed, right_closed


def compute_output_properties(
    operands: Sequence[ContinuousTensor],
    index_to_operand_dims: dict[str, list[tuple[int, int]]],
    out_indices: str,
) -> dict[str, str]:
    """Derive each *output* index's property from the operand dims that carry it.

    Intersecting pieces along a shared index combines their boundary kinds
    *conservatively*, boundary-by-boundary:

    * if **any** contributing dim is a pinpoint, the output index is a pinpoint
      (the point pins the coordinate regardless of the other intervals);
    * otherwise the output is an interval that is left-closed iff **every**
      contributing dim is left-closed, and right-closed iff **every**
      contributing dim is right-closed. (A single open boundary anywhere opens
      that side of the intersection.)
    """
    total: dict[str, str] = {}
    for index in out_indices:
        op_dims = index_to_operand_dims[index]
        properties = [operands[op].property[dim] for (op, dim) in op_dims]

        if PINPOINT in properties:
            total[index] = PINPOINT
            continue

        lc = all(left_closed(p) for p in properties)
        rc = all(right_closed(p) for p in properties)
        total[index] = ("[" if lc else "(") + ("]" if rc else ")")

    return total
