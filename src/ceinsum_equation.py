"""Step 0 — parse the einsum equation into an index → operand-dim map.

Turns ``"ij,i,j->i"`` into the structures the rest of the pipeline indexes by:
the per-operand index strings, the output index string, and a map from each
index letter to every ``(operand, dim)`` position that carries it.
"""

from __future__ import annotations


def parse_equation(
    equation: str, num_operands: int
) -> tuple[list[str], str, dict[str, list[tuple[int, int]]]]:
    """Parse ``equation`` and locate every occurrence of each index.

    Returns ``(in_indices, out_indices, index_to_operand_dims)`` where

    * ``in_indices`` — index string per operand, e.g. ``["ij", "i", "j"]``;
    * ``out_indices`` — the output index string, e.g. ``"i"``;
    * ``index_to_operand_dims`` — ``{index: [(op, dim), ...]}`` listing every
      position that carries the index, e.g.
      ``{"i": [(0, 0), (1, 0)], "j": [(0, 1), (2, 0)]}``.
    """
    if "->" not in equation:
        raise ValueError(f"equation must contain '->': {equation!r}")
    equation = equation.replace(" ", "")
    in_part, out_indices = equation.split("->")
    in_indices = in_part.split(",")

    if len(in_indices) != num_operands:
        raise ValueError(
            f"equation lists {len(in_indices)} operands but {num_operands} given"
        )

    index_to_operand_dims: dict[str, list[tuple[int, int]]] = {
        index: [] for index in set("".join(in_indices))
    }
    for op_idx, op_indices in enumerate(in_indices):
        for dim_idx, index in enumerate(op_indices):
            index_to_operand_dims[index].append((op_idx, dim_idx))

    return in_indices, out_indices, index_to_operand_dims
