"""Continuous einsum: public API.

This module is the thin orchestrator for the continuous-einsum pipeline. The
data model lives in :mod:`ctensor`; each pipeline step is implemented in its own
module and exposed as one high-level function, which :func:`ceinsum` calls in
order:

0. :func:`ceinsum_equation.parse_equation`        — parse the einsum string.
1. :func:`ceinsum_properties.compute_output_properties` — per-index output property.
2. :func:`ceinsum_mapping.run_mapping`            — join the operands' pieces.
3. :func:`ceinsum_output.build_output_pieces`     — group join tuples into pieces.
4. :func:`ceinsum_coordinates.compute_output_coordinates` — intersect coordinates.
5. :func:`ceinsum_values.compute_output_values`   — gather → multiply → scatter-add.
6. :func:`ceinsum_coalesce.coalesce`              — split overlaps, sum into disjoint pieces.

Example::

    t1 = continuous_tensor([(xs, xe), (ys, ye)], v1, property=["[)", "[)"])
    t2 = continuous_tensor([(a, b)],             v2, property=["[]"])
    t3 = continuous_tensor([(c,)],               v3, property=["P"])
    out = ceinsum("ij,i,j->i", t1, t2, t3)
"""

from __future__ import annotations

from ceinsum_coalesce import coalesce
from ceinsum_coordinates import compute_output_coordinates
from ceinsum_equation import parse_equation
from ceinsum_mapping import MappingBuilder, run_mapping
from ceinsum_output import build_output_pieces
from ceinsum_properties import compute_output_properties
from ceinsum_values import compute_output_values
from ctensor import ContinuousTensor, continuous_tensor

__all__ = ["ContinuousTensor", "continuous_tensor", "ceinsum"]


def ceinsum(
    equation: str,
    *operands: ContinuousTensor,
    builder: MappingBuilder | None = None,
) -> ContinuousTensor:
    """Continuous einsum over COO continuous tensors.

    Runs the four-step pipeline (see module docstring) by calling each step's
    high-level function in turn. Supports 1-, 2-, and 3-operand equations.

    ``builder`` chooses the piece-join implementation passed to step 2: the
    optimized searchsorted builder (default, ``None``) or the naive all-pair
    ``table_mapping.build_table_mapping``.

    Worked example (the same operands as ``tests/test_ceinsum.py::test_ij_i_j__i``)::

        t1 = ct([(_T(0,1), _T(2,3)),    # i: [0,2), [1,3)
                 (_T(0,5), _T(1,6))],   # j: [0,1), [5,6)
                _T(2, 3), ["[)", "[)"])
        t2 = ct([(_T(1,10), _T(4,12))], _T(10, 20), ["[]"])  # i: [1,4], [10,12]
        t3 = ct([(_T(0.5, 5.5),)],      _T(100, 200), ["P"]) # j: pts 0.5, 5.5
        out = ceinsum("ij,i,j->i", t1, t2, t3)

    The per-step ``# eg:`` comments below trace this example end to end.
    """
    # 0) Parse the equation into the index → operand-dim map.
    # eg: in=["ij","i","j"], out="i",
    #     index_to_operand_dims={"i":[(0,0),(1,0)], "j":[(0,1),(2,0)]}
    _in_indices, out_indices, index_to_operand_dims = parse_equation(
        equation, len(operands)
    )

    # 1) Output property per output index.
    # eg: i is interval in t1 ("[)") and t2 ("[]") → conservative AND → {"i": "[)"}
    index_properties = compute_output_properties(
        operands, index_to_operand_dims, out_indices
    )

    # 2) Mapping: join the operands' pieces under intersection conditions.
    # eg: 2 surviving join tuples (t1.0,t2.0,t3.0) and (t1.1,t2.0,t3.1) →
    #     piece_idx = ([0,1], [0,0], [0,1])   # one column per operand
    #     (i: [0,2)∩[1,4] and [1,3)∩[1,4];  j: 0.5∈[0,1) and 5.5∈[5,6))
    piece_idx = run_mapping(operands, index_to_operand_dims, builder)

    # 3) Output construction: group join tuples into output pieces.
    # eg: i is provided by t1 & t2; their piece pairs (0,0),(1,0) are distinct →
    #     num_out=2, inv=[0,1], rep=[0,1]
    inv, num_out, rep = build_output_pieces(
        piece_idx, index_to_operand_dims, out_indices
    )

    # 4) Output coordinates: intersect the contributing pieces per index.
    # eg: i start=max(t1.start,t2.start)=[1,1], end=min(t1.end,t2.end)=[2,3]
    #     → out_dims=[([1,1], [2,3])], out_property=("[)",)
    out_dims = compute_output_coordinates(
        operands, piece_idx, index_to_operand_dims, out_indices, index_properties, rep
    )
    out_property = tuple(index_properties[oi] for oi in out_indices)

    # 5) Values: product across operands per join tuple, scatter-added per piece.
    # eg: [2,3]·[10,10]·[100,200] = [2000,6000], scatter-add by inv → [2000,6000]
    out_values = compute_output_values(operands, piece_idx, inv, num_out)

    # eg: ContinuousTensor(dims=(([1,1],[2,3]),), values=[2000,6000], property=("[)",))
    out = ContinuousTensor(tuple(out_dims), out_values, out_property)

    # 6) Coalesce: contracting an interleaved index can leave output pieces that
    # overlap along an output dim (e.g. pinpoint j in "ij,i->i"). Rewrite them
    # into a disjoint set, summing values where they overlapped. A no-op when
    # the pieces are already disjoint.
    return coalesce(out)
