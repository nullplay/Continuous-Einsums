"""Continuous einsum: public API.

This module is the thin orchestrator for the continuous-einsum pipeline. The
data model lives in :mod:`ctensor`; the evaluation follows the format-aware
mask → product → merge strategy of the thesis chapter, one module per step:

0. :func:`ceinsum_equation.parse_equation`      — parse the einsum string.
1. :func:`ceinsum_mask.build_mask`              — **mask creation**: a sparse
   COO mask over the operands' piece positions; entry (a, b, ...) exists iff
   the pieces intersect on every shared index. Its value array carries the
   intersection *lengths* of the reduced all-interval indices, so contracting
   a continuous variable computes the integral ``∫ A·B dk``.
2. :func:`ceinsum_product.compute_product`      — **product**: per mask entry,
   the candidate value (operand values × mask measure) and its output
   coordinates (max of starts / min of ends through the mask positions).
3. :func:`ceinsum_merge.merge_discrete`         — **merge, discrete half**:
   sum candidates with identical output coordinates (unique + scatter-add).
   :func:`ceinsum_coalesce.coalesce` — **merge, continuous half**: when a
   reduction leaves output pieces that *partially* overlap, cut them into
   disjoint pieces and sum per region.

Example::

    t1 = continuous_tensor([(xs, xe), (ys, ye)], v1, property=["[)", "[)"])
    t2 = continuous_tensor([(a, b)],             v2, property=["[]"])
    t3 = continuous_tensor([(c,)],               v3, property=["P"])
    out = ceinsum("ij,i,j->i", t1, t2, t3)
"""

from __future__ import annotations

from ceinsum_coalesce import coalesce
from ceinsum_equation import parse_equation
from ceinsum_mask import MaskBuilder, build_mask
from ceinsum_merge import merge_discrete
from ceinsum_product import compute_output_properties, compute_product
from ctensor import ContinuousTensor, continuous_tensor

__all__ = ["ContinuousTensor", "continuous_tensor", "ceinsum"]


def ceinsum(
    equation: str,
    *operands: ContinuousTensor,
    builder: MaskBuilder | None = None,
) -> ContinuousTensor:
    """Continuous einsum over COO continuous tensors.

    Runs the mask → product → merge pipeline (see module docstring). Supports
    1-, 2-, and 3-operand equations. Contraction over an index whose carriers
    are all intervals integrates (measure-weighted); any pinpoint carrier
    makes it a plain sum.

    ``builder`` chooses the mask-creation strategy passed to step 1: the
    binary-search builder (default, ``None``) or the naive all-pair
    ``mask_dense.build_dense_mask``.

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
    in_indices, out_indices, index_to_operand_dims = parse_equation(
        equation, len(operands)
    )

    # A reduction (a contracted index — present in the inputs but not the
    # output) is what can leave the output with pieces that overlap along an
    # output dim; only then is coalescing needed. Without one the output pieces
    # are already disjoint, so we skip the step entirely.
    has_reduction = bool(set("".join(in_indices)) - set(out_indices))

    # Output property per output index.
    # eg: i is interval in t1 ("[)") and t2 ("[]") → conservative AND → {"i": "[)"}
    index_properties = compute_output_properties(
        operands, index_to_operand_dims, out_indices
    )
    out_property = tuple(index_properties[oi] for oi in out_indices)

    # 1) Mask: join the operands' pieces under intersection conditions.
    # eg: 2 surviving join tuples (t1.0,t2.0,t3.0) and (t1.1,t2.0,t3.1) →
    #     piece_idx = ([0,1], [0,0], [0,1])   # one column per operand
    #     (i: [0,2)∩[1,4] and [1,3)∩[1,4];  j: 0.5∈[0,1) and 5.5∈[5,6))
    # mask.values carries the integral measure of reduced interval indices
    # (None here: the reduced j has a pinpoint carrier, so it plainly sums).
    mask = build_mask(operands, index_to_operand_dims, out_indices, builder)

    # 2) Product: per-tuple candidate values and output coordinates.
    # eg: values = [2,3]·[10,10]·[100,200] = [2000,6000]
    #     i coords: start=max(t1.s,t2.s)=[1,1], end=min(t1.e,t2.e)=[2,3]
    #     key_cols = (t1 col, t2 col) — the operands providing i
    product = compute_product(
        operands, mask, index_to_operand_dims, out_indices, index_properties
    )

    # 3) Merge, discrete half: sum candidates sharing their output coordinates.
    # eg: provider piece pairs (0,0),(1,0) are distinct → two output pieces,
    #     ContinuousTensor(dims=(([1,1],[2,3]),), values=[2000,6000], ("[)",))
    out = merge_discrete(product, out_property)

    # Merge, continuous half: contracting an interleaved index can leave output
    # pieces that *partially* overlap along an output dim (e.g. pinpoint j in
    # "ij,i->i"). Rewrite them into a disjoint set, summing values where they
    # overlapped. Only a reduction can produce such overlaps, and coalesce is
    # itself a no-op when the pieces are already disjoint.
    if has_reduction:
        out = coalesce(out)
    return out
