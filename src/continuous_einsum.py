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
3. :func:`ceinsum_merge.merge`                  — **merge**: sum candidates
   that claim the same output region — identical coordinates and partially
   overlapping intervals alike — into a canonical disjoint COO tensor. Runs
   only when the einsum has a reduction; without one every candidate is
   already a distinct, disjoint output piece.

Example::

    t1 = continuous_tensor([(xs, xe), (ys, ye)], v1, property=["[)", "[)"])
    t2 = continuous_tensor([(a, b)],             v2, property=["[)"])
    t3 = continuous_tensor([(c,)],               v3, property=["P"])
    out = ceinsum("ij,i,j->i", t1, t2, t3)
"""

from __future__ import annotations

import torch

from ceinsum_equation import parse_equation
from ceinsum_mask import Mask, MaskBuilder, build_mask
from ceinsum_merge import merge
from ceinsum_product import compute_output_properties, compute_product
from ctensor import ContinuousTensor, continuous_tensor, is_pinpoint

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
        t2 = ct([(_T(1,10), _T(4,12))], _T(10, 20), ["[)"])  # i: [1,4), [10,12)
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
    # eg: i is interval in t1 and t2, j pinpoint via t3 → {"i": "[)"}
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

    # Fast path — pinned output: when the einsum reduces and some operand is
    # all-pinpoint with an index pattern equal to the output's (e.g.
    # "ij,ik,kj->ij" with a pinpoint A), that operand pins every output
    # coordinate: each candidate's output coords are exactly one of its
    # pieces' coords. Steps 2+3 then collapse — the product's coordinate
    # intersection is a plain copy, and the merge (dedup + interval cutting)
    # is a scatter-add keyed by that operand's piece index. Valid because a
    # tensor's pieces are pairwise distinct, so piece index ≡ coord tuple.
    # Without a reduction the default path already is the fast path (coords
    # copy through the mask, merge is skipped), and the scatter-add's
    # zero-filtering would only add a device sync.
    anchor_op = next(
        (o for o, idx in enumerate(in_indices)
         if out_indices and idx == out_indices
         and all(is_pinpoint(p) for p in operands[o].property)),
        None,
    )
    if anchor_op is not None and has_reduction:
        return _pinned_output(operands, mask, anchor_op)

    # 2) Product: per-tuple candidate values and output coordinates.
    # eg: values = [2,3]·[10,10]·[100,200] = [2000,6000]
    #     i coords: start=max(t1.s,t2.s)=[1,1], end=min(t1.e,t2.e)=[2,3]
    product = compute_product(
        operands, mask, index_to_operand_dims, out_indices, index_properties
    )

    # 3) Merge: sum candidates claiming the same output region — identical
    # coordinates (the contraction terms) and partially overlapping intervals
    # alike — into disjoint pieces. Only a reduction can make candidates
    # collide; without one they are already distinct, disjoint pieces and the
    # merge is the identity, so the tensor is assembled directly.
    # eg: contracted j puts both candidates on overlapping i intervals →
    #     sweep cuts them at breakpoints {1,2,3}: [1,2)→8000, [2,3)→6000
    if has_reduction:
        return merge(product.coords, product.values, out_property)
    return ContinuousTensor(
        tuple(tuple(spec) for spec in product.coords), product.values, out_property
    )


def _pinned_output(
    operands: tuple[ContinuousTensor, ...], mask: Mask, anchor_op: int
) -> ContinuousTensor:
    """Product + merge for an all-pinpoint anchor operand matching the output.

    Candidate values are the usual gather-multiply (times the mask measure);
    everything coordinate-side reduces to a scatter-add over the anchor's
    pieces: ``out[p] = Σ candidates with piece_idx[anchor][t] == p``. Pieces
    summing to exact zero are dropped, mirroring :func:`ceinsum_merge.merge`.
    """
    piece_idx = mask.piece_idx
    values = operands[0].values[piece_idx[0]]
    for op in range(1, len(operands)):
        values = values * operands[op].values[piece_idx[op]]
    if mask.values is not None:
        values = values * mask.values

    anchor = operands[anchor_op]
    out_vals = torch.zeros(anchor.nnz, dtype=values.dtype, device=values.device)
    out_vals.index_add_(0, piece_idx[anchor_op], values)
    keep = out_vals != 0
    dims = tuple((spec[0][keep],) for spec in anchor.dims)
    return ContinuousTensor(dims, out_vals[keep], anchor.property)
