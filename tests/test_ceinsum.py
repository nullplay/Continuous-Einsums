"""Concrete, hand-checked tests for the continuous einsum API (``ceinsum``).

Each test builds operands from explicit small numbers and asserts the exact
expected output tensor. No reference implementation, no synthesis — just
"feed these pieces, expect this result", across a variety of einsum equations,
operand dimensionalities, and property combinations (operand count kept ≤ 3).

Semantics being pinned down (see continuous_einsum module docstring). All
intervals are half-open ``[)``; pinpoints are ``P``.

* Interval/interval overlap: ``lo1 < hi2`` and ``lo2 < hi1`` (strict —
  touching intervals do not intersect).
* Point in interval ``[lo, hi)``: ``lo <= p < hi``. Pinpoint vs pinpoint:
  exact equality.
* Output piece identity = the tuple of piece indices of the operands providing
  the output indices (uniqueness by contributing pieces, not coordinate value).
* Output coordinate: ``start = max(starts)``, ``end = min(ends)``; if any
  provider is a pinpoint the output index is a pinpoint and the coord is copied.
* Output dim property: pinpoint if any provider is a pinpoint, else ``[)``.
* Output value: product of matched operands' values times the mask's integral
  measure, scatter-added per output piece. A reduced index whose carriers are
  all intervals integrates: each join tuple is weighted by the intersection
  length on that index (``∫ A·B dk``). A reduced index with any pinpoint
  carrier contracts by plain summation (no factor).

These call the real ``ceinsum`` and so are skipped until it is implemented.
"""

from __future__ import annotations

import pytest
import torch

from continuous_einsum import ContinuousTensor, ceinsum, continuous_tensor

DTYPE = torch.float64


def _ct(dims, values, property):
    return continuous_tensor(dims, values, property, dtype=DTYPE)


def _T(*xs):
    return torch.tensor(list(xs), dtype=DTYPE)


# ---------------------------------------------------------------------------
# Probe: skip the comparison tests until ceinsum is implemented.
# ---------------------------------------------------------------------------


def _ceinsum_ready() -> bool:
    a = _ct([(_T(0.0),)], _T(1.0), ["P"])
    try:
        ceinsum("i->i", a)
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


requires_ceinsum = pytest.mark.skipif(
    not _ceinsum_ready(), reason="ceinsum not implemented yet"
)


# ---------------------------------------------------------------------------
# Order-insensitive equality: canonicalize pieces by (coords..., value).
# ---------------------------------------------------------------------------


def _canonical(ct: ContinuousTensor) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for spec in ct.dims:
        for t in spec:
            cols.append(t.detach().cpu().to(torch.float64))
    cols.append(ct.values.detach().cpu().to(torch.float64))
    if ct.nnz == 0:
        return torch.empty((0, len(cols)), dtype=torch.float64)
    mat = torch.stack(cols, dim=1)
    order = torch.arange(mat.shape[0])
    for col in range(mat.shape[1] - 1, -1, -1):
        order = order[torch.argsort(mat[order, col], stable=True)]
    return mat[order]


def assert_ceinsum(out: ContinuousTensor, expected: ContinuousTensor, label: str = "") -> None:
    assert tuple(out.property) == tuple(expected.property), (
        label, out.property, expected.property,
    )
    assert out.nnz == expected.nnz, (label, out.nnz, expected.nnz)
    got = _canonical(out)
    want = _canonical(expected)
    assert got.shape == want.shape, (label, got.shape, want.shape)
    assert torch.allclose(got, want, atol=1e-9), (label, got, want)


# ===========================================================================
# i,i->i  — at most 3 property combinations.
# ===========================================================================


@requires_ceinsum
def test_i_i__i_interval_product():
    """[) ⨉ [) interval overlap product."""
    a = _ct([(_T(0.0, 5.0), _T(2.0, 7.0))], _T(2.0, 3.0), ["[)"])   # [0,2), [5,7)
    b = _ct([(_T(1.0, 6.0), _T(3.0, 8.0))], _T(10.0, 20.0), ["[)"])  # [1,3), [6,8)

    out = ceinsum("i,i->i", a, b)

    # [0,2)∩[1,3)=[1,2) v=2*10=20 ; [5,7)∩[6,8)=[6,7) v=3*20=60
    expected = _ct([(_T(1.0, 6.0), _T(2.0, 7.0))], _T(20.0, 60.0), ["[)"])
    assert_ceinsum(out, expected, "i,i->i interval")


@requires_ceinsum
def test_i_i__i_pinpoint_equality():
    """P ⨉ P sparse-style equality contraction."""
    a = _ct([(_T(1.0, 2.0, 5.0),)], _T(10.0, 20.0, 30.0), ["P"])
    b = _ct([(_T(2.0, 5.0, 9.0),)], _T(1.0, 2.0, 3.0), ["P"])

    out = ceinsum("i,i->i", a, b)

    # matches at 2 (20*1=20) and 5 (30*2=60); 1 and 9 unmatched
    expected = _ct([(_T(2.0, 5.0),)], _T(20.0, 60.0), ["P"])
    assert_ceinsum(out, expected, "i,i->i pinpoint")


@requires_ceinsum
def test_i_i__i_touching_halfopen_disjoint():
    """[) ⨉ [) intervals that merely touch at a boundary do not intersect."""
    a = _ct([(_T(0.0, 3.0), _T(1.0, 5.0))], _T(2.0, 4.0), ["[)"])  # [0,1), [3,5)
    b = _ct([(_T(1.0, 6.0), _T(2.0, 9.0))], _T(3.0, 7.0), ["[)"])  # [1,2), [6,9)

    out = ceinsum("i,i->i", a, b)

    # [0,1)∩[1,2)=∅ (touching) ; [3,5)∩[6,9)=∅ → empty output
    expected = _ct([(_T(), _T())], _T(), ["[)"])
    assert_ceinsum(out, expected, "i,i->i touching [)")


@requires_ceinsum
def test_rejects_non_halfopen_property():
    """Only "[)" and "P" property codes exist; others fail at construction."""
    for bad in ("(]", "[]", "()"):
        with pytest.raises(ValueError, match="invalid property"):
            _ct([(_T(0.0), _T(1.0))], _T(1.0), [bad])


# ===========================================================================
# Multi-operand / multi-dimensional einsums (operand count ≤ 3).
# ===========================================================================


@requires_ceinsum
def test_ij_i_j__i():
    """ij,i,j->i — the motivating example: overlap on i, point-in-interval on j."""
    t1 = _ct(
        [(_T(0.0, 1.0), _T(2.0, 3.0)),   # i: [0,2), [1,3)
         (_T(0.0, 5.0), _T(1.0, 6.0))],  # j: [0,1), [5,6)
        _T(2.0, 3.0),
        ["[)", "[)"],
    )
    t2 = _ct([(_T(1.0, 10.0), _T(4.0, 12.0))], _T(10.0, 20.0), ["[)"])  # i: [1,4),[10,12)
    t3 = _ct([(_T(0.5, 5.5),)], _T(100.0, 200.0), ["P"])               # j: pts 0.5, 5.5

    out = ceinsum("ij,i,j->i", t1, t2, t3)

    # (t1.0,t2.0): [0,2)∩[1,4)=[1,2), v=2*10*100=2000
    # (t1.1,t2.0): [1,3)∩[1,4)=[1,3), v=3*10*200=6000
    # j is contracted, so these two overlapping i-pieces are summed where they
    # overlap (coalesce step): [1,2)→2000+6000=8000, [2,3)→6000.
    expected = _ct([(_T(1.0, 2.0), _T(2.0, 3.0))], _T(8000.0, 6000.0), ["[)"])
    assert_ceinsum(out, expected, "ij,i,j->i")


@requires_ceinsum
def test_ij_j__i_reduction_scatter_add():
    """ij,j->i — contract pinpoint j; duplicate j matches sum into one i piece."""
    t1 = _ct(
        [(_T(0.0, 5.0), _T(2.0, 7.0)),  # i: [0,2), [5,7)
         (_T(1.0, 3.0),)],              # j pinpoints: 1.0, 3.0
        _T(2.0, 4.0),
        ["[)", "P"],
    )
    t2 = _ct([(_T(1.0, 1.0, 3.0),)], _T(10.0, 20.0, 30.0), ["P"])  # j: 1,1,3

    out = ceinsum("ij,j->i", t1, t2)

    # t1.p0 (j=1) matches t2.q0,q1 → piece (t1.0): 2*10 + 2*20 = 60, i=[0,2)
    # t1.p1 (j=3) matches t2.q2     → piece (t1.1): 4*30 = 120,      i=[5,7)
    expected = _ct([(_T(0.0, 5.0), _T(2.0, 7.0))], _T(60.0, 120.0), ["[)"])
    assert_ceinsum(out, expected, "ij,j->i reduction")


@requires_ceinsum
def test_ij_i__i_coalesce_overlapping_reduction():
    """ij,i->i — contracting pinpoint j leaves overlapping i-pieces that the
    coalesce step splits at every boundary and sums per region."""
    t1 = _ct(
        [(_T(0.0, 5.0, 3.0), _T(10.0, 15.0, 13.0)),  # i: [0,10), [5,15), [3,13)
         (_T(0.0, 1.0, 2.0),)],                       # j pinpoints: 0, 1, 2
        _T(1.0, 1.0, 1.0),
        ["[)", "P"],
    )
    t2 = _ct([(_T(2.0), _T(11.0))], _T(1.0), ["[)"])  # i: [2,11)

    out = ceinsum("ij,i->i", t1, t2)

    # raw (overlapping) contributions: [2,10):1, [5,11):1, [3,11):1.
    # summed per region: [2,3)=1, [3,5)=2, [5,10)=3, [10,11)=2.
    expected = _ct(
        [(_T(2.0, 3.0, 5.0, 10.0), _T(3.0, 5.0, 10.0, 11.0))],
        _T(1.0, 2.0, 3.0, 2.0),
        ["[)"],
    )
    assert_ceinsum(out, expected, "ij,i->i coalesce")


@requires_ceinsum
def test_ij_jk__ik_matmul():
    """ij,jk->ik — textbook matmul, contracting pinpoint j, keeping intervals i,k."""
    t1 = _ct(
        [(_T(0.0, 1.0), _T(1.0, 2.0)),  # i: [0,1), [1,2)
         (_T(0.0, 1.0),)],              # j pinpoints: 0.0, 1.0
        _T(2.0, 3.0),
        ["[)", "P"],
    )
    t2 = _ct(
        [(_T(0.0, 1.0, 0.0),),                            # j: 0.0, 1.0, 0.0
         (_T(10.0, 20.0, 30.0), _T(11.0, 21.0, 31.0))],  # k: [10,11),[20,21),[30,31)
        _T(5.0, 7.0, 11.0),
        ["P", "[)"],
    )

    out = ceinsum("ij,jk->ik", t1, t2)

    # j matches: (t1.0,t2.0) j=0, (t1.0,t2.2) j=0, (t1.1,t2.1) j=1
    #   i=[0,1) k=[10,11) v=2*5=10 ; i=[0,1) k=[30,31) v=2*11=22 ; i=[1,2) k=[20,21) v=3*7=21
    expected = _ct(
        [(_T(0.0, 0.0, 1.0), _T(1.0, 1.0, 2.0)),
         (_T(10.0, 30.0, 20.0), _T(11.0, 31.0, 21.0))],
        _T(10.0, 22.0, 21.0),
        ["[)", "[)"],
    )
    assert_ceinsum(out, expected, "ij,jk->ik")


@requires_ceinsum
def test_ij_ij__ij_pointwise_2d():
    """ij,ij->ij — 2-D pointwise overlap of interval boxes."""
    op0 = _ct(
        [(_T(0.0, 4.0), _T(2.0, 6.0)),       # i: [0,2), [4,6)
         (_T(10.0, 14.0), _T(12.0, 16.0))],  # j: [10,12), [14,16)
        _T(2.0, 3.0),
        ["[)", "[)"],
    )
    op1 = _ct(
        [(_T(1.0, 5.0), _T(3.0, 7.0)),       # i: [1,3), [5,7)
         (_T(11.0, 15.0), _T(13.0, 17.0))],  # j: [11,13), [15,17)
        _T(5.0, 7.0),
        ["[)", "[)"],
    )

    out = ceinsum("ij,ij->ij", op0, op1)

    # (p0,q0): i [0,2)∩[1,3)=[1,2) ; j [10,12)∩[11,13)=[11,12) ; v=2*5=10
    # (p1,q1): i [4,6)∩[5,7)=[5,6) ; j [14,16)∩[15,17)=[15,16) ; v=3*7=21
    expected = _ct(
        [(_T(1.0, 5.0), _T(2.0, 6.0)),
         (_T(11.0, 15.0), _T(12.0, 16.0))],
        _T(10.0, 21.0),
        ["[)", "[)"],
    )
    assert_ceinsum(out, expected, "ij,ij->ij")


@requires_ceinsum
def test_ijk_k__ij_3d_operand():
    """ijk,k->ij — 3-D operand, contract pinpoint k down to a 2-D output."""
    op0 = _ct(
        [(_T(0.0, 10.0), _T(2.0, 12.0)),   # i: [0,2), [10,12)
         (_T(0.0, 10.0), _T(2.0, 12.0)),   # j: [0,2), [10,12)
         (_T(1.0, 3.0),)],                 # k pinpoints: 1.0, 3.0
        _T(2.0, 4.0),
        ["[)", "[)", "P"],
    )
    op1 = _ct([(_T(1.0, 3.0, 9.0),)], _T(10.0, 30.0, 99.0), ["P"])  # k: 1,3,9

    out = ceinsum("ijk,k->ij", op0, op1)

    # p0(k=1)&q0(k=1): i=[0,2) j=[0,2) v=2*10=20
    # p1(k=3)&q1(k=3): i=[10,12) j=[10,12) v=4*30=120 ; k=9 unmatched
    expected = _ct(
        [(_T(0.0, 10.0), _T(2.0, 12.0)),
         (_T(0.0, 10.0), _T(2.0, 12.0))],
        _T(20.0, 120.0),
        ["[)", "[)"],
    )
    assert_ceinsum(out, expected, "ijk,k->ij")


@requires_ceinsum
def test_i_i_i__i_three_providers():
    """i,i,i->i — 3 interval operands intersect to one piece."""
    a = _ct([(_T(0.0), _T(10.0))], _T(2.0), ["[)"])  # [0,10)
    b = _ct([(_T(1.0), _T(8.0))], _T(3.0), ["[)"])   # [1,8)
    c = _ct([(_T(2.0), _T(9.0))], _T(5.0), ["[)"])   # [2,9)

    out = ceinsum("i,i,i->i", a, b, c)

    # intersect: start=max(0,1,2)=2, end=min(10,8,9)=8.
    expected = _ct([(_T(2.0), _T(8.0))], _T(30.0), ["[)"])
    assert_ceinsum(out, expected, "i,i,i->i three providers")


# ===========================================================================
# Integral semantics — the manuscript's worked examples and reduced-interval
# variables (mask value = intersection length, so contraction integrates).
# ===========================================================================


@requires_ceinsum
def test_manuscript_example1_pointwise_2d():
    """ij,ij->ij — manuscript Example 1: one output piece per intersecting
    pair of boxes; a1 and b1 do not intersect."""
    A = _ct(
        [(_T(0.0, 3.0), _T(2.0, 5.0)),   # i: [0,2), [3,5)
         (_T(0.0, 1.0), _T(2.0, 4.0))],  # j: [0,2), [1,4)
        _T(2.0, 3.0),
        ["[)", "[)"],
    )
    B = _ct(
        [(_T(1.0, 0.0), _T(4.0, 1.0)),   # i: [1,4), [0,1)
         (_T(1.0, 0.0), _T(3.0, 2.0))],  # j: [1,3), [0,2)
        _T(10.0, 20.0),
        ["[)", "[)"],
    )

    out = ceinsum("ij,ij->ij", A, B)

    # (a0,b0): [1,2)x[1,2):20  (a0,b1): [0,1)x[0,2):40  (a1,b0): [3,4)x[1,3):30
    expected = _ct(
        [(_T(1.0, 0.0, 3.0), _T(2.0, 1.0, 4.0)),
         (_T(1.0, 0.0, 1.0), _T(2.0, 2.0, 3.0))],
        _T(20.0, 40.0, 30.0),
        ["[)", "[)"],
    )
    assert_ceinsum(out, expected, "manuscript ex1")


@requires_ceinsum
def test_manuscript_example2_integral():
    """ik,k->i — manuscript Example 2: C_i = ∫_k A_ik·B_k dk. The reduced k is
    all-interval, so each join tuple is weighted by its k-overlap length."""
    A = _ct(
        [(_T(0.0, 3.0), _T(2.0, 5.0)),   # i: [0,2), [3,5)
         (_T(1.0, 0.0), _T(4.0, 6.0))],  # k: [1,4), [0,6)
        _T(2.0, 3.0),
        ["[)", "[)"],
    )
    B = _ct([(_T(0.0, 2.0), _T(1.0, 5.0))], _T(10.0, 20.0), ["[)"])  # k: [0,1), [2,5)

    out = ceinsum("ik,k->i", A, B)

    # (a0,b0): k [1,4)∩[0,1)=∅. (a0,b1): len([2,4))=2 → 2·20·2=80 on [0,2).
    # (a1,b0): len([0,1))=1 → 3·10·1=30 ; (a1,b1): len([2,5))=3 → 3·20·3=180;
    # both on i=[3,5) → 210.
    expected = _ct([(_T(0.0, 3.0), _T(2.0, 5.0))], _T(80.0, 210.0), ["[)"])
    assert_ceinsum(out, expected, "manuscript ex2")


@requires_ceinsum
def test_ij__i_single_operand_integral_coalesce():
    """ij->i — a single operand integrates over its own j extent; the
    overlapping i pieces are then coalesced."""
    A = _ct(
        [(_T(0.0, 1.0), _T(2.0, 3.0)),   # i: [0,2), [1,3)
         (_T(0.0, 1.0), _T(3.0, 4.0))],  # j: [0,3), [1,4)
        _T(2.0, 5.0),
        ["[)", "[)"],
    )

    out = ceinsum("ij->i", A)

    # piece0: 2·len([0,3))=6 on [0,2); piece1: 5·len([1,4))=15 on [1,3).
    # coalesced: [0,1):6, [1,2):21, [2,3):15.
    expected = _ct(
        [(_T(0.0, 1.0, 2.0), _T(1.0, 2.0, 3.0))], _T(6.0, 21.0, 15.0), ["[)"]
    )
    assert_ceinsum(out, expected, "ij->i integral")


@requires_ceinsum
def test_ij_i__i_interval_j_single_carrier_integral():
    """ij,i->i — the reduced interval j is carried by one operand inside a
    join: its measure is that piece's own j length."""
    A = _ct([(_T(0.0), _T(4.0)), (_T(0.0), _T(2.0))], _T(2.0), ["[)", "[)"])
    B = _ct([(_T(1.0), _T(3.0))], _T(10.0), ["[)"])

    out = ceinsum("ij,i->i", A, B)

    # i: [0,4)∩[1,3)=[1,3); MV = len(j=[0,2)) = 2 → 2·10·2 = 40.
    expected = _ct([(_T(1.0), _T(3.0))], _T(40.0), ["[)"])
    assert_ceinsum(out, expected, "ij,i->i integral")


def _integral_case_operands():
    A = _ct(
        [(_T(0.0, 3.0), _T(2.0, 5.0)), (_T(1.0, 0.0), _T(4.0, 6.0))],
        _T(2.0, 3.0),
        ["[)", "[)"],
    )
    B = _ct([(_T(0.0, 2.0), _T(1.0, 5.0))], _T(10.0, 20.0), ["[)"])
    return A, B


@requires_ceinsum
def test_builder_equivalence_on_integral_case():
    """The dense and binary-search mask builders agree on the MV path."""
    from mask_dense import build_dense_mask

    A, B = _integral_case_operands()
    out_default = ceinsum("ik,k->i", A, B)
    out_dense = ceinsum("ik,k->i", A, B, builder=build_dense_mask)
    assert_ceinsum(out_dense, out_default, "builder equivalence (dense)")


@requires_ceinsum
def test_db_join_builder_equivalence_on_integral_case():
    """The polars database-join builder agrees on the MV path."""
    pytest.importorskip("polars")
    from mask_db_join import build_db_join_mask

    A, B = _integral_case_operands()
    out_default = ceinsum("ik,k->i", A, B)
    out_db = ceinsum("ik,k->i", A, B, builder=build_db_join_mask)
    assert_ceinsum(out_db, out_default, "builder equivalence (db join)")
