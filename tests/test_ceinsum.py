"""Concrete, hand-checked tests for the continuous einsum API (``ceinsum``).

Each test builds operands from explicit small numbers and asserts the exact
expected output tensor. No reference implementation, no synthesis — just
"feed these pieces, expect this result", across a variety of einsum equations,
operand dimensionalities, and property combinations (operand count kept ≤ 3).

Semantics being pinned down (see continuous_einsum module docstring):

* Interval/interval overlap: ``lo1 OP hi2`` and ``lo2 OP hi1`` where ``OP`` is
  ``<=`` only when *both* touching boundaries are closed, else ``<``.
* Point in interval: ``lo OP p`` and ``p OP hi`` with the same closed→``<=``
  rule. Pinpoint vs pinpoint: exact equality.
* Output piece identity = the tuple of piece indices of the operands providing
  the output indices (uniqueness by contributing pieces, not coordinate value).
* Output coordinate: ``start = max(starts)``, ``end = min(ends)``; if any
  provider is a pinpoint the output index is a pinpoint and the coord is copied.
* Output dim property: pinpoint if any provider is a pinpoint; else interval,
  left-closed iff all providers left-closed, right-closed iff all right-closed.
* Output value: plain product of matched operands' values, scatter-added per
  output piece (no length/measure weighting).

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
def test_i_i__i_touching_closed_overlaps():
    """[] ⨉ [] intervals touching at a single closed point overlap."""
    a = _ct([(_T(0.0, 3.0), _T(1.0, 5.0))], _T(2.0, 4.0), ["[]"])  # [0,1], [3,5]
    b = _ct([(_T(1.0, 6.0), _T(2.0, 9.0))], _T(3.0, 7.0), ["[]"])  # [1,2], [6,9]

    out = ceinsum("i,i->i", a, b)

    # [0,1]∩[1,2]=[1,1] (touch, both closed) v=2*3=6 ; [3,5]∩[6,9]=∅
    expected = _ct([(_T(1.0), _T(1.0))], _T(6.0), ["[]"])
    assert_ceinsum(out, expected, "i,i->i touching []")


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
    t2 = _ct([(_T(1.0, 10.0), _T(4.0, 12.0))], _T(10.0, 20.0), ["[]"])  # i: [1,4],[10,12]
    t3 = _ct([(_T(0.5, 5.5),)], _T(100.0, 200.0), ["P"])               # j: pts 0.5, 5.5

    out = ceinsum("ij,i,j->i", t1, t2, t3)

    # (t1.0,t2.0): [0,2)∩[1,4]=[1,2), v=2*10*100=2000
    # (t1.1,t2.0): [1,3)∩[1,4]=[1,3), v=3*10*200=6000
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
def test_ij_ij__ij_pointwise_2d_leftopen():
    """ij,ij->ij — 2-D pointwise overlap with a left-open (] dimension."""
    op0 = _ct(
        [(_T(0.0, 4.0), _T(2.0, 6.0)),       # i: [0,2), [4,6)
         (_T(10.0, 14.0), _T(12.0, 16.0))],  # j: (10,12], (14,16]
        _T(2.0, 3.0),
        ["[)", "(]"],
    )
    op1 = _ct(
        [(_T(1.0, 5.0), _T(3.0, 7.0)),       # i: [1,3), [5,7)
         (_T(11.0, 15.0), _T(13.0, 17.0))],  # j: (11,13], (15,17]
        _T(5.0, 7.0),
        ["[)", "(]"],
    )

    out = ceinsum("ij,ij->ij", op0, op1)

    # (p0,q0): i [0,2)∩[1,3)=[1,2) ; j (10,12]∩(11,13]=(11,12] ; v=2*5=10
    # (p1,q1): i [4,6)∩[5,7)=[5,6) ; j (14,16]∩(15,17]=(15,16] ; v=3*7=21
    expected = _ct(
        [(_T(1.0, 5.0), _T(2.0, 6.0)),
         (_T(11.0, 15.0), _T(12.0, 16.0))],
        _T(10.0, 21.0),
        ["[)", "(]"],
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
def test_i_i_i__i_three_providers_property():
    """i,i,i->i — 3 interval operands; output property is the conservative AND."""
    a = _ct([(_T(0.0), _T(10.0))], _T(2.0), ["[]"])  # [0,10]
    b = _ct([(_T(1.0), _T(8.0))], _T(3.0), ["[]"])   # [1,8]
    c = _ct([(_T(2.0), _T(9.0))], _T(5.0), ["[)"])   # [2,9)

    out = ceinsum("i,i,i->i", a, b, c)

    # intersect: start=max(0,1,2)=2, end=min(10,8,9)=8.
    # left closed: all left-closed → "[" ; right closed: c is open → ")".
    expected = _ct([(_T(2.0), _T(8.0))], _T(30.0), ["[)"])
    assert_ceinsum(out, expected, "i,i,i->i property")
