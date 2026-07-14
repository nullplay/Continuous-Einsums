"""Tests for the interaction-table continuous einsum (``table_ceinsum``).

Covers, in order:

* the thesis chapter's two worked examples, pinned to their exact grids;
* pointwise agreement with the ground-truth product of operand evaluations
  and with the ``ceinsum`` pipeline (the two now share integral semantics:
  a reduced all-interval index weights each contribution by its overlap
  length);
* a brute-force loop reference for the integrated case (``ik,kj->ij``),
  which ``ceinsum`` must also match pointwise;
* structural properties: output dimension types, explicit zeros kept,
  unsupported brackets rejected.
"""

from __future__ import annotations

import pytest
import torch

from continuous_einsum import ceinsum
from ctensor import ContinuousTensor, continuous_tensor
from table_ceinsum import table_ceinsum

DTYPE = torch.float64


def _T(*xs):
    return torch.tensor(list(xs), dtype=DTYPE)


def _ct(dims, values, property):
    return continuous_tensor(dims, values, property, dtype=DTYPE)


def _eval_at(ct: ContinuousTensor, points: torch.Tensor) -> torch.Tensor:
    """Evaluate a continuous tensor at ``points`` (npts, ndim): sum of covering pieces."""
    npts = points.shape[0]
    mask = torch.ones(npts, ct.nnz, dtype=torch.bool)
    for d in range(ct.ndim):
        p = points[:, d][:, None]
        spec = ct.dims[d]
        if len(spec) == 1:
            mask &= p == spec[0][None, :]
        else:
            mask &= (spec[0][None, :] <= p) & (p < spec[1][None, :])
    return (mask.to(ct.values.dtype) * ct.values[None, :]).sum(dim=1)


def _interval_operand(n, lo, hi, seed, max_width=2.0):
    """1-D random half-open intervals in [lo, hi) with positive random values."""
    gen = torch.Generator().manual_seed(seed)
    s = lo + (hi - lo - max_width) * torch.rand(n, generator=gen, dtype=DTYPE)
    e = s + 0.05 + (max_width - 0.05) * torch.rand(n, generator=gen, dtype=DTYPE)
    v = 0.5 + torch.rand(n, generator=gen, dtype=DTYPE)
    return s, e, v


def _probe_points_1d(*cts):
    """Cell midpoints of all interval endpoints appearing in the 1-D operands."""
    pts = torch.unique(torch.cat([t for ct in cts for t in ct.dims[0]]))
    return ((pts[:-1] + pts[1:]) / 2)[:, None]


# ---------------------------------------------------------------------------
# Thesis worked examples, pinned exactly.
# ---------------------------------------------------------------------------


def test_example_A_elementwise_intervals():
    """i,i->i with interval operands: 7 cells, grid (0,20,60,40,0,60,0)."""
    A = _ct([(_T(0.0, 5.0), _T(4.0, 7.0))], _T(2.0, 3.0), ["[)"])
    B = _ct([(_T(1.0, 2.0), _T(3.0, 6.0))], _T(10.0, 20.0), ["[)"])

    res = table_ceinsum("i,i->i", A, B)

    cs, ce = res.cands["i"]
    assert torch.equal(cs, _T(0, 1, 2, 3, 4, 5, 6))
    assert torch.equal(ce, _T(1, 2, 3, 4, 5, 6, 7))
    assert torch.allclose(res.grid, _T(0, 20, 60, 40, 0, 60, 0))

    out = res.to_coo()
    assert out.property == ("[)",)
    assert out.nnz == 7  # explicit zeros kept


def test_example_B_matmul_integrated():
    """ik,kj->ij with A:(P,I), B:(I,I): k integrated with overlap-length weights."""
    A = _ct(
        [(_T(1.0, 1.0, 4.0),), (_T(0.0, 2.0, 0.0), _T(3.0, 5.0, 1.0))],
        _T(2.0, 3.0, 5.0),
        ["P", "[)"],
    )
    B = _ct(
        [(_T(1.0, 0.0), _T(4.0, 6.0)), (_T(0.0, 1.0), _T(2.0, 3.0))],
        _T(10.0, 20.0),
        ["[)", "[)"],
    )

    res = table_ceinsum("ik,kj->ij", A, B)

    assert torch.equal(res.cands["i"][0], _T(1.0, 4.0))
    assert torch.equal(res.cands["j"][0], _T(0.0, 1.0, 2.0))
    assert torch.equal(res.cands["j"][1], _T(1.0, 2.0, 3.0))
    expected = torch.tensor(
        [[100.0, 400.0, 300.0], [0.0, 100.0, 100.0]], dtype=DTYPE
    )
    assert torch.allclose(res.grid, expected)

    out = res.to_coo()
    assert out.property == ("P", "[)")
    assert out.nnz == 6  # 2 x 3 grid, the (i=4, j=[0,1)) zero included
    assert torch.allclose(out.values, expected.reshape(-1))


# ---------------------------------------------------------------------------
# Pointwise cross-checks on shared-semantics equations (no all-I contraction).
# ---------------------------------------------------------------------------


def test_pointwise_i_i__i_intervals():
    """i,i->i: table result == A(p)*B(p) pointwise == existing ceinsum."""
    sa, ea, va = _interval_operand(40, 0.0, 20.0, seed=1)
    sb, eb, vb = _interval_operand(35, 0.0, 20.0, seed=2)
    A = _ct([(sa, ea)], va, ["[)"])
    B = _ct([(sb, eb)], vb, ["[)"])

    out_t = table_ceinsum("i,i->i", A, B).to_coo()
    out_c = ceinsum("i,i->i", A, B)

    pts = _probe_points_1d(A, B)
    truth = _eval_at(A, pts) * _eval_at(B, pts)
    assert torch.allclose(_eval_at(out_t, pts), truth, atol=1e-9)
    assert torch.allclose(_eval_at(out_c, pts), truth, atol=1e-9)


def test_pointwise_ij_j__i_pinpoint_contraction():
    """ij,j->i: pinpoint j contracts by summation in both implementations."""
    gen = torch.Generator().manual_seed(3)
    n_a, n_b = 30, 25
    sa, ea, va = _interval_operand(n_a, 0.0, 15.0, seed=4)
    j_levels = _T(0.0, 1.0, 2.0, 3.0, 4.0)
    ja = j_levels[torch.randint(0, 5, (n_a,), generator=gen)]
    jb = j_levels[torch.randint(0, 5, (n_b,), generator=gen)]
    vb = 0.5 + torch.rand(n_b, generator=gen, dtype=DTYPE)

    A = _ct([(sa, ea), (ja,)], va, ["[)", "P"])
    B = _ct([(jb,)], vb, ["P"])

    out_t = table_ceinsum("ij,j->i", A, B).to_coo()
    out_c = ceinsum("ij,j->i", A, B)

    pts = _probe_points_1d(
        _ct([(sa, ea)], va, ["[)"]), _ct([(sa, ea)], va, ["[)"])
    )
    # ground truth: C(p) = sum_a [p in A_i(a)] * Av[a] * sum_b [jb == ja[a]] * Bv[b]
    b_sum_at = {float(j): float(vb[jb == j].sum()) for j in j_levels}
    covered = (sa[None, :] <= pts) & (pts < ea[None, :])
    weights = va * _T(*[b_sum_at[float(j)] for j in ja])
    truth = (covered.to(DTYPE) * weights[None, :]).sum(dim=1)

    assert torch.allclose(_eval_at(out_t, pts), truth, atol=1e-9)
    assert torch.allclose(_eval_at(out_c, pts), truth, atol=1e-9)


def test_pointwise_ij_ij__ij_mixed():
    """ij,ij->ij with i interval, j pinpoint: pointwise product, no contraction."""
    gen = torch.Generator().manual_seed(5)
    n_a, n_b = 25, 20
    sa, ea, va = _interval_operand(n_a, 0.0, 12.0, seed=6)
    sb, eb, vb = _interval_operand(n_b, 0.0, 12.0, seed=7)
    j_levels = _T(0.0, 1.0, 2.0)
    ja = j_levels[torch.randint(0, 3, (n_a,), generator=gen)]
    jb = j_levels[torch.randint(0, 3, (n_b,), generator=gen)]

    A = _ct([(sa, ea), (ja,)], va, ["[)", "P"])
    B = _ct([(sb, eb), (jb,)], vb, ["[)", "P"])

    out_t = table_ceinsum("ij,ij->ij", A, B).to_coo()
    out_c = ceinsum("ij,ij->ij", A, B)

    i_pts = _probe_points_1d(_ct([(sa, ea)], va, ["[)"]), _ct([(sb, eb)], vb, ["[)"]))
    pts = torch.cartesian_prod(i_pts[:, 0], j_levels)
    truth = _eval_at(A, pts) * _eval_at(B, pts)
    assert torch.allclose(_eval_at(out_t, pts), truth, atol=1e-9)
    assert torch.allclose(_eval_at(out_c, pts), truth, atol=1e-9)


# ---------------------------------------------------------------------------
# Brute-force reference for the integrated case.
# ---------------------------------------------------------------------------


def _brute_force_ik_kj(A: ContinuousTensor, B: ContinuousTensor) -> torch.Tensor:
    """Loop reference for ik,kj->ij with A:(P,I), B:(I,I): integrate over k."""
    ai = A.dims[0][0]
    aks, ake = A.dims[1]
    bks, bke = B.dims[0]
    bjs, bje = B.dims[1]
    pts_i = torch.unique(ai)
    pts_j = torch.unique(torch.cat([bjs, bje]))
    cjs, cje = pts_j[:-1], pts_j[1:]
    grid = torch.zeros(len(pts_i), len(cjs), dtype=A.values.dtype)
    for a in range(A.nnz):
        for b in range(B.nnz):
            w = min(float(ake[a]), float(bke[b])) - max(float(aks[a]), float(bks[b]))
            if w <= 0:
                continue
            i_idx = int((pts_i == ai[a]).nonzero())
            for c in range(len(cjs)):
                if bjs[b] <= cjs[c] and cje[c] <= bje[b]:
                    grid[i_idx, c] += float(A.values[a]) * float(B.values[b]) * w
    return grid


def test_integrated_matmul_vs_brute_force():
    gen = torch.Generator().manual_seed(8)
    n_a, n_b = 9, 8
    ai = _T(0.0, 1.0, 2.0)[torch.randint(0, 3, (n_a,), generator=gen)]
    aks = 5.0 * torch.rand(n_a, generator=gen, dtype=DTYPE)
    ake = aks + 0.1 + 2.0 * torch.rand(n_a, generator=gen, dtype=DTYPE)
    va = 0.5 + torch.rand(n_a, generator=gen, dtype=DTYPE)
    bks = 5.0 * torch.rand(n_b, generator=gen, dtype=DTYPE)
    bke = bks + 0.1 + 2.0 * torch.rand(n_b, generator=gen, dtype=DTYPE)
    bjs = 5.0 * torch.rand(n_b, generator=gen, dtype=DTYPE)
    bje = bjs + 0.1 + 2.0 * torch.rand(n_b, generator=gen, dtype=DTYPE)
    vb = 0.5 + torch.rand(n_b, generator=gen, dtype=DTYPE)

    A = _ct([(ai,), (aks, ake)], va, ["P", "[)"])
    B = _ct([(bks, bke), (bjs, bje)], vb, ["[)", "[)"])

    res = table_ceinsum("ik,kj->ij", A, B)
    assert torch.allclose(res.grid, _brute_force_ik_kj(A, B), atol=1e-9)

    # ceinsum shares the integral semantics: compare pointwise at every
    # (i level, j cell midpoint).
    out_c = ceinsum("ik,kj->ij", A, B)
    out_t = res.to_coo()
    pts_j = torch.unique(torch.cat([B.dims[1][0], B.dims[1][1]]))
    mid_j = (pts_j[:-1] + pts_j[1:]) / 2
    pts = torch.cartesian_prod(torch.unique(A.dims[0][0]), mid_j)
    assert torch.allclose(_eval_at(out_c, pts), _eval_at(out_t, pts), atol=1e-9)


def test_single_operand_interval_reduction():
    """ij->i with j interval: integrate each piece over its own j extent."""
    A = _ct(
        [(_T(0.0, 1.0), _T(2.0, 3.0)), (_T(0.0, 1.0), _T(3.0, 4.0))],
        _T(2.0, 5.0),
        ["[)", "[)"],
    )
    res = table_ceinsum("ij->i", A)
    # cells of i: [0,1) [1,2) [2,3); weights: piece0 j-length 3, piece1 j-length 3
    assert torch.allclose(res.grid, _T(2 * 3, 2 * 3 + 5 * 3, 5 * 3))

    # ceinsum agrees on the single-operand integral.
    out_c = ceinsum("ij->i", A)
    pts = _probe_points_1d(_ct([A.dims[0]], A.values, ["[)"]))
    assert torch.allclose(_eval_at(out_c, pts), _eval_at(res.to_coo(), pts), atol=1e-9)


# ---------------------------------------------------------------------------
# Structural properties.
# ---------------------------------------------------------------------------


def test_output_types_follow_carriers():
    """Output dim is P iff some input carrier is P; the P column wins in step 1."""
    A = _ct([(_T(1.0, 2.0),), (_T(0.0, 4.0), _T(3.0, 6.0))], _T(1.0, 1.0), ["P", "[)"])
    B = _ct([(_T(0.0, 3.0), _T(2.0, 5.0)), (_T(2.0, 5.0),)], _T(1.0, 1.0), ["[)", "P"])
    out = table_ceinsum("ik,kj->ij", A, B).to_coo()
    assert out.property == ("P", "P")


def test_rejects_unsupported_brackets():
    A = _ct([(_T(0.0), _T(1.0))], _T(1.0), ["[]"])
    with pytest.raises(NotImplementedError):
        table_ceinsum("i->i", A)


def test_repeated_output_index_reuses_column():
    """ij,j->ii: both output dims read the same candidate column for i."""
    A = _ct([(_T(1.0, 2.0),), (_T(0.5, 0.5),)], _T(3.0, 4.0), ["P", "P"])
    B = _ct([(_T(0.5),)], _T(10.0), ["P"])
    out = table_ceinsum("ij,j->ii", A, B).to_coo()
    assert out.property == ("P", "P")
    assert torch.equal(out.dims[0][0], out.dims[1][0])
    assert torch.allclose(out.values, _T(30.0, 40.0))
