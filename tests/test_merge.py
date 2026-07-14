"""Unit tests for the unified merge (``ceinsum_merge.merge``).

``merge`` takes candidates — per-dim coordinates (pinpoint or half-open
interval) plus a value each — and returns the canonical disjoint COO tensor:
candidates covering the same region are summed (identical coordinates and
partially overlapping intervals alike), exact-zero regions and uncovered gaps
are dropped. Covers the 1-D sweep, pinpoint grouping, and the N-D
cut-then-merge on boxes, including mixed pinpoint × interval dimensions and
randomized differential checks against pointwise evaluation.
"""

from __future__ import annotations

import torch

from ceinsum_merge import merge
from ctensor import ContinuousTensor, continuous_tensor

DTYPE = torch.float64


def _T(*xs):
    return torch.tensor(list(xs), dtype=DTYPE)


def _ct(dims, values, property):
    return continuous_tensor(dims, values, property, dtype=DTYPE)


def _merge_ct(ct: ContinuousTensor) -> ContinuousTensor:
    return merge(list(ct.dims), ct.values, ct.property)


def _canonical(ct: ContinuousTensor) -> torch.Tensor:
    cols = [t.detach().cpu().to(torch.float64) for spec in ct.dims for t in spec]
    cols.append(ct.values.detach().cpu().to(torch.float64))
    if ct.nnz == 0:
        return torch.empty((0, len(cols)), dtype=torch.float64)
    mat = torch.stack(cols, dim=1)
    order = torch.arange(mat.shape[0])
    for col in range(mat.shape[1] - 1, -1, -1):
        order = order[torch.argsort(mat[order, col], stable=True)]
    return mat[order]


def assert_merged(out: ContinuousTensor, expected: ContinuousTensor, label=""):
    assert tuple(out.property) == tuple(expected.property), (label, out.property)
    assert out.nnz == expected.nnz, (label, out.nnz, expected.nnz)
    got, want = _canonical(out), _canonical(expected)
    assert torch.allclose(got, want, atol=1e-9), (label, got, want)


def _eval_at(ct: ContinuousTensor, pts: torch.Tensor) -> torch.Tensor:
    """Pointwise evaluation: sum of pieces covering each point."""
    mask = torch.ones(pts.shape[0], ct.nnz, dtype=torch.bool)
    for d in range(ct.ndim):
        p = pts[:, d][:, None]
        spec = ct.dims[d]
        if len(spec) == 1:
            mask &= p == spec[0][None, :]
        else:
            mask &= (spec[0][None, :] <= p) & (p < spec[1][None, :])
    return (mask.to(ct.values.dtype) * ct.values[None, :]).sum(dim=1)


def _assert_disjoint(ct: ContinuousTensor, label=""):
    """No two output pieces may overlap with positive measure."""
    n = ct.nnz
    if n < 2:
        return
    ov = torch.ones(n, n, dtype=torch.bool)
    for d in range(ct.ndim):
        spec = ct.dims[d]
        if len(spec) == 1:
            ov &= spec[0][:, None] == spec[0][None, :]
        else:
            lo = torch.maximum(spec[0][:, None], spec[0][None, :])
            hi = torch.minimum(spec[1][:, None], spec[1][None, :])
            ov &= lo < hi
    ov.fill_diagonal_(False)
    assert not ov.any(), (label, "output pieces overlap")


def _measure(ct: ContinuousTensor) -> torch.Tensor:
    """Total mass: sum of value x (product of interval widths)."""
    m = ct.values.clone()
    for d in range(ct.ndim):
        spec = ct.dims[d]
        if len(spec) == 2:
            m = m * (spec[1] - spec[0])
    return m.sum()


# ---------------------------------------------------------------------------
# 1-D sweep.
# ---------------------------------------------------------------------------


def test_1d_partial_overlap():
    """[0,3):10 and [2,5):20 cut at breakpoints into 10 / 30 / 20."""
    ct = _ct([(_T(0.0, 2.0), _T(3.0, 5.0))], _T(10.0, 20.0), ["[)"])
    expected = _ct(
        [(_T(0.0, 2.0, 3.0), _T(2.0, 3.0, 5.0))], _T(10.0, 30.0, 20.0), ["[)"]
    )
    assert_merged(_merge_ct(ct), expected, "1d partial")


def test_1d_gap_dropped():
    """Disjoint pieces pass through; the uncovered gap is not emitted."""
    ct = _ct([(_T(0.0, 2.0), _T(1.0, 3.0))], _T(1.0, 2.0), ["[)"])
    assert_merged(_merge_ct(ct), ct, "1d gap")


def test_1d_touching_pieces_stay_separate():
    """Adjacent pieces with different values share a boundary, not a piece."""
    ct = _ct([(_T(0.0, 1.0), _T(1.0, 2.0))], _T(1.0, 2.0), ["[)"])
    assert_merged(_merge_ct(ct), ct, "1d touching")


def test_1d_zero_sum_region_dropped():
    """Where candidate values cancel exactly, the region is dropped."""
    ct = _ct([(_T(0.0, 1.0), _T(2.0, 3.0))], _T(5.0, -5.0), ["[)"])
    expected = _ct([(_T(0.0, 2.0), _T(1.0, 3.0))], _T(5.0, -5.0), ["[)"])
    assert_merged(_merge_ct(ct), expected, "1d zero sum")


def test_1d_identical_pieces_sum():
    ct = _ct([(_T(1.0, 1.0), _T(4.0, 4.0))], _T(3.0, 4.0), ["[)"])
    expected = _ct([(_T(1.0), _T(4.0))], _T(7.0), ["[)"])
    assert_merged(_merge_ct(ct), expected, "1d identical")


def test_empty_input():
    ct = _ct([(_T(), _T())], _T(), ["[)"])
    out = _merge_ct(ct)
    assert out.nnz == 0 and out.property == ("[)",)


# ---------------------------------------------------------------------------
# Pinpoint dimensions.
# ---------------------------------------------------------------------------


def test_pinpoint_groups_sweep_independently():
    """Pieces overlap only within equal pinpoint coordinates."""
    ct = _ct(
        [(_T(1.0, 1.0, 2.0),), (_T(0.0, 2.0, 0.0), _T(3.0, 5.0, 1.0))],
        _T(10.0, 20.0, 5.0),
        ["P", "[)"],
    )
    expected = _ct(
        [(_T(1.0, 1.0, 1.0, 2.0),),
         (_T(0.0, 2.0, 3.0, 0.0), _T(2.0, 3.0, 5.0, 1.0))],
        _T(10.0, 30.0, 20.0, 5.0),
        ["P", "[)"],
    )
    assert_merged(_merge_ct(ct), expected, "pin groups")


def test_all_pinpoint_coincident_sum():
    ct = _ct([(_T(1.0, 1.0, 2.0),)], _T(3.0, 4.0, 5.0), ["P"])
    expected = _ct([(_T(1.0, 2.0),)], _T(7.0, 5.0), ["P"])
    assert_merged(_merge_ct(ct), expected, "all pinpoint")


def test_multi_pinpoint_dims():
    """(P, P): only exact coordinate tuples merge."""
    ct = _ct(
        [(_T(1.0, 1.0, 1.0, 2.0),), (_T(5.0, 5.0, 6.0, 5.0),)],
        _T(1.0, 2.0, 4.0, 8.0),
        ["P", "P"],
    )
    expected = _ct(
        [(_T(1.0, 1.0, 2.0),), (_T(5.0, 6.0, 5.0),)],
        _T(3.0, 4.0, 8.0),
        ["P", "P"],
    )
    assert_merged(_merge_ct(ct), expected, "P,P dedup")


# ---------------------------------------------------------------------------
# N-D boxes (cut-then-merge).
# ---------------------------------------------------------------------------


def test_2d_seven_cell_figure():
    """[0,3)x[0,2):10 ⊎ [2,5)x[1,3):20 → seven cells, shared cell 30."""
    ct = _ct(
        [(_T(0.0, 2.0), _T(3.0, 5.0)), (_T(0.0, 1.0), _T(2.0, 3.0))],
        _T(10.0, 20.0),
        ["[)", "[)"],
    )
    expected = _ct(
        [
            (_T(0.0, 0.0, 2.0, 2.0, 2.0, 3.0, 3.0),
             _T(2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 5.0)),
            (_T(0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0),
             _T(1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0)),
        ],
        _T(10.0, 10.0, 10.0, 30.0, 20.0, 20.0, 20.0),
        ["[)", "[)"],
    )
    assert_merged(_merge_ct(ct), expected, "2d seven cells")


def test_2d_disjoint_boxes_unchanged():
    """Boxes overlapping in one dim but disjoint in the other must not sum."""
    ct = _ct(
        [(_T(0.0, 0.0), _T(2.0, 2.0)), (_T(0.0, 5.0), _T(1.0, 6.0))],
        _T(10.0, 20.0),
        ["[)", "[)"],
    )
    assert_merged(_merge_ct(ct), ct, "2d disjoint")


def test_2d_identical_boxes_sum():
    ct = _ct(
        [(_T(1.0, 1.0), _T(2.0, 2.0)), (_T(3.0, 3.0), _T(4.0, 4.0))],
        _T(5.0, 6.0),
        ["[)", "[)"],
    )
    expected = _ct([(_T(1.0), _T(2.0)), (_T(3.0), _T(4.0))], _T(11.0), ["[)", "[)"])
    assert_merged(_merge_ct(ct), expected, "2d identical")


def test_3d_boxes_corner_overlap():
    """Two 3-D boxes overlapping in a corner: [0,2)^3:1 ⊎ [1,3)^3:2 →
    8 + 8 cells sharing the cell [1,2)^3, which sums to 3 → 15 cells."""
    ct = _ct(
        [(_T(0.0, 1.0), _T(2.0, 3.0))] * 3,
        _T(1.0, 2.0),
        ["[)", "[)", "[)"],
    )
    out = _merge_ct(ct)
    assert out.nnz == 15
    _assert_disjoint(out, "3d corner")
    # the shared cell holds 1+2, everything else keeps its box's value
    pts = torch.tensor([[1.5, 1.5, 1.5], [0.5, 0.5, 0.5], [2.5, 2.5, 2.5],
                        [0.5, 1.5, 1.5], [2.5, 1.5, 1.5]], dtype=DTYPE)
    assert torch.equal(_eval_at(out, pts), _T(3.0, 1.0, 2.0, 1.0, 2.0))
    # mass is conserved: 8·1 + 8·2 = 24
    assert torch.allclose(_measure(out), _measure(ct))


def test_mixed_pinpoint_and_2d_boxes():
    """(P, [), [)): boxes merge only within their pinpoint group."""
    ct = _ct(
        [
            (_T(7.0, 7.0, 8.0),),                     # groups: 7, 7, 8
            (_T(0.0, 2.0, 0.0), _T(3.0, 5.0, 3.0)),   # i intervals
            (_T(0.0, 1.0, 0.0), _T(2.0, 3.0, 2.0)),   # j intervals
        ],
        _T(10.0, 20.0, 40.0),
        ["P", "[)", "[)"],
    )
    out = _merge_ct(ct)
    _assert_disjoint(out, "mixed P+2D")
    # group 7 is the seven-cell figure; group 8 is one untouched box
    assert out.nnz == 7 + 1
    pts = torch.tensor(
        [[7.0, 2.5, 1.5],   # shared cell of group 7 → 30
         [7.0, 0.5, 0.5],   # group 7, box 1 only → 10
         [7.0, 4.0, 2.5],   # group 7, box 2 only → 20
         [8.0, 1.5, 1.0],   # group 8 → 40
         [8.0, 4.0, 2.5]],  # group 8, outside its box → 0
        dtype=DTYPE,
    )
    assert torch.equal(_eval_at(out, pts), _T(30.0, 10.0, 20.0, 40.0, 0.0))
    assert torch.allclose(_measure(out), _measure(ct))


def test_2d_random_differential():
    """Random overlapping boxes: output is disjoint, pointwise-identical to
    the candidate sum, and mass-conserving."""
    gen = torch.Generator().manual_seed(7)
    n = 40
    s1 = 10 * torch.rand(n, generator=gen, dtype=DTYPE)
    e1 = s1 + 0.2 + 1.5 * torch.rand(n, generator=gen, dtype=DTYPE)
    s2 = 10 * torch.rand(n, generator=gen, dtype=DTYPE)
    e2 = s2 + 0.2 + 1.5 * torch.rand(n, generator=gen, dtype=DTYPE)
    v = torch.randn(n, generator=gen, dtype=DTYPE)
    ct = _ct([(s1, e1), (s2, e2)], v, ["[)", "[)"])

    out = _merge_ct(ct)
    _assert_disjoint(out, "2d random")
    pts = 13 * torch.rand(700, 2, generator=gen, dtype=DTYPE) - 1
    assert torch.allclose(_eval_at(out, pts), _eval_at(ct, pts), atol=1e-9)
    assert torch.allclose(_measure(out), _measure(ct), atol=1e-9)


def test_3d_random_differential_with_pinpoint():
    """Random (P, [), [)) candidates with duplicate coordinates mixed in."""
    gen = torch.Generator().manual_seed(11)
    n = 90
    gk = torch.randint(0, 3, (n,), generator=gen).to(DTYPE)
    s1 = 8 * torch.rand(n, generator=gen, dtype=DTYPE)
    e1 = s1 + 0.2 + 2 * torch.rand(n, generator=gen, dtype=DTYPE)
    s2 = 8 * torch.rand(n, generator=gen, dtype=DTYPE)
    e2 = s2 + 0.2 + 2 * torch.rand(n, generator=gen, dtype=DTYPE)
    v = torch.randn(n, generator=gen, dtype=DTYPE)
    # duplicate a third of the candidates exactly (the dedup path)
    dup = slice(0, n // 3)
    ct = _ct(
        [(torch.cat([gk, gk[dup]]),),
         (torch.cat([s1, s1[dup]]), torch.cat([e1, e1[dup]])),
         (torch.cat([s2, s2[dup]]), torch.cat([e2, e2[dup]]))],
        torch.cat([v, v[dup]]),
        ["P", "[)", "[)"],
    )
    out = _merge_ct(ct)
    _assert_disjoint(out, "3d random")
    g_pts = torch.randint(0, 3, (600,), generator=gen).to(DTYPE)
    xy = 10 * torch.rand(600, 2, generator=gen, dtype=DTYPE) - 1
    pts = torch.cat([g_pts[:, None], xy], dim=1)
    assert torch.allclose(_eval_at(out, pts), _eval_at(ct, pts), atol=1e-9)
    assert torch.allclose(_measure(out), _measure(ct), atol=1e-9)


def test_scalar_output():
    out = merge([], _T(1.0, 2.0, 3.0), ())
    assert out.ndim == 0 and out.nnz == 1
    assert torch.allclose(out.values, _T(6.0))
