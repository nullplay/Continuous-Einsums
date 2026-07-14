"""Unit tests for the merge step's coalesce (``ceinsum_merge.coalesce``).

Coalesce rewrites a COO continuous tensor whose pieces may overlap into an
equivalent one with disjoint pieces, summing values where pieces overlapped:
a sweep-line over breakpoints in 1-D, and per-dimension cutting followed by
the discrete merge in N-D. Exact-zero regions and uncovered gaps are dropped.
"""

from __future__ import annotations

import torch

from ceinsum_merge import coalesce
from ctensor import ContinuousTensor, continuous_tensor

DTYPE = torch.float64


def _T(*xs):
    return torch.tensor(list(xs), dtype=DTYPE)


def _ct(dims, values, property):
    return continuous_tensor(dims, values, property, dtype=DTYPE)


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


def assert_coalesced(out: ContinuousTensor, expected: ContinuousTensor, label=""):
    assert tuple(out.property) == tuple(expected.property), (label, out.property)
    assert out.nnz == expected.nnz, (label, out.nnz, expected.nnz)
    got, want = _canonical(out), _canonical(expected)
    assert torch.allclose(got, want, atol=1e-9), (label, got, want)


# ---------------------------------------------------------------------------
# 1-D sweep-line.
# ---------------------------------------------------------------------------


def test_1d_partial_overlap():
    """[0,3):10 and [2,5):20 cut at breakpoints into 10 / 30 / 20."""
    ct = _ct([(_T(0.0, 2.0), _T(3.0, 5.0))], _T(10.0, 20.0), ["[)"])
    expected = _ct(
        [(_T(0.0, 2.0, 3.0), _T(2.0, 3.0, 5.0))], _T(10.0, 30.0, 20.0), ["[)"]
    )
    assert_coalesced(coalesce(ct), expected, "1d partial")


def test_1d_gap_dropped():
    """Disjoint pieces pass through; the uncovered gap between them is not
    emitted."""
    ct = _ct([(_T(0.0, 2.0), _T(1.0, 3.0))], _T(1.0, 2.0), ["[)"])
    assert_coalesced(coalesce(ct), ct, "1d gap")


def test_1d_touching_pieces_stay_separate():
    """Adjacent pieces with different values share a boundary, not a piece."""
    ct = _ct([(_T(0.0, 1.0), _T(1.0, 2.0))], _T(1.0, 2.0), ["[)"])
    assert_coalesced(coalesce(ct), ct, "1d touching")


def test_1d_zero_sum_region_dropped():
    """Where candidate values cancel exactly, the region is dropped."""
    ct = _ct([(_T(0.0, 1.0), _T(2.0, 3.0))], _T(5.0, -5.0), ["[)"])
    expected = _ct([(_T(0.0, 2.0), _T(1.0, 3.0))], _T(5.0, -5.0), ["[)"])
    assert_coalesced(coalesce(ct), expected, "1d zero sum")


def test_1d_identical_pieces_sum():
    ct = _ct([(_T(1.0, 1.0), _T(4.0, 4.0))], _T(3.0, 4.0), ["[)"])
    expected = _ct([(_T(1.0), _T(4.0))], _T(7.0), ["[)"])
    assert_coalesced(coalesce(ct), expected, "1d identical")


def test_empty_input():
    ct = _ct([(_T(), _T())], _T(), ["[)"])
    out = coalesce(ct)
    assert out.nnz == 0 and out.property == ("[)",)


# ---------------------------------------------------------------------------
# Pinpoint grouping.
# ---------------------------------------------------------------------------


def test_pinpoint_groups_sweep_independently():
    """Pieces overlap only within equal pinpoint coordinates; groups do not
    bleed into each other."""
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
    assert_coalesced(coalesce(ct), expected, "pin groups")


def test_all_pinpoint_coincident_sum():
    ct = _ct([(_T(1.0, 1.0, 2.0),)], _T(3.0, 4.0, 5.0), ["P"])
    expected = _ct([(_T(1.0, 2.0),)], _T(7.0, 5.0), ["P"])
    assert_coalesced(coalesce(ct), expected, "all pinpoint")


# ---------------------------------------------------------------------------
# N-D cut-then-merge.
# ---------------------------------------------------------------------------


def test_2d_seven_cell_figure():
    """The manuscript's 2-D cut figure: [0,3)x[0,2):10 ⊎ [2,5)x[1,3):20 →
    seven cells, the shared cell [2,3)x[1,2) summing to 30."""
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
    assert_coalesced(coalesce(ct), expected, "2d seven cells")


def test_2d_disjoint_boxes_unchanged():
    """Boxes overlapping in one dim but disjoint in the other must not sum
    (and, being disjoint, come back unchanged)."""
    ct = _ct(
        [(_T(0.0, 0.0), _T(2.0, 2.0)), (_T(0.0, 5.0), _T(1.0, 6.0))],
        _T(10.0, 20.0),
        ["[)", "[)"],
    )
    out = coalesce(ct)
    assert_coalesced(out, ct, "2d disjoint")


def test_2d_identical_boxes_sum():
    ct = _ct(
        [(_T(1.0, 1.0), _T(2.0, 2.0)), (_T(3.0, 3.0), _T(4.0, 4.0))],
        _T(5.0, 6.0),
        ["[)", "[)"],
    )
    expected = _ct([(_T(1.0), _T(2.0)), (_T(3.0), _T(4.0))], _T(11.0), ["[)", "[)"])
    assert_coalesced(coalesce(ct), expected, "2d identical")


def test_2d_with_pinpoint_group():
    """A pinpoint dim separates otherwise-overlapping interval boxes."""
    ct = _ct(
        [(_T(7.0, 7.0, 8.0),), (_T(0.0, 1.0, 0.0), _T(2.0, 3.0, 2.0))],
        _T(1.0, 1.0, 9.0),
        ["P", "[)"],
    )
    expected = _ct(
        [(_T(7.0, 7.0, 7.0, 8.0),),
         (_T(0.0, 1.0, 2.0, 0.0), _T(1.0, 2.0, 3.0, 2.0))],
        _T(1.0, 2.0, 1.0, 9.0),
        ["P", "[)"],
    )
    assert_coalesced(coalesce(ct), expected, "2d pin group")
