"""Regression: band-lead detection must not pair conditions across dimensions.

A 2-D point-in-box join (A all-pinpoint, B all-interval) emits two
containments whose condition list can pair one x cond with one y cond into a
shape-valid but dimensionally-mixed "overlap" lead. Before the build-time
interval check, that produced crossing windows (``repeats can not be
negative``) or, when it didn't crash, a uselessly loose band.
"""

from __future__ import annotations

import pytest
import torch

from mask_binary_search import build_binary_search_mask
from mask_dense import build_dense_mask
from synth_dataset import INTERVAL, PINPOINT, create_nd_pieces

SPACE_MAX = 64.0


@pytest.mark.parametrize("n", [100, 1000, 5000])
def test_point_in_box_2d_lead_not_mispaired(n: int) -> None:
    (a_x,), (a_y,) = create_nd_pieces(
        n, (PINPOINT, PINPOINT), (SPACE_MAX, SPACE_MAX), 0.5, seed=11
    )
    (b_xs, b_xe), (b_ys, b_ye) = create_nd_pieces(
        n, (INTERVAL, INTERVAL), (SPACE_MAX, SPACE_MAX), 0.5, seed=22
    )
    op = {"A_x": a_x, "A_y": a_y, "B_x_s": b_xs, "B_x_e": b_xe,
          "B_y_s": b_ys, "B_y_e": b_ye}
    output = ("A", "B")
    eqs = [
        "B_x_s[B] <= A_x[A]", "A_x[A] < B_x_e[B]",
        "B_y_s[B] <= A_y[A]", "A_y[A] < B_y_e[B]",
    ]

    got = build_binary_search_mask(op, output, eqs)()
    ref = build_dense_mask(op, output, eqs)()

    got_rows = torch.stack([c.cpu().to(torch.long) for c in got], dim=1)
    order = torch.argsort(got_rows[:, 0] * n + got_rows[:, 1])
    got_rows = got_rows[order]
    ref_rows = ref.cpu().to(torch.long)
    order = torch.argsort(ref_rows[:, 0] * n + ref_rows[:, 1])
    ref_rows = ref_rows[order]

    assert got_rows.shape == ref_rows.shape
    assert torch.equal(got_rows, ref_rows)
