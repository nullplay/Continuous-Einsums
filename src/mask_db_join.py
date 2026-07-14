"""Mask builder: database join via polars.

The third mask-creation strategy of the thesis chapter: finding the tuples of
pieces that satisfy the intersection conditions is precisely a database join,
so the whole step is handed to a database library. Each join axis becomes one
frame (its coordinate tensors as columns plus a row-index column), the
condition DSL translates to column predicates, and axes fold together with
``join_where`` (an inequality join; a plain cross join when no condition
links the new axis).

Shares the ``(op, output, cond)`` DSL and output layout of
:mod:`mask_dense` / :mod:`mask_binary_search`: the returned closure yields a
tuple of 1-D long index columns, one per output axis. ``polars`` is imported
inside the builder, so this module is importable without it.
"""

from __future__ import annotations

from typing import Callable

import torch

from mask_binary_search import _Cond, _Expr, _parse_cond


def _piece_col(axis: str) -> str:
    return f"__piece_{axis}"


def build_db_join_mask(
    op: dict[str, torch.Tensor],
    output: tuple[str, ...],
    cond: list[str],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    import polars as pl

    parsed: list[_Cond] = [_parse_cond(c) for c in cond]

    # Each tensor belongs to exactly one axis; collect the mapping so every
    # axis's frame can gather its columns (and its extent). A condition may
    # touch one axis (a per-row filter), two (an ordinary join predicate), or
    # more (a predicate applied once all its axes are joined).
    tensor_axis: dict[str, str] = {}
    for c in parsed:
        if not c.axes():
            raise ValueError(f"condition references no axis: {c}")
        for expr in (c.left, c.right):
            for t in expr.terms:
                if t.tensor is None:
                    continue
                prev = tensor_axis.setdefault(t.tensor, t.axis)
                if prev != t.axis:
                    raise ValueError(
                        f"tensor {t.tensor!r} used with two axes ({prev}, {t.axis})"
                    )
    for ax in output:
        if ax not in tensor_axis.values():
            raise NotImplementedError(
                f"output axis {ax!r} is referenced by no tensor: extent unknown"
            )

    device = next(iter(op.values())).device

    def _pl_expr(expr: _Expr) -> "pl.Expr":
        acc = None
        for t in expr.terms:
            if t.tensor is None:
                e = pl.lit(t.coef)
            elif t.coef == 1.0:
                e = pl.col(t.tensor)
            else:
                e = pl.col(t.tensor) * t.coef
            acc = e if acc is None else acc + e
        return acc if acc is not None else pl.lit(0.0)

    def _pl_pred(c: _Cond) -> "pl.Expr":
        left, right = _pl_expr(c.left), _pl_expr(c.right)
        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[c.op]

    def db_join_mask() -> tuple[torch.Tensor, ...]:
        applied: set[int] = set()

        frames: dict[str, pl.LazyFrame] = {}
        for ax in output:
            cols = {
                name: op[name].detach().cpu().numpy()
                for name, a in tensor_axis.items()
                if a == ax
            }
            n = len(next(iter(cols.values())))
            cols[_piece_col(ax)] = torch.arange(n).numpy()
            frame = pl.LazyFrame(cols)
            # Single-axis conditions are per-row filters on the axis's frame.
            for i, c in enumerate(parsed):
                if i not in applied and c.axes() == {ax}:
                    frame = frame.filter(_pl_pred(c))
                    applied.add(i)
            frames[ax] = frame

        joined = frames[output[0]]
        joined_axes = {output[0]}
        for ax in output[1:]:
            preds = []
            for i, c in enumerate(parsed):
                if (
                    i not in applied
                    and ax in c.axes()
                    and c.axes() <= joined_axes | {ax}
                ):
                    preds.append(_pl_pred(c))
                    applied.add(i)
            if preds:
                joined = joined.join_where(frames[ax], *preds)
            else:
                joined = joined.join(frames[ax], how="cross")
            joined_axes.add(ax)
            # A condition whose axes all joined through *other* conditions
            # becomes a plain filter.
            for i, c in enumerate(parsed):
                if i not in applied and c.axes() <= joined_axes:
                    joined = joined.filter(_pl_pred(c))
                    applied.add(i)

        df = joined.select([_piece_col(ax) for ax in output]).collect()
        return tuple(
            torch.from_numpy(df[_piece_col(ax)].to_numpy().copy())
            .to(dtype=torch.long, device=device)
            for ax in output
        )

    return db_join_mask
