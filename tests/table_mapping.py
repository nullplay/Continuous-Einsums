"""Declarative builder for the brute-force table mapping pipeline.

A case is described by:

* ``op``: ``dict[str, torch.Tensor]`` mapping a *tensor name* to a 1-D tensor
  of values. Tensor names are arbitrary identifiers used inside conditions.

* ``output``: ordered collection of *axis names* (strings). Each axis is an
  iteration domain; the axis at position ``k`` lives on dimension ``k`` of the
  broadcasted boolean table and is also column ``k`` of the ``nonzero`` output.

* ``cond``: collection of Python expression strings using ``tensor[axis]`` to
  reference values. Each expression must be a single comparison
  (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) whose two sides may use
  ``+``, ``-``, ``*``, ``/`` over indexed-tensor references and numeric
  constants. The same axis name may index different tensors as long as those
  tensors have the same length; the same tensor may be indexed by different
  axes (e.g. ``A[i] == A[j]`` for a self-join).

``build_table_mapping(op, output, cond)`` returns a 0-arg callable that:

1. Broadcasts each indexed tensor along its axis so the conditions can be
   evaluated as full N-dimensional boolean tensors,
2. ANDs them together,
3. Returns ``torch.nonzero(mask, as_tuple=False).to(torch.long)``.

The resulting callable composes with :func:`torch.compile` exactly like
hand-written closures: the only ops emitted are reshape/broadcast/arith/cmp/
and/nonzero.
"""

from __future__ import annotations

import ast
from typing import Callable, Mapping, Sequence

import torch


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}

_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


def _reshape_for_axis(t: torch.Tensor, axis_pos: int, n_axes: int) -> torch.Tensor:
    shape = [1] * n_axes
    shape[axis_pos] = t.shape[0]
    return t.reshape(shape)


def _eval(node, op, axis_pos):
    if isinstance(node, ast.Expression):
        return _eval(node.body, op, axis_pos)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparisons are not supported")
        left = _eval(node.left, op, axis_pos)
        right = _eval(node.comparators[0], op, axis_pos)
        return _CMPOPS[type(node.ops[0])](left, right)
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](
            _eval(node.left, op, axis_pos),
            _eval(node.right, op, axis_pos),
        )
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval(node.operand, op, axis_pos))
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise ValueError("only `tensor[axis]` references are supported")
        if not isinstance(node.slice, ast.Name):
            raise ValueError("only single-name axis subscripts are supported")
        tensor_name = node.value.id
        axis_name = node.slice.id
        if tensor_name not in op:
            raise ValueError(f"unknown tensor {tensor_name!r}")
        if axis_name not in axis_pos:
            raise ValueError(
                f"axis {axis_name!r} not declared in output "
                f"{tuple(axis_pos)}"
            )
        return _reshape_for_axis(op[tensor_name], axis_pos[axis_name], len(axis_pos))
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def build_table_mapping(
    op: Mapping[str, torch.Tensor],
    output: Sequence[str],
    cond: Sequence[str],
) -> Callable[[], torch.Tensor]:
    output = tuple(output)
    cond = tuple(cond)
    if not cond:
        raise ValueError("at least one condition is required")
    if len(output) < 1:
        raise ValueError("at least one output axis is required")
    if len(set(output)) != len(output):
        raise ValueError(f"duplicate axis names in output: {output}")

    axis_pos = {name: i for i, name in enumerate(output)}
    parsed = tuple(ast.parse(c, mode="eval") for c in cond)

    def table_mapping() -> torch.Tensor:
        mask = _eval(parsed[0], op, axis_pos)
        for tree in parsed[1:]:
            mask = mask & _eval(tree, op, axis_pos)
        return torch.nonzero(mask, as_tuple=False).to(torch.long)

    return table_mapping
