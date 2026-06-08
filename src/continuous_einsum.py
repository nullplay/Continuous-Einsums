"""Continuous einsum: data model + public API (implementation pending).

This module defines the *shape* of the API that ``tests/test_ceinsum.py``
exercises. Only the data container (:class:`ContinuousTensor` /
:func:`continuous_tensor`) is implemented here; :func:`ceinsum` is a stub that
raises ``NotImplementedError`` until the four-step pipeline is built.

Data model
----------
A continuous tensor is stored COO-style: ``nnz`` "pieces", each piece carrying
one coordinate spec per dimension plus a single value. A dimension is either:

* an **interval** — two endpoint arrays ``(start, end)``, each length ``nnz``;
* a **pinpoint** — one coordinate array ``(coord,)``, length ``nnz``.

``property[d]`` is a code per dimension:

* ``"[)"`` ``"(]"`` ``"[]"`` ``"()"`` — interval, the brackets giving the
  closed/open-ness of the (start, end) boundaries;
* ``"P"`` — pinpoint.

Example::

    t1 = continuous_tensor([(xs, xe), (ys, ye)], v1, property=["[)", "[)"])
    t2 = continuous_tensor([(a, b)],             v2, property=["[]"])
    t3 = continuous_tensor([(c,)],               v3, property=["P"])
    out = ceinsum("ij,i,j->i", t1, t2, t3)

The four-step pipeline (target of the pending implementation)
-------------------------------------------------------------
1. **Mapping** — from the einsum string + per-dim properties, derive the
   intersection conditions linking the pieces of each operand (interval/interval
   overlap, point-in-interval, pinpoint equality) and enumerate the surviving
   piece tuples ``(p0, p1, ...)``. This reuses the ``op``/``output``/``cond``
   DSL of :mod:`table_mapping` / :mod:`table_opt_mapping`.
2. **Output construction** — group the mapping tuples by the piece indices of
   the operands that *provide each output index*; each unique group is one
   output piece. Assign a contiguous output id per tuple.
3. **Output coordinates** — per output index, intersect the contributing
   pieces (``start = max`` of starts, ``end = min`` of ends). If any provider
   is a pinpoint, copy the pinpoint coordinate.
4. **Values** — ``out.value[out_id] += prod(operand.value[piece])`` over all
   operands (gather → multiply → scatter-add).

The whole pipeline is meant to be one ``torch.compile`` region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

PINPOINT = "P"
INTERVAL_PROPERTIES = ("[)", "(]", "[]", "()")
VALID_PROPERTIES = INTERVAL_PROPERTIES + (PINPOINT,)


def is_pinpoint(prop: str) -> bool:
    return prop == PINPOINT


def left_closed(prop: str) -> bool:
    """Interval start boundary is closed (``[``)."""
    return prop[0] == "["


def right_closed(prop: str) -> bool:
    """Interval end boundary is closed (``]``)."""
    return prop[1] == "]"


@dataclass(frozen=True)
class ContinuousTensor:
    """COO-style continuous tensor.

    Attributes
    ----------
    dims:
        One tuple per dimension. ``(coord,)`` for a pinpoint dim,
        ``(start, end)`` for an interval dim. Every tensor is 1-D, length
        ``nnz``.
    values:
        1-D tensor, length ``nnz``.
    property:
        One code per dimension (see module docstring).
    """

    dims: tuple[tuple[torch.Tensor, ...], ...]
    values: torch.Tensor
    property: tuple[str, ...]

    @property
    def ndim(self) -> int:
        return len(self.property)

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ContinuousTensor(ndim={self.ndim}, nnz={self.nnz}, "
            f"property={list(self.property)})"
        )


def continuous_tensor(
    dims: Sequence[Sequence[torch.Tensor] | torch.Tensor],
    values: torch.Tensor,
    property: Sequence[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> ContinuousTensor:
    """Construct a :class:`ContinuousTensor`, validating shapes against ``property``.

    ``dims[d]`` is either a single 1-D tensor (pinpoint shorthand) / a
    1-tuple ``(coord,)`` when ``property[d] == "PP"``, or a 2-tuple
    ``(start, end)`` for an interval dim.
    """
    property = tuple(property)
    if len(dims) != len(property):
        raise ValueError(
            f"dims has {len(dims)} entries but property has {len(property)}"
        )
    for p in property:
        if p not in VALID_PROPERTIES:
            raise ValueError(
                f"invalid property {p!r}; expected one of {VALID_PROPERTIES}"
            )

    def _as_tensor(x: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t

    values = _as_tensor(values)
    nnz = int(values.shape[0])

    norm_dims: list[tuple[torch.Tensor, ...]] = []
    for d, (spec, prop) in enumerate(zip(dims, property)):
        if isinstance(spec, torch.Tensor):
            spec = (spec,)
        spec = tuple(_as_tensor(t) for t in spec)
        expected = 1 if is_pinpoint(prop) else 2
        if len(spec) != expected:
            raise ValueError(
                f"dim {d} (property {prop!r}) expects {expected} endpoint "
                f"array(s), got {len(spec)}"
            )
        for t in spec:
            if t.ndim != 1 or int(t.shape[0]) != nnz:
                raise ValueError(
                    f"dim {d} endpoint must be 1-D length {nnz}, got shape "
                    f"{tuple(t.shape)}"
                )
        norm_dims.append(spec)

    return ContinuousTensor(tuple(norm_dims), values, property)


def ceinsum(equation: str, *operands: ContinuousTensor) -> ContinuousTensor:
    """Continuous einsum (NOT YET IMPLEMENTED).

    See the module docstring for the intended four-step pipeline. Currently a
    stub so the test suite can define the target contract before the optimized,
    ``torch.compile``-friendly implementation lands.
    """
    raise NotImplementedError(
        "ceinsum is not implemented yet; see continuous_einsum module docstring "
        "for the planned four-step (mapping → output construct → coordinates → "
        "values) pipeline."
    )
