"""Step 3 — merge: reduce the candidates into the output by coordinates.

The discrete half (:func:`merge_discrete`) sums the candidates that share the
*same* output coordinates: a ``torch.unique`` over the product step's grouping
key assigns each join tuple an output piece id, and an ``index_add_`` performs
the accumulating scatter. This produces a valid COO output whose pieces may
still *partially* overlap along interval dimensions when the einsum has a
reduction; :func:`coalesce` (the continuous half) rewrites those into disjoint
pieces.
"""

from __future__ import annotations

import torch

from ceinsum_product import Product
from ctensor import ContinuousTensor


def merge_discrete(product: Product, out_property: tuple[str, ...]) -> ContinuousTensor:
    """Sum candidates with identical output coordinates into one piece each.

    Tuples are grouped by the *provider piece columns* rather than the float
    coordinates: tuples agreeing on every output-providing operand necessarily
    share their output coordinates, and integer keys are robust.
    """
    P = int(product.values.shape[0])
    device = product.values.device

    if product.key_cols:
        key = torch.stack(product.key_cols, dim=1)  # (P, |providers|)
        _uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        num_out = int(_uniq.shape[0])
    else:
        # Scalar output (full contraction): all tuples collapse to one piece.
        inv = torch.zeros(P, dtype=torch.long, device=device)
        num_out = 1 if P > 0 else 0

    # One representative tuple per output piece supplies the coordinates
    # (every tuple in a group shares them).
    rep = torch.empty(num_out, dtype=torch.long, device=device)
    rep[inv] = torch.arange(P, dtype=torch.long, device=device)

    values = torch.zeros(num_out, dtype=product.values.dtype, device=device)
    values.index_add_(0, inv, product.values)

    dims = tuple(tuple(col[rep] for col in spec) for spec in product.coords)
    return ContinuousTensor(dims, values, out_property)
