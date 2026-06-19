"""Pure (Dash-free) layer for the continuous-einsum visualizer.

Converts editable piece tables <-> ContinuousTensor, runs ceinsum, and turns a
ContinuousTensor into a Plotly figure. Kept free of Dash so it can be unit
tested headlessly; ``app.py`` wires these into the GUI.

Geometry per piece (decided by the dim's property code):
  * interval x interval -> rectangle
  * interval x pinpoint -> a line (segment at the pinpoint coordinate)
  * pinpoint x pinpoint -> a point
  * single interval dim -> a segment on a baseline; single pinpoint -> a dot
Open boundaries are drawn with hollow endpoint markers, closed with filled.
"""

from __future__ import annotations

import os
import re
import sys

import plotly.graph_objects as go
import torch

# Make src/ importable whether run from repo root or gui/.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from continuous_einsum import ceinsum  # noqa: E402
from ctensor import (  # noqa: E402
    ContinuousTensor,
    continuous_tensor,
    is_pinpoint,
    left_closed,
    right_closed,
)
from ceinsum_equation import parse_equation  # noqa: E402

DT = torch.float64
PROP_OPTIONS = ["[)", "(]", "[]", "()", "P"]
PALETTE = ["#27ae60", "#2980b9", "#000000"]   # operand colors: green, blue, black
OUTPUT_COLOR = "#c0392b"                       # output pieces: always red

# Every plot (inputs and output) uses the same box so shapes match.
FIG_W = 440
FIG_H_2D = 440
FIG_H_1D = 220

# Table columns are fixed (lo/hi per up-to-2 dims + value); 1-D operands ignore
# the dim-1 columns, pinpoint dims use only the lo column as the coordinate.
TABLE_COLUMNS = ["lo0", "hi0", "lo1", "hi1", "value"]


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    if v is None or v == "":
        return default
    return float(v)


_TERM_RE = re.compile(r"([A-Za-z]\w*)\(([^()]*)\)$")


def _parse_term(term: str) -> tuple[str, str]:
    """Parse one ``name(i,j)`` factor into ``(name, "ij")``."""
    m = _TERM_RE.match(term)
    if m:
        idx = "".join(c for c in m.group(2) if c.isalnum())
        return m.group(1), idx
    if re.fullmatch(r"[A-Za-z]\w*", term):
        return term, ""                       # bare scalar tensor, no indices
    raise ValueError(f"cannot parse term {term!r}")


def parse_math_equation(text: str) -> tuple[str, str, list[str], list[str]]:
    """Parse ``y(i) = A(i,j) * x(j)`` into names + concatenated index strings.

    Returns ``(out_name, out_indices, in_names, in_indices)`` where the index
    strings use the numpy-style concatenation (``"ij"``) the backend expects.
    Raises ``ValueError`` on malformed input.
    """
    s = (text or "").replace(" ", "")
    if "=" not in s:
        raise ValueError("equation needs '='  (e.g. y(i) = A(i,j) * x(j))")
    lhs, rhs = s.split("=", 1)
    out_name, out_idx = _parse_term(lhs)
    if not rhs:
        raise ValueError("right-hand side is empty")
    in_names, in_indices = [], []
    for term in rhs.split("*"):
        if not term:
            raise ValueError("empty factor in product")
        name, idx = _parse_term(term)
        in_names.append(name)
        in_indices.append(idx)
    return out_name, out_idx, in_names, in_indices


def to_numpy_equation(out_indices: str, in_indices: list[str]) -> str:
    """``("i", ["ij", "j"]) -> "ij,j->i"`` for the numpy-style backend."""
    return ",".join(in_indices) + "->" + out_indices


def _interval_overlap(
    lo_a: float, hi_a: float, lo_b: float, hi_b: float,
    lc: bool, rc: bool,
) -> bool:
    """Do two same-property intervals share at least one point?

    ``lc``/``rc`` are the (shared) closed-ness of the left/right boundary. Two
    intervals that merely touch at a single coordinate overlap only when both
    adjoining boundaries are closed (e.g. ``[a,b]`` and ``[b,c]`` share ``b``,
    but ``[a,b)`` and ``[b,c)`` do not).
    """
    if lo_a > hi_b or lo_b > hi_a:
        return False
    if lo_a == hi_b or lo_b == hi_a:        # touching at a single point
        return lc and rc
    return True


def piece_overlaps(a: dict, b: dict, props: list[str]) -> bool:
    """True if table rows ``a`` and ``b`` describe overlapping regions.

    A region is the product of its per-dim intervals/points, so the regions
    intersect iff they intersect along *every* dimension. Pinpoint dims overlap
    only when their coordinates are equal.
    """
    for d, prop in enumerate(props):
        if is_pinpoint(prop):
            if _f(a, f"lo{d}") != _f(b, f"lo{d}"):
                return False
        else:
            lo_a, hi_a = sorted((_f(a, f"lo{d}"), _f(a, f"hi{d}")))
            lo_b, hi_b = sorted((_f(b, f"lo{d}"), _f(b, f"hi{d}")))
            if not _interval_overlap(lo_a, hi_a, lo_b, hi_b,
                                     left_closed(prop), right_closed(prop)):
                return False
    return True


def table_columns_for_props(props: list[str]) -> list[dict]:
    """Dash-table columns matching the per-dim properties.

    A pinpoint dim ``d`` shows a single ``coord{d}`` column (stored in ``lo{d}``);
    an interval dim shows ``lo{d}``/``hi{d}``. The trailing ``value`` is always
    present. Dimension index is the column-name suffix, so a 1-D operand on dim 1
    would read ``lo1``/``hi1``.
    """
    cols: list[dict] = []
    for d, p in enumerate(props):
        if is_pinpoint(p):
            cols.append({"name": f"coord{d}", "id": f"lo{d}", "type": "numeric"})
        else:
            cols.append({"name": f"lo{d}", "id": f"lo{d}", "type": "numeric"})
            cols.append({"name": f"hi{d}", "id": f"hi{d}", "type": "numeric"})
    cols.append({"name": "value", "id": "value", "type": "numeric"})
    return cols


def tensor_to_rows(ct: ContinuousTensor) -> list[dict]:
    """Flatten a ContinuousTensor's pieces into editable table rows."""
    rows: list[dict] = []
    for p in range(ct.nnz):
        row: dict = {}
        for d in range(ct.ndim):
            spec = ct.dims[d]
            row[f"lo{d}"] = round(float(spec[0][p]), 3)
            if not is_pinpoint(ct.property[d]):
                row[f"hi{d}"] = round(float(spec[1][p]), 3)
        row["value"] = round(float(ct.values[p]), 6)
        rows.append(row)
    return rows


def rows_to_tensor(rows: list[dict], props: list[str]) -> ContinuousTensor:
    """Build a ContinuousTensor from table rows + per-dim property codes."""
    ndim = len(props)
    n = len(rows)
    dims: list[tuple[torch.Tensor, ...]] = []
    if n == 0:
        for d in range(ndim):
            z = torch.zeros(0, dtype=DT)
            dims.append((z,) if is_pinpoint(props[d]) else (z, z.clone()))
        return continuous_tensor(dims, torch.zeros(0, dtype=DT), props, dtype=DT)

    values = torch.tensor([_f(r, "value", 1.0) for r in rows], dtype=DT)
    for d in range(ndim):
        lo = torch.tensor([_f(r, f"lo{d}") for r in rows], dtype=DT)
        if is_pinpoint(props[d]):
            dims.append((lo,))                      # coord = lo column
        else:
            hi = torch.tensor([_f(r, f"hi{d}") for r in rows], dtype=DT)
            dims.append((lo, hi))
    return continuous_tensor(dims, values, props, dtype=DT)


def run_einsum(
    equation: str,
    operand_rows: list[list[dict]],
    operand_props: list[list[str]],
) -> tuple[ContinuousTensor, list[str], str]:
    """Run ceinsum on the operands described by the tables. Returns
    ``(output_tensor, in_indices, out_indices)``."""
    in_indices, out_indices, _ = parse_equation(equation, len(operand_rows))
    operands = [rows_to_tensor(operand_rows[i], operand_props[i])
                for i in range(len(operand_rows))]
    out = ceinsum(equation, *operands)
    return out, in_indices, out_indices


# ---------------------------------------------------------------------------
# Figure building.
# ---------------------------------------------------------------------------


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _dim_span(ct: ContinuousTensor, p: int, d: int) -> tuple[float, float, bool]:
    """(lo, hi, is_point) for piece ``p`` along dim ``d``."""
    spec = ct.dims[d]
    if is_pinpoint(ct.property[d]):
        c = float(spec[0][p])
        return c, c, True
    return float(spec[0][p]), float(spec[1][p]), False


def _endpoint_symbols(prop: str) -> tuple[str, str]:
    return ("circle" if left_closed(prop) else "circle-open",
            "circle" if right_closed(prop) else "circle-open")


def tensor_figure(
    ct: ContinuousTensor,
    indices: str,
    title: str,
    color: str,
) -> go.Figure:
    """Render a 1-D or 2-D ContinuousTensor as labelled geometric pieces."""
    fig = go.Figure()
    ndim = ct.ndim
    n = ct.nnz

    # Scalar / unsupported dim count: just show the values as text.
    if ndim == 0 or ndim > 2:
        txt = "  ".join(f"{float(ct.values[p]):g}" for p in range(n)) or "(empty)"
        fig.add_annotation(text=f"{ndim}-D result: {txt}", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title=title, width=FIG_W, height=FIG_H_1D,
                          margin=dict(l=40, r=20, t=40, b=40))
        return fig

    for p in range(n):
        val = float(ct.values[p])
        label = f"{val:g}"
        lo0, hi0, pt0 = _dim_span(ct, p, 0)
        if ndim == 1:
            cx = lo0
            if pt0:
                fig.add_trace(go.Scatter(
                    x=[lo0], y=[0], mode="markers",
                    marker=dict(size=12, color=color), showlegend=False,
                    hovertemplate=f"coord={lo0:g}<br>val={label}<extra></extra>"))
            else:
                ls, rs = _endpoint_symbols(ct.property[0])
                fig.add_trace(go.Scatter(
                    x=[lo0, hi0], y=[0, 0], mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=12, color=color, symbol=[ls, rs]),
                    showlegend=False,
                    hovertemplate=(f"[{lo0:g}, {hi0:g}] {ct.property[0]}"
                                   f"<br>val={label}<extra></extra>")))
                cx = (lo0 + hi0) / 2
            fig.add_trace(go.Scatter(x=[cx], y=[0.12], mode="text", text=[label],
                          textfont=dict(size=13, color="#222"),
                          showlegend=False, hoverinfo="skip"))
            continue

        # ndim == 2
        lo1, hi1, pt1 = _dim_span(ct, p, 1)
        if not pt0 and not pt1:                         # rectangle
            xs = [lo0, hi0, hi0, lo0, lo0]
            ys = [lo1, lo1, hi1, hi1, lo1]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                line=dict(color=color, width=2), fillcolor=_rgba(color, 0.25),
                showlegend=False, hoveron="fills",
                hovertemplate=(f"{indices[0]}:[{lo0:g},{hi0:g}] "
                               f"{indices[1]}:[{lo1:g},{hi1:g}]"
                               f"<br>val={label}<extra></extra>")))
            cx, cy = (lo0 + hi0) / 2, (lo1 + hi1) / 2
        elif pt0 and not pt1:                           # vertical line
            fig.add_trace(go.Scatter(x=[lo0, lo0], y=[lo1, hi1], mode="lines",
                          line=dict(color=color, width=4), showlegend=False))
            cx, cy = lo0, (lo1 + hi1) / 2
        elif not pt0 and pt1:                           # horizontal line
            fig.add_trace(go.Scatter(x=[lo0, hi0], y=[lo1, lo1], mode="lines",
                          line=dict(color=color, width=4), showlegend=False))
            cx, cy = (lo0 + hi0) / 2, lo1
        else:                                           # point
            fig.add_trace(go.Scatter(x=[lo0], y=[lo1], mode="markers",
                          marker=dict(size=13, color=color), showlegend=False))
            cx, cy = lo0, lo1
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="text", text=[label],
                      textfont=dict(size=13, color="#222"),
                      showlegend=False, hoverinfo="skip"))

    xlab = indices[0] if len(indices) >= 1 else "x"
    ylab = indices[1] if ndim == 2 else ""
    fig.update_layout(
        title=title, width=FIG_W, height=FIG_H_2D if ndim == 2 else FIG_H_1D,
        margin=dict(l=45, r=20, t=40, b=40),
        xaxis_title=xlab, yaxis_title=ylab, shapes=[],
        dragmode="drawrect" if ndim == 2 else "drawline",
        newshape=dict(line=dict(color=color), fillcolor=_rgba(color, 0.15)),
        plot_bgcolor="#fafafa",
    )
    # Fixed domain: every tensor (inputs and output) lives in [0, 100) per axis.
    fig.update_xaxes(range=[0, 100], autorange=False)
    if ndim == 2:
        fig.update_yaxes(range=[0, 100], autorange=False)
    else:
        fig.update_yaxes(range=[-0.5, 0.5], autorange=False,
                         showticklabels=False, zeroline=True)
    return fig


def blank_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, width=FIG_W, height=FIG_H_1D,
                      margin=dict(l=45, r=20, t=40, b=40), plot_bgcolor="#fafafa")
    return fig
