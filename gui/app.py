"""Continuous-einsum visualizer (Plotly Dash).

Run:  python gui/app.py   then open http://127.0.0.1:8050

Each operand gets a panel: pick a property per dimension, draw pieces directly
on the canvas (rectangle for 2-D, line for 1-D — use the modebar draw tools) or
edit the table. The output tensor recomputes live on every change — there is no
Run button. Drawing a shape appends a row to that operand's table (default
value 1); the table is the source of truth and the canvas re-renders from it.
Pieces that would overlap an existing piece are silently ignored.
"""

from __future__ import annotations

from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from ce_viz import (
    OUTPUT_COLOR,
    PALETTE,
    PROP_OPTIONS,
    TABLE_COLUMNS,
    blank_figure,
    parse_math_equation,
    piece_overlaps,
    rows_to_tensor,
    run_einsum,
    table_columns_for_props,
    tensor_figure,
    tensor_to_rows,
    to_numpy_equation,
)

MAX_OPERANDS = 3
DRAW_CONFIG = {
    "modeBarButtonsToAdd": ["drawrect", "drawline", "eraseshape"],
    "displaylogo": False,
}

# Preloaded example: y(i) = A(i) * x(i) interval overlap. A has two [) intervals,
# x has three. Overlaps -> output [20,40) v=10 and [70,80) v=21.
DEFAULT_EQUATION = "y(i) = A(i) * x(i)"
DEFAULT_TABLES = [
    [{"lo0": 10, "hi0": 40, "lo1": 0, "hi1": 0, "value": 2},
     {"lo0": 60, "hi0": 80, "lo1": 0, "hi1": 0, "value": 3}],
    [{"lo0": 20, "hi0": 50, "lo1": 0, "hi1": 0, "value": 5},
     {"lo0": 70, "hi0": 90, "lo1": 0, "hi1": 0, "value": 7},
     {"lo0": 0, "hi0": 5, "lo1": 0, "hi1": 0, "value": 11}],
    [],
]
DEFAULT_PROPS = [("[)", "[)"), ("[)", "[)"), ("[)", "[)")]
# Per-operand arity of the default equation (drives initial table columns).
_DEFAULT_IN_INDICES = parse_math_equation(DEFAULT_EQUATION)[3]

app = Dash(__name__)
# WSGI entry point for production servers (gunicorn wsgi:server).
server = app.server


def _operand_panel(i: int) -> html.Div:
    return html.Div(
        id=f"op{i}-panel",
        style={"border": "1px solid #ccc", "borderRadius": "6px",
               "padding": "8px", "margin": "6px", "flex": "1", "minWidth": "320px"},
        children=[
            html.H4(id=f"op{i}-label"),
            html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center"},
                     children=[
                html.Span("dim0:"),
                dcc.Dropdown(id=f"op{i}-prop0", options=PROP_OPTIONS,
                             value=DEFAULT_PROPS[i][0], clearable=False,
                             style={"width": "90px"}),
                html.Span("dim1:", id=f"op{i}-prop1-lbl"),
                dcc.Dropdown(id=f"op{i}-prop1", options=PROP_OPTIONS,
                             value=DEFAULT_PROPS[i][1], clearable=False,
                             style={"width": "90px"}),
            ]),
            dcc.Graph(id=f"op{i}-graph", config=DRAW_CONFIG),
            dash_table.DataTable(
                id=f"op{i}-table",
                columns=table_columns_for_props(
                    list(DEFAULT_PROPS[i][:len(_DEFAULT_IN_INDICES[i])])
                    if i < len(_DEFAULT_IN_INDICES) else list(DEFAULT_PROPS[i][:1])),
                data=DEFAULT_TABLES[i],
                editable=True, row_deletable=True,
                style_table={"overflowX": "auto"},
            ),
            html.Button("+ row", id=f"op{i}-addrow", n_clicks=0,
                        style={"marginTop": "4px"}),
        ],
    )


app.layout = html.Div(style={"fontFamily": "monospace", "padding": "10px"}, children=[
    html.H2("Continuous Einsum Visualizer"),
    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
        html.Span("equation:"),
        dcc.Input(id="equation", value=DEFAULT_EQUATION, type="text",
                  style={"width": "320px", "fontFamily": "monospace"}),
        html.Span(id="status", style={"marginLeft": "12px", "color": "#444"}),
    ]),
    dcc.Store(id="eq-store"),
    dcc.Store(id="prev-indices", data=_DEFAULT_IN_INDICES),
    html.Div(id="operands-row", style={"display": "flex", "flexWrap": "wrap"},
             children=[_operand_panel(i) for i in range(MAX_OPERANDS)]),
    html.H3("Output", style={"textAlign": "center"}),
    html.Div(style={"display": "flex", "justifyContent": "center", "gap": "24px",
                    "alignItems": "flex-start"},
             children=[
        dcc.Graph(id="out-graph"),
        html.Div(style={"minWidth": "240px"}, children=[
            dash_table.DataTable(
                id="out-table", columns=[], data=[],
                editable=True, row_deletable=True,
                style_table={"overflowX": "auto"},
            ),
        ]),
    ]),
])


# --- changing a tensor's indices wipes only that tensor's pieces ------------
# A piece's dims are tied to its tensor's index spec, so switching A(i,j) to
# A(i) must drop A's stale 2-D boxes — but only A's. Tensors whose indices are
# unchanged (incl. a renamed tensor or an edited output) keep their pieces.
@app.callback(
    [Output(f"op{i}-table", "data", allow_duplicate=True) for i in range(MAX_OPERANDS)]
    + [Output("prev-indices", "data")],
    Input("equation", "value"),
    State("prev-indices", "data"),
    prevent_initial_call=True,
)
def clear_changed_tensor(eq_str, prev):
    try:
        _out_name, _out_idx, _names, new_idx = parse_math_equation(eq_str)
    except Exception:
        # Incomplete/invalid mid-edit: leave every tensor's pieces alone.
        return [no_update] * MAX_OPERANDS + [no_update]
    prev = prev or []
    cleared = []
    for i in range(MAX_OPERANDS):
        old = prev[i] if i < len(prev) else None
        new = new_idx[i] if i < len(new_idx) else None
        cleared.append([] if old != new else no_update)
    return cleared + [new_idx]


# --- equation -> which panels are shown, labels, dim-1 visibility -----------
@app.callback(
    [Output("eq-store", "data")]
    + [Output(f"op{i}-panel", "style") for i in range(MAX_OPERANDS)]
    + [Output(f"op{i}-label", "children") for i in range(MAX_OPERANDS)]
    + [Output(f"op{i}-prop1", "style") for i in range(MAX_OPERANDS)]
    + [Output(f"op{i}-prop1-lbl", "style") for i in range(MAX_OPERANDS)],
    Input("equation", "value"),
)
def configure(eq_str):
    base = {"border": "1px solid #ccc", "borderRadius": "6px", "padding": "8px",
            "margin": "6px", "flex": "1", "minWidth": "320px"}
    hidden = {"display": "none"}
    try:
        out_name, out_indices, in_names, in_indices = parse_math_equation(eq_str)
        assert in_names
    except Exception:
        styles = [hidden] * MAX_OPERANDS
        eq = {"in_indices": [], "out_indices": "", "num": 0, "in_names": [],
              "out_name": "", "eq_str": "",
              "warn": "enter an equation like  y(i) = A(i,j) * x(j)"}
        return (eq, *styles, *[""] * MAX_OPERANDS,
                *[hidden] * MAX_OPERANDS, *[hidden] * MAX_OPERANDS)

    num = len(in_indices)
    panel_styles, labels, prop1_styles, prop1_lbls = [], [], [], []
    warn = ""
    for i in range(MAX_OPERANDS):
        if i < num:
            idx = in_indices[i]
            ndim = len(idx)
            panel_styles.append(base)
            labels.append(f"{in_names[i]}({','.join(idx)})")
            show_d1 = {} if ndim >= 2 else hidden
            prop1_styles.append({"width": "90px"} if ndim >= 2 else hidden)
            prop1_lbls.append(show_d1)
            if ndim > 2:
                warn = f"{in_names[i]} has >2 dims; the GUI only draws 1-D/2-D operands"
        else:
            panel_styles.append(hidden)
            labels.append("")
            prop1_styles.append(hidden)
            prop1_lbls.append(hidden)
    if num > MAX_OPERANDS:
        warn = f"only {MAX_OPERANDS} input tensors supported (got {num})"
    eq = {"in_indices": in_indices, "out_indices": out_indices, "num": num,
          "in_names": in_names, "out_name": out_name,
          "eq_str": to_numpy_equation(out_indices, in_indices), "warn": warn}
    return (eq, *panel_styles, *labels, *prop1_styles, *prop1_lbls)


# --- per-operand: render preview from table + props -------------------------
def _operand_figure(rows, p0, p1, eq, i):
    """Build operand ``i``'s figure straight from its table rows + props."""
    if not eq or i >= eq["num"]:
        return blank_figure("(unused)")
    idx = eq["in_indices"][i]
    name = eq["in_names"][i]
    title = f"{name}({','.join(idx)})"
    ndim = len(idx)
    if ndim > 2:
        return blank_figure(f"{title}  (3-D not drawable)")
    props = [p0, p1][:ndim]
    ct = rows_to_tensor(rows or [], props)
    return tensor_figure(ct, idx, title, PALETTE[i])


def _register_operand_callbacks(i: int) -> None:
    @app.callback(
        Output(f"op{i}-graph", "figure"),
        Input(f"op{i}-table", "data"),
        Input(f"op{i}-prop0", "value"),
        Input(f"op{i}-prop1", "value"),
        Input("eq-store", "data"),
    )
    def render(rows, p0, p1, eq, _i=i):
        return _operand_figure(rows, p0, p1, eq, _i)

    @app.callback(
        Output(f"op{i}-table", "columns"),
        Input(f"op{i}-prop0", "value"),
        Input(f"op{i}-prop1", "value"),
        Input("eq-store", "data"),
    )
    def columns(p0, p1, eq, _i=i):
        if not eq or _i >= eq["num"]:
            raise PreventUpdate
        ndim = min(len(eq["in_indices"][_i]), 2)
        return table_columns_for_props([p0, p1][:ndim])

    @app.callback(
        Output(f"op{i}-table", "data", allow_duplicate=True),
        Output(f"op{i}-graph", "figure", allow_duplicate=True),
        Input(f"op{i}-graph", "relayoutData"),
        State(f"op{i}-table", "data"),
        State(f"op{i}-prop0", "value"),
        State(f"op{i}-prop1", "value"),
        State("eq-store", "data"),
        prevent_initial_call=True,
    )
    def on_draw(relayout, rows, p0, p1, eq, _i=i):
        if not relayout or not relayout.get("shapes"):
            raise PreventUpdate
        ndim = len(eq["in_indices"][_i]) if eq and _i < eq["num"] else 1
        props = [p0, p1][:ndim]
        rows = list(rows or [])
        added = False
        for sh in relayout["shapes"]:
            x0, x1 = sorted([sh.get("x0", 0.0), sh.get("x1", 0.0)])
            y0, y1 = sorted([sh.get("y0", 0.0), sh.get("y1", 0.0)])
            row = {c: None for c in TABLE_COLUMNS}
            row["lo0"], row["hi0"] = round(x0, 3), round(x1, 3)
            if ndim >= 2:
                row["lo1"], row["hi1"] = round(y0, 3), round(y1, 3)
            row["value"] = 1
            # Silently drop a drawn piece that overlaps any existing piece.
            if any(piece_overlaps(row, other, props) for other in rows):
                continue
            rows.append(row)
            added = True
        # Always rebuild the figure from the table's truth so a rejected
        # (overlapping) shape — which lives only on the client canvas — is
        # wiped. If nothing was added, leave the table untouched.
        fig = _operand_figure(rows, p0, p1, eq, _i)
        return (rows if added else no_update), fig

    @app.callback(
        Output(f"op{i}-table", "data", allow_duplicate=True),
        Input(f"op{i}-addrow", "n_clicks"),
        State(f"op{i}-table", "data"),
        prevent_initial_call=True,
    )
    def add_row(_n, rows, _i=i):
        rows = list(rows or [])
        rows.append({c: (1 if c == "value" else 0) for c in TABLE_COLUMNS})
        return rows


for _i in range(MAX_OPERANDS):
    _register_operand_callbacks(_i)


def _check_property_consistency(in_indices, operand_props):
    """Raise if an index is carried by interval dims with mismatched properties.

    Pieces sharing an index are intersected along it, so their boundary kinds
    must agree (e.g. intersecting a ``[)`` dim with a ``[]`` dim is ambiguous).
    Pinpoints may coexist with any interval (point-in-interval is well defined).
    """
    index_props: dict[str, set[str]] = {}
    for i, idx in enumerate(in_indices):
        for d, letter in enumerate(idx):
            if d < len(operand_props[i]):
                index_props.setdefault(letter, set()).add(operand_props[i][d])
    for letter, props in index_props.items():
        intervals = {p for p in props if p != "P"}
        if len(intervals) > 1:
            raise ValueError(
                f"property mismatch on index '{letter}': "
                f"{sorted(intervals)} cannot be intersected"
            )


# --- Reactive: recompute the einsum and render the output on every change ----
@app.callback(
    Output("out-graph", "figure"),
    Output("out-table", "data"),
    Output("out-table", "columns"),
    Output("status", "children", allow_duplicate=True),
    Input("eq-store", "data"),
    [Input(f"op{i}-table", "data") for i in range(MAX_OPERANDS)],
    [Input(f"op{i}-prop0", "value") for i in range(MAX_OPERANDS)],
    [Input(f"op{i}-prop1", "value") for i in range(MAX_OPERANDS)],
    prevent_initial_call=True,
)
def run(*args):
    # Dash flattens the list-comprehension Inputs into individual positional
    # args: (eq, t0..t2, p0_0..p0_2, p1_0..p1_2).
    eq = args[0]
    mid = args[1:]
    M = MAX_OPERANDS
    tables = list(mid[0:M])
    prop0s = list(mid[M:2 * M])
    prop1s = list(mid[2 * M:3 * M])
    if not eq or eq["num"] == 0:
        return (blank_figure("Output"), [], [],
                (eq or {}).get("warn", "invalid equation"))
    warn = eq.get("warn", "")
    try:
        num = eq["num"]
        operand_rows, operand_props = [], []
        for i in range(num):
            idx = eq["in_indices"][i]
            ndim = len(idx)
            props = [prop0s[i], prop1s[i]][:ndim]
            operand_rows.append(tables[i] or [])
            operand_props.append(props)
        _check_property_consistency(eq["in_indices"], operand_props)
        out, _in, out_idx = run_einsum(
            eq["eq_str"], operand_rows, operand_props)
        out_name = eq.get("out_name") or "y"
        out_title = (f"{out_name}({','.join(out_idx)})" if out_idx
                     else f"{out_name} (scalar)")
        fig = tensor_figure(out, out_idx, out_title, OUTPUT_COLOR)
        out_cols = table_columns_for_props(list(out.property))
        out_rows = tensor_to_rows(out)
        status = f"OK — {out.nnz} output piece(s), property {list(out.property)}"
        return fig, out_rows, out_cols, (f"{warn}  |  {status}" if warn else status)
    except Exception as e:  # surface the error in the UI instead of crashing
        return blank_figure("Output"), [], [], f"Error: {e}"


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
