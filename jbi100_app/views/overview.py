# jb100_app/views/overview.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import html, dcc, callback, Input, Output, State

from ..data import get_data
from .. import config

_bundle = get_data()
_services_df = _bundle.services_df.copy()

# ---------- styling constants ----------
SEVERE_TH = 0.60
HIGH_TH = 0.30

EVENT_STYLE = {
    "flu": {"color": "rgba(213, 94, 0, 0.35)", "line_dash": "dot", "label": "Flu"},
    "strike": {"color": "rgba(0, 114, 178, 0.35)", "line_dash": "dash", "label": "Strike"},
    "donation": {"color": "rgba(0, 158, 115, 0.35)", "line_dash": "dashdot", "label": "Donation"},
    "none": {"color": "rgba(0,0,0,0.0)", "line_dash": "solid", "label": "None"},
}

SELECTION_LINE = {"color": "rgba(20,20,20,0.85)", "dash": "dash", "width": 2.5}
EVENT_BAND_ALPHA = 0.10  # 0.08–0.12

# --- brushing style ---
BRUSH_FILL = "rgba(20,20,20,0.08)"
BRUSH_LINE = {"color": "rgba(20,20,20,0.55)", "dash": "dot", "width": 1.5}


def layout():
    return html.Div(
        children=[
            html.Div(
                className="graph_card",
                children=[
                    html.H6("Overview · Bed Pressure (Shortage Rate)"),
                    html.Div(
                        id="overview-kpi-row",
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                    ),
                    html.Div(
                        id="overview-hover-preview",
                        style={"marginTop": "8px", "color": "#555"},
                        children="Tip: Hover a cell to preview; click a cell to lock selection.",
                    ),
                    dcc.Graph(
                        id="overview-heatmap",
                        config={"displayModeBar": False},
                        style={"height": "520px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        className="graph_card",
                        style={"flex": "1", "minWidth": "360px"},
                        children=[
                            html.H6("Weekly trend (mean shortage rate)"),
                            dcc.Graph(
                                id="overview-trend",
                                figure=go.Figure(),
                                # 开启少量 modebar 按钮，便于刷取/清除
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d", "autoScale2d"],
                                },
                                style={"height": "320px"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="graph_card",
                        style={"flex": "1", "minWidth": "360px"},
                        children=[
                            html.H6("Weekly refused patients (sum)"),
                            dcc.Graph(
                                id="overview-refused",
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d", "autoScale2d"],
                                },
                                style={"height": "320px"},
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


# ---------- helpers ----------
def _filter_services(df: pd.DataFrame, week_range, services, events) -> pd.DataFrame:
    out = df.copy()
    if week_range and len(week_range) == 2:
        w0, w1 = week_range
        out = out[(out["week"] >= w0) & (out["week"] <= w1)]
    if services:
        out = out[out["service"].isin(services)]
    if events:
        out = out[out["event"].isin(events)]
    return out


def _kpi_card(title: str, value: str) -> html.Div:
    return html.Div(
        style={
            "background": "white",
            "border": "1px solid #e5e5e5",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "minWidth": "220px",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#666"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "600"}),
        ],
    )


def _risk_label(z: float) -> str:
    if np.isnan(z):
        return ""
    if z >= SEVERE_TH:
        return "Severe"
    if z >= HIGH_TH:
        return "High"
    return "Normal"


def _aggregate_service_week(df: pd.DataFrame) -> pd.DataFrame:
    """Make service-week unique (fixes inconsistent hover/preview when multiple rows exist)."""
    if df.empty:
        return df

    num_sum = ["patients_refused", "patients_request", "patients_admitted"]
    num_mean = ["shortage_rate", "utilization", "bed_pressure", "available_beds", "capacity_delta"]

    present_sum = [c for c in num_sum if c in df.columns]
    present_mean = [c for c in num_mean if c in df.columns]

    agg = {c: "sum" for c in present_sum}
    agg.update({c: "mean" for c in present_mean})
    if "event" in df.columns:
        agg["event"] = lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]

    out = df.groupby(["service", "week"], as_index=False).agg(agg)
    return out


def _build_custom_grid(df_sw: pd.DataFrame, services_idx, weeks_idx) -> np.ndarray:
    fields = ["patients_refused", "utilization", "bed_pressure", "available_beds", "patients_request", "event"]
    fields = [f for f in fields if f in df_sw.columns]

    lookup = df_sw.set_index(["service", "week"])[fields].to_dict("index") if len(df_sw) else {}

    grid = []
    for s in services_idx:
        row = []
        for w in weeks_idx:
            item = lookup.get((s, w))
            if not item:
                row.append([None, None, None, None, None, None, None])
                continue

            refused = item.get("patients_refused") if "patients_refused" in item else None
            req = item.get("patients_request") if "patients_request" in item else None
            refused_pct = None
            try:
                if req not in (None, 0) and refused is not None:
                    refused_pct = float(refused) / float(req)
            except Exception:
                refused_pct = None

            row.append(
                [
                    refused,
                    item.get("utilization"),
                    item.get("bed_pressure"),
                    item.get("available_beds"),
                    req,
                    item.get("event"),
                    refused_pct,
                ]
            )
        grid.append(row)
    return np.array(grid, dtype=object)


def _severe_outline(pivot: pd.DataFrame) -> go.Scatter:
    if pivot.size == 0:
        return go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip", showlegend=False)

    sev_x, sev_y = [], []
    for i, s in enumerate(pivot.index):
        for j, w in enumerate(pivot.columns):
            z = pivot.values[i, j]
            try:
                if z is not None and not np.isnan(z) and float(z) >= SEVERE_TH:
                    sev_x.append(w)
                    sev_y.append(s)
            except Exception:
                pass

    return go.Scatter(
        x=sev_x,
        y=sev_y,
        mode="markers",
        marker=dict(symbol="square-open", size=18, color="rgba(30,30,30,0.35)", line=dict(width=1)),
        hoverinfo="skip",
        showlegend=False,
    )


def _event_strip(df_sw: pd.DataFrame, weeks_idx) -> go.Heatmap:
    keys = list(EVENT_STYLE.keys()) or ["none"]
    if df_sw.empty or "event" not in df_sw.columns:
        week_event = {}
    else:
        week_event = (
            df_sw.groupby("week")["event"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else (s.iloc[0] if len(s) else "none"))
            .to_dict()
        )

    z, text = [], []
    for w in weeks_idx:
        ev = week_event.get(w, "none")
        if ev not in EVENT_STYLE:
            ev = "none"
        text.append(EVENT_STYLE[ev]["label"])
        z.append(keys.index(ev))

    if len(keys) == 1:
        cs = [(0.0, EVENT_STYLE[keys[0]]["color"]), (1.0, EVENT_STYLE[keys[0]]["color"])]
    else:
        cs = [(i / (len(keys) - 1), EVENT_STYLE[k]["color"]) for i, k in enumerate(keys)]

    return go.Heatmap(
        z=[z] if z else [[0]],
        x=weeks_idx if weeks_idx else [1],
        y=[""],
        colorscale=cs,
        showscale=False,
        hovertemplate="Event: <b>%{text}</b><extra></extra>",
        text=[text] if text else [[""]],
    )


def _add_event_bands(fig_trend, fig_ref, df_sw: pd.DataFrame) -> None:
    if df_sw.empty or "event" not in df_sw.columns:
        return

    ev_df = df_sw[df_sw["event"].notna() & (df_sw["event"] != "none")]
    if ev_df.empty:
        return

    week_event = (
        ev_df.groupby("week")["event"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .to_dict()
    )
    weeks_sorted = sorted(int(w) for w in week_event.keys() if pd.notna(w))
    if not weeks_sorted:
        return

    ranges = []
    w0 = w1 = weeks_sorted[0]
    cur_ev = week_event[w0]
    for w in weeks_sorted[1:]:
        ev = week_event[w]
        if ev == cur_ev and w == w1 + 1:
            w1 = w
        else:
            ranges.append((w0, w1, cur_ev))
            w0 = w1 = w
            cur_ev = ev
    ranges.append((w0, w1, cur_ev))

    for a, b, ev in ranges:
        if ev not in EVENT_STYLE:
            continue
        base = EVENT_STYLE[ev]["color"]
        fill = base.replace("0.35", f"{EVENT_BAND_ALPHA:.2f}") if "0.35" in base else base
        x0, x1 = a - 0.5, b + 0.5
        for fig in (fig_trend, fig_ref):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, opacity=1.0, line_width=0, layer="below")

    fig_trend.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text="Background bands indicate event periods.",
        showarrow=False,
        font=dict(size=10, color="rgba(90,90,90,0.85)"),
        align="left",
    )


def _add_brush_band(fig: go.Figure, w0: int, w1: int) -> None:
    if w0 is None or w1 is None:
        return
    a, b = (w0, w1) if w0 <= w1 else (w1, w0)
    fig.add_vrect(
        x0=a - 0.5,
        x1=b + 0.5,
        fillcolor=BRUSH_FILL,
        opacity=1.0,
        line_width=BRUSH_LINE["width"],
        line_dash=BRUSH_LINE["dash"],
        line_color=BRUSH_LINE["color"],
        layer="above",
    )


def _effective_week_range(week_range, store) -> tuple | None:
    """Intersect slider week-range with brushed range (if any)."""
    base = None
    if week_range and len(week_range) == 2:
        base = (int(week_range[0]), int(week_range[1]))

    brush = ((store or {}).get("brush") or {})
    bw0 = brush.get("w0")
    bw1 = brush.get("w1")
    if bw0 is None or bw1 is None:
        return base

    bw0, bw1 = int(bw0), int(bw1)
    if bw0 > bw1:
        bw0, bw1 = bw1, bw0

    if base is None:
        return (bw0, bw1)

    w0 = max(base[0], bw0)
    w1 = min(base[1], bw1)
    if w0 > w1:
        # empty intersection: fall back to brushed range (better UX than blank)
        return (bw0, bw1)
    return (w0, w1)


def _parse_brush_from_relayout(relayout: dict | None) -> tuple[int | None, int | None, bool]:
    """
    Returns (w0, w1, cleared)
    cleared=True when user resets axes/autorange.
    """
    if not relayout or not isinstance(relayout, dict):
        return None, None, False

    # reset/autoscale signals (common patterns)
    if relayout.get("xaxis.autorange") is True:
        return None, None, True
    if relayout.get("autosize") is True and "xaxis.range[0]" not in relayout and "xaxis.range" not in relayout:
        # sometimes emitted during reset; treat as clear if no explicit range
        return None, None, True

    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        try:
            w0 = int(float(relayout["xaxis.range[0]"]))
            w1 = int(float(relayout["xaxis.range[1]"]))
            return w0, w1, False
        except Exception:
            return None, None, False

    if "xaxis.range" in relayout and isinstance(relayout["xaxis.range"], (list, tuple)) and len(relayout["xaxis.range"]) == 2:
        try:
            w0 = int(float(relayout["xaxis.range"][0]))
            w1 = int(float(relayout["xaxis.range"][1]))
            return w0, w1, False
        except Exception:
            return None, None, False

    return None, None, False


# ---------- callbacks ----------
@callback(
    Output("overview-heatmap", "figure"),
    Output("overview-trend", "figure"),
    Output("overview-refused", "figure"),
    Output("overview-kpi-row", "children"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("event-filter", "value"),
    # 关键：必须是 Input 才能在 brush 更新后自动重算
    Input("global-store", "data"),
)
def overview_update(week_range, services, events, store):
    eff_range = _effective_week_range(week_range, store)

    df = _filter_services(_services_df, eff_range, services, events)
    df_sw = _aggregate_service_week(df)

    pivot = (
        df_sw.pivot_table(index="service", columns="week", values="shortage_rate", aggfunc="mean")
        .reindex(index=config.SERVICES_ORDER)
        .sort_index(axis=1)
    )
    services_idx = list(pivot.index)
    weeks_idx = list(pivot.columns)

    custom_grid = _build_custom_grid(df_sw, services_idx, weeks_idx)

    hm = go.Heatmap(
        z=pivot.values,
        x=weeks_idx,
        y=services_idx,
        colorscale=getattr(config, "SHORTAGE_COLOR_SCALE", None),
        zmin=0,
        zmax=float(np.nanmax(pivot.values)) if pivot.size else 1,
        customdata=custom_grid,
        colorbar=dict(title="shortage_rate"),
        hovertemplate=(
            "<b>%{y}</b> · week %{x}<br>"
            "shortage_rate=<b>%{z:.2f}</b> (<i>%{text}</i>)<br>"
            "refused=%{customdata[0]} / request=%{customdata[4]} (%{customdata[6]:.0%})<br>"
            "utilization=%{customdata[1]}<br>"
            "bed_pressure=%{customdata[2]}<br>"
            "beds=%{customdata[3]}<br>"
            "event=%{customdata[5]}<extra></extra>"
        ),
    )
    if pivot.size:
        hm.text = np.vectorize(_risk_label)(pivot.values.astype(float))
    else:
        hm.text = []

    fig_hm = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.07, 0.93],
        vertical_spacing=0.02,
    )
    fig_hm.add_trace(_event_strip(df_sw, weeks_idx), row=1, col=1)
    fig_hm.add_trace(hm, row=2, col=1)
    fig_hm.add_trace(_severe_outline(pivot), row=2, col=1)

    fig_hm.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig_hm.update_xaxes(title_text="Week", row=2, col=1, side="bottom")
    fig_hm.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
    fig_hm.update_yaxes(title_text="Service", row=2, col=1)
    fig_hm.add_annotation(
        x=0,
        y=1.12,
        xref="paper",
        yref="paper",
        text="Event strip (context)",
        showarrow=False,
        font=dict(size=11, color="rgba(80,80,80,0.8)"),
        align="left",
    )

    # --- trend ---
    trend = df_sw.groupby("week", as_index=False)["shortage_rate"].mean().sort_values("week")
    fig_trend = px.line(
        trend,
        x="week",
        y="shortage_rate",
        labels={"week": "Week", "shortage_rate": "Mean shortage rate"},
        template="plotly",
    )
    fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10), dragmode="zoom")
    if len(df_sw):
        fig_trend.update_yaxes(range=[0, max(0.01, float(df_sw["shortage_rate"].max()))])

    # --- refused ---
    refused = (
        df_sw.groupby("week", as_index=False)["patients_refused"].sum().sort_values("week")
        if "patients_refused" in df_sw.columns
        else pd.DataFrame({"week": [], "patients_refused": []})
    )
    fig_ref = px.bar(
        refused,
        x="week",
        y="patients_refused",
        labels={"week": "Week", "patients_refused": "Refused (sum)"},
    )
    fig_ref.update_layout(margin=dict(l=10, r=10, t=10, b=10), dragmode="zoom")

    # pinned selection line (existing behavior)
    sel_week = ((store or {}).get("selection") or {}).get("week")
    if sel_week is not None:
        for fig in (fig_trend, fig_ref):
            fig.add_vline(
                x=sel_week,
                line_width=SELECTION_LINE["width"],
                line_dash=SELECTION_LINE["dash"],
                line_color=SELECTION_LINE["color"],
            )

    # brushing band overlay (NEW)
    brush = ((store or {}).get("brush") or {})
    bw0, bw1 = brush.get("w0"), brush.get("w1")
    if bw0 is not None and bw1 is not None:
        bw0, bw1 = int(bw0), int(bw1)
        _add_brush_band(fig_trend, bw0, bw1)
        _add_brush_band(fig_ref, bw0, bw1)
        _add_brush_band(fig_hm, bw0, bw1)
        fig_trend.add_annotation(
            x=0.01,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"Brushed weeks: {min(bw0,bw1)}–{max(bw0,bw1)} (double-click / reset axes to clear)",
            showarrow=False,
            font=dict(size=10, color="rgba(90,90,90,0.85)"),
            align="left",
        )

    _add_event_bands(fig_trend, fig_ref, df_sw)

    # KPIs
    if len(df_sw) == 0:
        kpis = [
            _kpi_card("Max shortage_rate", "—"),
            _kpi_card("Total refused", "—"),
            _kpi_card("Worst service-week", "—"),
        ]
        return fig_hm, fig_trend, fig_ref, kpis

    max_row = df_sw.loc[df_sw["shortage_rate"].idxmax()]
    total_refused = int(df_sw["patients_refused"].sum()) if "patients_refused" in df_sw.columns else 0
    kpis = [
        _kpi_card("Max shortage_rate (risk)", f"{float(max_row['shortage_rate']):.2f}"),
        _kpi_card("Total refused (impact)", f"{total_refused}"),
        _kpi_card("Worst service-week", f"{max_row['service']} · week {int(max_row['week'])}"),
    ]
    return fig_hm, fig_trend, fig_ref, kpis


# --- brushing controller (time-range brush via zoom/drag on either trend/refused) ---
@callback(
    Output("global-store", "data", allow_duplicate=True),
    Input("overview-trend", "relayoutData"),
    Input("overview-refused", "relayoutData"),
    State("global-store", "data"),
    prevent_initial_call=True,
)
def overview_brush_update(relayout_trend, relayout_ref, store):
    store = store or {}

    # decide which relayout to use (prefer the one that has actual range info)
    candidates = [r for r in (relayout_trend, relayout_ref) if isinstance(r, dict) and len(r)]
    if not candidates:
        return store

    relayout = None
    for r in candidates:
        if "xaxis.range[0]" in r or "xaxis.range" in r or "xaxis.autorange" in r:
            relayout = r
            break
    if relayout is None:
        relayout = candidates[0]

    w0, w1, cleared = _parse_brush_from_relayout(relayout)
    if cleared:
        store.pop("brush", None)
        return store

    if w0 is not None and w1 is not None:
        store["brush"] = {"w0": int(w0), "w1": int(w1)}
    return store


@callback(
    Output("overview-hover-preview", "children"),
    Input("overview-heatmap", "hoverData"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("event-filter", "value"),
)
def overview_hover_preview(hoverData, week_range, services, events):
    if not hoverData or "points" not in hoverData or not hoverData["points"]:
        return "Tip: Hover a cell to preview; click a cell to lock selection."

    pts = hoverData.get("points", [])
    p = next((q for q in pts if ("x" in q and "y" in q and "z" in q)), None)
    if p is None:
        return "Tip: Hover a cell to preview; click a cell to lock selection."

    service = p.get("y")
    week = p.get("x")
    z = p.get("z")

    try:
        week_i = int(week)
    except Exception:
        week_i = week

    df = _aggregate_service_week(_filter_services(_services_df, week_range, services, events))
    row = df[(df["service"] == service) & (df["week"] == week_i)]

    refused_txt = "—"
    ev_label = "—"
    if not row.empty:
        refused = row["patients_refused"].iloc[0] if "patients_refused" in row.columns else None
        refused_txt = "—" if pd.isna(refused) else str(int(refused))
        ev_raw = row["event"].iloc[0] if "event" in row.columns else None
        ev_label = EVENT_STYLE.get(str(ev_raw), EVENT_STYLE["none"])["label"] if ev_raw is not None else "—"

    z_txt = "—" if z is None else f"{float(z):.2f}"
    return f"Preview → service={service}, week={week_i}, shortage_rate={z_txt}, refused={refused_txt}, event={ev_label}"


@callback(
    Output("global-store", "data", allow_duplicate=True),
    Input("overview-heatmap", "clickData"),
    State("global-store", "data"),
    prevent_initial_call=True,
)
def overview_lock_selection(clickData, store):
    store = store or {}
    if not clickData or "points" not in clickData or not clickData["points"]:
        return store

    pts = clickData.get("points", [])
    p = next((q for q in pts if ("x" in q and "y" in q and "z" in q)), None)
    if p is None:
        return store

    service = p.get("y")
    week = p.get("x")
    try:
        week = int(week)
    except Exception:
        pass

    store["selection"] = {"service": service, "week": week}
    return store


@callback(
    Output("overview-selected-table", "children"),
    Input("global-store", "data"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("event-filter", "value"),
)
def render_selected_summary(store, week_range, services, events):
    store = store or {}
    sel = (store.get("selection") or {})
    sel_service = sel.get("service")
    sel_week = sel.get("week")

    if sel_service is None or sel_week is None:
        return html.Div(
            children=[
                html.H6("Selected cell details"),
                html.Div("Click a cell in the heatmap to pin details here.", style={"color": "#666"}),
            ],
            style={"padding": "6px 4px"},
        )

    df = _aggregate_service_week(_filter_services(_services_df, week_range, services, events))
    row = df[(df["service"] == sel_service) & (df["week"] == int(sel_week))]
    if row.empty:
        return html.Div(
            children=[
                html.H6("Selected cell details"),
                html.Div(
                    f"No data found for service={sel_service}, week={sel_week} under current filters.",
                    style={"color": "#a00"},
                ),
            ],
            style={"padding": "6px 4px"},
        )

    r = row.iloc[0]

    def fmt(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, float)):
            return f"{x:.2f}" if abs(x) < 1000 and (float(x) % 1 != 0) else str(int(x))
        return str(x)

    items = [
        ("Service", r.get("service")),
        ("Week", r.get("week")),
        ("Event", r.get("event")),
        ("Shortage rate", r.get("shortage_rate")),
        ("Bed pressure", r.get("bed_pressure")),
        ("Utilization", r.get("utilization")),
        ("Available beds", r.get("available_beds")),
        ("Patients request", r.get("patients_request")),
        ("Patients admitted", r.get("patients_admitted")),
        ("Patients refused", r.get("patients_refused")),
        ("Capacity delta", r.get("capacity_delta")),
    ]

    table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "marginTop": "6px"},
        children=[
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(k, style={"fontWeight": "600", "padding": "6px 10px", "width": "220px"}),
                            html.Td(fmt(v), style={"padding": "6px 10px"}),
                        ],
                        style={"borderTop": "1px solid #eee"},
                    )
                    for k, v in items
                ]
            )
        ],
    )

    return html.Div(
        children=[
            html.H6("Selected cell details"),
            html.Div(f"Pinned: {sel_service} · week {sel_week}", style={"color": "#666", "marginTop": "2px"}),
            table,
        ],
        style={"padding": "6px 4px"},
    )
