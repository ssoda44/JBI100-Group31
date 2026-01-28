# jbi100_app/views/diagnosis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, no_update

from ..data import get_data
from .. import config

# =====================================================
# Configuration & Constants
# =====================================================
ROLL_WIN = 6
DEMAND_THRESHOLD = 25   
WASTAGE_THRESHOLD = 15    
CAPACITY_THRESHOLD = -20.0 

CHART_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=80, r=80, t=40, b=40), # Balanced margins for dual y-axis
    hovermode="x unified"
)

# =====================================================
# Internal Helpers
# =====================================================
def _to_int(s): return pd.to_numeric(s, errors="coerce").astype("Int64")
def _norm_list(xs): return [str(x).strip().lower() for x in xs] if xs else []

def _add_highlight_line(fig, week):
    """Adds a vertical reference line to synchronize week-level analysis."""
    if week is not None and week > 0:
        fig.add_vline(
            x=week, 
            line_width=2, 
            line_dash="dash", 
            line_color="rgba(0,0,0,0.6)",
            annotation_text=f"Week {week}",
            annotation_position="top left"
        )

def _kpi_card(title, value, subtitle, color, reason=None):
    """Renders a styled metric card with a compact inline reasoning label."""
    card_children = [
        html.Div(title, style={"fontSize": "14px", "color": "#666", "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": color}),
        html.Div(subtitle, style={"fontSize": "12px", "color": "#999"}),
    ]
    if reason:
        card_children.append(
            html.Div(
                f"⚠ {reason}", 
                style={
                    "fontSize": "10px", "fontWeight": "bold", "color": color,
                    "marginTop": "4px", "fontStyle": "italic",
                    "borderTop": "1px solid #f0f0f0", "paddingTop": "2px"
                }
            )
        )
    return html.Div(style={
        "flex": "1", "padding": "20px", "borderRadius": "10px", 
        "backgroundColor": "white", "borderLeft": f"5px solid {color}",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "display": "flex", "flexDirection": "column", "justifyContent": "center",
        "minHeight": "115px" 
    }, children=card_children)

# =====================================================
# Layout
# =====================================================
def layout():
    print("DEBUG: Diagnosis Layout is being rendered")
    return html.Div([
        html.Div([
            html.Div("Diagnosis & Root-Cause", style={"fontSize": "24px", "fontWeight": 700}),
            html.Div("Analyzing bed wastage and supply-demand deviations.", 
                     style={"opacity": 0.7, "fontSize": "14px", "marginTop": "2px"}),
        ], style={"marginBottom": "20px"}),

        html.Div(id="diagnosis-root-cause-row", style={"display": "flex", "gap": "15px", "marginBottom": "25px"}),

        html.Div([
            html.Label("Deep-Dive Week Selection:", style={"marginRight": "10px", "fontWeight": "bold"}),
            dcc.Input(id="diag-selected-week", type="number", value=1, min=1, max=52, style={"width": "80px", "padding": "5px"}),
        ], style={"marginBottom": "20px", "background": "#f9f9f9", "padding": "15px", "borderRadius": "8px"}),

        html.Div([
            dcc.Graph(id="diag-fig-supply-demand-delta", config={"displayModeBar": False}),
        ], style={"marginBottom": "20px", "border": "1px solid #eee", "padding": "10px"}),

        html.Div([
            html.Div([dcc.Graph(id="diag-fig-waste-trend", config={"displayModeBar": False})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-waste-dist", config={"displayModeBar": False})], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        html.Div([
            html.Div([dcc.Graph(id="diag-fig-request-trend", config={"displayModeBar": False})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-capacity-dist", config={"displayModeBar": False})], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),
    ])

# =====================================================
# Callbacks
# =====================================================

@callback(
    Output("diag-selected-week", "value"),
    Input("shared-selected-week", "data")
)
def sync_week_from_global_store(global_week):
    print("Syncing diag week from global store:", global_week)
    return global_week if global_week is not None else 1

@callback(
    Output("diagnosis-root-cause-row", "children"),
    Input("diag-selected-week", "value"),
    Input("service-filter", "value"),
    Input("week-range", "value"),
)
def update_root_cause_indicators(selected_week, services, week_range):
    print("selected_week:", selected_week )
    if selected_week is None:
        return html.Div("Select a week to see metrics.", style={"padding": "20px", "color": "#666"})

    bundle = get_data()
    df_svc = bundle.services_df.copy()
    if services:
        sv_set = set(_norm_list(services))
        df_svc = df_svc[df_svc["service"].astype(str).str.strip().str.lower().isin(sv_set)]

    df_svc["unoccupied_beds"] = df_svc["available_beds"] - df_svc["patients_admitted"]
    f_svc = df_svc.groupby("week")[["patients_request", "available_beds", "unoccupied_beds"]].sum()

    f_svc["req_base"] = f_svc["patients_request"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["req_delta"] = f_svc["patients_request"] - f_svc["req_base"]
    f_svc["beds_base"] = f_svc["available_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["beds_delta"] = f_svc["available_beds"] - f_svc["beds_base"]

    if selected_week not in f_svc.index:
        return html.Div(f"No data for week {selected_week}")

    wd = f_svc.loc[selected_week]
    top_waste_row = df_svc[df_svc["week"] == selected_week].sort_values("unoccupied_beds", ascending=False)
    top_dept = top_waste_row.iloc[0]["service"] if not top_waste_row.empty else "N/A"
    top_val = top_waste_row.iloc[0]["unoccupied_beds"] if not top_waste_row.empty else 0

    demand_reason = "Sudden demand surge" if wd["req_delta"] > DEMAND_THRESHOLD else None
    wastage_reason = f"Misallocation in {top_dept}" if top_val > WASTAGE_THRESHOLD else None
    capacity_reason = "Sudden supply shortage" if wd["beds_delta"] < CAPACITY_THRESHOLD else None

    return [
        _kpi_card("Demand Pressure", f"{wd['req_delta']:+.1f}", "Req vs Base", "#d62728" if wd["req_delta"] > 0 else "#2ca02c", demand_reason),
        _kpi_card("Capacity Change", f"{wd['beds_delta']:+.1f}", "Beds vs Base", "#d62728" if wd["beds_delta"] < 0 else "#2ca02c", capacity_reason),
        _kpi_card("Primary Wastage", f"{int(top_val)} Beds", f"Highest: {top_dept}", "#1f77b4", wastage_reason)
    ]

@callback(
    Output("diag-fig-supply-demand-delta", "figure"),
    Output("diag-fig-waste-trend", "figure"),
    Output("diag-fig-waste-dist", "figure"),
    Output("diag-fig-request-trend", "figure"),
    Output("diag-fig-capacity-dist", "figure"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("diag-selected-week", "value"),
)
def update_diagnosis_trends(week_range, services, selected_week):
    print("Updating diagnosis trends for week_range:", week_range, "services:", services, "selected_week:", selected_week)
    if not week_range: week_range = [1, 52]
    wmin, wmax = int(week_range[0]), int(week_range[1])
    bundle = get_data()
    df_svc = bundle.services_df.copy()
    df_svc = df_svc[df_svc["week"].between(wmin, wmax)]
    if services:
        sv_set = set(_norm_list(services))
        df_svc = df_svc[df_svc["service"].astype(str).str.strip().str.lower().isin(sv_set)]

    df_svc["unoccupied_beds"] = df_svc["available_beds"] - df_svc["patients_admitted"]
    f_svc = df_svc.groupby("week")[["patients_request", "available_beds", "unoccupied_beds"]].sum().reindex(range(wmin, wmax+1))

    f_svc["req_base"] = f_svc["patients_request"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["req_delta"] = f_svc["patients_request"] - f_svc["req_base"]
    f_svc["beds_base"] = f_svc["available_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["beds_delta"] = f_svc["available_beds"] - f_svc["beds_base"]
    f_svc["unocc_base"] = f_svc["unoccupied_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)

    # --- SYNCHRONIZED DUAL Y-AXIS LOGIC ---
    # Find max absolute deviation to center both axes at 0
    max_req = f_svc["req_delta"].abs().max() or 1
    max_beds = f_svc["beds_delta"].abs().max() or 1

    fig_sd = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["req_delta"], name="Demand Δ", marker_color="#2ca02c"), secondary_y=False)
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["beds_delta"], name="Supply Δ", marker_color="#9467bd"), secondary_y=True)
    
    # Synchronize zeros by setting symmetric ranges based on each axis's own max
    fig_sd.update_yaxes(range=[-max_req * 1.1, max_req * 1.1], secondary_y=False, title_text="Demand Deviation")
    fig_sd.update_yaxes(range=[-max_beds * 1.1, max_beds * 1.1], secondary_y=True, title_text="Supply Deviation")
    fig_sd.update_layout(title="Demand vs Supply Deviations (Synchronized Zeros)", barmode='group', **CHART_LAYOUT)

    # Remaining Charts
    fig_waste = go.Figure()
    fig_waste.add_trace(go.Scatter(x=f_svc.index, y=f_svc["unoccupied_beds"], name="Unoccupied", line=dict(color="#7f7f7f", width=3)))
    fig_waste.add_trace(go.Scatter(x=f_svc.index, y=f_svc["unocc_base"], name="Baseline", line=dict(color="gray", dash="dash")))
    fig_waste.update_layout(title="Hospital Bed Wastage Trend", **CHART_LAYOUT)

    fig_dist = px.bar(df_svc, x="week", y="unoccupied_beds", color="service", title="Wastage by Department")
    fig_dist.update_layout(**CHART_LAYOUT)

    fig_req = go.Figure([
        go.Scatter(x=f_svc.index, y=f_svc["patients_request"], name="Total Requests", line=dict(color="#2ca02c")),
        go.Scatter(x=f_svc.index, y=f_svc["req_base"], name="Baseline", line=dict(color="gray", dash="dash"))
    ])
    fig_req.update_layout(title="Patient Request Volume", **CHART_LAYOUT)

    fig_cap = px.bar(df_svc, x="week", y="available_beds", color="service", title="Capacity Distribution")
    fig_cap.update_layout(**CHART_LAYOUT)

    for f in [fig_sd, fig_waste, fig_dist, fig_req, fig_cap]:
        f.update_yaxes(automargin=True)
        _add_highlight_line(f, selected_week)

    return fig_sd, fig_waste, fig_dist, fig_req, fig_cap

