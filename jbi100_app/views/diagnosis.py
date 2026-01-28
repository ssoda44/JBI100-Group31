# jbi100_app/views/diagnosis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, no_update, ctx, State

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
    margin=dict(l=80, r=80, t=40, b=40), 
    hovermode="x unified"
)

# =====================================================
# Internal Helpers
# =====================================================
def _norm_list(xs): 
    """Normalizes input lists for consistent filtering."""
    return [str(x).strip().lower() for x in xs] if xs else []

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
    """Defines the Diagnosis view layout with unique container IDs to prevent cross-view crosstalk."""
    return html.Div(id="diagnosis-view-container", children=[
        html.Div([
            html.Div("Diagnosis & Root-Cause", style={"fontSize": "24px", "fontWeight": 700}),
            html.Div("Analyzing bed wastage and deviations. Brush any graph to sync zoom. Y-axis on the first chart is locked.", 
                     style={"opacity": 0.7, "fontSize": "14px", "marginTop": "2px"}),
        ], style={"marginBottom": "20px"}),

        html.Div(id="diagnosis-root-cause-row", style={"display": "flex", "gap": "15px", "marginBottom": "25px"}),

        html.Div([
            html.Label("Deep-Dive Week Selection:", style={"marginRight": "10px", "fontWeight": "bold"}),
            dcc.Input(id="diag-selected-week", type="number", value=1, min=1, max=52, style={"width": "80px", "padding": "5px"}),
        ], style={"marginBottom": "20px", "background": "#f9f9f9", "padding": "15px", "borderRadius": "8px"}),

        html.Div([
            dcc.Graph(id="diag-fig-supply-demand-delta", config={"displayModeBar": True}),
        ], style={"marginBottom": "20px", "border": "1px solid #eee", "padding": "10px"}),

        html.Div([
            html.Div([dcc.Graph(id="diag-fig-waste-trend", config={"displayModeBar": True})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-waste-dist", config={"displayModeBar": True})], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        html.Div([
            html.Div([dcc.Graph(id="diag-fig-request-trend", config={"displayModeBar": True})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-capacity-dist", config={"displayModeBar": True})], style={"flex": "1"}),
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
    """Syncs the selected week from the global application store."""
    return global_week if global_week is not None else 1

@callback(
    Output("diagnosis-root-cause-row", "children"),
    Input("diag-selected-week", "value"),
    Input("service-filter", "value"),
    Input("week-range", "value"),
)
def update_root_cause_indicators(selected_week, services, week_range):
    """Updates KPI cards based on filtered metrics and baseline deviations."""
    if selected_week is None:
        return no_update

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
    """Generates all diagnosis charts with locked Y-axes on the supply-demand chart."""
    if selected_week is None:
        return [no_update] * 5

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

    # Calculate Deviations
    f_svc["req_base"] = f_svc["patients_request"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["req_delta"] = f_svc["patients_request"] - f_svc["req_base"]
    f_svc["beds_base"] = f_svc["available_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["beds_delta"] = f_svc["available_beds"] - f_svc["beds_base"]

    # Symmetric Range Calculation for Zero Alignment
    max_req = f_svc["req_delta"].abs().max() or 1
    max_beds = f_svc["beds_delta"].abs().max() or 1

    fig_sd = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["req_delta"], name="Demand Δ", marker_color="#2ca02c"), secondary_y=False)
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["beds_delta"], name="Supply Δ", marker_color="#9467bd"), secondary_y=True)
    
    # LOCK Y-AXIS adjustability to prevent zero-line misalignment
    fig_sd.update_yaxes(range=[-max_req * 1.1, max_req * 1.1], secondary_y=False, title_text="Demand Deviation", fixedrange=True)
    fig_sd.update_yaxes(range=[-max_beds * 1.1, max_beds * 1.1], secondary_y=True, title_text="Supply Deviation", fixedrange=True)
    
    fig_sd.update_layout(
        title="Demand vs Supply Deviations (Locked Y-Axes)", 
        barmode='group', 
        meta={'max_req': float(max_req), 'max_beds': float(max_beds)},
        **CHART_LAYOUT
    )

    # Standard Figures
    fig_waste = go.Figure(go.Scatter(x=f_svc.index, y=f_svc["unoccupied_beds"], name="Unoccupied", line=dict(color="#7f7f7f", width=3)))
    fig_waste.update_layout(title="Hospital Bed Wastage Trend", **CHART_LAYOUT)

    fig_dist = px.bar(df_svc, x="week", y="unoccupied_beds", color="service", title="Wastage by Department")
    fig_dist.update_layout(**CHART_LAYOUT)

    fig_req = go.Figure(go.Scatter(x=f_svc.index, y=f_svc["patients_request"], name="Total Requests", line=dict(color="#2ca02c")))
    fig_req.update_layout(title="Patient Request Volume", **CHART_LAYOUT)

    fig_cap = px.bar(df_svc, x="week", y="available_beds", color="service", title="Capacity Distribution")
    fig_cap.update_layout(**CHART_LAYOUT)

    figs = [fig_sd, fig_waste, fig_dist, fig_req, fig_cap]
    for f in figs:
        f.update_yaxes(automargin=True)
        _add_highlight_line(f, selected_week)

    return figs

# =====================================================
# Synchronized Zoom Callback
# =====================================================
@callback(
    Output("diag-fig-supply-demand-delta", "figure", allow_duplicate=True),
    Output("diag-fig-waste-trend", "figure", allow_duplicate=True),
    Output("diag-fig-waste-dist", "figure", allow_duplicate=True),
    Output("diag-fig-request-trend", "figure", allow_duplicate=True),
    Output("diag-fig-capacity-dist", "figure", allow_duplicate=True),
    [Input(f"diag-fig-{s}", "relayoutData") for s in ["supply-demand-delta", "waste-trend", "waste-dist", "request-trend", "capacity-dist"]],
    [State(f"diag-fig-{s}", "figure") for s in ["supply-demand-delta", "waste-trend", "waste-dist", "request-trend", "capacity-dist"]],
    prevent_initial_call=True
)
def sync_diagnosis_zoom(*args):
    """Synchronizes X-axis across all charts while preserving locked Y-axes on chart 1."""
    triggered_id = ctx.triggered_id
    if not triggered_id or "diag-fig" not in triggered_id:
        return [no_update] * 5

    mid = len(args) // 2
    relayout_list = args[:mid]
    figure_list = list(args[mid:])

    # Verification: Ensure figures exist (State check) to avoid TypeError
    if any(f is None for f in figure_list):
        return [no_update] * 5

    rel_data = next((r for r in relayout_list if r is not None), None)
    if not rel_data:
        return [no_update] * 5

    # Determine X-axis range (Zoom or Reset)
    new_range = None
    if 'xaxis.range[0]' in rel_data:
        new_range = [rel_data['xaxis.range[0]'], rel_data['xaxis.range[1]']]
    elif 'xaxis.autorange' in rel_data:
        new_range = None

    # Apply X-axis synchronization
    for i, f in enumerate(figure_list):
        if not isinstance(f, dict): continue
        if 'layout' not in f: f['layout'] = {}
        if 'xaxis' not in f['layout']: f['layout']['xaxis'] = {}
        
        if new_range:
            f['layout']['xaxis']['range'] = new_range
            f['layout']['xaxis']['autorange'] = False
        else:
            f['layout']['xaxis']['autorange'] = True

        # Ensure Chart 1 preserves its locked y-axes even after an auto-reset
        if i == 0:
            meta = f.get('layout', {}).get('meta', {})
            m_req, m_beds = meta.get('max_req', 1), meta.get('max_beds', 1)
            if new_range is None:
                if 'yaxis' not in f['layout']: f['layout']['yaxis'] = {}
                if 'yaxis2' not in f['layout']: f['layout']['yaxis2'] = {}
                f['layout']['yaxis']['range'] = [-m_req * 1.1, m_req * 1.1]
                f['layout']['yaxis2']['range'] = [-m_beds * 1.1, m_beds * 1.1]
                f['layout']['yaxis']['fixedrange'] = True
                f['layout']['yaxis2']['fixedrange'] = True

    return figure_list