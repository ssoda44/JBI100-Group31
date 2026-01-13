# jbi100_app/views/diagnosis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output

from ..data import get_data
from .. import config

# =====================================================
# Configuration & Constants
# =====================================================
ROLL_WIN = 6

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
            line_color="rgba(0,0,0,0.4)",
            annotation_text=f"Week {week}",
            annotation_position="top left"
        )

def layout():
    return html.Div([
        html.Div([
            html.Div("Diagnosis", style={"fontSize": "18px", "fontWeight": 700}),
            html.Div("Root-cause analysis focusing on Bed Wastage and Supply-Demand deviations.", 
                     style={"opacity": 0.7, "fontSize": "12px", "marginTop": "2px"}),
        ], style={"marginBottom": "10px"}),

        # 1. Selection & Highlight Controls
        html.Div([
            html.Label("Highlight Week for Deep-Dive:", style={"marginRight": "10px", "fontWeight": "bold"}),
            dcc.Input(
                id="diag-selected-week", 
                type="number", 
                value=1, 
                min=1, 
                max=52,
                style={"width": "80px", "padding": "5px"}
            )
        ], style={"marginBottom": "20px", "background": "#f9f9f9", "padding": "15px", "borderRadius": "8px"}),

        # 2. ROW 1: Demand vs Bed Supply Deviation (Full Width)
        html.Div([
            dcc.Graph(id="diag-fig-supply-demand-delta", config={"displayModeBar": False}),
        ], style={"marginBottom": "20px", "border": "1px solid #eee", "padding": "10px"}),

        # 3. ROW 2: Bed Wastage Analysis (Trend and Distribution side-by-side)
        
        html.Div([
            html.Div([dcc.Graph(id="diag-fig-waste-trend", config={"displayModeBar": False})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-waste-dist", config={"displayModeBar": False})], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        # 4. ROW 3: Basic Trends (Demand and Bed Capacity)
        html.Div([
            html.Div([dcc.Graph(id="diag-fig-request-trend", config={"displayModeBar": False})], style={"flex": "1"}),
            html.Div([dcc.Graph(id="diag-fig-capacity-dist", config={"displayModeBar": False})], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),
    ])

# Callback to sync with the Global Store from Incidents View
@callback(
    Output("diag-selected-week", "value"),
    Input("shared-selected-week", "data")
)
def sync_week_from_global_store(global_week):
    return global_week

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
def update_diagnosis(week_range, services, selected_week):
    if not week_range: week_range = [1, 52]
    wmin, wmax = int(week_range[0]), int(week_range[1])
    
    # --- Data Loading ---
    bundle = get_data()
    df_svc = bundle.services_df.copy()
    
    # Filtering
    df_svc = df_svc[df_svc["week"].between(wmin, wmax)]
    if services:
        sv_set = set(_norm_list(services))
        df_svc = df_svc[df_svc["service"].astype(str).str.strip().str.lower().isin(sv_set)]

    # --- 1. Aggregation Logic (Sums for Whole Numbers) ---
    df_svc["unoccupied_beds"] = df_svc["available_beds"] - df_svc["patients_admitted"]
    
    # Weekly Totals (Global Hospital View)
    f_svc = df_svc.groupby("week")[["patients_request", "available_beds", "unoccupied_beds"]].sum().reindex(range(wmin, wmax+1))

    # --- 2. Calculate Rolling Baselines based on Sums ---
    f_svc["req_base"] = f_svc["patients_request"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["req_delta"] = f_svc["patients_request"] - f_svc["req_base"]
    f_svc["beds_base"] = f_svc["available_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)
    f_svc["beds_delta"] = f_svc["available_beds"] - f_svc["beds_base"]
    f_svc["unocc_base"] = f_svc["unoccupied_beds"].rolling(ROLL_WIN, min_periods=1).mean().shift(1)

    # --- FIGURE CONSTRUCTION ---

    # LINE 1: Demand vs Bed Supply Deviation
    fig_sd = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["req_delta"], name="Total Demand Δ", marker_color="#2ca02c", offsetgroup=1), secondary_y=False)
    fig_sd.add_trace(go.Bar(x=f_svc.index, y=f_svc["beds_delta"], name="Total Bed Supply Δ", marker_color="#9467bd", offsetgroup=2), secondary_y=True)
    r_lim = max(f_svc["req_delta"].abs().max(), 0.1) * 1.1
    b_lim = max(f_svc["beds_delta"].abs().max(), 0.1) * 1.1
    fig_sd.update_layout(title="Total Demand vs Bed Supply Deviations", barmode='group', template="plotly_white", height=350)
    fig_sd.update_yaxes(range=[-r_lim, r_lim], secondary_y=False, zeroline=True)
    fig_sd.update_yaxes(range=[-b_lim, b_lim], secondary_y=True, zeroline=True)

    # LINE 2: Bed Wastage Trend
    fig_waste = go.Figure()
    fig_waste.add_trace(go.Scatter(x=f_svc.index, y=f_svc["unoccupied_beds"], name="Unoccupied", line=dict(color="#7f7f7f", width=3)))
    fig_waste.add_trace(go.Scatter(x=f_svc.index, y=f_svc["unocc_base"], name="Baseline", line=dict(color="gray", dash="dash")))
    fig_waste.update_layout(title="Total Bed Wastage Analysis", template="plotly_white", height=350, yaxis_title="Empty Beds")

    # LINE 2: Wastage Distribution (Stacked Bar)
    fig_dist = px.bar(df_svc, x="week", y="unoccupied_beds", color="service", title="Wastage Distribution by Department", template="plotly_white", height=350)
    fig_dist.update_layout(barmode='stack')

    # LINE 3: Demand Trend
    fig_req = go.Figure([
        go.Scatter(x=f_svc.index, y=f_svc["patients_request"], name="Total Requests", line=dict(color="#2ca02c", width=2)),
        go.Scatter(x=f_svc.index, y=f_svc["req_base"], name="Baseline", line=dict(color="gray", dash="dash"))
    ])
    fig_req.update_layout(title="Total Patient Demand Trend", template="plotly_white", height=300)

    # LINE 3: Bed Capacity Distribution
    
    fig_cap = px.bar(df_svc, x="week", y="available_beds", color="service", title="Bed Capacity Distribution", template="plotly_white", height=300)
    fig_cap.update_layout(barmode='stack', yaxis_title="Total Available Beds")

    # Highlight vertical line for all figures
    for f in [fig_sd, fig_waste, fig_dist, fig_req, fig_cap]:
        _add_highlight_line(f, selected_week)

    return fig_sd, fig_waste, fig_dist, fig_req, fig_cap