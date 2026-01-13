# jb100_app/views/impact.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, callback, Input, Output

from ..data import get_data
from .. import config


# =====================================================
# Bundle
# =====================================================
_bundle = get_data()
_services_df = getattr(_bundle, "services_df", pd.DataFrame()).copy()
_patients_df = getattr(_bundle, "patients_df", pd.DataFrame()).copy()


# =====================================================
# Small utils
# =====================================================
def _norm_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    return [str(x).strip().lower() for x in xs]


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _empty_fig(title: str, note: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly",
        height=height,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text=title, x=0.02, xanchor="left"),
    )
    fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, opacity=0.75)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _kpi_card(title: str, value: str, sub: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "opacity": 0.7, "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 700, "lineHeight": "24px"}),
            html.Div(sub, style={"fontSize": "12px", "opacity": 0.7, "marginTop": "6px"}),
        ],
        style={
            "border": "1px solid #eee",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "background": "white",
            "minHeight": "78px",
        },
    )


# =====================================================
# 1) Weekly services outcome table (aligns to incidents filters)
#    - IMPORTANT: Do NOT filter by event here (baseline must not be cut by event filter)
#    - We will use week_mask derived from incidents weekly table for event-focus.
# =====================================================
def _weekly_services_table(
    services_df: pd.DataFrame,
    week_range: List[int],
    services: List[str],
) -> Tuple[pd.DataFrame, bool]:
    """
    Returns week-level table with:
      week
      shortage_rate (if exists)
      staff_morale_mean (if exists as staff_morale)
    """
    wmin, wmax = int(week_range[0]), int(week_range[1])
    full = pd.DataFrame({"week": list(range(wmin, wmax + 1))})

    if services_df is None or services_df.empty or "week" not in services_df.columns:
        out = full.copy()
        out["shortage_rate"] = np.nan
        out["staff_morale_mean"] = np.nan
        return out, False

    df = services_df.copy()
    df["week"] = _to_int(df["week"])
    df = df[df["week"].between(wmin, wmax)]

    if services and "service" in df.columns:
        sv = set(_norm_list(services))
        df = df[_norm_series(df["service"]).isin(sv)]

    morale_available = "staff_morale" in df.columns

    if df.empty:
        out = full.copy()
        if "shortage_rate" in services_df.columns:
            out["shortage_rate"] = np.nan
        if morale_available:
            out["staff_morale_mean"] = np.nan
        return out, morale_available

    agg_spec = {}
    if "shortage_rate" in df.columns:
        agg_spec["shortage_rate"] = "mean"
    if morale_available:
        df["staff_morale"] = pd.to_numeric(df["staff_morale"], errors="coerce")
        agg_spec["staff_morale"] = "mean"

    if not agg_spec:
        out = full.copy()
        out["shortage_rate"] = np.nan
        out["staff_morale_mean"] = np.nan
        return out, morale_available

    out = df.groupby("week", as_index=False).agg(agg_spec)
    if "staff_morale" in out.columns:
        out = out.rename(columns={"staff_morale": "staff_morale_mean"})

    out = full.merge(out, on="week", how="left")
    return out, morale_available


# =====================================================
# 2) Weekly patient satisfaction (optional)
#    - No event filter here; event focus will be applied via week_mask
# =====================================================
def _weekly_patient_satisfaction(
    patients_df: pd.DataFrame,
    week_range: List[int],
    services: List[str],
) -> pd.DataFrame:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    out = pd.DataFrame({"week": list(range(wmin, wmax + 1))})

    if patients_df is None or patients_df.empty or "week" not in patients_df.columns:
        return out

    df = patients_df.copy()
    df["week"] = _to_int(df["week"])
    df = df[df["week"].between(wmin, wmax)]

    if services and "service" in df.columns:
        sv = set(_norm_list(services))
        df = df[_norm_series(df["service"]).isin(sv)]

    sat_col = None
    for c in ["patient_satisfaction", "satisfaction", "satisfaction_score", "patient_score"]:
        if c in df.columns:
            sat_col = c
            break

    if sat_col is None or df.empty:
        return out

    df[sat_col] = pd.to_numeric(df[sat_col], errors="coerce")
    df = df[df[sat_col].notna()]
    if df.empty:
        return out

    w = df.groupby("week", as_index=False).agg(
        patient_satisfaction_mean=(sat_col, "mean"),
        patient_n=(sat_col, "size"),
    )
    return out.merge(w, on="week", how="left")


# =====================================================
# 3) Build incident flags FROM incidents.py detection (authoritative)
#    - Uses same menu filters and same "event filter affects table" semantics.
# =====================================================
def _incident_weeks_from_incidents_py(
    services_df: pd.DataFrame,
    staff_week_service_df: pd.DataFrame,
    week_range: List[int],
    services: List[str],
    events: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      flag_df: week, is_incident, metric_trigger, severity
      week_mask: Series(index=week, bool)  (event focus: weeks where selected event appears)
    """
    # Local import to reduce risk of circular-import issues
    from . import incidents as inc

    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))

    # build weekly table exactly like incidents page (does NOT cut KPI by event)
    weekly_df, metrics = inc._build_weekly_table(services_df, week_range, services, events or [])
    staff_ps = inc._staff_per_service(staff_week_service_df, week_range, services)

    # run detection (authoritative)
    incidents_df = inc._detect_incidents(weekly_df, metrics, week_range, staff_ps)

    # apply the SAME event-filter-to-table semantics as incidents.py update callback
    # (events is optional; empty => no constraint)
    if events and incidents_df is not None and (not incidents_df.empty) and ("event_fact" in incidents_df.columns):
        ev_set = set(_norm_list(events))
        incidents_df = incidents_df[
            incidents_df["event_fact"].astype(str).str.strip().str.lower().isin(ev_set)
        ]

    # detected weeks set
    detected_weeks: set[int] = set()
    metric_trigger_map: dict[int, str] = {}
    severity_map: dict[int, float] = {}

    if incidents_df is not None and not incidents_df.empty and "week" in incidents_df.columns:
        # incidents_df week is like "W05"
        def _wk_to_int(wk: str) -> Optional[int]:
            try:
                return int("".join([c for c in str(wk) if c.isdigit()]))
            except Exception:
                return None

        tmp = incidents_df.copy()
        tmp["week_int"] = tmp["week"].apply(_wk_to_int)
        tmp = tmp[tmp["week_int"].notna()]

        # per week pick the row with max severity as trigger
        if "severity" in tmp.columns:
            # ensure numeric
            tmp["severity_num"] = pd.to_numeric(tmp["severity"], errors="coerce")
        else:
            tmp["severity_num"] = np.nan

        for w in weeks:
            sub = tmp[tmp["week_int"] == w]
            if sub.empty:
                continue
            detected_weeks.add(w)
            # trigger = max severity row
            sub = sub.sort_values("severity_num", ascending=False)
            metric_trigger_map[w] = str(sub.iloc[0].get("metric", "shortage_rate"))
            try:
                severity_map[w] = float(sub.iloc[0].get("severity_num", np.nan))
            except Exception:
                severity_map[w] = np.nan

    flag_df = pd.DataFrame({"week": weeks})
    flag_df["is_incident"] = flag_df["week"].apply(lambda w: w in detected_weeks)
    flag_df["metric_trigger"] = flag_df["week"].apply(lambda w: metric_trigger_map.get(w, ""))
    flag_df["severity"] = flag_df["week"].apply(lambda w: severity_map.get(w, np.nan))

    # event-focus week mask:
    # if events selected => keep only weeks where selected event appears in ANY selected service
    if events:
        ev_set = set(_norm_list(events))
        wm = (
            weekly_df.assign(
                _hit=weekly_df["event_fact"].astype(str).str.strip().str.lower().isin(ev_set)
            )
            .groupby("week")["_hit"]
            .any()
            .reindex(weeks)
            .fillna(False)
        )
    else:
        wm = pd.Series(True, index=weeks)

    return flag_df, wm


# =====================================================
# UI
# =====================================================
def layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Impact", style={"fontSize": "18px", "fontWeight": 700}),
                    html.Div(
                        "Impact uses the SAME incident detection as Incidents (authoritative).",
                        style={"opacity": 0.7, "fontSize": "12px", "marginTop": "2px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(id="impact-kpi-1"),
                    html.Div(id="impact-kpi-2"),
                    html.Div(id="impact-kpi-3"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(200px, 1fr))", "gap": "10px"},
            ),
            html.Div(
                [
                    dcc.Graph(id="impact-fig-1", config={"displayModeBar": False}),
                    dcc.Graph(id="impact-fig-2", config={"displayModeBar": False}),
                    dcc.Graph(id="impact-fig-3", config={"displayModeBar": False}),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "12px", "marginTop": "10px"},
            ),
            html.Div(
                id="impact-note",
                style={"fontSize": "11px", "opacity": 0.6, "marginTop": "6px"},
            ),
        ],
        style={"width": "100%"},
    )


# =====================================================
# Callback
# =====================================================
@callback(
    Output("impact-kpi-1", "children"),
    Output("impact-kpi-2", "children"),
    Output("impact-kpi-3", "children"),
    Output("impact-fig-1", "figure"),
    Output("impact-fig-2", "figure"),
    Output("impact-fig-3", "figure"),
    Output("impact-note", "children"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("event-filter", "value"),
)
def update_impact(
    week_range: List[int],
    services: List[str],
    events: List[str],
):
    # defaults consistent with other pages
    if not week_range or len(week_range) != 2:
        week_range = [1, 52]
    if not services:
        services = list(getattr(config, "SERVICES_ORDER", []))
    events = events or []  # empty => no constraint

    # Authoritative incident flags from incidents.py
    flag_df, week_mask = _incident_weeks_from_incidents_py(
        _services_df,
        getattr(_bundle, "staff_week_service_df", pd.DataFrame()).copy(),
        week_range,
        services,
        events,
    )

    # Weekly outcomes (no event cut; apply week_mask afterwards)
    weekly_services, morale_available = _weekly_services_table(_services_df, week_range, services)
    weekly_sat = _weekly_patient_satisfaction(_patients_df, week_range, services)

    # merge
    w = flag_df.merge(weekly_services, on="week", how="left").merge(weekly_sat, on="week", how="left")

    # event focus: if events selected, restrict analysis to those weeks only
    if events:
        keep_weeks = set(week_mask[week_mask == True].index.tolist())
        w = w[w["week"].isin(keep_weeks)].copy()

    # If after filtering nothing remains
    if w.empty:
        k1 = _kpi_card("Outcome", "n/a", "No weeks under current filters.")
        k2 = _kpi_card("Staff morale (mean)", "n/a", "No weeks under current filters.")
        k3 = _kpi_card("Incident weeks", "0", "No weeks under current filters.")
        fig1 = _empty_fig("Outcome: incident vs normal", "No data under current filters.")
        fig2 = _empty_fig("Staff morale: incident vs normal", "No data under current filters.")
        fig3 = _empty_fig("Sync: staff morale vs outcome", "No data under current filters.", height=340)
        note = "Impact incident definition: uses Incidents detected weeks. No weeks left after event focus."
        return k1, k2, k3, fig1, fig2, fig3, note

    inc = w[w["is_incident"] == True]
    nor = w[w["is_incident"] == False]

    def _m(x: pd.Series) -> float:
        try:
            return float(pd.to_numeric(x, errors="coerce").mean())
        except Exception:
            return np.nan

    inc_n = int(inc.shape[0])
    nor_n = int(nor.shape[0])

    # outcome
    outcome_col = "patient_satisfaction_mean" if "patient_satisfaction_mean" in w.columns else None
    if outcome_col is None or w[outcome_col].isna().all():
        outcome_col = "shortage_rate" if "shortage_rate" in w.columns else "severity"
        outcome_name = "Patient satisfaction" if outcome_col == "patient_satisfaction_mean" else (
            "Shortage rate (weekly mean)" if outcome_col == "shortage_rate" else "Incident severity (from Incidents)"
        )
    else:
        outcome_name = "Patient satisfaction"

    out_inc = _m(inc[outcome_col]) if outcome_col in inc.columns else np.nan
    out_nor = _m(nor[outcome_col]) if outcome_col in nor.columns else np.nan

    # morale
    morale_inc = _m(inc["staff_morale_mean"]) if "staff_morale_mean" in inc.columns else np.nan
    morale_nor = _m(nor["staff_morale_mean"]) if "staff_morale_mean" in nor.columns else np.nan

    k1 = _kpi_card(
        outcome_name,
        "n/a" if pd.isna(out_inc) else f"{out_inc:.3f}",
        "Δ vs normal: n/a" if (pd.isna(out_inc) or pd.isna(out_nor)) else f"Δ vs normal: {out_inc - out_nor:+.3f}",
    )
    k2 = _kpi_card(
        "Staff morale (mean)",
        "n/a" if pd.isna(morale_inc) else f"{morale_inc:.3f}",
        "Δ vs normal: n/a" if (pd.isna(morale_inc) or pd.isna(morale_nor)) else f"Δ vs normal: {morale_inc - morale_nor:+.3f}",
    )
    k3 = _kpi_card(
        "Incident weeks",
        f"{inc_n}",
        f"Normal weeks: {nor_n}",
    )

    # Fig 1: outcome incident vs normal
    if outcome_col in w.columns and w[outcome_col].notna().any():
        fig1 = go.Figure()
        fig1.add_trace(go.Box(y=nor[outcome_col], name="Normal", boxpoints="all", jitter=0.35))
        fig1.add_trace(go.Box(y=inc[outcome_col], name="Incident", boxpoints="all", jitter=0.35))
        fig1.update_layout(
            template="plotly",
            height=320,
            margin=dict(l=36, r=18, t=54, b=36),
            title=dict(text=f"{outcome_name}: incident vs normal", x=0.02, xanchor="left"),
        )
        fig1.update_yaxes(automargin=True)
    else:
        fig1 = _empty_fig(f"{outcome_name}: incident vs normal", "No outcome data under current filters.")

    # Fig 2: morale incident vs normal
    if not morale_available:
        fig2 = _empty_fig("Staff morale (mean): incident vs normal", "Column staff_morale not found in services_df.")
    elif "staff_morale_mean" in w.columns and w["staff_morale_mean"].notna().any():
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=nor["staff_morale_mean"], name="Normal", boxpoints="all", jitter=0.35))
        fig2.add_trace(go.Box(y=inc["staff_morale_mean"], name="Incident", boxpoints="all", jitter=0.35))
        fig2.update_layout(
            template="plotly",
            height=320,
            margin=dict(l=36, r=18, t=54, b=36),
            title=dict(text="Staff morale (mean): incident vs normal", x=0.02, xanchor="left"),
        )
        fig2.update_yaxes(automargin=True)
    else:
        fig2 = _empty_fig("Staff morale (mean): incident vs normal", "No staff morale available under current filters.")

    # Fig 3: sync scatter (morale vs outcome), highlight incident
    if (
        morale_available
        and "staff_morale_mean" in w.columns
        and w["staff_morale_mean"].notna().any()
        and outcome_col in w.columns
        and w[outcome_col].notna().any()
    ):
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=nor["staff_morale_mean"], y=nor[outcome_col], mode="markers", name="Normal"))
        if inc.shape[0] > 0:
            fig3.add_trace(
                go.Scatter(
                    x=inc["staff_morale_mean"],
                    y=inc[outcome_col],
                    mode="markers",
                    name="Incident",
                    marker_symbol="diamond",
                    marker_size=10,
                )
            )
        fig3.update_layout(
            template="plotly",
            height=340,
            margin=dict(l=36, r=18, t=54, b=36),
            title=dict(text=f"Sync: staff morale vs {outcome_name}", x=0.02, xanchor="left"),
        )
        fig3.update_xaxes(title_text="staff_morale_mean", automargin=True)
        fig3.update_yaxes(title_text=outcome_name, automargin=True)
    else:
        fig3 = _empty_fig("Sync: staff morale vs outcome", "Not enough data to compute sync under current filters.", height=340)

    # Build note with incidents.py knobs (authoritative)
    try:
        from . import incidents as inc
        note = (
            f"Impact incident definition: weeks are INCIDENT iff they are detected in Incidents "
            f"(rolling window={inc.ROLL_WIN}, thresholds: UNUSUAL≥{inc.SOFT_Z_TH}, RARE≥{inc.SEVERE_Z_TH}). "
            f"Event focus applied: {'ON' if bool(events) else 'OFF'}."
        )
    except Exception:
        note = "Impact incident definition: uses Incidents detected weeks. (Could not read incidents.py knobs.)"

    return k1, k2, k3, fig1, fig2, fig3, note
