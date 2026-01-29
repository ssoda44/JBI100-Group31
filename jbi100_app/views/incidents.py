# jbi100_app/views/incidents.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from dash import no_update

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dash import html, dcc, callback, Input, Output, State
from dash import dash_table

from ..data import get_data
from .. import config


# =====================================================
# Load bundle
# =====================================================
_bundle = get_data()
_services_df = getattr(_bundle, "services_df", pd.DataFrame()).copy()
_staff_week_service_df = getattr(_bundle, "staff_week_service_df", pd.DataFrame()).copy()


# =====================================================
# Tuning
# =====================================================
ROLL_WIN = 6
SOFT_Z_TH = 1.0
SEVERE_Z_TH = 2.0
MIN_POINTS = 8
Z_CLAMP = 12.0

# Direction: only detect "worse" weeks (one-sided)
HIGH_WORSE = {"shortage_rate", "patients_refused"}

# brushing style (week window)
BRUSH_FILL = "rgba(20,20,20,0.08)"
BRUSH_LINE = {"color": "rgba(20,20,20,0.55)", "dash": "dot", "width": 1.5}


# =====================================================
# Helpers
# =====================================================
def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _norm_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    return [str(x).strip().lower() for x in xs]


def _empty_fig(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly",
        height=320,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text=title, x=0.02, xanchor="left"),
    )
    fig.add_annotation(
        text=note,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        opacity=0.75,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _format_week_axis(fig: go.Figure, wmin: int, wmax: int) -> None:
    # ✅ BUGFIX: do NOT hard-set xaxis range, otherwise zoom/brush will always snap back.
    fig.update_xaxes(
        title_text="Week",
        tickmode="linear",
        tick0=wmin,
        dtick=4,
        automargin=True,
    )


def _available_metrics(df: pd.DataFrame) -> List[str]:
    want = ["shortage_rate", "patients_refused"]
    return [m for m in want if m in df.columns]


def _make_week_mask(weekly_df: pd.DataFrame, events: List[str]) -> Optional[pd.Series]:
    if not events:
        return None
    if weekly_df is None or weekly_df.empty or "week" not in weekly_df.columns or "event_fact" not in weekly_df.columns:
        return None

    ev_set = set(_norm_list(events))
    tmp = weekly_df[["week", "event_fact"]].copy()
    tmp["event_fact"] = tmp["event_fact"].astype(str).str.strip().str.lower()
    mask = tmp.groupby("week")["event_fact"].apply(lambda s: bool(set(s.dropna()).intersection(ev_set)))
    return mask


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


def _parse_brush_from_relayout(relayout: Dict[str, Any] | None) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Returns (w0, w1, cleared)

    """
    if not relayout or not isinstance(relayout, dict):
        return None, None, False

    # standard zoom drag emits xaxis.range[0/1]
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        try:
            w0 = int(float(relayout["xaxis.range[0]"]))
            w1 = int(float(relayout["xaxis.range[1]"]))
            return w0, w1, False
        except Exception:
            return None, None, False

    # sometimes xaxis.range = [a,b]
    if "xaxis.range" in relayout and isinstance(relayout["xaxis.range"], (list, tuple)) and len(relayout["xaxis.range"]) == 2:
        try:
            w0 = int(float(relayout["xaxis.range"][0]))
            w1 = int(float(relayout["xaxis.range"][1]))
            return w0, w1, False
        except Exception:
            return None, None, False

    return None, None, False


def _effective_week_range(menu_week_range: List[int], store: dict | None) -> List[int]:
    """
    Intersect menu slider week-range with global-store brush (if any).
    If intersection is empty, fallback to brush range (better UX than blank).
    """
    if not menu_week_range or len(menu_week_range) != 2:
        base = None
    else:
        base = (int(menu_week_range[0]), int(menu_week_range[1]))

    brush = ((store or {}).get("brush") or {})
    bw0 = brush.get("w0")
    bw1 = brush.get("w1")
    if bw0 is None or bw1 is None:
        if base is None:
            return [1, 52]
        return [base[0], base[1]]

    bw0, bw1 = int(bw0), int(bw1)
    if bw0 > bw1:
        bw0, bw1 = bw1, bw0

    if base is None:
        return [bw0, bw1]

    w0 = max(base[0], bw0)
    w1 = min(base[1], bw1)
    if w0 > w1:
        return [bw0, bw1]
    return [w0, w1]


def _selected_ids_from_points(sel_sev: dict | None, sel_evt: dict | None) -> Optional[List[str]]:
    picked = set()
    for sel in (sel_sev, sel_evt):
        if sel and isinstance(sel, dict) and "points" in sel:
            for p in sel.get("points", []):
                if "customdata" in p and p["customdata"] is not None:
                    picked.add(p["customdata"])
    return sorted(picked) if picked else None


# =====================================================
# Robust baseline z (median + MAD)
# =====================================================
def _rolling_mean(v: pd.Series, win: int) -> pd.Series:
    return v.rolling(win, min_periods=3).mean()


def _rolling_mad(v: pd.Series, win: int) -> pd.Series:
    mean = _rolling_mean(v, win)
    dev = (v - mean).abs()
    return dev.rolling(win, min_periods=3).mean()


def _robust_z(v: pd.Series, win: int) -> pd.DataFrame:
    v_clean = v.replace(0, np.nan)
    x = v_clean.astype(float)
    mean = _rolling_mean(x, win).shift(1)
    mad = _rolling_mad(x, win).shift(1)

    pos = mad[(mad > 0) & mad.notna()]
    if pos.size == 0:
        z = pd.Series(np.nan, index=x.index)
    else:
        mad_floor = max(float(np.nanmedian(pos)) * 0.25, 1e-3)
        denom = 1.4826 * mad.clip(lower=mad_floor) + 1e-9
        z = (x - mean) / denom

    return pd.DataFrame({"value": x, "baseline": mean, "z_raw": z})


def _rarity(severity: float) -> str:
    if severity >= SEVERE_Z_TH:
        return "RARE"
    if severity >= SOFT_Z_TH:
        return "UNUSUAL"
    return "NORMAL"


def _severity_scatter_fig(
    incidents_df: pd.DataFrame,
    week_range: List[int],
    selected_ids: List[str] = None,
    brush: dict | None = None,
) -> go.Figure:
    """Plots severity Z-values over time for all services in one plot."""
    wmin, wmax = int(week_range[0]), int(week_range[1])
    if incidents_df is None or incidents_df.empty:
        return _empty_fig("Incident Severity Breakdown", "No Unusual/Rare incidents detected.")

    df = incidents_df.copy()
    df["week_num"] = df["week"].apply(lambda x: int(str(x).replace("W", "")))

    fig = go.Figure()
    services = sorted(df["service"].unique())
    colors = px.colors.qualitative.Plotly

    for i, svc in enumerate(services):
        svc_df = df[df["service"] == svc]

        opacity = 1.0
        if selected_ids is not None:
            opacity = [1.0 if row_id in selected_ids else 0.2 for row_id in svc_df["id"]]

        fig.add_trace(
            go.Scatter(
                x=svc_df["week_num"],
                y=svc_df["severity"],
                mode="markers",
                name=str(svc),
                customdata=svc_df["id"],
                marker=dict(
                    size=10,
                    line=dict(width=1, color="DarkSlateGrey"),
                    color=colors[i % len(colors)],
                    opacity=opacity,
                ),
                text=svc_df["metric"] + " (" + svc_df["rarity"] + ")",
                hovertemplate="<b>%{name}</b><br>Week: %{x}<br>Z-Score: %{y}<br>Metric: %{text}<extra></extra>",
            )
        )

    fig.add_hline(y=SOFT_Z_TH, line_dash="dash", line_color="orange", annotation_text="Unusual Threshold")
    fig.add_hline(y=SEVERE_Z_TH, line_dash="dash", line_color="red", annotation_text="Rare Threshold")

    fig.update_layout(
        template="plotly",
        height=320,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text="Incident Severity Breakdown (Z-Score by Service)", x=0.02, xanchor="left"),
        xaxis_title="Week",
        yaxis_title="Severity (Z-score)",
        legend=dict(orientation="h", y=1.1, x=0.0),
        hovermode="closest",
        clickmode="event+select",
        dragmode="select",
        uirevision="incidents-v1",
    )
    _format_week_axis(fig, wmin, wmax)

    if brush and brush.get("w0") is not None and brush.get("w1") is not None:
        _add_brush_band(fig, int(brush["w0"]), int(brush["w1"]))
        fig.add_annotation(
            x=0.01,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"Brushed weeks: {min(int(brush['w0']), int(brush['w1']))}–{max(int(brush['w0']), int(brush['w1']))} (use modebar reset to clear)",
            showarrow=False,
            font=dict(size=10, color="rgba(90,90,90,0.85)"),
            align="left",
        )

    return fig


def _service_event_point_fig(
    incidents_df: pd.DataFrame,
    selected_ids: List[str] = None,
    brush: dict | None = None,
) -> go.Figure:
    """Point plot with services on x-axis and Z-value on y-axis, grouped by events."""
    if incidents_df is None or incidents_df.empty:
        return _empty_fig("Service-Event Severity Impact", "No incidents detected.")

    df = incidents_df.copy()
    df["event_fact"] = df["event_fact"].replace("none", "No specific event")

    fig = go.Figure()
    events = sorted(df["event_fact"].unique())
    colors = px.colors.qualitative.Antique

    for i, evt in enumerate(events):
        evt_df = df[df["event_fact"] == evt]

        opacity = 1.0
        if selected_ids is not None:
            opacity = [1.0 if row_id in selected_ids else 0.2 for row_id in evt_df["id"]]

        fig.add_trace(
            go.Scatter(
                x=evt_df["service"],
                y=evt_df["severity"],
                mode="markers",
                name=str(evt),
                customdata=evt_df["id"],
                marker=dict(
                    size=12,
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                    color=colors[i % len(colors)],
                    opacity=opacity,
                ),
                text=evt_df["week"] + ": " + evt_df["metric"],
                hovertemplate="<b>Event: %{fullData.name}</b><br>Service: %{x}<br>Severity Z: %{y}<br>Details: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=60, b=60),
        title=dict(text="Severity Impact by Service & Event Type", x=0.02, xanchor="left"),
        xaxis_title="Medical Service",
        yaxis_title="Severity (Z-score)",
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
        clickmode="event+select",
        dragmode="select",
        uirevision="incidents-v1",
    )

    fig.add_hline(y=SOFT_Z_TH, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_hline(y=SEVERE_Z_TH, line_dash="dash", line_color="red", opacity=0.5)

    # (optional) show brush as annotation only (x-axis is categorical here; no vrect)
    if brush and brush.get("w0") is not None and brush.get("w1") is not None:
        fig.add_annotation(
            x=0.01,
            y=1.04,
            xref="paper",
            yref="paper",
            text=f"Brushed weeks: {min(int(brush['w0']), int(brush['w1']))}–{max(int(brush['w0']), int(brush['w1']))}",
            showarrow=False,
            font=dict(size=10, color="rgba(90,90,90,0.85)"),
            align="left",
        )

    return fig


# =====================================================
# Weekly table builder
# =====================================================
def _build_weekly_table(
    services_df: pd.DataFrame,
    week_range: List[int],
    services: List[str],
    events: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if services_df is None or services_df.empty:
        return pd.DataFrame(columns=["week", "service", "event_fact", "event_flag"]), []

    df = services_df.copy()
    if not {"week", "service"}.issubset(df.columns):
        return pd.DataFrame(columns=["week", "service", "event_fact", "event_flag"]), []

    wmin, wmax = int(week_range[0]), int(week_range[1])
    df["week"] = _to_int(df["week"])
    df = df[df["week"].between(wmin, wmax)]

    if services:
        sv = set(_norm_list(services))
        df = df[df["service"].astype(str).str.strip().str.lower().isin(sv)]

    metrics = _available_metrics(df)

    if df.empty:
        svc_list = [s for s in (services or list(getattr(config, "SERVICES_ORDER", []))) if s] or ["(none)"]
        out = pd.MultiIndex.from_product([list(range(wmin, wmax + 1)), svc_list], names=["week", "service"]).to_frame(index=False)
        for m in metrics:
            out[m] = np.nan
        out["event_fact"] = "none"
        out["event_flag"] = 0
        return out, metrics

    out = df.groupby(["week", "service"], as_index=False).agg({m: "mean" for m in metrics})

    if "event" in df.columns:
        ev = (
            df.groupby(["week", "service"])["event"]
            .apply(lambda x: str(x.dropna().iloc[0]).strip().lower() if x.dropna().shape[0] else "none")
            .reset_index()
            .rename(columns={"event": "event_fact"})
        )
    else:
        ev = out[["week", "service"]].copy()
        ev["event_fact"] = "none"

    out = out.merge(ev, on=["week", "service"], how="left")
    out["event_fact"] = out["event_fact"].fillna("none").astype(str).str.strip().str.lower()

    ev_set = set(_norm_list(events)) if events else None
    if ev_set is None:
        out["event_flag"] = (out["event_fact"] != "none").astype(int)
    else:
        out["event_flag"] = out["event_fact"].isin(ev_set).astype(int)

    svc_list = sorted(out["service"].dropna().astype(str).unique().tolist())
    full = pd.MultiIndex.from_product([list(range(wmin, wmax + 1)), svc_list], names=["week", "service"]).to_frame(index=False)
    out = full.merge(out, on=["week", "service"], how="left")
    out["event_fact"] = out["event_fact"].fillna("none").astype(str).str.strip().str.lower()
    out["event_flag"] = out["event_flag"].fillna(0).astype(int)

    return out, metrics


def _staff_per_service(
    staff_week_service_df: pd.DataFrame,
    week_range: List[int],
    services: List[str],
) -> pd.DataFrame:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))

    if (
        staff_week_service_df is None
        or staff_week_service_df.empty
        or not {"week", "service", "staff_present_total"}.issubset(staff_week_service_df.columns)
    ):
        svc_list = [s for s in (services or list(getattr(config, "SERVICES_ORDER", []))) if s] or ["(none)"]
        return pd.MultiIndex.from_product([weeks, svc_list], names=["week", "service"]).to_frame(index=False)

    df = staff_week_service_df.copy()
    df["week"] = _to_int(df["week"])
    df = df[df["week"].between(wmin, wmax)]

    if services:
        sv = set(_norm_list(services))
        df = df[df["service"].astype(str).str.strip().str.lower().isin(sv)]

    agg = df.groupby(["week", "service"], as_index=False).agg(
        staff_present_total=("staff_present_total", "sum"),
        patients_request=("patients_request", "sum") if "patients_request" in df.columns else ("staff_present_total", "size"),
    )

    # simulate missing staff data for every 3rd week
    is_3n_week = (agg["week"] % 3 == 0)
    agg.loc[is_3n_week, "staff_present_total"] = np.nan

    denom = agg["patients_request"].replace({0: np.nan}).astype(float)
    agg["staff_to_request"] = agg["staff_present_total"].astype(float) / denom

    agg = agg.sort_values(["service", "week"]).reset_index(drop=True)
    agg["staff_delta"] = np.nan

    for svc, g in agg.groupby("service", sort=False):
        idx = g.index
        s_values = g["staff_to_request"]
        rolling_base = s_values.rolling(window=ROLL_WIN, min_periods=1).mean().shift(1)
        agg.loc[idx, "staff_delta"] = (s_values - rolling_base).values

    svc_names = sorted(agg["service"].dropna().astype(str).unique().tolist())
    weeks_full = pd.MultiIndex.from_product([weeks, svc_names], names=["week", "service"]).to_frame(index=False)

    out = weeks_full.merge(
        agg[["week", "service", "staff_to_request", "staff_delta"]],
        on=["week", "service"],
        how="left",
    )

    return out


def _staff_agg(staff_week_service_df: pd.DataFrame, week_range: List[int], services: List[str]) -> pd.DataFrame:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))
    out = pd.DataFrame({"week": weeks})

    if (
        staff_week_service_df is None
        or staff_week_service_df.empty
        or not {"week", "service", "staff_present_total"}.issubset(staff_week_service_df.columns)
    ):
        return out

    df = staff_week_service_df.copy()
    df["week"] = _to_int(df["week"])
    df = df[df["week"].between(wmin, wmax)]

    if services:
        sv = set(_norm_list(services))
        df = df[df["service"].astype(str).str.strip().str.lower().isin(sv)]

    if df.empty:
        return out

    agg = df.groupby("week", as_index=False).agg(
        staff_present_total=("staff_present_total", "sum"),
        patients_request=("patients_request", "sum") if "patients_request" in df.columns else ("staff_present_total", "size"),
    )
    denom = agg["patients_request"].replace({0: np.nan}).astype(float)
    agg["staff_to_request"] = agg["staff_present_total"].astype(float) / denom

    out = out.merge(agg[["week", "staff_to_request"]], on="week", how="left")
    return out


@dataclass
class IncidentRow:
    id: str
    service: str
    week: int
    metric: str
    value: float
    baseline: float
    severity: float
    rarity: str
    event_fact: str
    staff_to_request: float
    staff_delta: float


def _detect_incidents(
    weekly_df: pd.DataFrame,
    metrics: List[str],
    week_range: List[int],
    staff_ps: pd.DataFrame,
) -> pd.DataFrame:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))

    staff_map = (
        staff_ps.set_index(["week", "service"])
        if staff_ps is not None and not staff_ps.empty and {"week", "service"}.issubset(staff_ps.columns)
        else None
    )

    def _staff_get(w: int, svc: str, col: str) -> float:
        if staff_map is None:
            return np.nan
        try:
            return float(staff_map.loc[(w, svc), col])
        except Exception:
            return np.nan

    rows: List[IncidentRow] = []

    for svc, g in weekly_df.groupby("service", sort=False):
        g = g.sort_values("week")
        ev_fact = g.set_index("week")["event_fact"].reindex(weeks).fillna("none").astype(str)

        for m in metrics:
            s = g.set_index("week")[m].reindex(weeks).astype(float)
            if s.dropna().shape[0] < MIN_POINTS:
                continue

            dfz = _robust_z(s, ROLL_WIN)
            for w in weeks:
                z_raw = dfz.loc[w, "z_raw"]
                if pd.isna(z_raw):
                    continue

                z_raw = float(z_raw)
                sev = z_raw if m in HIGH_WORSE else abs(z_raw)
                if (pd.isna(sev)) or (sev <= 0):
                    continue

                label = _rarity(float(sev))
                if label == "NORMAL":
                    continue

                rows.append(
                    IncidentRow(
                        id=f"{svc}-{w}-{m}",
                        service=str(svc),
                        week=w,
                        metric=m,
                        value=float(dfz.loc[w, "value"]) if not pd.isna(dfz.loc[w, "value"]) else np.nan,
                        baseline=float(dfz.loc[w, "baseline"]) if not pd.isna(dfz.loc[w, "baseline"]) else np.nan,
                        severity=float(sev),
                        rarity=label,
                        event_fact=str(ev_fact.loc[w]).strip().lower(),
                        staff_to_request=_staff_get(w, str(svc), "staff_to_request"),
                        staff_delta=_staff_get(w, str(svc), "staff_delta"),
                    )
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "service",
                "week",
                "metric",
                "value",
                "baseline",
                "severity",
                "rarity",
                "event_fact",
                "staff_to_request",
                "staff_delta",
            ]
        )

    rarity_rank = {"RARE": 0, "UNUSUAL": 1, "NORMAL": 2}
    rows.sort(key=lambda r: (rarity_rank.get(r.rarity, 9), -r.severity, r.service, -r.week))

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["week_num"] = df["week"].astype(int)
    df["week"] = df["week"].apply(lambda x: f"W{int(x):02d}")
    df["severity"] = df["severity"].map(lambda v: round(float(np.clip(float(v), 0, Z_CLAMP)), 2))
    df["value"] = df["value"].map(lambda v: None if pd.isna(v) else round(float(v), 3))
    df["baseline"] = df["baseline"].map(lambda v: None if pd.isna(v) else round(float(v), 3))
    df["staff_to_request"] = df["staff_to_request"].map(lambda v: None if pd.isna(v) else round(float(v), 4))
    df["staff_delta"] = df["staff_delta"].map(lambda v: None if pd.isna(v) else round(float(v), 4))
    return df


def _aggregate_weekly(
    weekly_df: pd.DataFrame,
    metrics: List[str],
    week_range: List[int],
    week_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))
    out = pd.DataFrame({"week": weeks})

    if weekly_df is None or weekly_df.empty:
        out["event_flag"] = 0
        for m in metrics:
            out[m] = np.nan
        return out

    g = weekly_df.groupby("week", as_index=False).agg({m: "mean" for m in metrics})
    out = out.merge(g, on="week", how="left")

    if "event_flag" in weekly_df.columns:
        ev = weekly_df.groupby("week")["event_flag"].max().reset_index()
        out = out.merge(ev, on="week", how="left")
        out["event_flag"] = out["event_flag"].fillna(0).astype(int)
    else:
        out["event_flag"] = 0

    if week_mask is not None:
        m = week_mask.reindex(out["week"]).fillna(False).to_numpy()
        for col in metrics:
            if col in out.columns:
                out.loc[~m, col] = np.nan
        out.loc[~m, "event_flag"] = 0

    return out


def _add_event_bands(fig: go.Figure, weekly_show: pd.DataFrame, wmin: int, wmax: int) -> None:
    if weekly_show is None or weekly_show.empty or "event_flag" not in weekly_show.columns:
        return
    tmp = weekly_show.set_index("week")["event_flag"].reindex(range(wmin, wmax + 1)).fillna(0).astype(int)
    for w, f in tmp.items():
        if int(f) == 1:
            fig.add_vrect(x0=w - 0.5, x1=w + 0.5, opacity=0.12, line_width=0)


def _kpi_fig(
    weekly_full: pd.DataFrame,
    weekly_show: pd.DataFrame,
    metric: str,
    week_range: List[int],
    brush: dict | None = None,
) -> go.Figure:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    if weekly_full is None or weekly_full.empty or metric not in weekly_full.columns:
        return _empty_fig("KPI", "No data under current filters.")

    s_full = weekly_full.set_index("week")[metric].reindex(range(wmin, wmax + 1)).astype(float)
    if s_full.dropna().empty:
        return _empty_fig("KPI", "No data under current filters.")
    dfz_full = _robust_z(s_full, ROLL_WIN)

    s_show = (
        weekly_show.set_index("week")[metric].reindex(range(wmin, wmax + 1)).astype(float)
        if weekly_show is not None and not weekly_show.empty
        else s_full * np.nan
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfz_full.index, y=s_show, mode="lines+markers", name=f"{metric} (avg)", connectgaps=True))
    fig.add_trace(
        go.Scatter(x=dfz_full.index, y=dfz_full["baseline"], mode="lines", name=f"Baseline ({ROLL_WIN}w, median)", connectgaps=True)
    )
    fig.update_layout(
        template="plotly",
        height=320,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text=f"{metric}", x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.02, x=0.0),
        hovermode="x unified",
        dragmode="zoom",
        uirevision="incidents-v1",
    )
    _add_event_bands(fig, weekly_show, wmin, wmax)
    _format_week_axis(fig, wmin, wmax)
    fig.update_yaxes(automargin=True)

    if brush and brush.get("w0") is not None and brush.get("w1") is not None:
        _add_brush_band(fig, int(brush["w0"]), int(brush["w1"]))

    return fig


def _staff_fig(
    staff_full: pd.DataFrame,
    staff_show: pd.DataFrame,
    week_range: List[int],
    brush: dict | None = None,
) -> go.Figure:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    if staff_full is None or staff_full.empty or "staff_to_request" not in staff_full.columns:
        return _empty_fig("Staff", "No staff context under current filters.")

    s_full = staff_full.set_index("week")["staff_to_request"].reindex(range(wmin, wmax + 1)).astype(float)
    if s_full.dropna().empty:
        return _empty_fig("Staff", "No staff context under current filters.")
    dfz_full = _robust_z(s_full, ROLL_WIN)

    s_show = (
        staff_show.set_index("week")["staff_to_request"].reindex(range(wmin, wmax + 1)).astype(float)
        if staff_show is not None and (not staff_show.empty) and ("staff_to_request" in staff_show.columns)
        else s_full * np.nan
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dfz_full.index, y=s_show.replace(0, np.nan), mode="lines+markers", name="Staffing Ratio", connectgaps=True)
    )
    fig.add_trace(go.Scatter(x=dfz_full.index, y=dfz_full["baseline"], mode="lines", name=f"Baseline ({ROLL_WIN}w, mean)", connectgaps=True))
    fig.update_layout(
        template="plotly",
        height=320,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text="Staff context", x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.02, x=0.0),
        hovermode="x unified",
        dragmode="zoom",
        uirevision="incidents-v1",
    )
    _format_week_axis(fig, wmin, wmax)
    fig.update_yaxes(automargin=True, rangemode="tozero")

    if brush and brush.get("w0") is not None and brush.get("w1") is not None:
        _add_brush_band(fig, int(brush["w0"]), int(brush["w1"]))

    return fig


def _heatmap_fig(
    weekly_full: pd.DataFrame,
    week_mask: Optional[pd.Series],
    metrics: List[str],
    week_range: List[int],
    brush: dict | None = None,
) -> go.Figure:
    wmin, wmax = int(week_range[0]), int(week_range[1])
    weeks = list(range(wmin, wmax + 1))

    if weekly_full is None or weekly_full.empty:
        return _empty_fig("Fingerprint", "No data.")

    keep, mat = [], []
    for m in metrics:
        if m not in weekly_full.columns:
            continue
        s = weekly_full.set_index("week")[m].reindex(weeks).astype(float)
        if s.dropna().shape[0] < MIN_POINTS:
            continue
        z = _robust_z(s, ROLL_WIN)["z_raw"].reindex(weeks).astype(float)

        if week_mask is not None:
            mm = week_mask.reindex(weeks).fillna(False).to_numpy()
            z = z.where(mm, np.nan)

        if z.dropna().empty:
            continue
        keep.append(m)
        mat.append(z.fillna(0.0).clip(-Z_CLAMP, Z_CLAMP).values)

    if not keep:
        return _empty_fig("Fingerprint", "Not enough data under current filters.")

    zmat = np.vstack(mat)
    fig = go.Figure(
        data=go.Heatmap(
            z=zmat,
            x=weeks,
            y=keep,
            zmid=0,
            colorscale="RdBu",
            hovertemplate="Metric=%{y}<br>Week=%{x}<br>Z=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly",
        height=320,
        margin=dict(l=36, r=18, t=54, b=36),
        title=dict(text="Incident fingerprint (Z-score heatmap)", x=0.02, xanchor="left"),
        dragmode="zoom",
        uirevision="incidents-v1",
    )
    _format_week_axis(fig, wmin, wmax)
    fig.update_yaxes(automargin=True)

    if brush and brush.get("w0") is not None and brush.get("w1") is not None:
        _add_brush_band(fig, int(brush["w0"]), int(brush["w1"]))

    return fig


# =====================================================
# Layout
# =====================================================
def make_incidents_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Incidents", style={"fontSize": "18px", "fontWeight": 700}),
                    html.Div(
                        "Table is filtered by menu (week/service/event). Charts follow event-filter. "
                        "Brush time window by zooming any KPI chart; select incidents by box/lasso in scatter plots.",
                        style={"opacity": 0.7, "fontSize": "12px", "marginTop": "2px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div("Detected incidents", style={"fontWeight": 600, "fontSize": "14px"}),
                    html.Div(id="inc-summary", style={"opacity": 0.7, "fontSize": "12px"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
            ),
            dash_table.DataTable(
                id="incidents-table",
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Service", "id": "service"},
                    {"name": "Week", "id": "week"},
                    {"name": "Metric", "id": "metric"},
                    {"name": "Value", "id": "value", "type": "numeric"},
                    {"name": "Baseline", "id": "baseline", "type": "numeric"},
                    {"name": "Severity Z", "id": "severity", "type": "numeric"},
                    {"name": "Rarity", "id": "rarity"},
                    {"name": "Event", "id": "event_fact"},
                    {"name": "Staff to request", "id": "staff_to_request", "type": "numeric"},
                    {"name": "Staff Δ", "id": "staff_delta", "type": "numeric"},
                ],
                data=[],
                page_size=12,
                sort_action="native",
                row_selectable="single",
                style_table={"overflowX": "auto", "marginTop": "8px"},
                style_cell={"fontSize": "13px", "padding": "8px 10px", "whiteSpace": "nowrap"},
                style_header={"fontWeight": 600, "borderBottom": "1px solid #ddd"},
                style_data={"borderBottom": "1px solid #f3f3f3"},
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="inc-fig1", config={"displayModeBar": True, "displaylogo": False}),
                        style={"flex": "1 1 320px", "minWidth": "320px"},
                    ),
                    html.Div(
                        dcc.Graph(id="inc-fig2", config={"displayModeBar": True, "displaylogo": False}),
                        style={"flex": "1 1 320px", "minWidth": "320px"},
                    ),
                    html.Div(
                        dcc.Graph(id="inc-fig3", config={"displayModeBar": True, "displaylogo": False}),
                        style={"flex": "1 1 320px", "minWidth": "320px"},
                    ),
                    html.Div(
                        dcc.Graph(id="inc-fig-severity", config={"displayModeBar": True, "displaylogo": False}),
                        style={"flex": "1 1 100%", "minWidth": "320px"},
                    ),
                    html.Div(
                        dcc.Graph(id="inc-fig-service-event", config={"displayModeBar": True, "displaylogo": False}),
                        style={"flex": "1 1 100%", "minWidth": "320px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "stretch", "paddingTop": "10px"},
            ),
        ],
        style={"width": "100%"},
    )


def layout() -> html.Div:
    return make_incidents_panel()


# =====================================================
# Brush controller
# =====================================================
@callback(
    Output("global-store", "data", allow_duplicate=True),
    Input("inc-fig1", "relayoutData"),
    Input("inc-fig2", "relayoutData"),
    Input("inc-fig3", "relayoutData"),
    Input("inc-fig-severity", "relayoutData"),
    State("global-store", "data"),
    prevent_initial_call=True,
)
def incidents_time_brush(rel1, rel2, rel3, rel4, store):
    store = store or {}
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    relayout = {"inc-fig1": rel1, "inc-fig2": rel2, "inc-fig3": rel3, "inc-fig-severity": rel4}.get(trig)

    if not isinstance(relayout, dict) or not relayout:
        return no_update

    # reset axes -> clear brush
    if relayout.get("xaxis.autorange") is True:
        store.pop("brush", None)
        return store

    w0, w1, _ = _parse_brush_from_relayout(relayout)
    if w0 is None or w1 is None:
        return no_update

    store["brush"] = {"w0": int(w0), "w1": int(w1)}
    return store


@callback(
    Output("inc-summary", "children"),
    Output("incidents-table", "data"),
    Output("inc-fig1", "figure"),
    Output("inc-fig2", "figure"),
    Output("inc-fig3", "figure"),
    Output("inc-fig-severity", "figure"),
    Output("inc-fig-service-event", "figure"),
    Input("week-range", "value"),
    Input("service-filter", "value"),
    Input("event-filter", "value"),
    Input("global-store", "data"),
    Input("inc-fig-severity", "selectedData"),
    Input("inc-fig-service-event", "selectedData"),
)
def update_incidents(week_range: List[int], services: List[str], events: List[str], store, sel_sev, sel_evt):
    # effective time window = menu slider ∩ brushed range
    eff_week_range = _effective_week_range(week_range, store)

    if not services:
        services = list(getattr(config, "SERVICES_ORDER", []))
    events = events or []

    weekly_df, metrics = _build_weekly_table(_services_df, eff_week_range, services, events)
    staff_ps = _staff_per_service(_staff_week_service_df, eff_week_range, services)
    staff_ag_full = _staff_agg(_staff_week_service_df, eff_week_range, services)

    if weekly_df is None or weekly_df.empty or not metrics:
        return (
            "0 incident(s).",
            [],
            _empty_fig("KPI", ""),
            _empty_fig("Staff", ""),
            _empty_fig("Fingerprint", ""),
            _empty_fig("Severity", ""),
            _empty_fig("Service-Event", ""),
        )

    incidents_df = _detect_incidents(weekly_df, metrics, eff_week_range, staff_ps)

    # event filter already applied in weekly table; keep extra guard for incidents_df
    if events and (incidents_df is not None) and (not incidents_df.empty) and ("event_fact" in incidents_df.columns):
        ev_set = set(_norm_list(events))
        incidents_df = incidents_df[incidents_df["event_fact"].astype(str).str.strip().str.lower().isin(ev_set)]

    selected_ids = _selected_ids_from_points(sel_sev, sel_evt)

    # weekly aggregation + event masking for KPI visibility (same as your old logic)
    week_mask = _make_week_mask(weekly_df, events)
    weekly_full = _aggregate_weekly(weekly_df, metrics, eff_week_range, week_mask=None)
    weekly_show = _aggregate_weekly(weekly_df, metrics, eff_week_range, week_mask=week_mask)

    staff_ag_show = staff_ag_full.copy()
    if week_mask is not None and (staff_ag_show is not None) and (not staff_ag_show.empty) and ("week" in staff_ag_show.columns):
        mm = week_mask.reindex(staff_ag_show["week"]).fillna(False).to_numpy()
        if "staff_to_request" in staff_ag_show.columns:
            staff_ag_show.loc[~mm, "staff_to_request"] = np.nan

    brush = (store or {}).get("brush")

    primary = "shortage_rate" if "shortage_rate" in metrics else metrics[0]
    fig1 = _kpi_fig(weekly_full, weekly_show, primary, eff_week_range, brush=brush)
    fig2 = _staff_fig(staff_ag_full, staff_ag_show, eff_week_range, brush=brush)
    fig3 = _heatmap_fig(weekly_full, week_mask, metrics, eff_week_range, brush=brush)

    fig_severity = _severity_scatter_fig(incidents_df, eff_week_range, selected_ids, brush=brush)
    fig_service_event = _service_event_point_fig(incidents_df, selected_ids, brush=brush)

    # table brushing: filter rows to selected ids (if any)
    incidents_show = incidents_df
    if selected_ids:
        incidents_show = incidents_df[incidents_df["id"].isin(selected_ids)]

    # summary text
    w0, w1 = int(eff_week_range[0]), int(eff_week_range[1])
    if selected_ids:
        summary = f"{len(incidents_show)} selected / {len(incidents_df)} total incidents (weeks {w0}–{w1})."
    else:
        summary = f"{len(incidents_df)} incident(s) detected (UNUSUAL/RARE) (weeks {w0}–{w1})."

    return summary, incidents_show.to_dict("records"), fig1, fig2, fig3, fig_severity, fig_service_event


# =====================================================
# Table row selection -> global-store selection week (for cross-page linking)
# =====================================================
@callback(
    Output("global-store", "data", allow_duplicate=True),
    Input("incidents-table", "selected_rows"),
    State("incidents-table", "data"),
    State("global-store", "data"),
    prevent_initial_call=True,
)
def sync_selection_to_store(selected_rows, table_data, store):
    if not selected_rows or not table_data:
        return no_update
    store = store or {}
    store.setdefault("selection", store.get("selection", {}))
    try:
        row = table_data[selected_rows[0]]
        wk = int(str(row["week"]).replace("W", ""))
        store["selection"]["week"] = wk
        store["selection"]["service"] = row.get("service")
        return store
    except Exception:
        return no_update
