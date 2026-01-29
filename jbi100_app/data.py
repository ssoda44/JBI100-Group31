from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from. import config


@dataclass(frozen=True)
class DataBundle:
    services_df: pd.DataFrame
    staff_week_service_role_df: pd.DataFrame
    staff_week_service_df: pd.DataFrame
    patients_df: pd.DataFrame


def _read_csv(path) -> pd.DataFrame:
    path = str(path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find data file: {path}\n"
            f"Check paths in config.py (DATA_DIR / *_PATH)."
        ) from e


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return numer / denom


def _add_patient_week_bins(patients_df: pd.DataFrame) -> pd.DataFrame:
    df = patients_df.copy()

    # Dates
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

    # Core numeric
    df["age"] = _to_float(df["age"])
    df["satisfaction"] = _to_float(df["satisfaction"])
    df["stay_duration"] = _to_float(df["stay_duration"])

    # ISO week from arrival_date (1–53); keep as int
    # If arrival_date missing, week will be NA.
    iso_week = df["arrival_date"].dt.isocalendar().week # type: ignore
    df["week"] = iso_week.astype("Int64")

    # age_bin: 0–17 / 18–39 / 40–64 / 65+
    age_bins = [-np.inf, 17, 39, 64, np.inf]
    age_labels = ["0–17", "18–39", "40–64", "65+"]
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    # los_bin: 0–2 / 3–5 / 6–10 / 11+
    los_bins = [-np.inf, 2, 5, 10, np.inf]
    los_labels = ["0–2", "3–5", "6–10", "11+"]
    df["los_bin"] = pd.cut(df["stay_duration"], bins=los_bins, labels=los_labels, right=True)

    return df


def _prep_services(services_df: pd.DataFrame) -> pd.DataFrame:
    df = services_df.copy()

    # Types
    df["week"] = _to_int(df["week"])
    df["month"] = _to_int(df["month"])
    df["service"] = df["service"].astype(str)

    # Core metrics (already present, but normalize types)
    for col in [
        "available_beds",
        "patients_request",
        "patients_admitted",
        "patients_refused",
        "patient_satisfaction",
        "staff_morale",
        "bed_pressure",
        "shortage_rate",
        "utilization",
    ]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    # Event normalization
    if "event" in df.columns:
        df["event"] = df["event"].fillna("none").astype(str)
    else:
        df["event"] = "none"

    # Derived helpers for UI
    if "available_beds" in df.columns and "patients_request" in df.columns:
        df["capacity_delta"] = df["available_beds"] - df["patients_request"]
    else:
        df["capacity_delta"] = np.nan

    # Default incident flag (threshold can be changed in UI later; this is a convenient default)
    thr = getattr(config, "DEFAULT_SHORTAGE_THRESHOLD", 0.2)
    if "patients_refused" in df.columns and "shortage_rate" in df.columns:
        df["is_incident_default"] = (df["patients_refused"] > 0) | (df["shortage_rate"] >= thr)
    else:
        df["is_incident_default"] = False

    # Keep consistent ordering keys
    df = df.sort_values(["service", "week"], kind="mergesort").reset_index(drop=True)

    return df


def _prep_staff_schedule(schedule_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = schedule_df.copy()

    df["week"] = _to_int(df["week"])
    df["service"] = df["service"].astype(str)
    df["role"] = df["role"].astype(str)

    # present should be 0/1
    df["present"] = _to_int(df["present"]).fillna(0).astype(int)

    # week × service × role
    staff_week_service_role = (
        df.groupby(["week", "service", "role"], as_index=False)["present"]
        .sum()
        .rename(columns={"present": "staff_present_count"}) # type: ignore
        .sort_values(["service", "week", "role"], kind="mergesort")
        .reset_index(drop=True)
    )

    # week × service (total)
    staff_week_service = (
        df.groupby(["week", "service"], as_index=False)["present"]
        .sum()
        .rename({"present": "staff_present_total"}, axis=1) # type: ignore
        .sort_values(["service", "week"], kind="mergesort")
        .reset_index(drop=True)
    )

    return staff_week_service_role, staff_week_service


def get_data() -> DataBundle:
    """
    Loads and preprocesses cleaned datasets into a stable set of DataFrames
    that your Dash views can directly consume.

    Outputs:
      - services_df: service×week master table (shortage_rate is main metric)
      - staff_week_service_role_df: week×service×role present counts
      - staff_week_service_df: week×service total present counts + ratios
      - patients_df: patient-level table with derived week + bins
    """
    # --- Read data ---
    patients_raw = _read_csv(config.PATIENTS_CLEANED_PATH)
    services_raw = _read_csv(config.SERVICES_CLEANED_PATH)
    staff_schedule_raw = _read_csv(config.STAFF_SCHEDULE_CLEANED_PATH)

    # staff_cleaned is optional (Phase 1); ignore by default
    # staff_raw = _read_csv(config.STAFF_CLEANED_PATH)

    # --- Preprocess ---
    services_df = _prep_services(services_raw)
    patients_df = _add_patient_week_bins(patients_raw)

    staff_week_service_role_df, staff_week_service_df = _prep_staff_schedule(staff_schedule_raw)

    # --- Add staffing ratios (join to services) ---
    # Use services' demand figures as denominator; avoid divide by zero.
    join_cols = ["week", "service"]
    tmp = services_df[join_cols + ["patients_request", "patients_admitted"]].copy()

    staff_week_service_df = staff_week_service_df.merge(tmp, on=join_cols, how="left")

    staff_week_service_df["staff_to_request"] = _safe_div(
        staff_week_service_df["staff_present_total"], staff_week_service_df["patients_request"]
    )
    staff_week_service_df["staff_to_admitted"] = _safe_div(
        staff_week_service_df["staff_present_total"], staff_week_service_df["patients_admitted"]
    )

    return DataBundle(
        services_df=services_df,
        staff_week_service_role_df=staff_week_service_role_df,
        staff_week_service_df=staff_week_service_df,
        patients_df=patients_df,
    )
