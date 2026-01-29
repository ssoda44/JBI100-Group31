# jb100_app/config.py
from pathlib import Path

# =========================
# Paths
# =========================

DATA_DIR = Path(r"C:\Users\19670\Desktop\JBI100-Group31\Hospital Beds Management")

PATIENTS_CLEANED_PATH = DATA_DIR / "patients_cleaned.csv"
SERVICES_CLEANED_PATH = DATA_DIR / "services_cleaned.csv"
STAFF_CLEANED_PATH = DATA_DIR / "staff_cleaned.csv"          # Phase 1 可不用
STAFF_SCHEDULE_CLEANED_PATH = DATA_DIR / "staff_schedule_cleaned.csv"


# =========================
# Core dashboard parameters
# =========================
DEFAULT_SHORTAGE_THRESHOLD = 0.2
TOP_N_INCIDENTS = 10


# =========================
# Domain orderings (for plots)
# =========================
SERVICES_ORDER = [
    "ICU",
    "emergency",
    "general_medicine",
    "surgery"
]

EVENT_ORDER = [
    "none",
    "flu",
    "donation",
    "strike"
]

ROLE_ORDER = [
    "doctor",
    "nurse",
    "assistant"
]


# =========================
# Color schemes (semantic!)
# =========================

# Service colors
SERVICE_COLORS = {
    "icu": "#1f77b4",              
    "emergency": "#d62728",        
    "general_medicine": "#2ca02c", 
    "surgery": "#9467bd"           #
}

# Event colors
EVENT_COLORS = {
    "none": "#bdbdbd",
    "flu": "#ff7f0e",
    "donation": "#8c564b",
    "strike": "#e377c2"
}

# Staff role colors（T3）
ROLE_COLORS = {
    "doctor": "#1f77b4",
    "nurse": "#ff7f0e",
    "assistant": "#2ca02c"
}

# Sequential scale for shortage_rate (low → high)
SHORTAGE_COLOR_SCALE = [
    "#edf8fb",
    "#b3cde3",
    "#8c96c6",
    "#8856a7",
    "#810f7c"
]


# =========================
# Binning definitions
# =========================
AGE_BINS = [0, 17, 39, 64, 120]
AGE_BIN_LABELS = ["0–17", "18–39", "40–64", "65+"]

LOS_BINS = [0, 2, 5, 10, 365]
LOS_BIN_LABELS = ["0–2", "3–5", "6–10", "11+"]


# =========================
# Text / UI labels
# =========================
LAYER_LABELS = {
    "overview": "Overview · Capacity Risk",
    "incidents": "Incidents · Abnormal Weeks",
    "diagnosis": "Diagnosis · Supply & Staffing",
    "impact": "Impact · Patient Experience"
}
