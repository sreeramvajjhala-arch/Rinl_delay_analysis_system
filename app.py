"""
RINL Visakhapatnam Steel Plant
AI-Based Delay Analysis & Predictive Maintenance System

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep this file, requirements.txt, and sample_delays_data.csv in the same folder.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

CSV_FILE = "sample_delays_data.csv"

TEXT_COLUMNS = [
    "SHOP_CODE", "MATERIAL", "RAKE", "EQPT", "SUB_EQPT", "REMARKS",
    "DELAY_DET_CODE", "AGENCY_CODE", "CONTINUED", "EXPECTED_DOC", "DELAY_ID",
]
NUMERIC_COLUMNS = ["DELAY_DURN", "EFF_DURATION", "CUM_DELAY", "DELAY_FREQ"]
DATE_COLUMNS = ["DEL_DATE", "DELAY_FROM", "DELAY_TO"]

PAGE_LIST = [
    "🏠 Executive Dashboard",
    "🏭 Shop Analysis",
    "⚙️ Equipment Analysis",
    "👷 Agency Analysis",
    "⏱️ Duration Analysis",
    "🔁 Conveyor Analysis",
    "📝 Delay Reasons",
    "🌧️ Season / Monsoon",
    "🤖 Predictive Maintenance",
    "📥 Download Report",
]

st.set_page_config(
    page_title="RINL Delay Analysis System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --rinl-navy: #061a33;
        --rinl-blue: #0b3a68;
        --rinl-mid: #1f5f99;
        --rinl-orange: #f58220;
        --rinl-grey: #64748b;
        --card-border: #d9e2ec;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(245,130,32,.12), transparent 28%),
            linear-gradient(180deg, #eef3f8 0%, #f8fafc 55%, #edf2f7 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061a33 0%, #0b315d 55%, #123f70 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stDateInput label {
        font-weight: 800 !important;
        letter-spacing: .2px;
    }

    .main .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    .rinl-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #061a33 0%, #0b3a68 58%, #f58220 125%);
        padding: 26px 30px;
        border-radius: 22px;
        margin-bottom: 22px;
        box-shadow: 0 18px 38px rgba(6, 26, 51, 0.22);
    }

    .rinl-hero:after {
        content: '';
        position: absolute;
        right: -80px;
        top: -90px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(255,255,255,.12);
    }

    .rinl-kicker {
        color: #ffcb91;
        font-size: 13px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-bottom: 8px;
    }

    .rinl-hero h1 {
        color: white;
        font-size: 34px;
        line-height: 1.12;
        margin: 0;
        font-weight: 900;
        letter-spacing: -0.5px;
    }

    .rinl-hero p {
        color: #e8eef5;
        font-size: 15.5px;
        margin: 10px 0 0 0;
        max-width: 900px;
    }

    .status-strip {
        background: rgba(255,255,255,.86);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 18px;
        color: #0f2742;
        box-shadow: 0 6px 22px rgba(15, 39, 66, .08);
    }

    .metric-card {
        background: rgba(255,255,255,.96);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        border: 1px solid #e2e8f0;
        border-top: 5px solid var(--rinl-orange);
        box-shadow: 0 10px 25px rgba(15, 39, 66, 0.08);
        min-height: 116px;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        transition: .16s ease;
        box-shadow: 0 14px 30px rgba(15, 39, 66, 0.12);
    }

    .metric-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .65px;
    }

    .metric-value {
        color: #061a33;
        font-size: 25px;
        line-height: 1.15;
        font-weight: 900;
        margin-top: 10px;
        word-break: break-word;
    }

    .section-title {
        color: #061a33;
        font-size: 23px;
        font-weight: 900;
        border-left: 7px solid var(--rinl-orange);
        padding-left: 13px;
        margin: 18px 0 14px 0;
        letter-spacing: -.2px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 12px 12px 0 0;
        padding: 10px 18px;
        border: 1px solid #dbe4ee;
        font-weight: 800;
    }

    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_white"
COLOR_SEQ = ["#0b3a68", "#1f5f99", "#f58220", "#64748b", "#22a06b", "#b42318", "#7c3aed"]


# ----------------------------- Data Pipeline -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    drop_cols = [c for c in df.columns if c == "" or c.startswith("UNNAMED")]
    return df.drop(columns=drop_cols, errors="ignore")


@st.cache_data(show_spinner="Loading delay data...")
def load_data(csv_file: str = CSV_FILE) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        st.error(f"{csv_file} not found. Keep app.py and {csv_file} in the same folder.")
        st.stop()

    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin1", low_memory=False)
    return normalize_columns(df)


@st.cache_data(show_spinner="Cleaning and engineering features...")
def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = (
            df[col]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN", "NaT": "UNKNOWN"})
        )

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    for col in DATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    if df["DEL_DATE"].notna().any():
        df = df[df["DEL_DATE"].notna()].copy()

    df = df.drop_duplicates().reset_index(drop=True)

    df["MONTH"] = df["DEL_DATE"].dt.month.fillna(0).astype(int)
    df["YEAR"] = df["DEL_DATE"].dt.year.fillna(0).astype(int)
    df["MONTH_NAME"] = df["DEL_DATE"].dt.month_name().fillna("Unknown")
    df["PERIOD"] = df["DEL_DATE"].dt.to_period("M").astype(str).replace("NaT", "Unknown")
    df["DAY_NAME"] = df["DEL_DATE"].dt.day_name().fillna("Unknown")

    season_conditions = [
        df["MONTH"].isin([3, 4, 5]),
        df["MONTH"].isin([6, 7, 8, 9]),
        df["MONTH"].isin([10, 11]),
        df["MONTH"].isin([12, 1, 2]),
    ]
    df["SEASON"] = np.select(season_conditions, ["Summer", "Monsoon", "Post-Monsoon", "Winter"], default="Unknown")

    text_blob = (
        df["EQPT"].astype(str) + " " + df["SUB_EQPT"].astype(str) + " " + df["REMARKS"].astype(str)
    ).str.upper()

    df["IS_CONVEYOR"] = text_blob.str.contains(r"CONVEYOR|\bCONV\b|BELT|BCN|TCN|PULLEY", regex=True, na=False)

    df["DELAY_CATEGORY"] = pd.cut(
        df["DELAY_DURN"],
        bins=[-0.1, 30, 120, 360, np.inf],
        labels=["Minor Delay", "Moderate Delay", "Major Delay", "Critical Delay"],
    ).astype(str)

    lower_blob = text_blob.str.lower()
    reason_rules = [
        ("Conveyor", r"conveyor|belt|\bconv\b|bcn|tcn|pulley"),
        ("Mechanical", r"motor|bearing|roller|gearbox|alignment|lubrication|mechanical"),
        ("Electrical", r"power|cable|trip|panel|sensor|starter|electrical|supply"),
        ("Material Handling", r"material|rake|loading|unloading|chute|hopper|blockage"),
        ("Operational", r"operation|operator|process|production|operational"),
        ("Agency Related", r"agency|contractor|maintenance delay"),
    ]
    df["DELAY_REASON_CATEGORY"] = "Other"
    for category, pattern in reason_rules:
        df.loc[df["DELAY_REASON_CATEGORY"].eq("Other") & lower_blob.str.contains(pattern, regex=True, na=False), "DELAY_REASON_CATEGORY"] = category

    score = np.zeros(len(df), dtype=np.int16)
    score += np.where(df["DELAY_DURN"] > 30, 10, 0)
    score += np.where(df["DELAY_DURN"] > 120, 20, 0)
    score += np.where(df["DELAY_DURN"] > 360, 30, 0)
    score += np.where(df["DELAY_FREQ"] >= 2, 10, 0)
    score += np.where(df["DELAY_FREQ"] >= 5, 20, 0)
    score += np.where(df["CUM_DELAY"] > 120, 10, 0)
    score += np.where(df["CUM_DELAY"] > 360, 20, 0)
    score += np.where(df["IS_CONVEYOR"], 10, 0)
    score += np.where(df["SEASON"].eq("Monsoon"), 10, 0)
    score += np.where(df["IS_CONVEYOR"] & df["SEASON"].eq("Monsoon"), 10, 0)
    df["RISK_SCORE"] = np.clip(score, 0, 100)

    df["RISK_LEVEL"] = pd.cut(
        df["RISK_SCORE"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
    ).astype(str)

    return df


def risk_level_from_score(score: float) -> str:
    if score < 25:
        return "Low Risk"
    if score < 50:
        return "Medium Risk"
    if score < 75:
        return "High Risk"
    return "Critical Risk"


def format_minutes(value: float) -> str:
    if value >= 60:
        return f"{value / 60:,.1f} hr"
    return f"{value:,.0f} min"


@st.cache_resource(show_spinner="Training predictive model...")
def train_prediction_model(df: pd.DataFrame):
    required = [
        "SHOP_CODE", "EQPT", "SUB_EQPT", "AGENCY_CODE", "MONTH", "SEASON",
        "DELAY_FREQ", "DELAY_DURN", "EFF_DURATION", "IS_CONVEYOR", "RISK_LEVEL",
    ]
    if not set(required).issubset(df.columns) or len(df) < 50 or df["RISK_LEVEL"].nunique() < 2:
        return None, "Rule-based scoring active: dataset is too small or has only one risk class."

    sample_df = df[required].copy()
    if len(sample_df) > 30000:
        sample_df = sample_df.sample(30000, random_state=42)

    X = sample_df.drop(columns=["RISK_LEVEL"])
    y = sample_df["RISK_LEVEL"]
    categorical = ["SHOP_CODE", "EQPT", "SUB_EQPT", "AGENCY_CODE", "SEASON"]
    numeric = ["MONTH", "DELAY_FREQ", "DELAY_DURN", "EFF_DURATION", "IS_CONVEYOR"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
                ("num", "passthrough", numeric),
            ]
        )
        model = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )),
        ])
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return model, f"Random Forest trained successfully. Validation accuracy: {accuracy:.2%}."
    except Exception as exc:
        return None, f"Rule-based scoring active. ML training could not complete: {exc}"


# ----------------------------- UI Helpers -----------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="rinl-hero">
            <div class="rinl-kicker">Industrial AI Dashboard • RINL / Vizag Steel</div>
            <h1>AI-Based Delay Analysis & Predictive Maintenance System</h1>
            <p>Control-room style analytics for delay duration, shop impact, equipment reliability, agency responsibility, seasonal risk and maintenance prioritization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def metric_card(title: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def show_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    total_duration = float(df["DELAY_DURN"].sum()) if total else 0
    avg_duration = float(df["DELAY_DURN"].mean()) if total else 0
    critical_count = int(df["RISK_LEVEL"].isin(["High Risk", "Critical Risk"]).sum()) if total else 0
    conveyor_count = int(df["IS_CONVEYOR"].sum()) if total else 0
    affected_equipment = int(df["EQPT"].nunique()) if total else 0
    top_shop = df.groupby("SHOP_CODE")["DELAY_DURN"].sum().idxmax() if total else "N/A"
    top_equipment = df.groupby("EQPT")["DELAY_DURN"].sum().idxmax() if total else "N/A"
    top_agency = df.groupby("AGENCY_CODE")["DELAY_DURN"].sum().idxmax() if total else "N/A"

    data = [
        ("Delay Records", f"{total:,}"),
        ("Total Delay", format_minutes(total_duration)),
        ("Average Delay", format_minutes(avg_duration)),
        ("High / Critical Cases", f"{critical_count:,}"),
        ("Conveyor Cases", f"{conveyor_count:,}"),
        ("Affected Equipment", f"{affected_equipment:,}"),
        ("Top Shop", str(top_shop)),
        ("Top Agency", str(top_agency)),
    ]
    for start in range(0, len(data), 4):
        cols = st.columns(4)
        for col, (title, value) in zip(cols, data[start:start + 4]):
            col.markdown(metric_card(title, value), unsafe_allow_html=True)


def clean_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        colorway=COLOR_SEQ,
        height=430,
        margin=dict(l=15, r=15, t=58, b=20),
        title_font=dict(size=18, family="Inter, Segoe UI", color="#061a33"),
        font=dict(family="Inter, Segoe UI", size=12, color="#263648"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, title_font=dict(size=12), tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#e2e8f0", title_font=dict(size=12), tickfont=dict(size=11))
    return fig


def plot_bar(data: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, horizontal: bool = False) -> None:
    if data.empty:
        st.info("No data available for this chart.")
        return
    if horizontal:
        fig = px.bar(data, x=y, y=x, color=color, orientation="h", title=title, text_auto=".2s")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(data, x=x, y=y, color=color, title=title, text_auto=".2s")
    st.plotly_chart(clean_fig(fig), use_container_width=True)


def plot_line(data: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if data.empty:
        st.info("No data available for this chart.")
        return
    fig = px.line(data, x=x, y=y, color=color, markers=True, title=title)
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    st.plotly_chart(clean_fig(fig), use_container_width=True)


def plot_pie(data: pd.DataFrame, names: str, values: str, title: str) -> None:
    if data.empty:
        st.info("No data available for this chart.")
        return
    fig = px.pie(data, names=names, values=values, title=title, hole=0.45, color_discrete_sequence=COLOR_SEQ)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(clean_fig(fig), use_container_width=True)


def plot_heatmap(data: pd.DataFrame, x: str, y: str, z: str, title: str) -> None:
    if data.empty:
        st.info("No data available for this heatmap.")
        return
    pivot = data.pivot_table(index=y, columns=x, values=z, aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, text_auto=".2s", aspect="auto", title=title, color_continuous_scale="Blues")
    st.plotly_chart(clean_fig(fig), use_container_width=True)


def empty_guard(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("No records found for the selected filters.")
        return True
    return False


def compact_group(df: pd.DataFrame, group_cols: Iterable[str], top_n: int = 20) -> pd.DataFrame:
    return (
        df.groupby(list(group_cols), dropna=False, as_index=False)
        .agg(
            DELAY_COUNT=("DELAY_DURN", "count"),
            TOTAL_DELAY=("DELAY_DURN", "sum"),
            AVG_DELAY=("DELAY_DURN", "mean"),
            AVG_RISK_SCORE=("RISK_SCORE", "mean"),
        )
        .sort_values("TOTAL_DELAY", ascending=False)
        .head(top_n)
    )


def get_recommendation(row: pd.Series) -> str:
    risk = str(row.get("RISK_LEVEL", ""))
    reason = str(row.get("DELAY_REASON_CATEGORY", ""))
    season = str(row.get("SEASON", ""))
    is_conveyor = bool(row.get("IS_CONVEYOR", False))

    if is_conveyor and season == "Monsoon" and risk in ["High Risk", "Critical Risk"]:
        return "Priority monsoon inspection: belt alignment, rollers, pulleys, motor load and lubrication."
    if reason == "Mechanical":
        return "Inspect bearings, gearbox, motor health, lubrication and alignment."
    if reason == "Electrical":
        return "Check supply quality, panels, cables, sensors, starters and trip logs."
    if reason == "Material Handling":
        return "Inspect chute, hopper, loading/unloading flow and blockage points."
    if risk in ["High Risk", "Critical Risk"]:
        return "Schedule preventive maintenance and monitor repeated delay frequency."
    return "Continue routine monitoring and preventive inspection."


# ----------------------------- Sidebar -----------------------------
def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown("# 🏭 RINL")
    st.sidebar.markdown("Delay Intelligence System")
    page = st.sidebar.radio("Navigation", PAGE_LIST, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    filtered = df.copy()
    if filtered["DEL_DATE"].notna().any():
        min_date = filtered["DEL_DATE"].min().date()
        max_date = filtered["DEL_DATE"].max().date()
        selected_range = st.sidebar.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            filtered = filtered[
                (filtered["DEL_DATE"].dt.date >= start_date) &
                (filtered["DEL_DATE"].dt.date <= end_date)
            ]

    with st.sidebar.expander("Plant filters", expanded=True):
        for label, column in [("Shop", "SHOP_CODE"), ("Equipment", "EQPT"), ("Sub-equipment", "SUB_EQPT"), ("Agency", "AGENCY_CODE"), ("Season", "SEASON"), ("Risk level", "RISK_LEVEL")]:
            if column in filtered.columns:
                options = sorted(filtered[column].astype(str).dropna().unique().tolist())
                selected = st.multiselect(label, options, default=[])
                if selected:
                    filtered = filtered[filtered[column].astype(str).isin(selected)]

    st.sidebar.markdown("---")
    st.sidebar.caption("Optimized Streamlit prototype for RINL internship project.")
    return page, filtered


# ----------------------------- Pages -----------------------------
def executive_dashboard(df: pd.DataFrame) -> None:
    section("Executive Dashboard")
    show_kpis(df)
    if empty_guard(df):
        return

    tab1, tab2, tab3 = st.tabs(["Overview", "Risk", "Heatmaps"])

    with tab1:
        monthly = df.groupby("PERIOD", as_index=False).agg(TOTAL_DELAY=("DELAY_DURN", "sum"), DELAY_COUNT=("DELAY_DURN", "count"))
        monthly = monthly.sort_values("PERIOD")
        c1, c2 = st.columns(2)
        with c1:
            plot_line(monthly, "PERIOD", "TOTAL_DELAY", "Monthly Delay Trend")
        with c2:
            category = compact_group(df, ["DELAY_CATEGORY"], 10)
            plot_pie(category, "DELAY_CATEGORY", "TOTAL_DELAY", "Delay Duration Mix")

        c3, c4 = st.columns(2)
        with c3:
            shops = compact_group(df, ["SHOP_CODE"], 10)
            plot_bar(shops, "SHOP_CODE", "TOTAL_DELAY", "Top Shops by Delay Duration", horizontal=True)
        with c4:
            eqpt = compact_group(df, ["EQPT"], 10)
            plot_bar(eqpt, "EQPT", "TOTAL_DELAY", "Top Equipment by Delay Duration", horizontal=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            risk = compact_group(df, ["RISK_LEVEL"], 10)
            plot_bar(risk, "RISK_LEVEL", "DELAY_COUNT", "Risk Level Distribution")
        with c2:
            reason = compact_group(df, ["DELAY_REASON_CATEGORY"], 10)
            plot_pie(reason, "DELAY_REASON_CATEGORY", "TOTAL_DELAY", "Delay Reason Contribution")

        pareto = compact_group(df, ["EQPT"], 15).sort_values("TOTAL_DELAY", ascending=False)
        pareto["CUMULATIVE_PERCENT"] = pareto["TOTAL_DELAY"].cumsum() / pareto["TOTAL_DELAY"].sum() * 100
        fig = go.Figure()
        fig.add_bar(x=pareto["EQPT"], y=pareto["TOTAL_DELAY"], name="Total delay")
        fig.add_scatter(x=pareto["EQPT"], y=pareto["CUMULATIVE_PERCENT"], name="Cumulative %", yaxis="y2", mode="lines+markers")
        fig.update_layout(title="Pareto Analysis: Equipment Delay Contribution", yaxis_title="Delay minutes", yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]))
        st.plotly_chart(clean_fig(fig), use_container_width=True)

    with tab3:
        heat = df.groupby(["SHOP_CODE", "SEASON"], as_index=False)["DELAY_DURN"].sum()
        plot_heatmap(heat, "SEASON", "SHOP_CODE", "DELAY_DURN", "Shop vs Season Delay Heatmap")
        daily = df.groupby(["DAY_NAME", "RISK_LEVEL"], as_index=False)["DELAY_DURN"].sum()
        plot_heatmap(daily, "RISK_LEVEL", "DAY_NAME", "DELAY_DURN", "Day vs Risk Delay Heatmap")


def shop_analysis(df: pd.DataFrame) -> None:
    section("Shop-wise Delay Analysis")
    if empty_guard(df):
        return
    summary = compact_group(df, ["SHOP_CODE"], 50)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(summary.head(15), "SHOP_CODE", "TOTAL_DELAY", "Shop-wise Total Delay", horizontal=True)
    with c2:
        plot_bar(summary.sort_values("DELAY_COUNT", ascending=False).head(15), "SHOP_CODE", "DELAY_COUNT", "Shop-wise Delay Count", horizontal=True)

    top_shops = summary.head(5)["SHOP_CODE"].astype(str).tolist()
    monthly_shop = df[df["SHOP_CODE"].astype(str).isin(top_shops)].groupby(["PERIOD", "SHOP_CODE"], as_index=False)["DELAY_DURN"].sum()
    plot_line(monthly_shop.sort_values("PERIOD"), "PERIOD", "DELAY_DURN", "Monthly Trend of Top Shops", "SHOP_CODE")


def equipment_analysis(df: pd.DataFrame) -> None:
    section("Equipment-wise Delay Analysis")
    if empty_guard(df):
        return
    eqpt = compact_group(df, ["EQPT"], 80)
    eqpt["RISK_LEVEL"] = eqpt["AVG_RISK_SCORE"].apply(risk_level_from_score)
    sub = compact_group(df, ["EQPT", "SUB_EQPT"], 100)
    sub["RISK_LEVEL"] = sub["AVG_RISK_SCORE"].apply(risk_level_from_score)

    c1, c2 = st.columns(2)
    with c1:
        plot_bar(eqpt.head(15), "EQPT", "TOTAL_DELAY", "Top Equipment by Total Delay", horizontal=True)
    with c2:
        plot_bar(eqpt.sort_values("AVG_RISK_SCORE", ascending=False).head(15), "EQPT", "AVG_RISK_SCORE", "Top Equipment by Risk Score", "RISK_LEVEL", True)

    st.markdown("#### Equipment Risk Table")
    st.dataframe(eqpt.head(80), use_container_width=True, hide_index=True)
    st.markdown("#### Sub-equipment Summary")
    st.dataframe(sub.head(100), use_container_width=True, hide_index=True)


def agency_analysis(df: pd.DataFrame) -> None:
    section("Agency-wise Delay Analysis")
    if empty_guard(df):
        return
    summary = compact_group(df, ["AGENCY_CODE"], 50)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(summary.head(15), "AGENCY_CODE", "TOTAL_DELAY", "Agency-wise Total Delay", horizontal=True)
    with c2:
        plot_bar(summary.sort_values("DELAY_COUNT", ascending=False).head(15), "AGENCY_CODE", "DELAY_COUNT", "Agency-wise Delay Count", horizontal=True)


def duration_analysis(df: pd.DataFrame) -> None:
    section("Duration-wise Delay Analysis")
    if empty_guard(df):
        return
    summary = compact_group(df, ["DELAY_CATEGORY"], 10)
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(summary, "DELAY_CATEGORY", "DELAY_COUNT", "Delay Count by Duration Category")
    with c2:
        plot_pie(summary, "DELAY_CATEGORY", "TOTAL_DELAY", "Duration Contribution by Category")

    st.dataframe(summary, use_container_width=True, hide_index=True)
    critical = df[df["DELAY_CATEGORY"].eq("Critical Delay")].sort_values("DELAY_DURN", ascending=False)
    st.markdown("#### Critical Delay Records")
    cols = ["DEL_DATE", "SHOP_CODE", "EQPT", "SUB_EQPT", "AGENCY_CODE", "DELAY_DURN", "CUM_DELAY", "REMARKS", "RISK_LEVEL"]
    st.dataframe(critical[[c for c in cols if c in critical.columns]].head(100), use_container_width=True, hide_index=True)


def conveyor_analysis(df: pd.DataFrame) -> None:
    section("Conveyor-wise Delay Analysis")
    if empty_guard(df):
        return
    conv = df[df["IS_CONVEYOR"]].copy()
    if conv.empty:
        st.info("No conveyor-related delays found for the selected filters.")
        return
    show_kpis(conv)
    summary = compact_group(conv, ["SHOP_CODE", "EQPT", "SUB_EQPT"], 80)
    summary["RISK_LEVEL"] = summary["AVG_RISK_SCORE"].apply(risk_level_from_score)

    c1, c2 = st.columns(2)
    with c1:
        plot_bar(summary.head(20), "EQPT", "TOTAL_DELAY", "Top Conveyor Equipment by Delay", horizontal=True)
    with c2:
        season = compact_group(conv, ["SEASON"], 10)
        plot_pie(season, "SEASON", "TOTAL_DELAY", "Conveyor Delay by Season")

    st.dataframe(summary.head(100), use_container_width=True, hide_index=True)
    st.markdown("#### Conveyor Delay Records")
    cols = ["DEL_DATE", "SHOP_CODE", "EQPT", "SUB_EQPT", "DELAY_DURN", "REMARKS", "SEASON", "RISK_LEVEL"]
    st.dataframe(conv[[c for c in cols if c in conv.columns]].head(150), use_container_width=True, hide_index=True)


def description_analysis(df: pd.DataFrame) -> None:
    section("Delay Description / Reason Analysis")
    if empty_guard(df):
        return
    reason = compact_group(df, ["DELAY_REASON_CATEGORY"], 20)
    code = compact_group(df, ["DELAY_DET_CODE"], 20)

    c1, c2 = st.columns(2)
    with c1:
        plot_pie(reason, "DELAY_REASON_CATEGORY", "TOTAL_DELAY", "Reason Category Distribution")
    with c2:
        plot_bar(code.head(15), "DELAY_DET_CODE", "DELAY_COUNT", "Top Delay Detail Codes", horizontal=True)

    remarks = (
        df[df["REMARKS"].ne("UNKNOWN")]
        .groupby("REMARKS", as_index=False)
        .agg(DELAY_COUNT=("DELAY_DURN", "count"), TOTAL_DELAY=("DELAY_DURN", "sum"))
        .sort_values("DELAY_COUNT", ascending=False)
        .head(60)
    )
    st.markdown("#### Most Common Delay Remarks")
    st.dataframe(remarks, use_container_width=True, hide_index=True)


def season_analysis(df: pd.DataFrame) -> None:
    section("Season / Monsoon Analysis")
    if empty_guard(df):
        return
    season = compact_group(df, ["SEASON"], 10)
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(season, "SEASON", "TOTAL_DELAY", "Season-wise Total Delay")
    with c2:
        plot_pie(season, "SEASON", "DELAY_COUNT", "Season-wise Delay Count")

    monsoon = df[df["SEASON"].eq("Monsoon")].copy()
    st.markdown("#### Monsoon Hotspots")
    c3, c4 = st.columns(2)
    with c3:
        monsoon_shop = compact_group(monsoon, ["SHOP_CODE"], 15)
        plot_bar(monsoon_shop, "SHOP_CODE", "TOTAL_DELAY", "Monsoon Shop-wise Delay", horizontal=True)
    with c4:
        monsoon_eq = compact_group(monsoon, ["EQPT"], 15)
        plot_bar(monsoon_eq, "EQPT", "AVG_RISK_SCORE", "Monsoon Equipment Risk", horizontal=True)

    likely = compact_group(monsoon[monsoon["IS_CONVEYOR"]], ["SHOP_CODE", "EQPT", "SUB_EQPT"], 50)
    likely["RISK_LEVEL"] = likely["AVG_RISK_SCORE"].apply(risk_level_from_score)
    st.markdown("#### Likely Conveyor Delays During Monsoon")
    st.dataframe(likely, use_container_width=True, hide_index=True)


def predictive_maintenance(df: pd.DataFrame) -> None:
    section("Predictive Maintenance")
    if empty_guard(df):
        return

    model, status = train_prediction_model(df)
    st.info(status)

    equipment_risk = (
        df.groupby(["SHOP_CODE", "EQPT", "SUB_EQPT"], as_index=False)
        .agg(
            DELAY_FREQUENCY=("DELAY_FREQ", "sum"),
            DELAY_COUNT=("DELAY_DURN", "count"),
            TOTAL_DELAY_DURATION=("DELAY_DURN", "sum"),
            AVG_DELAY_DURATION=("DELAY_DURN", "mean"),
            AVG_RISK_SCORE=("RISK_SCORE", "mean"),
            CONVEYOR_CASES=("IS_CONVEYOR", "sum"),
        )
        .sort_values(["AVG_RISK_SCORE", "TOTAL_DELAY_DURATION"], ascending=False)
    )
    equipment_risk["RISK_LEVEL"] = equipment_risk["AVG_RISK_SCORE"].apply(risk_level_from_score)

    rep = df.sort_values("RISK_SCORE", ascending=False).drop_duplicates(["SHOP_CODE", "EQPT", "SUB_EQPT"])
    rep = rep[["SHOP_CODE", "EQPT", "SUB_EQPT", "DELAY_REASON_CATEGORY", "SEASON", "IS_CONVEYOR", "RISK_LEVEL"]]
    equipment_risk = equipment_risk.merge(rep, on=["SHOP_CODE", "EQPT", "SUB_EQPT"], how="left", suffixes=("", "_REP"))
    equipment_risk["MAINTENANCE_RECOMMENDATION"] = equipment_risk.apply(get_recommendation, axis=1)

    c1, c2 = st.columns(2)
    with c1:
        plot_bar(equipment_risk.head(20), "EQPT", "AVG_RISK_SCORE", "Top Predictive Risk Equipment", "RISK_LEVEL", True)
    with c2:
        risk_dist = compact_group(df, ["RISK_LEVEL"], 10)
        plot_pie(risk_dist, "RISK_LEVEL", "DELAY_COUNT", "Maintenance Risk Distribution")

    st.markdown("#### High Priority Equipment Risk List")
    display_cols = [
        "SHOP_CODE", "EQPT", "SUB_EQPT", "DELAY_FREQUENCY", "DELAY_COUNT",
        "TOTAL_DELAY_DURATION", "AVG_DELAY_DURATION", "AVG_RISK_SCORE", "RISK_LEVEL",
        "MAINTENANCE_RECOMMENDATION",
    ]
    st.dataframe(equipment_risk[display_cols].head(120), use_container_width=True, hide_index=True)

    st.markdown("#### Single Equipment Check")
    c1, c2, c3 = st.columns(3)
    with c1:
        shop = st.selectbox("Shop", sorted(df["SHOP_CODE"].astype(str).unique()))
    subset = df[df["SHOP_CODE"].astype(str).eq(str(shop))]
    with c2:
        eqpt = st.selectbox("Equipment", sorted(subset["EQPT"].astype(str).unique()))
    subset2 = subset[subset["EQPT"].astype(str).eq(str(eqpt))]
    with c3:
        sub = st.selectbox("Sub-equipment", sorted(subset2["SUB_EQPT"].astype(str).unique()))

    selected = subset2[subset2["SUB_EQPT"].astype(str).eq(str(sub))]
    if not selected.empty:
        avg_score = float(selected["RISK_SCORE"].mean())
        risk = risk_level_from_score(avg_score)
        representative = selected.sort_values("RISK_SCORE", ascending=False).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", risk)
        m2.metric("Avg Risk Score", f"{avg_score:.1f}/100")
        m3.metric("Total Delay", format_minutes(selected["DELAY_DURN"].sum()))
        m4.metric("Delay Count", f"{len(selected):,}")
        st.success(get_recommendation(representative))


def create_management_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df.groupby(["SHOP_CODE", "EQPT", "SUB_EQPT", "AGENCY_CODE", "SEASON", "DELAY_REASON_CATEGORY"], dropna=False, as_index=False)
        .agg(
            DELAY_COUNT=("DELAY_DURN", "count"),
            TOTAL_DELAY_DURATION=("DELAY_DURN", "sum"),
            AVG_DELAY_DURATION=("DELAY_DURN", "mean"),
            TOTAL_DELAY_FREQUENCY=("DELAY_FREQ", "sum"),
            AVG_RISK_SCORE=("RISK_SCORE", "mean"),
        )
        .sort_values("TOTAL_DELAY_DURATION", ascending=False)
    )
    report["RISK_LEVEL"] = report["AVG_RISK_SCORE"].apply(risk_level_from_score)
    return report


def download_report_page(df: pd.DataFrame) -> None:
    section("Download Report")
    if empty_guard(df):
        return
    report = create_management_report(df)
    st.markdown("#### Filtered Management Summary")
    st.dataframe(report.head(250), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Summary Report CSV",
        report.to_csv(index=False).encode("utf-8"),
        "rinl_delay_summary_report.csv",
        "text/csv",
    )
    st.download_button(
        "Download Filtered Raw Data CSV",
        df.to_csv(index=False).encode("utf-8"),
        "rinl_filtered_delay_data.csv",
        "text/csv",
    )


# ----------------------------- Main -----------------------------
def main() -> None:
    render_header()
    raw = load_data()
    df = prepare_data(raw)

    required = ["DEL_DATE", "SHOP_CODE", "EQPT", "SUB_EQPT", "DELAY_DURN", "AGENCY_CODE", "REMARKS"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        st.warning("Expected columns missing from CSV: " + ", ".join(missing))

    page, filtered = sidebar_filters(df)

    period_text = "N/A"
    if len(filtered) and filtered["DEL_DATE"].notna().any():
        period_text = f"{filtered['DEL_DATE'].min().date()} to {filtered['DEL_DATE'].max().date()}"

    st.markdown(
        f"""
        <div class="status-strip">
            <b>Active Records:</b> {len(filtered):,} &nbsp; | &nbsp;
            <b>Data Period:</b> {period_text} &nbsp; | &nbsp;
            <b>Current View:</b> {page}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "🏠 Executive Dashboard":
        executive_dashboard(filtered)
    elif page == "🏭 Shop Analysis":
        shop_analysis(filtered)
    elif page == "⚙️ Equipment Analysis":
        equipment_analysis(filtered)
    elif page == "👷 Agency Analysis":
        agency_analysis(filtered)
    elif page == "⏱️ Duration Analysis":
        duration_analysis(filtered)
    elif page == "🔁 Conveyor Analysis":
        conveyor_analysis(filtered)
    elif page == "📝 Delay Reasons":
        description_analysis(filtered)
    elif page == "🌧️ Season / Monsoon":
        season_analysis(filtered)
    elif page == "🤖 Predictive Maintenance":
        predictive_maintenance(filtered)
    elif page == "📥 Download Report":
        download_report_page(filtered)


if __name__ == "__main__":
    main()
