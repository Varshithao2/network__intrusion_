"""
app.py
-------
Streamlit dashboard for the ML-Based Network Intrusion Detection and
Security Monitoring System.

Styled as a dark-mode SOC/SIEM operations console:
  - Top header bar (branding + live status)
  - 5-tab navigation: OVERVIEW, TRAFFIC ANALYSIS, THREAT DETECTION,
    MODEL PERFORMANCE, SECURITY ALERTS
  - KPI summary grid + attack distribution / recent predictions on Overview
  - Footer disclaimer banner

Educational / prototype project. It does NOT perform real-time packet
capture and does NOT block any traffic -- it only analyzes traffic data
(CSV files) that you provide and displays predictions + risk levels.

Run with:  streamlit run app.py
"""

import os
import json
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.predict import models_available, load_artifacts, predict_batch
from src.data_preprocessing import load_csv
from src.risk_assessment import build_alert
from src.evaluation import why_precision_recall_matter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Network Intrusion Detection",
    layout="wide",
)

# =======================================================================
# THEME -- dark SOC/SIEM palette
# =======================================================================
BG_BASE = "#0B0F19"
BG_CARD = "#121826"
BORDER = "#1F2937"
GREEN = "#10B981"
CYAN = "#06B6D4"
RED = "#EF4444"
AMBER = "#F59E0B"
TEXT_PRIMARY = "#FFFFFF"
TEXT_MUTED = "#94A3B8"

st.markdown(f"""
<style>
html, body, [class*="css"], .stApp {{
    background-color: {BG_BASE} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'Consolas', 'Courier New', monospace;
}}
section[data-testid="stSidebar"] {{ display: none; }}
header[data-testid="stHeader"] {{ background-color: {BG_BASE} !important; }}
h1, h2, h3, h4, h5 {{ color: {TEXT_PRIMARY} !important; }}
p, span, label, div {{ color: {TEXT_PRIMARY}; }}

/* ---- Top header bar ---- */
.ids-header {{
    display: flex; justify-content: space-between; align-items: center;
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 24px; margin-bottom: 14px;
}}
.ids-header-title {{ font-size: 1.35rem; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: 0.5px; }}
.ids-header-sub {{ font-size: 0.72rem; color: {TEXT_MUTED}; letter-spacing: 1px; margin-top: 2px; }}
.version-tag {{
    display: inline-block; margin-top: 6px; font-size: 0.68rem; font-weight: 700;
    color: {CYAN}; border: 1px solid {CYAN}; border-radius: 4px; padding: 1px 8px;
    letter-spacing: 1px;
}}
.header-right {{ text-align: right; }}
.status-badge {{
    font-size: 0.78rem; font-weight: 700; color: {GREEN}; letter-spacing: 1px;
}}
.led {{
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    background-color: {GREEN}; margin-right: 6px;
    box-shadow: 0 0 8px {GREEN};
}}
.header-meta {{ font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: 4px; }}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {{
    color: {TEXT_MUTED} !important; font-weight: 700; letter-spacing: 0.5px;
    font-family: 'Consolas', 'Courier New', monospace;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {GREEN} !important; border-bottom: 3px solid {GREEN} !important;
}}
div[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}

/* ---- KPI cards ---- */
.kpi-card {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 18px; height: 118px; display: flex; flex-direction: column; justify-content: space-between;
}}
.kpi-label {{ font-size: 0.72rem; color: {TEXT_MUTED}; letter-spacing: 1px; text-transform: uppercase; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ font-size: 0.72rem; color: {TEXT_MUTED}; }}
.kpi-white {{ color: {TEXT_PRIMARY}; }}
.kpi-green {{ color: {GREEN}; }}
.kpi-red {{ color: {RED}; }}
.kpi-cyan {{ color: {CYAN}; }}

/* ---- Risk badges ---- */
.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-weight: 700; font-size: 0.72rem; letter-spacing: 0.5px;
}}
.badge-high {{ background-color: rgba(239,68,68,0.15); color: {RED}; border: 1px solid {RED}; }}
.badge-medium {{ background-color: rgba(245,158,11,0.15); color: {AMBER}; border: 1px solid {AMBER}; }}
.badge-low {{ background-color: rgba(16,185,129,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}

/* ---- Containers ---- */
.panel {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 16px;
}}
.panel-title {{
    font-size: 0.85rem; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 10px; border-left: 3px solid {GREEN}; padding-left: 8px;
}}

/* ---- Recent predictions table ---- */
.pred-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; font-family: 'Consolas', monospace; }}
.pred-table th {{
    text-align: left; color: {TEXT_MUTED}; font-size: 0.68rem; letter-spacing: 1px;
    border-bottom: 1px solid {BORDER}; padding: 6px 8px; text-transform: uppercase;
}}
.pred-table td {{ padding: 6px 8px; border-bottom: 1px solid {BORDER}; color: {TEXT_PRIMARY}; }}
.pred-table tr:hover td {{ background-color: rgba(255,255,255,0.03); }}

/* ---- Attack bar rows ---- */
.bar-row {{ margin-bottom: 10px; }}
.bar-label {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {TEXT_MUTED}; margin-bottom: 3px; }}
.bar-track {{ background-color: #1A2131; border-radius: 4px; height: 10px; overflow: hidden; }}
.bar-fill {{ background-color: {RED}; height: 10px; }}

/* ---- Buttons / inputs ---- */
.stButton>button, .stDownloadButton>button {{
    background-color: transparent; color: {GREEN}; border: 1px solid {GREEN};
    border-radius: 5px; font-weight: 700; font-family: 'Consolas', monospace;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ background-color: {GREEN}; color: {BG_BASE}; }}
.stTextInput input, .stSelectbox div, .stNumberInput input {{
    background-color: {BG_CARD} !important; color: {TEXT_PRIMARY} !important; border: 1px solid {BORDER} !important;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}
div[data-testid="stExpander"] {{ border: 1px solid {BORDER} !important; border-radius: 8px; background-color: {BG_CARD}; }}

/* ---- Footer ---- */
.footer-banner {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-left: 4px solid {AMBER};
    border-radius: 8px; padding: 12px 18px; margin-top: 24px; font-size: 0.75rem; color: {TEXT_MUTED};
}}
</style>
""", unsafe_allow_html=True)

GRAYS_ON_DARK = [RED, "#F87171", "#FCA5A5", "#7F1D1D", "#B91C1C", "#450A0A"]


def badge(risk: str) -> str:
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(risk, "badge-medium")
    return f'<span class="badge {cls}">{risk}</span>'


def style_plot(fig, title=None):
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY, family="Consolas, monospace"),
        title=dict(text=title) if title else fig.layout.title,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=45, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


def load_training_summary():
    path = os.path.join(OUTPUTS_DIR, "training_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def ensure_models():
    """Train the default demo model when model artifacts are missing."""
    if models_available():
        return

    with st.spinner("No trained model found. Training the demo model..."):
        completed = subprocess.run(
            [sys.executable, "-m", "src.train_model"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

    if completed.returncode != 0 or not models_available():
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(details or "Model training did not create the expected artifacts.")


# =======================================================================
# Bootstrap
# =======================================================================
try:
    ensure_models()
except Exception as exc:
    st.error(f"Could not prepare the detection model: {exc}")
    st.stop()

artifacts = get_artifacts()
summary = load_training_summary()

if "last_batch_results" not in st.session_state:
    st.session_state["last_batch_results"] = None
results = st.session_state["last_batch_results"]

# =======================================================================
# HEADER BAR
# =======================================================================
total_flows = len(results) if results is not None else 0
dataset_label = "CIC-IDS2017-format CSV" if results is not None else "No dataset loaded"

st.markdown(f"""
<div class="ids-header">
    <div>
        <div class="ids-header-title">🛡️ NETWORK INTRUSION DETECTION</div>
        <div class="ids-header-sub">FORTINET SECURITY ASSOCIATE · AICTE&ndash;EDUSKILLS</div>
        <div class="version-tag">PROTOTYPE v1.0</div>
    </div>
    <div class="header-right">
        <div class="status-badge"><span class="led"></span>IDS ACTIVE</div>
        <div class="header-meta">DATASET: {dataset_label}</div>
        <div class="header-meta">TOTAL FLOWS: {total_flows:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =======================================================================
# NAVIGATION
# =======================================================================
tab_overview, tab_traffic, tab_threats, tab_model, tab_alerts = st.tabs(
    ["OVERVIEW", "TRAFFIC ANALYSIS", "THREAT DETECTION", "MODEL PERFORMANCE", "SECURITY ALERTS"]
)

# =======================================================================
# TAB: OVERVIEW
# =======================================================================
with tab_overview:
    with st.expander("📁 Load Traffic Data (upload a CIC-IDS2017-format CSV)", expanded=(results is None)):
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        run_clicked = st.button("▶ RUN ANALYSIS", type="primary", disabled=uploaded is None)
        if uploaded is not None and run_clicked:
            try:
                raw_df = load_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()
            with st.spinner("Preprocessing and running inference..."):
                try:
                    results = predict_batch(raw_df, artifacts)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()
            st.session_state["last_batch_results"] = results
            st.rerun()

    # ---- KPI Grid (2 x 3) ----
    if results is not None:
        total = len(results)
        normal_ct = int((results["Prediction"] == "Normal").sum())
        attack_ct = total - normal_ct
        normal_pct = round(normal_ct / total * 100, 1) if total else 0
        attack_pct = round(attack_ct / total * 100, 1) if total else 0
        attack_types = results.loc[results["Prediction"] == "Attack", "Attack_Type"] \
            if "Attack_Type" in results.columns else pd.Series(dtype=object)
        n_types = attack_types.nunique()
        top_types = attack_types.value_counts().head(3).index.tolist()
        high_risk_ct = int((results["Risk_Level"] == "HIGH").sum()) if "Risk_Level" in results.columns else 0
    else:
        total = normal_ct = attack_ct = n_types = high_risk_ct = 0
        normal_pct = attack_pct = 0
        top_types = []

    if summary:
        m = summary["binary_metrics"]
        acc_pct = f"{m['accuracy']*100:.1f}%"
        f1_val = f"{m['f1_score']:.3f}"
    else:
        acc_pct, f1_val = "N/A", "N/A"

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    with r1c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Flows Analyzed</div>
            <div class="kpi-value kpi-white">{total:,}</div>
            <div class="kpi-sub">Dataset: CIC-IDS2017 format</div>
        </div>""", unsafe_allow_html=True)

    with r1c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Normal Traffic</div>
            <div class="kpi-value kpi-green">{normal_ct:,}</div>
            <div class="kpi-sub">{normal_pct}% of analyzed flows</div>
        </div>""", unsafe_allow_html=True)

    with r1c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Attack Flows</div>
            <div class="kpi-value kpi-red">{attack_ct:,}</div>
            <div class="kpi-sub">{attack_pct}% of analyzed flows</div>
        </div>""", unsafe_allow_html=True)

    with r2c1:
        sub = ", ".join(top_types) if top_types else "No attacks detected"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Attack Types Detected</div>
            <div class="kpi-value kpi-white">{n_types}</div>
            <div class="kpi-sub">Top: {sub}</div>
        </div>""", unsafe_allow_html=True)

    with r2c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value kpi-green">{acc_pct}</div>
            <div class="kpi-sub">Random Forest · F1: {f1_val}</div>
        </div>""", unsafe_allow_html=True)

    with r2c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">High Risk Alerts</div>
            <div class="kpi-value kpi-red">{high_risk_ct:,}</div>
            <div class="kpi-sub">Requires immediate review</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ---- Bottom breakdown: distribution bars | recent predictions ----
    bcol1, bcol2 = st.columns([1, 1.3])

    with bcol1:
        st.markdown('<div class="panel"><div class="panel-title">Attack Type Distribution</div>', unsafe_allow_html=True)
        if results is not None and n_types > 0:
            type_counts = attack_types.value_counts().head(8)
            max_count = type_counts.max()
            for name, count in type_counts.items():
                pct_width = int(count / max_count * 100)
                st.markdown(f"""
                <div class="bar-row">
                    <div class="bar-label"><span>{name}</span><span>{count:,}</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct_width}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94A3B8;">No attack traffic to display. Load data above.</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bcol2:
        st.markdown('<div class="panel"><div class="panel-title">Recent Flow Predictions</div>', unsafe_allow_html=True)
        if results is not None:
            recent = results.tail(12).iloc[::-1].copy()
            time_col = next((c for c in ["Timestamp", "timestamp"] if c in recent.columns), None)
            src_col = next((c for c in ["Source IP", "Src IP", "Source", "Src"] if c in recent.columns), None)
            port_col = next((c for c in ["Destination Port", "Dst Port"] if c in recent.columns), None)

            rows_html = ""
            for i, row in recent.iterrows():
                t = str(row[time_col]) if time_col else f"#{i}"
                s = str(row[src_col]) if src_col else "N/A"
                p = str(row[port_col]) if port_col else "N/A"
                typ = row.get("Attack_Type", row.get("Prediction", "N/A"))
                risk = row.get("Risk_Level", "LOW")
                conf = row.get("Confidence", 0)
                rows_html += (
                    f"<tr><td>{t}</td><td>{s}</td><td>{p}</td><td>{typ}</td>"
                    f"<td>{badge(risk)}</td><td>{conf:.1f}%</td></tr>"
                )

            st.markdown(f"""
            <table class="pred-table">
                <tr><th>Time</th><th>Source</th><th>Dst Port</th><th>Type</th><th>Risk</th><th>Conf</th></tr>
                {rows_html}
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94A3B8;">No predictions yet. Load data above.</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =======================================================================
# TAB: TRAFFIC ANALYSIS
# =======================================================================
with tab_traffic:
    if results is None:
        st.info("Load traffic data from the OVERVIEW tab first.")
    else:
        sub_dist, sub_feature, sub_corr = st.tabs(["Distribution", "Feature Explorer", "Correlation"])

        with sub_dist:
            col1, col2 = st.columns(2)
            with col1:
                counts = results["Prediction"].value_counts().reset_index()
                counts.columns = ["Prediction", "Count"]
                fig = px.pie(counts, names="Prediction", values="Count", hole=0.45,
                             title="Normal vs Attack",
                             color="Prediction",
                             color_discrete_map={"Normal": GREEN, "Attack": RED})
                st.plotly_chart(style_plot(fig), use_container_width=True)
            with col2:
                if n_types > 0:
                    type_counts = attack_types.value_counts().reset_index()
                    type_counts.columns = ["Attack Type", "Count"]
                    fig2 = px.bar(type_counts, x="Attack Type", y="Count", title="Attack Type Distribution",
                                  color_discrete_sequence=[RED])
                    st.plotly_chart(style_plot(fig2), use_container_width=True)
                else:
                    st.success("No attacks detected in the analyzed traffic.")

            if "Destination Port" in results.columns:
                top_ports = results[results["Prediction"] == "Attack"]["Destination Port"] \
                    .value_counts().head(10).reset_index()
                top_ports.columns = ["Destination Port", "Count"]
                top_ports["Destination Port"] = top_ports["Destination Port"].astype(str)
                fig_ports = px.bar(top_ports, x="Destination Port", y="Count",
                                    title="Top 10 Targeted Ports", color_discrete_sequence=[CYAN])
                st.plotly_chart(style_plot(fig_ports), use_container_width=True)

        with sub_feature:
            numeric_cols = [c for c in artifacts["feature_columns"] if c in results.columns]
            if numeric_cols:
                f1c, f2c = st.columns(2)
                with f1c:
                    stat_feature = st.selectbox("Feature", numeric_cols)
                with f2c:
                    chart_kind = st.radio("Chart", ["Histogram", "Box Plot"], horizontal=True)
                if chart_kind == "Histogram":
                    fig3 = px.histogram(results, x=stat_feature, color="Prediction", barmode="overlay",
                                         nbins=40, color_discrete_map={"Normal": GREEN, "Attack": RED},
                                         title=f"Distribution: {stat_feature}")
                else:
                    fig3 = px.box(results, x="Prediction", y=stat_feature, color="Prediction",
                                   color_discrete_map={"Normal": GREEN, "Attack": RED},
                                   title=f"{stat_feature} by Prediction")
                st.plotly_chart(style_plot(fig3), use_container_width=True)
                st.dataframe(results[numeric_cols].describe().T, use_container_width=True)
            else:
                st.info("No numeric feature columns available.")

        with sub_corr:
            numeric_cols = [c for c in artifacts["feature_columns"] if c in results.columns]
            if len(numeric_cols) >= 2:
                top_n = st.slider("Top-variance features", 4, min(20, len(numeric_cols)), min(10, len(numeric_cols)))
                selected = results[numeric_cols].var().sort_values(ascending=False).head(top_n).index.tolist()
                corr = results[selected].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                      title="Feature Correlation Heatmap", aspect="auto")
                st.plotly_chart(style_plot(fig_corr), use_container_width=True)
            else:
                st.info("Not enough numeric features for correlation.")

# =======================================================================
# TAB: THREAT DETECTION
# =======================================================================
with tab_threats:
    if results is None:
        st.info("Load traffic data from the OVERVIEW tab first.")
    else:
        f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
        with f1:
            filter_choice = st.selectbox("Filter", ["All", "Attacks Only", "High Risk Only"])
        with f2:
            min_conf = st.slider("Min. confidence %", 0, 100, 0)
        with f3:
            search_term = st.text_input("Search attack type", placeholder="e.g. DoS, Port Scan...")

        display_candidates = ["Source IP", "Src IP", "Destination IP", "Dst IP", "Source Port",
                               "Src Port", "Destination Port", "Protocol", "Prediction",
                               "Attack_Type", "Risk_Level", "Confidence"]
        available_cols = [c for c in display_candidates if c in results.columns]
        for c in ["Prediction", "Attack_Type", "Risk_Level", "Confidence"]:
            if c not in available_cols and c in results.columns:
                available_cols.append(c)

        view_df = results[available_cols].copy()
        mask = pd.Series(True, index=results.index)
        if filter_choice == "Attacks Only":
            mask &= results["Prediction"] == "Attack"
        elif filter_choice == "High Risk Only":
            mask &= results["Risk_Level"] == "HIGH"
        if "Confidence" in results.columns:
            mask &= results["Confidence"] >= min_conf
        if search_term and "Attack_Type" in results.columns:
            mask &= results["Attack_Type"].astype(str).str.contains(search_term, case=False, na=False)
        view_df = view_df[mask]

        st.dataframe(view_df, use_container_width=True, height=420,
                     column_config={"Confidence": st.column_config.ProgressColumn(
                         "Confidence", format="%.1f%%", min_value=0, max_value=100)}
                     if "Confidence" in view_df.columns else None)
        st.caption(f"Showing {len(view_df):,} of {len(results):,} analyzed flows.")

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ DOWNLOAD FULL RESULTS (CSV)", data=csv_bytes,
                            file_name="ids_prediction_results.csv", mime="text/csv")

# =======================================================================
# TAB: MODEL PERFORMANCE
# =======================================================================
with tab_model:
    if summary is None:
        st.warning("No training summary found. Run `python -m src.train_model` to generate one.")
    else:
        st.markdown('<div class="panel"><div class="panel-title">Binary Model (Normal vs Attack)</div>', unsafe_allow_html=True)
        m = summary["binary_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi-value kpi-green">{m["accuracy"]:.2%}</div><div class="kpi-label">Accuracy</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-value kpi-cyan">{m["precision"]:.2%}</div><div class="kpi-label">Precision</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-value kpi-cyan">{m["recall"]:.2%}</div><div class="kpi-label">Recall</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-value kpi-white">{m["f1_score"]:.3f}</div><div class="kpi-label">F1-Score</div>', unsafe_allow_html=True)
        st.caption(f"Training time: {m['training_time_seconds']:.2f}s · "
                   f"Prediction time: {m['predict_time_seconds']:.4f}s · Test samples: {m['n_test_samples']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Why Precision / Recall / F1 matter (not just Accuracy)"):
            st.write(why_precision_recall_matter())

        cm_path = os.path.join(OUTPUTS_DIR, "binary_confusion_matrix.csv")
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path)
            fig = px.imshow(cm.values, text_auto=True, color_continuous_scale="Greens",
                             title="Confusion Matrix - Binary Model", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(style_plot(fig), use_container_width=True)

        fi_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path)
            fig_fi = px.bar(fi.head(15), x="importance", y="feature", orientation="h",
                             title="Top 15 Feature Importances", color_discrete_sequence=[CYAN])
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plot(fig_fi), use_container_width=True)

        if summary.get("has_multiclass_model"):
            st.markdown('<div class="panel"><div class="panel-title">Multiclass Model (Attack Category)</div>', unsafe_allow_html=True)
            mm = summary["multiclass_metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="kpi-value kpi-green">{mm["accuracy"]:.2%}</div><div class="kpi-label">Accuracy</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi-value kpi-cyan">{mm["precision"]:.2%}</div><div class="kpi-label">Precision</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="kpi-value kpi-cyan">{mm["recall"]:.2%}</div><div class="kpi-label">Recall</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="kpi-value kpi-white">{mm["f1_score"]:.3f}</div><div class="kpi-label">F1-Score</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if summary.get("comparison_results"):
            st.markdown('<div class="panel"><div class="panel-title">Model Comparison</div>', unsafe_allow_html=True)
            comp_rows = [{"Model": "RandomForest_Binary", **{k: m[k] for k in ("accuracy", "precision", "recall", "f1_score")}}]
            for name, cmet in summary["comparison_results"].items():
                comp_rows.append({"Model": name, **{k: cmet[k] for k in ("accuracy", "precision", "recall", "f1_score")}})
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df.style.format({c: "{:.2%}" for c in ("accuracy", "precision", "recall", "f1_score")}),
                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =======================================================================
# TAB: SECURITY ALERTS
# =======================================================================
with tab_alerts:
    if results is None:
        st.info("Load traffic data from the OVERVIEW tab first.")
    else:
        high_risk = results[results["Risk_Level"] == "HIGH"] if "Risk_Level" in results.columns else pd.DataFrame()
        if len(high_risk) == 0:
            st.markdown('<div class="panel">✔ No HIGH risk threats detected in the analyzed traffic.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="panel"><span class="kpi-red" style="font-size:1.3rem;font-weight:800;">'
                        f'🚨 {len(high_risk)} HIGH RISK ALERT(S)</span></div>', unsafe_allow_html=True)
            for _, row in high_risk.head(50).iterrows():
                alert = build_alert(row.get("Attack_Type", "Attack"), row.get("Confidence", 0) / 100)
                st.markdown(f"""
                <div class="panel" style="border-left:4px solid {RED};">
                    <div style="font-weight:800; color:{RED}; margin-bottom:6px;">🚨 HIGH RISK ALERT</div>
                    <div>Attack Type: <b>{alert['attack_type']}</b></div>
                    <div>Risk Level: {badge(alert['risk_level'])}</div>
                    <div>Confidence: <b>{alert['confidence']}%</b></div>
                    <div style="margin-top:8px; color:{TEXT_MUTED};">
                        <b>Recommended Action:</b> {alert['recommended_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if len(high_risk) > 50:
                st.caption(f"... and {len(high_risk) - 50} more. Download the full CSV from Threat Detection.")

# =======================================================================
# FOOTER
# =======================================================================
