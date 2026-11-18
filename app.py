import hashlib
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from modules.auth import render_auth_page

# ==========================================================
# COMMUNITY SAFETY & TRUST AI PLATFORM
# Main Dashboard
# ==========================================================

st.set_page_config(
    page_title="Community Safety & Trust AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "currency_cnn_best.keras"
MODULE_NAV_ITEMS = (
    ("Dashboard", None),
    ("Currency", "currency"),
    ("Suspicious Activity", "activity"),
    ("Fake News", "scam"),
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0f172a;
        --muted: #475569;
        --subtle: #64748b;
        --line: #e2e8f0;
        --surface: #ffffff;
        --surface-alt: #f8fafc;
        --brand-dark: #0f172a;
        --brand-hover: #1e293b;
        --accent-teal: #0f766e;
        --green: #047857;
        --amber: #b45309;
        --red: #b91c1c;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--ink);
    }

    .stApp {
        color: var(--ink);
        background-color: #f8fafc;
        background-image: 
            linear-gradient(rgba(15, 23, 42, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.025) 1px, transparent 1px);
        background-size: 32px 32px;
    }

    /* Auth Page Theme when .st-key-auth_card is present */
    .stApp:has(.st-key-auth_card) {
        background-color: #F5F8FF !important;
        background-image: 
            radial-gradient(circle at 12% 14%, rgba(37, 99, 235, 0.09) 0%, transparent 45%),
            radial-gradient(circle at 88% 86%, rgba(6, 182, 212, 0.09) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(15, 118, 110, 0.04) 0%, transparent 50%),
            linear-gradient(rgba(37, 99, 235, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(37, 99, 235, 0.025) 1px, transparent 1px) !important;
        background-size: auto, auto, auto, 36px 36px, 36px 36px !important;
    }

    .stApp:has(.st-key-auth_card) [data-testid="stSidebar"],
    .stApp:has(.st-key-auth_card) [data-testid="collapsedControl"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 12px rgba(15, 23, 42, 0.03);
    }

    [data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1380px;
        padding-top: 1.5rem;
    }

    .block-container {
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    h1, h2, h3, h4, p { letter-spacing: -0.01em; }
    h1, h2, h3, h4 { color: var(--ink); font-weight: 750; }
    p, li { color: var(--muted); }
    hr { border-color: var(--line); }

    .topbar {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        min-height: 48px;
    }

    .brand {
        align-items: center;
        display: flex;
        gap: 0.75rem;
    }

    .brand-mark {
        align-items: center;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        display: grid;
        font-size: 1.25rem;
        height: 40px;
        justify-content: center;
        width: 40px;
    }

    .brand-name {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        line-height: 1.25;
    }

    .brand-caption, .eyebrow, .section-label span, .module-tech,
    .status-online, .status-planned, .metric-label, .card-kicker {
        font-family: 'DM Mono', monospace;
    }

    .brand-caption {
        color: var(--subtle);
        font-size: 0.64rem;
        letter-spacing: 0.04em;
    }

    .eyebrow {
        color: var(--accent-teal);
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .hero-title {
        color: var(--ink);
        font-size: 2.75rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.4rem 0 0.6rem;
    }

    .hero-copy, .feature-copy {
        color: var(--muted);
        line-height: 1.65;
        max-width: 620px;
        font-size: 0.95rem;
    }

    .section-label {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin: 1.8rem 0 0.9rem;
    }

    .section-label h2 {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        color: var(--ink);
    }

    .section-label span {
        color: var(--subtle);
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Light Theme Cards */
    .module-card, .metric-card, .analysis-card, .pipeline-card, .planned-panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
    }

    .module-card {
        display: flex;
        flex-direction: column;
        min-height: 252px;
        overflow: hidden;
        padding: 1.35rem;
        position: relative;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }

    .module-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 12px 28px -4px rgba(15, 23, 42, 0.09);
        transform: translateY(-2px);
    }

    .module-card.theme-currency {
        --card-accent: #b45309;
        background: linear-gradient(155deg, #fffdf7 0%, #ffffff 100%);
        border-color: #fed7aa;
    }

    .module-card.theme-activity {
        --card-accent: #0f766e;
        background: linear-gradient(155deg, #f0fdfa 0%, #ffffff 100%);
        border-color: #99f6e4;
    }

    .module-card.theme-scam {
        --card-accent: #047857;
        background: linear-gradient(155deg, #f0fdf4 0%, #ffffff 100%);
        border-color: #a7f3d0;
    }

    .module-card .module-icon {
        align-items: center;
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 12px;
        display: flex;
        font-size: 1.25rem;
        height: 42px;
        justify-content: center;
        width: 42px;
    }

    .module-title {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 750;
        line-height: 1.3;
        margin-top: 0.9rem;
    }

    .module-tech {
        color: var(--card-accent);
        font-size: 0.65rem;
        letter-spacing: 0.04em;
        margin-top: 0.22rem;
        text-transform: uppercase;
        font-weight: 600;
    }

    .module-description {
        color: var(--muted);
        flex: 1;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: 0.8rem 0 0.65rem;
    }

    .status-online, .status-planned {
        font-size: 0.64rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
    }

    .status-online { color: var(--green); }
    .status-planned { color: var(--amber); }

    /* Metrics */
    .metric-card {
        min-height: 80px;
        padding: 0.9rem 1rem;
    }

    .metric-label {
        color: var(--subtle);
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .metric-value.teal { color: var(--accent-teal); }
    .metric-value.green { color: var(--green); }
    .metric-value.amber { color: var(--amber); }
    .metric-value.slate { color: var(--muted); }

    .feature-title {
        color: var(--ink);
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.4rem 0 0.55rem;
    }

    /* Results Area */
    .result-empty {
        align-items: center;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 238px;
        text-align: center;
        padding: 1.5rem;
    }

    .result-empty.result-real {
        background: #f0fdf4;
        border: 1px solid #86efac;
        box-shadow: 0 4px 16px rgba(4, 120, 87, 0.06);
    }

    .result-empty.result-fake {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        box-shadow: 0 4px 16px rgba(185, 28, 28, 0.06);
    }

    .result-empty.result-unclear {
        background: #fffdf7;
        border: 1px solid #fed7aa;
        box-shadow: 0 4px 16px rgba(180, 83, 9, 0.06);
    }

    .result-word {
        color: var(--green);
        font-size: 2.6rem;
        font-weight: 800;
    }

    .result-word.fake { color: var(--red); }
    .result-word.unclear { color: var(--amber); }

    .confidence-number {
        color: var(--ink);
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .confidence-label {
        color: var(--subtle);
        font-size: 0.65rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .gauge { margin: 0.7rem auto 0.3rem; position: relative; width: 148px; }
    .gauge svg { display: block; transform: rotate(-90deg); }
    .gauge-bg, .gauge-fill { fill: none; stroke-linecap: round; stroke-width: 8; }
    .gauge-bg { stroke: #e2e8f0; }
    .gauge-fill { stroke: var(--amber); }
    .gauge-value { align-items: center; display: flex; flex-direction: column; inset: 0; justify-content: center; position: absolute; }

    /* Flow / Pipeline */
    .pipeline {
        align-items: center;
        display: flex;
        gap: 0.35rem;
        justify-content: space-between;
        margin: 0.2rem auto 1.4rem;
        max-width: 960px;
        width: 100%;
    }

    .pipeline-step { align-items: center; display: flex; flex: 1; gap: 0.32rem; min-width: 0; }
    .pipeline-node {
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        color: var(--ink);
        display: flex;
        flex-direction: column;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        gap: 0.3rem;
        justify-content: center;
        min-height: 68px;
        min-width: 0;
        padding: 0.5rem 0.35rem;
        text-align: center;
        width: 100%;
    }

    .flow-label { color: var(--subtle); font-size: 0.56rem; letter-spacing: 0.08em; font-weight: 600; }
    .flow-detail { color: var(--ink); font-family: 'Manrope', sans-serif; font-size: 0.78rem; font-weight: 700; line-height: 1.25; }
    .pipeline-arrow { color: #94a3b8; flex: 0 0 auto; font-size: 0.95rem; }

    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 14px;
        color: #92400e;
        font-size: 0.72rem;
        line-height: 1.55;
        padding: 0.75rem 0.95rem;
    }

    .planned-panel { min-height: 260px; padding: 1.8rem; text-align: center; }
    .planned-icon { font-size: 2.25rem; margin-bottom: 0.7rem; }
    .planned-title { color: var(--ink); font-size: 1.35rem; font-weight: 750; }
    .planned-copy { color: var(--muted); line-height: 1.65; margin: 0.65rem auto 1.2rem; max-width: 580px; }
    .planned-chip {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        color: var(--subtle);
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        padding: 0.42rem 0.72rem;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* High-Contrast Executive Button Palette */
    .stButton > button,
    .stFormSubmitButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        border-radius: 999px !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        min-height: 2.5rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.18s ease !important;
    }

    /* Primary buttons: Deep Slate / Carbon (Dashboard) */
    .stButton > button[kind="primary"]:not(.st-key-auth_card *),
    .stFormSubmitButton > button[kind="primary"]:not(.st-key-auth_card *),
    button[data-testid="baseButton-primary"]:not(.st-key-auth_card *) {
        background: #0f172a !important;
        border: 1px solid #0f172a !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12) !important;
    }

    .stButton > button[kind="primary"]:not(.st-key-auth_card *):hover,
    .stFormSubmitButton > button[kind="primary"]:not(.st-key-auth_card *):hover,
    button[data-testid="baseButton-primary"]:not(.st-key-auth_card *):hover {
        background: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary buttons: Clean White with Slate outline */
    .stButton > button[kind="secondary"],
    .stFormSubmitButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]),
    .stFormSubmitButton > button:not([kind="primary"]),
    button[data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }

    .stButton > button[kind="secondary"]:hover,
    .stFormSubmitButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover,
    .stFormSubmitButton > button:not([kind="primary"]):hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06) !important;
        transform: translateY(-1px) !important;
    }

    /* Sign Out buttons: Soft Alert Red */
    div.st-key-topbar_signout button,
    div.st-key-sidebar_signout button {
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        color: #991b1b !important;
    }

    div.st-key-topbar_signout button:hover,
    div.st-key-sidebar_signout button:hover {
        background: #fee2e2 !important;
        border-color: #f87171 !important;
        color: #7f1d1d !important;
    }

    /* Segmented Control & Tabs */
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }

    [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0f172a !important;
        border-bottom-color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: #f8fafc;
        border-color: #0f172a;
    }

    [data-testid="stImage"] img {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] { color: var(--subtle); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; font-size: 1.05rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)


def release_activity_camera():
    capture = st.session_state.get("activity_camera")
    if capture is not None:
        capture.release()
    st.session_state["activity_camera"] = None
    st.session_state["activity_camera_initial_frame"] = None
    st.session_state["activity_camera_previous_frame"] = None
    st.session_state["activity_camera_unchanged_reads"] = 0


def set_module(module):
    if module != "activity":
        release_activity_camera()
    st.session_state["module"] = module
    st.rerun()


def render_topbar(active_module=None):
    user = st.session_state.get("user")
    user_name = (user.get("full_name") or user.get("username") or "Member") if user else "Member"
    username = user.get("username", "") if user else ""

    top_col1, top_col2 = st.columns([3.2, 2.8])
    with top_col1:
        st.markdown(
            '<div class="topbar"><div class="brand"><div class="brand-mark">🛡️</div>'
            '<div><div class="brand-name">COMMUNITY SAFETY<br>& TRUST AI</div>'
            '<div class="brand-caption">AI-powered safety intelligence</div></div></div></div>',
            unsafe_allow_html=True,
        )
    with top_col2:
        user_c1, user_c2 = st.columns([1.8, 1])
        with user_c1:
            st.markdown(
                f'<div style="text-align:right; padding-top:0.45rem; font-family:\'DM Mono\', monospace; font-size:0.75rem; color:#059669;">'
                f'👤 <strong>{user_name}</strong> <span style="color:#64748b;">(@{username})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with user_c2:
            if st.button("Sign Out", key="topbar_signout", width="stretch"):
                release_activity_camera()
                st.session_state["authenticated"] = False
                st.session_state["user"] = None
                st.session_state["module"] = None
                st.rerun()

    navigation_columns = st.columns(len(MODULE_NAV_ITEMS), gap="small")
    for column, (label, module) in zip(navigation_columns, MODULE_NAV_ITEMS):
        with column:
            if st.button(
                label,
                key=f"navigate_{module or 'dashboard'}",
                type="primary" if module == active_module else "secondary",
                width="stretch",
            ) and module != active_module:
                set_module(module)


def render_sidebar(active_module=None):
    with st.sidebar:
        # App Branding Header
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.75rem; padding:0.4rem 0 1rem; border-bottom:1px solid #e2e8f0; margin-bottom:1.2rem;">
                <div style="font-size:1.6rem; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:12px; width:44px; height:44px; display:grid; place-items:center;">🛡️</div>
                <div>
                    <div style="font-size:0.92rem; font-weight:800; color:#0f172a; line-height:1.2; letter-spacing:0.04em;">COMMUNITY SAFETY</div>
                    <div style="font-size:0.64rem; color:#0f766e; font-family:'DM Mono', monospace; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">Trust AI Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # User profile chip
        user = st.session_state.get("user")
        if user:
            user_name = user.get("full_name") or user.get("username") or "Member"
            username = user.get("username", "")
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:0.75rem 0.85rem; margin-bottom:1.2rem; box-shadow:0 1px 3px rgba(15,23,42,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.62rem; color:#64748b; font-family:'DM Mono', monospace; text-transform:uppercase; letter-spacing:0.06em;">Active Session</span>
                        <span style="font-size:0.62rem; color:#047857; font-weight:700; font-family:'DM Mono', monospace;">● Online</span>
                    </div>
                    <div style="font-size:0.92rem; font-weight:750; color:#0f172a; margin-top:0.25rem;">👤 {user_name}</div>
                    <div style="font-size:0.72rem; color:#475569; font-family:'DM Mono', monospace;">@{username}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation section
        st.markdown(
            '<div style="font-size:0.68rem; font-weight:700; color:#64748b; font-family:\'DM Mono\', monospace; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">AI Modules</div>',
            unsafe_allow_html=True,
        )

        sidebar_modules = [
            ("📊 Dashboard Overview", None),
            ("💵 Fake Currency Detector", "currency"),
            ("👁️ Suspicious Activity", "activity"),
            ("📰 Fake News Detection", "scam"),
        ]

        for label, mod in sidebar_modules:
            is_active = (mod == active_module)
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                label,
                key=f"sidebar_nav_{mod or 'dashboard'}",
                type=btn_type,
                use_container_width=True,
            ) and mod != active_module:
                set_module(mod)

        st.divider()

        # System Status & AI Models Info
        st.markdown(
            '<div style="font-size:0.68rem; font-weight:700; color:#64748b; font-family:\'DM Mono\', monospace; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">Intelligence Engines</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:0.45rem; font-size:0.72rem; font-family:'DM Mono', monospace;">
                <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; padding:0.45rem 0.65rem; border-radius:10px; border:1px solid #e2e8f0;">
                    <span style="color:#334155;">Currency CNN</span>
                    <span style="color:#b45309; font-weight:700;">88.7% Acc</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; padding:0.45rem 0.65rem; border-radius:10px; border:1px solid #e2e8f0;">
                    <span style="color:#334155;">YOLOv11 nano</span>
                    <span style="color:#0f766e; font-weight:700;">80 Classes</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; padding:0.45rem 0.65rem; border-radius:10px; border:1px solid #e2e8f0;">
                    <span style="color:#334155;">Fake News NLP</span>
                    <span style="color:#047857; font-weight:700;">Linear SVM</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; padding:0.45rem 0.65rem; border-radius:10px; border:1px solid #e2e8f0;">
                    <span style="color:#334155;">Cloud Auth</span>
                    <span style="color:#334155; font-weight:700;">MongoDB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Sign Out in sidebar
        if st.button("🚪 Sign Out", key="sidebar_signout", use_container_width=True):
            release_activity_camera()
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["module"] = None
            st.rerun()


def render_metric_card(label, value, tone=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {tone}">{value}</div></div>', unsafe_allow_html=True)


def render_module_card(icon, title, technology, engine, description, module, planned):
    state = "COMING SOON" if planned else "ACTIVE"
    css_class = "status-planned" if planned else "status-online"
    theme = {"currency": "theme-currency", "activity": "theme-activity", "scam": "theme-scam"}[module]
    st.markdown(f'<div class="module-card {theme}"><div class="module-icon">{icon}</div><div class="module-title">{title}</div><div class="module-tech">{technology}</div><div class="module-description">{description}</div><div class="{css_class}">● {state}</div></div>', unsafe_allow_html=True)
    button_label = {
        "currency": "Open Currency Detector →",
        "activity": "Open Suspicious Activity →",
        "scam": "Open Fake News Detection →",
    }[module] if not planned else "View module →"
    if st.button(button_label, key=f"open_{module}", width="stretch"):
        set_module(module)


def render_pipeline(steps):
    content = ""
    for index, step in enumerate(steps):
        arrow = '<div class="pipeline-arrow">→</div>' if index < len(steps) - 1 else ""
        content += f'<div class="pipeline-step"><div class="pipeline-node">{step}</div>{arrow}</div>'
    st.markdown(f'<div class="pipeline">{content}</div>', unsafe_allow_html=True)


def render_disclaimer(text):
    st.markdown(f'<div class="disclaimer">{text}</div>', unsafe_allow_html=True)


def render_dashboard():
    render_topbar()
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown('<div class="eyebrow">Community Safety Center / Control Room</div><div class="hero-title">Detect. Analyze.<br><span style="color:#0f766e">Respond responsibly.</span></div><div class="hero-copy">A unified intelligence workspace for visual screening, activity awareness, and digital trust. <strong>Building safer communities through responsible AI.</strong></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="hero-visual"><span class="hero-hand robot">🦾</span><div class="hero-core">AI</div><span class="hero-hand human">🫲</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><h2>AI modules</h2></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_module_card("💵", "Fake Currency Detector", "CNN • Computer Vision", "CNN", "Visual screening of Indian currency notes with the trained CNN.", "currency", False)
    with col2:
        render_module_card("👁️", "Suspicious Activity Detection", "YOLO • Object Detection + Rules", "YOLO", "Review detected objects and crowd counts from an image; no intent or behavior is inferred.", "activity", False)
    with col3:
        render_module_card("📰", "Fake News Detection", "TF-IDF + Linear SVM • NLP", "SVM", "Classify English or Hindi news text with a signed, non-probability decision score.", "scam", False)


@st.cache_resource
def load_currency_model_ui():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_activity_model_ui():
    from modules.suspicious_activity import load_person_detector
    return load_person_detector()


@st.cache_data
def load_activity_alarm_audio():
    from modules.suspicious_activity import create_alarm_audio
    return create_alarm_audio()


def preprocess_currency_image_ui(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


def render_confidence_gauge(confidence):
    percentage = confidence * 100
    radius = 53
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - confidence)
    level = "UNCLEAR" if confidence < .6 else "MODERATE CONFIDENCE" if confidence < .8 else "HIGH CONFIDENCE"
    st.markdown(f"<div class=\"gauge\"><svg viewBox=\"0 0 128 128\"><circle class=\"gauge-bg\" cx=\"64\" cy=\"64\" r=\"{radius}\"/><circle class=\"gauge-fill\" cx=\"64\" cy=\"64\" r=\"{radius}\" stroke-dasharray=\"{circumference:.2f}\" stroke-dashoffset=\"{offset:.2f}\"/></svg><div class=\"gauge-value\"><div class=\"confidence-number\">{percentage:.2f}%</div><div class=\"confidence-label\">confidence</div></div></div><div class=\"confidence-state\">{level}</div>", unsafe_allow_html=True)


def predict_currency_image_ui(uploaded_file):
    model = load_currency_model_ui()
    image_array = preprocess_currency_image_ui(uploaded_file)
    fake_probability = float(model.predict(image_array, verbose=0)[0][0])
    # Validation analysis selected 0.45 to reduce FAKE -> REAL false negatives.
    predicted_label = 1 if fake_probability >= 0.45 else 0
    confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability
    prediction = "FAKE" if predicted_label == 1 else "REAL"
    if confidence < .60:
        prediction = "UNCLEAR"
    return prediction, confidence


def render_dual_side(side_label, upload_key, upload_card_key, result_card_key):
    uploaded_file = None
    image_is_valid = True
    with st.container(key=upload_card_key):
        st.markdown(f'<div class="scan-side-heading">{side_label}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            f"{side_label} image",
            type=["jpg", "jpeg", "png", "avif", "webp"],
            key=upload_key,
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                display_image = Image.open(uploaded_file).convert("RGB")
                st.image(display_image, caption=uploaded_file.name, width="stretch")
            except Exception as error:
                image_is_valid = False
                st.error(f"Unable to display this image: {error}")

    side_result = None
    with st.container(key=result_card_key):
        st.markdown(f'<div class="scan-side-heading">{side_label} Result</div>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown('<div class="result-empty result-pending"><div style="color:#334155;font-weight:650">Awaiting image input</div></div>', unsafe_allow_html=True)
        elif not image_is_valid:
            st.info("No result for this image. Upload a readable currency image.")
        else:
            try:
                prediction, confidence = predict_currency_image_ui(uploaded_file)
                if prediction == "UNCLEAR":
                    result_style = "result-unclear"
                    word_class = "unclear"
                else:
                    result_style = "result-fake" if prediction == "FAKE" else "result-real"
                    word_class = "fake" if prediction == "FAKE" else ""
                st.markdown(
                    f'<div class="result-empty {result_style}"><div class="result-word {word_class}">{prediction}</div></div>',
                    unsafe_allow_html=True,
                )
                render_confidence_gauge(confidence)
                side_result = (prediction, confidence)
            except Exception as error:
                st.error(f"Unable to analyze this image: {error}")
    return uploaded_file, side_result


def render_currency_detector():
    render_topbar("currency")
    st.markdown('<div class="eyebrow">CNN / Computer Vision</div><div class="feature-title">Fake Currency<br><span style="color:#b45309">Detector</span></div><div class="feature-copy">AI-powered Indian currency visual screening. Upload a note and receive a model-based Real/Fake signal.</div>', unsafe_allow_html=True)
    scan_mode = st.segmented_control(
        "Scan mode",
        options=["Single-Side Scan", "Dual-Side Scan"],
        default="Single-Side Scan",
        key="currency_scan_mode",
        width="stretch",
    )

    if scan_mode == "Single-Side Scan":
        left, right = st.columns([1.05, .95], gap="large")
        with left:
            st.markdown('<div class="section-label"><h2>Scan currency note</h2><span>Supported formats</span></div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload currency image", type=["jpg", "jpeg", "png", "avif", "webp"], key="currency_upload_ui")
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    source_image = Image.open(uploaded_file)
                    source_format = source_image.format or Path(uploaded_file.name).suffix.lstrip(".").upper()
                    display_image = source_image.convert("RGB")
                    st.image(display_image, caption=uploaded_file.name, width="stretch")
                    for column, label, value in zip(st.columns(3), ["File", "Format", "Dimensions"], [uploaded_file.name[:18], source_format, f"{display_image.width} × {display_image.height}"]):
                        with column:
                            render_metric_card(label, value)
                except Exception as error:
                    st.error(f"Unable to display this image: {error}")
        with right:
            st.markdown('<div class="eyebrow">Model output</div><div class="feature-title">AI <span style="color:#0f766e">Result</span></div><div class="feature-copy">The CNN evaluates the image independently of its denomination folder.</div>', unsafe_allow_html=True)
            if uploaded_file is None:
                st.markdown('<div class="result-empty result-pending"><div style="font-size:2rem;color:#64748b">◌</div><div style="color:#1e293b;font-weight:700;margin-top:.55rem">Awaiting image input</div><div style="color:#475569;font-size:.78rem;margin-top:.35rem">Your prediction will appear here</div></div>', unsafe_allow_html=True)
            else:
                try:
                    prediction, confidence = predict_currency_image_ui(uploaded_file)
                    if prediction == "UNCLEAR":
                        st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#92400e;font-size:.75rem;margin-top:.45rem">Please scan the note again.</div></div>', unsafe_allow_html=True)
                    else:
                        word_class = "fake" if prediction == "FAKE" else ""
                        result_style = "result-fake" if prediction == "FAKE" else "result-real"
                        st.markdown(f'<div class="result-empty {result_style}"><div class="result-word {word_class}">{prediction}</div><div style="color:#475569;font-size:.72rem;margin-top:.25rem">CNN classification</div></div>', unsafe_allow_html=True)
                    render_confidence_gauge(confidence)
                    for column, label, value, tone in zip(st.columns(4), ["Input", "Color", "Model", "Output"], ["224 × 224", "RGB", "CNN", "REAL / FAKE"], ["", "", "teal", ""]):
                        with column:
                            render_metric_card(label, value, tone)
                except Exception as error:
                    st.error(f"Unable to analyze this image: {error}")
    else:
        st.markdown(
            """
            <style>
            .st-key-currency-front-upload, .st-key-currency-back-upload {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
                margin-bottom: 0.9rem;
                padding: 1.1rem;
            }
            .st-key-currency-front-result, .st-key-currency-back-result {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
                padding: 1.1rem;
            }
            .scan-side-heading {
                color: #0f172a;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }
            .st-key-currency-final-result {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
                padding: 1.25rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        front_column, back_column = st.columns(2, gap="large")
        with front_column:
            front_file, front_result = render_dual_side(
                "Front Side",
                "currency_front_upload",
                "currency-front-upload",
                "currency-front-result",
            )
        with back_column:
            back_file, back_result = render_dual_side(
                "Back Side",
                "currency_back_upload",
                "currency-back-upload",
                "currency-back-result",
            )

        st.markdown('<div class="section-label"><h2>Final Result</h2></div>', unsafe_allow_html=True)
        with st.container(key="currency-final-result"):
            if front_file is None or back_file is None:
                st.info("Upload both sides of the currency note to see a final result.")
            elif front_result is None or back_result is None:
                st.info("A final result will appear when both uploaded images are readable and processed successfully.")
            elif "UNCLEAR" in (front_result[0], back_result[0]):
                st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#92400e;font-size:.85rem;margin-top:.45rem">At least one side was unclear. Please scan both sides again clearly with good lighting and focus.</div></div>', unsafe_allow_html=True)
            elif front_result[0] != back_result[0]:
                st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#92400e;font-size:.85rem;margin-top:.45rem">The two scanned sides produced different predictions. Please scan both sides again with clear lighting and focus.</div></div>', unsafe_allow_html=True)
            else:
                combined_prediction = f"LIKELY {front_result[0]}"
                combined_confidence = (front_result[1] + back_result[1]) / 2
                result_style = "result-fake" if front_result[0] == "FAKE" else "result-real"
                word_class = "fake" if front_result[0] == "FAKE" else ""
                st.markdown(
                    f'<div class="result-empty {result_style}"><div class="result-word {word_class}">{combined_prediction}</div></div>',
                    unsafe_allow_html=True,
                )
                render_confidence_gauge(combined_confidence)

    st.markdown('<div class="section-label"><h2>AI analysis flow</h2></div>', unsafe_allow_html=True)
    render_pipeline(['<span class="flow-label">IMAGE</span><span class="flow-detail">Upload</span>', '<span class="flow-label">PREPROCESS</span><span class="flow-detail">RGB • 224 × 224</span>', '<span class="flow-label">CNN</span><span class="flow-detail">Visual Analysis</span>', '<span class="flow-label">RESULT</span><span class="flow-detail">REAL / FAKE</span>'])
    st.markdown('<div class="section-label"><h2>Responsible AI</h2><span>Use with care</span></div>', unsafe_allow_html=True)
    render_disclaimer("This AI result is an image-based prediction and should not be treated as a definitive bank-grade counterfeit verification.")


def render_planned_module(module):
    activity = module == "activity"
    icon = "👁️" if activity else "📰"
    title = "Suspicious Activity Detector" if activity else "Fake News / Scam Detector"
    technology = "YOLO • Computer Vision • Rule Engine" if activity else "BERT • NLP"
    description = "A future visual safety workspace for webcam and video analysis, object detection, event timelines, and rule-based alerts." if activity else "A future digital trust workspace for message and news analysis, suspicious indicators, and explainable text classification."
    steps = ["VIDEO / WEBCAM", "PREPROCESSING", "YOLO", "RULE ENGINE", "SAFETY ALERT"] if activity else ["TEXT INPUT", "NLP PREPROCESSING", "BERT", "INDICATORS", "TRUST SIGNAL"]
    responsible = "AI alerts will be decision-support signals and should be verified by a human." if activity else "AI classification may be uncertain and should not replace independent verification."
    render_topbar(module)
    st.markdown(f'<div class="eyebrow">{technology}</div><div class="feature-title">{icon} {title}</div><div class="feature-copy">{description}</div>', unsafe_allow_html=True)
    theme = "theme-activity" if activity else "theme-scam"
    st.markdown(f'<div class="planned-panel {theme}"><div class="planned-icon">{icon}</div><div class="planned-title">Module under development</div><div class="planned-copy">This interface is prepared for the future {technology} integration. No live detection or prediction is being generated yet.</div><div class="planned-chip">Planned / Coming Soon</div></div><div class="section-label"><h2>Planned AI pipeline</h2><span>Architecture ready</span></div>', unsafe_allow_html=True)
    render_pipeline(steps)
    st.markdown('<div class="section-label"><h2>Responsible AI</h2><span>Future safeguards</span></div>', unsafe_allow_html=True)
    render_disclaimer(responsible)


def render_knife_alert(knife_detections, alarm_enabled, cooldown_seconds, event_key):
    from modules.suspicious_activity import claim_alert_event, should_play_alarm

    if not knife_detections:
        return []

    now = time.monotonic()
    last_alarm = st.session_state.get("activity_last_alarm_time")
    if event_key.startswith("image:"):
        recent_events = st.session_state.setdefault("activity_uploaded_alert_events", set())
    else:
        recent_events = st.session_state.setdefault("activity_recent_alert_events", [])
    is_new_event = claim_alert_event(event_key, recent_events)
    labels = ", ".join(f"'{item['class_name']}' ({item['confidence']:.0%})" for item in knife_detections)
    st.warning(
        f"Experimental visual alert — generic COCO knife label: {labels}. "
        "This does not confirm a weapon, threat, or crime."
    )
    if should_play_alarm(alarm_enabled, is_new_event, now, last_alarm, cooldown_seconds):
        try:
            st.audio(load_activity_alarm_audio(), format="audio/wav", autoplay=True)
            st.session_state["activity_last_alarm_time"] = now
        except Exception as error:
            st.caption(f"Sound playback is unavailable ({error}); the visual alert remains active.")
        else:
            st.caption("If browser audio is blocked, the visual alert remains visible.")
    elif alarm_enabled and not is_new_event:
        st.caption("Sound already played for this input; repeated reruns will not replay it.")
    elif alarm_enabled:
        st.caption(f"Sound cooldown active ({cooldown_seconds} seconds).")
    return knife_detections


def render_activity_detection_result(
    frame_rgb,
    person_model,
    crowd_threshold,
    detection_confidence,
    selected_class_ids,
    knife_class_id,
    knife_confidence,
    alarm_enabled,
    cooldown_seconds,
    image_size,
    iou_threshold,
    event_key,
    analysis=None,
):
    from modules.suspicious_activity import annotate_detections, crowding_rule_triggered, detect_frame

    if analysis is None:
        _, person_count, detections, knife_detections = detect_frame(
            frame_rgb,
            person_model,
            confidence_threshold=detection_confidence,
            knife_class_id=knife_class_id,
            knife_confidence_threshold=knife_confidence,
            image_size=image_size,
            iou_threshold=iou_threshold,
            max_detections=300,
        )
    else:
        person_count, detections, knife_detections = analysis
    counted_detections = detections + [
        detection for detection in knife_detections
        if detection not in detections
    ]
    if selected_class_ids:
        selected_ids = set(selected_class_ids)
        visible_detections = [item for item in counted_detections if item["class_id"] in selected_ids]
    else:
        visible_detections = counted_detections
    annotated = annotate_detections(
        frame_rgb,
        visible_detections,
        knife_class_id=knife_class_id,
    )
    class_counts = {}
    for detection in visible_detections:
        class_name = detection["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    original_column, annotated_column = st.columns(2, gap="medium")
    with original_column:
        st.image(frame_rgb, caption="Original image", width="stretch")
    with annotated_column:
        st.image(annotated, caption="Detected objects", width="stretch")

    metrics = st.columns(3)
    with metrics[0]:
        render_metric_card("Total objects", len(visible_detections))
    with metrics[1]:
        render_metric_card("Unique classes", len(class_counts))
    with metrics[2]:
        render_metric_card("People", person_count)
    if crowding_rule_triggered(person_count, crowd_threshold):
        st.warning(f"Crowding review indicator: {person_count} people (threshold {crowd_threshold}).")
    else:
        st.caption(f"Crowding review threshold: {crowd_threshold} people.")
    if class_counts:
        st.caption("Per-class counts")
        st.dataframe(
            [
                {"Class": name.title(), "Count": count}
                for name, count in sorted(class_counts.items())
            ],
            hide_index=True,
            width="stretch",
            height=35 * len(class_counts) + 38,
        )
        st.caption("Detections")
        st.dataframe(
            [
                {
                    "Class": item["class_name"].title(),
                    "Confidence": f"{item['confidence']:.1%}",
                    "Bounding box (x1, y1, x2, y2)": str(item["box"]),
                }
                for item in visible_detections
            ],
            hide_index=True,
            width="stretch",
            height=35 * len(visible_detections) + 38,
        )
    else:
        st.info("No objects detected above the current confidence threshold. Try lowering confidence or using a larger inference image size.")
    if knife_class_id is not None:
        render_knife_alert(knife_detections, alarm_enabled, cooldown_seconds, event_key)
    return person_count, knife_detections


@st.fragment(run_every=1.0)
def render_live_camera_frame(crowd_threshold, detection_confidence, selected_class_ids, knife_class_id, knife_confidence, alarm_enabled, cooldown_seconds):
    from modules.suspicious_activity import camera_diagnostics, camera_frames_unchanged, read_camera_frame

    capture = st.session_state.get("activity_camera")
    if capture is None:
        st.info("Start a camera connection to view live detections.")
        return

    try:
        initial_frame = st.session_state.get("activity_camera_initial_frame")
        if initial_frame is not None:
            frame_rgb = initial_frame
            st.session_state["activity_camera_initial_frame"] = None
        else:
            frame_rgb = read_camera_frame(capture)
        previous_frame = st.session_state.get("activity_camera_previous_frame")
        frame_unchanged = camera_frames_unchanged(previous_frame, frame_rgb)
        if frame_unchanged:
            unchanged_reads = st.session_state.get("activity_camera_unchanged_reads", 0) + 1
        else:
            unchanged_reads = 0
        st.session_state["activity_camera_unchanged_reads"] = unchanged_reads
        st.session_state["activity_camera_previous_frame"] = frame_rgb.copy()
        if unchanged_reads >= 2:
            raise RuntimeError(
                "Camera returned identical pixel data for three consecutive frames. "
                "The feed may be frozen or showing a test placeholder; check the source and restart it."
            )
        diagnostics = camera_diagnostics(
            capture,
            frame_rgb,
            st.session_state.get("activity_camera_source", "unknown camera"),
        )
        st.caption(
            f"Camera: {diagnostics['source']} · Backend: {diagnostics['backend']} · "
            f"Frame: {diagnostics['frame_dimensions']} · Read: "
            f"{'success' if diagnostics['frame_read_success'] else 'failed'} · "
            f"Mean brightness: {diagnostics['mean_brightness']:.1f}/255 · "
            f"Content changing: {'no' if frame_unchanged else 'checking first pair' if previous_frame is None else 'yes'}"
        )
        if diagnostics["mean_brightness"] < 5:
            st.warning("The frame is nearly black. Check that the phone camera app is actively streaming from its video endpoint, the lens is uncovered, and the scene is lit. Verify the URL and protocol shown in the camera app; a successful connection alone does not confirm a usable video feed.")
        person_model = load_activity_model_ui()
        event_key = "camera:" + hashlib.sha256(frame_rgb.tobytes()).hexdigest()
        render_activity_detection_result(
            frame_rgb,
            person_model,
            crowd_threshold,
            detection_confidence,
            selected_class_ids,
            knife_class_id,
            knife_confidence,
            alarm_enabled,
            cooldown_seconds,
            640,
            0.7,
            event_key,
        )
    except Exception as error:
        release_activity_camera()
        st.error(f"Camera or detection error: {error}")
        st.caption("Camera status: stopped and released · Frame read success: failed · Frame dimensions and brightness unavailable. Check that the configured URL is the phone app's actual video stream endpoint, not only its web/control page.")


def render_activity_detector():
    from modules.suspicious_activity import detect_frame, open_camera, probe_local_cameras

    render_topbar("activity")
    st.markdown('<div class="eyebrow">YOLO / Computer Vision</div><div class="feature-title">Suspicious Activity Detection</div><div class="feature-copy">Review supported object detections and crowd-count signals. The model does not infer intent or behavior.</div>', unsafe_allow_html=True)
    mode = "Image Upload"
    try:
        person_model = load_activity_model_ui()
    except Exception as error:
        st.error(f"Unable to load the configured YOLO model: {error}")
        return

    model_names = person_model.names
    class_names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
    model_path = getattr(person_model, "ckpt_path", None)
    model_filename = Path(str(model_path)).name if model_path else "configured YOLO weights"
    class_ids_by_name = {
        str(name): int(class_id)
        for class_id, name in (model_names.items() if isinstance(model_names, dict) else enumerate(model_names))
    }
    knife_class_id = next(
        (class_id for name, class_id in class_ids_by_name.items() if name.strip().lower() == "knife"),
        None,
    )
    with st.expander("Detection settings", expanded=True):
        controls = st.columns(4)
        with controls[0]:
            detection_confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.18, step=0.01, key="activity_detection_confidence")
        with controls[1]:
            iou_threshold = st.slider("IoU / NMS threshold", min_value=0.30, max_value=0.90, value=0.70, step=0.05, key="activity_iou_threshold", help="Higher values suppress fewer overlapping nearby detections.")
        with controls[2]:
            image_size = st.selectbox("Inference image size", options=[640, 960, 1280], index=0, key="activity_image_size", help="Use 960 or 1280 for high-resolution images when seeking small objects; higher sizes take longer and may not help low-resolution images.")
        with controls[3]:
            crowd_threshold = st.number_input("Crowding review threshold", min_value=1, max_value=50, value=5, step=1, key="activity_crowd_threshold")
        knife_confidence = 0.65
        alarm_enabled = False
        cooldown_seconds = 10
        if knife_class_id is not None:
            alert_controls = st.columns(3)
            with alert_controls[0]:
                knife_confidence = st.slider("Experimental knife-label threshold", min_value=0.05, max_value=0.99, value=0.65, step=0.05, key="activity_knife_confidence")
            with alert_controls[1]:
                alarm_enabled = st.checkbox("Audible alert", value=True, key="activity_alarm_enabled")
            with alert_controls[2]:
                cooldown_seconds = st.number_input("Alert sound cooldown (seconds)", min_value=3, max_value=120, value=10, step=1, key="activity_alarm_cooldown")
        selected_class_names = st.multiselect(
            "Object classes to show",
            options=class_names,
            default=[],
            key="activity_class_filter",
            help="Display filter only. Leave empty to show every detected class.",
        )
    selected_class_ids = [class_ids_by_name[name] for name in selected_class_names] or None
    st.caption(f"Model: {model_filename} · {len(class_names)} supported classes.")
    with st.expander("Supported model classes"):
        st.caption(", ".join(class_names))
    if "mirror" not in {name.strip().lower() for name in class_names}:
        st.caption("Mirrors are not included in this model’s classes; detecting them may require custom-trained weights.")
    if knife_class_id is not None:
        st.caption("The generic COCO “knife” label is experimental and does not confirm a weapon, threat, or crime.")
    else:
        st.caption("This model has no knife class; knife alerts are unavailable.")

    if mode == "Image Upload":
        release_activity_camera()
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            key="activity_image_upload",
            help="YOLO will check all supported model classes.",
        )
        if uploaded_file is None:
            st.caption("Supported formats: JPG, PNG, WEBP.")
            return
        try:
            image_bytes = uploaded_file.getvalue()
            source_image = Image.open(uploaded_file).convert("RGB")
        except (OSError, ValueError):
            st.error("Unable to read this image. Please upload a valid JPG, PNG, or WEBP file.")
            return
        source_hash = hashlib.sha256(image_bytes).hexdigest()
        analyze_clicked = st.button(
            "Analyze Image",
            key="activity_analyze_image",
            type="primary",
            width="stretch",
        )
        if analyze_clicked:
            try:
                with st.spinner("Detecting supported objects..."):
                    _, person_count, detections, knife_detections = detect_frame(
                        np.asarray(source_image),
                        person_model,
                        confidence_threshold=detection_confidence,
                        knife_class_id=knife_class_id,
                        knife_confidence_threshold=knife_confidence,
                        image_size=image_size,
                        iou_threshold=iou_threshold,
                        max_detections=300,
                    )
                st.session_state["activity_image_analysis"] = {
                    "source_hash": source_hash,
                    "detection_confidence": detection_confidence,
                    "knife_confidence": knife_confidence,
                    "image_size": image_size,
                    "iou_threshold": iou_threshold,
                    "person_count": person_count,
                    "detections": detections,
                    "knife_detections": knife_detections,
                }
            except Exception as error:
                st.session_state.pop("activity_image_analysis", None)
                st.error(f"Unable to analyze this image: {error}")
                return

        analysis = st.session_state.get("activity_image_analysis")
        analysis_is_current = (
            analysis
            and analysis["source_hash"] == source_hash
            and analysis["detection_confidence"] == detection_confidence
            and analysis["knife_confidence"] == knife_confidence
            and analysis["image_size"] == image_size
            and analysis["iou_threshold"] == iou_threshold
        )
        if not analysis_is_current:
            st.image(source_image, caption=uploaded_file.name, width="stretch")
            if not analyze_clicked:
                st.caption("Select **Analyze Image** to run detection or apply changed inference settings.")
            return
        try:
            render_activity_detection_result(
                np.asarray(source_image),
                person_model,
                int(crowd_threshold),
                detection_confidence,
                selected_class_ids,
                knife_class_id,
                knife_confidence,
                alarm_enabled,
                int(cooldown_seconds),
                image_size,
                iou_threshold,
                "image:" + source_hash,
                analysis=(
                    analysis["person_count"],
                    analysis["detections"],
                    analysis["knife_detections"],
                ),
            )
        except Exception as error:
            st.error(f"Unable to analyze this image: {error}")
        return

    st.markdown('<div class="section-label"><h2>Mobile / local camera</h2><span>One frame per second</span></div>', unsafe_allow_html=True)
    source_type = st.radio("Camera source", ["Mobile IP stream", "Local webcam"], horizontal=True, key="activity_camera_source_type")
    local_camera_backend = "MSMF"
    if source_type == "Mobile IP stream":
        camera_source = st.text_input(
            "Camera stream URL",
            placeholder="Paste the exact URL shown by your camera app or Iriun Webcam",
            key="activity_camera_url",
            help="Use the exact stream URL displayed by the camera app or virtual webcam driver. Do not guess a default IP or path; protocol and endpoint names vary by app setup.",
        ).strip()
        st.caption("For a mobile or virtual camera setup, use the exact stream URL and protocol shown by that app. This field does not assume a fixed default endpoint.")
        source_value = camera_source
    else:
        source_value = int(st.number_input("Webcam device index", min_value=0, max_value=10, value=0, step=1, key="activity_camera_device_index", help="Do not assume index 1 is the active webcam. Use the diagnosis below and select the index that actually shows changing live image content."))
        local_camera_backend = st.selectbox(
            "Capture backend",
            options=["Any", "MSMF", "DirectShow"],
            key="activity_camera_backend",
            help="Windows virtual cameras may appear only through a specific backend. The app tests the selected backend and only accepts a real, changing image frame.",
        )
        scan_cameras = st.button(
            "Scan local cameras (indices 0-3)",
            key="activity_scan_local_cameras",
            disabled=st.session_state.get("activity_camera") is not None,
        )
        if scan_cameras:
            with st.spinner("Testing camera indices 0-3 on MSMF and DirectShow; each device will be released after its probe..."):
                st.session_state["activity_camera_scan_results"] = probe_local_cameras(
                    device_indices=(0, 1, 2, 3),
                    frame_count=4,
                )
        scan_results = st.session_state.get("activity_camera_scan_results")
        if scan_results:
            st.dataframe(scan_results, hide_index=True, width="stretch")
            st.caption("OpenCV device indexes are backend-specific and do not reliably map to Windows virtual-camera names. Select the device/backend that shows usable, changing image content. If none do, the Iriun Webcam driver or connection must be repaired before integration can work.")

    start_column, stop_column = st.columns(2)
    with start_column:
        start_camera = st.button("Start Camera", key="activity_start_camera", width="stretch", disabled=st.session_state.get("activity_camera") is not None)
    with stop_column:
        stop_camera = st.button("Stop Camera", key="activity_stop_camera", width="stretch", disabled=st.session_state.get("activity_camera") is None)

    if start_camera:
        if isinstance(source_value, str) and not source_value:
            st.error("Enter a camera stream URL before starting.")
        else:
            release_activity_camera()
            try:
                with st.spinner("Connecting to camera..."):
                    if source_type == "Local webcam":
                        capture, initial_frame = open_camera(source_value, backend=local_camera_backend)
                    else:
                        capture, initial_frame = open_camera(source_value)
                st.session_state["activity_camera"] = capture
                st.session_state["activity_camera_initial_frame"] = initial_frame
                st.session_state["activity_camera_source_label"] = source_type
                st.session_state["activity_camera_source"] = source_value
                st.rerun()
            except Exception as error:
                if source_type == "Local webcam":
                    guidance = (
                        f" Local device index {source_value} returned no usable live video. "
                        "Try another index or backend. If using Iriun Webcam, confirm the virtual camera driver is installed, the stream is active, and the Windows device is actually exposing a live RGB frame; no index is assumed to be Iriun."
                    )
                else:
                    guidance = (
                        " Check the exact stream URL and protocol shown by your camera app or Iriun Webcam driver. "
                        "The URL is version- and setup-specific; this app does not assume a fixed default endpoint."
                    )
                st.error(f"Unable to connect to camera: {error}{guidance}")
    if stop_camera:
        release_activity_camera()
        st.rerun()

    if st.session_state.get("activity_camera") is not None:
        st.caption(f"Connected: {st.session_state.get('activity_camera_source_label', 'camera')}. Stop the camera before leaving this mode.")
        render_live_camera_frame(int(crowd_threshold), detection_confidence, selected_class_ids, knife_class_id, knife_confidence, alarm_enabled, int(cooldown_seconds))

    st.caption("Object labels and crowd counts are approximate visual signals and require human context.")


def render_fake_news_detector():
    from modules.fake_news_detection import predict_news

    render_topbar("scam")
    st.markdown(
        '<div class="eyebrow">TF-IDF + Linear SVM / Text Analysis</div>'
        '<div class="feature-title">Fake News Detection</div>'
        '<div class="feature-copy">Classify English or Hindi news text with the existing trained model.</div>',
        unsafe_allow_html=True,
    )
    input_column, result_column = st.columns([1.1, 0.9], gap="large")
    with input_column:
        st.markdown(
            '<div class="section-label"><h2>News article</h2>'
            '<span>English or Hindi</span></div>',
            unsafe_allow_html=True,
        )
        language_label = st.selectbox(
            "Article language",
            options=("English", "Hindi"),
            key="fake_news_language",
        )
        language = language_label.casefold()
        news_text = st.text_area(
            "Article text",
            height=260,
            placeholder=(
                "Paste the article text to classify..."
                if language == "english"
                else "वर्गीकृत करने के लिए समाचार लेख का पाठ यहाँ लिखें..."
            ),
            key="fake_news_text",
        )
        analyze_clicked = st.button(
            "Analyze Article",
            key="fake_news_analyze",
            type="primary",
            width="stretch",
        )

    input_fingerprint = hashlib.sha256(
        f"{language}\0{news_text}".encode("utf-8")
    ).hexdigest()
    if analyze_clicked:
        st.session_state.pop("fake_news_prediction", None)
        if not news_text.strip():
            st.warning("Enter non-empty news text before running the classifier.")
        else:
            try:
                prediction = predict_news(language, news_text)
            except ValueError as error:
                st.warning(f"Unable to classify this input: {error}")
            except FileNotFoundError as error:
                st.error(f"Fake-news model artifacts are unavailable: {error}")
            else:
                st.session_state["fake_news_prediction"] = {
                    "input_fingerprint": input_fingerprint,
                    "result": prediction,
                }

    with result_column:
        st.markdown(
            '<div class="section-label"><h2>Model output</h2>'
            '<span>Decision support</span></div>',
            unsafe_allow_html=True,
        )
        stored_prediction = st.session_state.get("fake_news_prediction")
        if (
            stored_prediction
            and stored_prediction["input_fingerprint"] == input_fingerprint
        ):
            result = stored_prediction["result"]
            if result["label"] == 1:
                st.error(f"Model prediction: {result['prediction']}")
            else:
                st.success(f"Model prediction: {result['prediction']}")
            st.metric("Signed decision score", f"{result['decision_score']:+.3f}")
            st.caption(result["decision_score_explanation"])
        else:
            st.markdown(
                '<div class="result-empty"><div style="color:#334155;font-weight:650">'
                'Awaiting article text</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("The result updates only after you select Analyze Article.")

    st.markdown(
        '<div class="section-label"><h2>AI analysis flow</h2></div>',
        unsafe_allow_html=True,
    )
    render_pipeline(
        [
            '<span class="flow-label">TEXT</span><span class="flow-detail">English / Hindi</span>',
            '<span class="flow-label">FEATURES</span><span class="flow-detail">Language TF-IDF</span>',
            '<span class="flow-label">MODEL</span><span class="flow-detail">Linear SVM</span>',
            '<span class="flow-label">OUTPUT</span><span class="flow-detail">REAL / FAKE</span>',
        ]
    )
    render_disclaimer(
        "This dataset-trained classifier is not independent fact-checking. Its signed "
        "Linear SVM decision score is not a calibrated probability; verify claims "
        "against reliable sources."
    )


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["authenticated"]:
    render_auth_page()
    st.stop()

if "module" not in st.session_state:
    st.session_state["module"] = None
active_module = st.session_state["module"]
render_sidebar(active_module)
if active_module is None:
    render_dashboard()
elif active_module == "currency":
    render_currency_detector()
elif active_module == "activity":
    render_activity_detector()
elif active_module == "scam":
    render_fake_news_detector()
else:
    render_topbar(active_module)
    st.error(f"Unknown module selection: {active_module}")