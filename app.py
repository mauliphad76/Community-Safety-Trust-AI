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
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#f4f1ff; --muted:#9692aa; --subtle:#68647b; --line:rgba(255,255,255,.09); --purple:#a875ff; --bright:#c49aff; --green:#66e3a5; --amber:#f5bd69; --cyan:#54d9ff; --mint:#66e3a5; --red:#ff7e96; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { color:var(--ink); background-color:#08090d; background-image:linear-gradient(rgba(84,217,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(84,217,255,.025) 1px,transparent 1px),radial-gradient(circle at 92% 3%,rgba(127,77,216,.18),transparent 27%),radial-gradient(circle at 7% 28%,rgba(42,104,165,.11),transparent 25%); background-size:42px 42px,42px 42px,auto,auto; }
    [data-testid="stHeader"] { background:transparent; } [data-testid="stMainBlockContainer"] { max-width:1380px; padding-top:2rem; } .block-container { padding-left:3.5rem; padding-right:3.5rem; }
    h1,h2,h3,h4,p { letter-spacing:0; } h1,h2,h3,h4 { color:var(--ink); } h1,h2,h3 { font-weight:800; } p,li { color:var(--muted); } hr { border-color:var(--line); }
    .topbar { align-items:center; display:flex; justify-content:space-between; margin-bottom:2.4rem; } .brand { align-items:center; display:flex; gap:.8rem; } .brand-mark { align-items:center; background:linear-gradient(145deg,#352052,#1a1625); border:1px solid rgba(196,154,255,.35); border-radius:13px; box-shadow:0 0 28px rgba(168,117,255,.16); display:grid; font-size:1.35rem; height:42px; justify-content:center; width:42px; } .brand-name { color:var(--ink); font-size:.87rem; font-weight:800; letter-spacing:.11em; line-height:1.25; } .brand-caption,.online,.eyebrow,.section-label span,.module-tech,.status-online,.status-planned,.metric-label,.card-kicker { font-family:'DM Mono',monospace; } .brand-caption { color:var(--subtle); font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; } .online { color:var(--green); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; } .online:before { background:var(--green); border-radius:50%; box-shadow:0 0 12px var(--green); content:''; display:inline-block; height:7px; margin-right:.45rem; width:7px; }
    .eyebrow { color:var(--bright); font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; } .hero-title { color:var(--ink); font-size:clamp(2.1rem,4vw,4.3rem); font-weight:800; line-height:1.02; margin:.55rem 0 .85rem; } .hero-copy,.feature-copy { color:var(--muted); line-height:1.7; max-width:640px; } .hero-copy { font-size:1rem; } .hero-copy strong { color:var(--ink); } .hero-orbit { color:var(--bright); font-size:5rem; opacity:.75; text-align:right; text-shadow:0 0 45px rgba(168,117,255,.4); }
    .section-label { align-items:center; display:flex; justify-content:space-between; margin:2.5rem 0 1rem; } .section-label h2 { font-size:1.05rem; margin:0; } .section-label span { color:var(--subtle); font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; }
    .module-card,.metric-card,.analysis-card,.pipeline-card,.planned-panel { background:linear-gradient(145deg,rgba(32,27,48,.9),rgba(17,15,25,.86)); border:1px solid var(--line); border-radius:20px; box-shadow:0 22px 60px rgba(0,0,0,.2); } .module-card { min-height:270px; overflow:hidden; padding:1.35rem; position:relative; } .module-card:after { background:radial-gradient(circle,var(--card-glow,rgba(168,117,255,.16)),transparent 67%); content:''; height:150px; position:absolute; right:-65px; top:-75px; width:150px; } .module-card.theme-currency { --card-accent:var(--amber); --card-glow:rgba(245,189,105,.18); } .module-card.theme-activity { --card-accent:var(--cyan); --card-glow:rgba(84,217,255,.16); } .module-card.theme-scam { --card-accent:var(--mint); --card-glow:rgba(102,227,165,.16); } .module-card .module-icon { align-items:center; background:color-mix(in srgb,var(--card-accent, var(--purple)) 13%,transparent); border:1px solid color-mix(in srgb,var(--card-accent, var(--purple)) 28%,transparent); border-radius:13px; display:flex; font-size:1.25rem; height:43px; justify-content:center; width:43px; } .module-title { color:var(--ink); font-size:1.16rem; font-weight:700; margin-top:1.2rem; } .module-tech { color:var(--card-accent,var(--bright)); font-size:.65rem; letter-spacing:.04em; margin-top:.28rem; text-transform:uppercase; } .module-description { color:var(--muted); font-size:.78rem; line-height:1.6; margin:1.25rem 0; min-height:50px; } .status-online,.status-planned { font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; } .status-online { color:var(--green); } .status-planned { color:var(--amber); } .module-meta { bottom:1.3rem; color:var(--subtle); display:flex; font-family:'DM Mono',monospace; font-size:.58rem; gap:1.2rem; letter-spacing:.07em; position:absolute; text-transform:uppercase; } .module-meta strong { color:var(--card-accent,var(--muted)); display:block; font-size:.68rem; margin-top:.22rem; }
    .metric-card { min-height:92px; padding:1.1rem 1.25rem; } .metric-label { color:var(--subtle); font-size:.62rem; letter-spacing:.1em; text-transform:uppercase; } .metric-value { color:var(--ink); font-size:1.65rem; font-weight:800; margin-top:.3rem; } .metric-value.purple { color:var(--bright); } .metric-value.green { color:var(--green); } .metric-value.amber { color:var(--amber); }
    .feature-title { color:var(--ink); font-size:clamp(1.8rem,3vw,3rem); font-weight:800; line-height:1.05; margin:.45rem 0 .6rem; } .analysis-card { min-height:390px; padding:1.35rem; } .card-kicker { color:var(--subtle); font-size:.63rem; letter-spacing:.12em; text-transform:uppercase; } .upload-zone,.result-empty { align-items:center; border-radius:15px; display:flex; justify-content:center; min-height:210px; text-align:center; } .upload-zone { background:rgba(168,117,255,.055); border:1px dashed rgba(196,154,255,.28); padding:1.1rem; } .result-empty { background:rgba(255,255,255,.025); border:1px solid var(--line); flex-direction:column; } .result-word { color:var(--green); font-size:2.6rem; font-weight:800; letter-spacing:.04em; } .result-word.fake { color:var(--red); } .confidence-number { color:var(--ink); font-family:'DM Mono',monospace; font-size:1.7rem; } .confidence-label { color:var(--muted); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; } .gauge { margin:.9rem auto .35rem; position:relative; width:154px; } .gauge svg { display:block; transform:rotate(-90deg); } .gauge-bg,.gauge-fill { fill:none; stroke-linecap:round; stroke-width:10; } .gauge-bg { stroke:rgba(255,255,255,.08); } .gauge-fill { stroke:var(--bright); } .gauge-value { align-items:center; display:flex; flex-direction:column; inset:0; justify-content:center; position:absolute; }
    .pipeline-card { padding:1.25rem; } .pipeline { align-items:center; display:flex; gap:.35rem; justify-content:space-between; overflow-x:auto; padding:.8rem 0 .15rem; } .pipeline-step { align-items:center; display:flex; flex:1; gap:.35rem; min-width:100px; } .pipeline-node { align-items:center; background:rgba(168,117,255,.12); border:1px solid rgba(168,117,255,.27); border-radius:11px; color:var(--bright); display:flex; font-family:'DM Mono',monospace; font-size:.58rem; justify-content:center; min-height:55px; padding:.45rem; text-align:center; width:100%; } .pipeline-arrow { color:var(--subtle); font-size:1.2rem; } .disclaimer { background:rgba(245,189,105,.06); border:1px solid rgba(245,189,105,.18); border-radius:14px; color:#c9bda9; font-size:.72rem; line-height:1.55; padding:.85rem 1rem; } .planned-panel { min-height:300px; padding:2rem; text-align:center; } .planned-panel.theme-activity { border-color:rgba(84,217,255,.3); box-shadow:0 0 40px rgba(84,217,255,.08); } .planned-panel.theme-scam { border-color:rgba(102,227,165,.3); box-shadow:0 0 40px rgba(102,227,165,.08); } .planned-icon { font-size:2.5rem; margin-bottom:.8rem; } .planned-title { color:var(--ink); font-size:1.5rem; font-weight:800; } .planned-copy { color:var(--muted); line-height:1.7; margin:.7rem auto 1.4rem; max-width:580px; } .planned-chip { border:1px solid rgba(245,189,105,.3); border-radius:999px; color:var(--amber); display:inline-block; font-family:'DM Mono',monospace; font-size:.64rem; letter-spacing:.1em; padding:.5rem .8rem; text-transform:uppercase; }
    .stButton > button { background:rgba(168,117,255,.11); border:1px solid rgba(196,154,255,.25); border-radius:11px; color:var(--ink); font-family:'Manrope',sans-serif; font-size:.78rem; font-weight:700; min-height:2.45rem; transition:all .2s ease; } .stButton > button:hover { background:rgba(168,117,255,.23); border-color:var(--bright); color:white; transform:translateY(-1px); } [data-testid="stFileUploaderDropzone"] { background:transparent; border:0; } [data-testid="stImage"] img { border:1px solid var(--line); border-radius:14px; } [data-testid="stMetric"] { background:rgba(255,255,255,.025); border:1px solid var(--line); border-radius:12px; padding:.75rem; } [data-testid="stMetricLabel"] { color:var(--subtle); } [data-testid="stMetricValue"] { color:var(--ink); font-family:'DM Mono',monospace; font-size:1.05rem; }
    @media (max-width:760px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero-orbit { display:none; } .pipeline { align-items:stretch; flex-direction:column; } .pipeline-step { width:100%; } .pipeline-arrow { display:none; } }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#f1f4f5; --muted:#a8b2b8; --subtle:#79878e; --line:rgba(232,242,246,.13); --purple:#74cfdf; --bright:#9adce7; --green:#74d6ac; --amber:#e9ae70; --cyan:#83cbd8; --mint:#83d5b5; --red:#e98984; }
    html,body,[class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { background:#080d11; background-image:radial-gradient(ellipse at 4% 10%,rgba(151,104,67,.12),transparent 38%),radial-gradient(ellipse at 96% 8%,rgba(60,126,142,.14),transparent 36%),linear-gradient(180deg,#0b1116 0%,#080d11 58%,#070b0e 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stMainBlockContainer"] { max-width:1380px; padding-top:1.6rem; }
    .block-container { padding-left:3rem; padding-right:3rem; }
    h1,h2,h3,h4,p { letter-spacing:0; }
    h1,h2,h3,h4 { color:var(--ink); }
    p,li { color:var(--muted); }
    hr { border-color:var(--line); }
    .topbar { align-items:center; display:flex; justify-content:flex-start; margin-bottom:1.2rem; min-height:48px; }
    .brand { align-items:center; display:flex; gap:.75rem; }
    .brand-mark { align-items:center; background:linear-gradient(145deg,rgba(238,247,249,.13),rgba(121,160,171,.06)); border:1px solid rgba(226,240,244,.19); border-radius:12px; box-shadow:inset 0 1px rgba(255,255,255,.12),0 8px 22px rgba(0,0,0,.18); display:grid; font-size:1.15rem; height:38px; justify-content:center; width:38px; }
    .brand-name { color:var(--ink); font-size:.82rem; font-weight:700; letter-spacing:.07em; line-height:1.3; }
    .brand-caption,.eyebrow,.section-label span,.module-tech,.status-online,.status-planned,.metric-label,.card-kicker { font-family:'DM Mono',monospace; }
    .brand-caption { color:var(--subtle); font-size:.62rem; letter-spacing:.04em; }
    .online { display:none; }
    .eyebrow { color:#9ccbd2; font-size:.66rem; letter-spacing:.13em; text-transform:uppercase; }
    .hero-title { color:var(--ink); font-size:3rem; font-weight:700; line-height:1.06; margin:.5rem 0 .7rem; }
    .hero-copy,.feature-copy { color:var(--muted); line-height:1.65; max-width:600px; }
    .hero-copy { font-size:.92rem; }
    .hero-copy strong { color:#d6e0e3; font-weight:600; }
    .hero-visual { align-items:center; animation:hero-arrive .7s ease-out both; display:flex; height:158px; justify-content:center; margin:.1rem auto 0; max-width:400px; position:relative; }
    .hero-visual:before,.hero-visual:after { border:1px solid rgba(178,219,225,.17); border-radius:50%; content:''; height:126px; left:50%; position:absolute; top:50%; transform:translate(-50%,-50%); width:126px; }
    .hero-visual:after { height:155px; opacity:.6; width:260px; }
    .hero-core { align-items:center; animation:core-pulse 4.8s ease-in-out infinite; background:radial-gradient(circle at 38% 30%,rgba(232,250,252,.26),rgba(89,145,156,.13) 43%,rgba(14,27,33,.8) 75%); border:1px solid rgba(206,237,240,.45); border-radius:50%; box-shadow:0 10px 32px rgba(0,0,0,.27),inset 0 1px rgba(255,255,255,.3),0 0 22px rgba(113,197,207,.16); color:#e8f7f7; display:flex; font-family:'DM Mono',monospace; font-size:1.65rem; font-weight:500; height:72px; justify-content:center; position:relative; text-shadow:0 1px 8px rgba(255,255,255,.25); width:72px; z-index:1; }
    .hero-hand { filter:drop-shadow(0 5px 10px rgba(0,0,0,.35)); font-size:3.1rem; position:relative; z-index:2; }
    .hero-hand.robot { margin-right:-.1rem; transform:rotate(-14deg); }
    .hero-hand.human { margin-left:-.1rem; transform:rotate(8deg); }
    .section-label { align-items:center; display:flex; justify-content:space-between; margin:1.5rem 0 .9rem; }
    .section-label h2 { font-size:1.02rem; font-weight:600; margin:0; }
    .section-label span { color:var(--subtle); font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; }
    .module-card,.metric-card,.analysis-card,.pipeline-card,.planned-panel { background:linear-gradient(145deg,rgba(245,250,251,.075),rgba(153,183,189,.035)); border:1px solid var(--line); border-radius:26px; box-shadow:0 16px 38px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.14); backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px); }
    .module-card { display:flex; flex-direction:column; min-height:252px; overflow:hidden; padding:1.35rem; position:relative; transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease; }
    .module-card:before { background:linear-gradient(115deg,rgba(255,255,255,.075),transparent 42%); content:''; inset:0; pointer-events:none; position:absolute; }
    .module-card:after { background:radial-gradient(circle,var(--card-glow),transparent 68%); content:''; height:180px; opacity:.72; position:absolute; right:-90px; top:-105px; width:180px; }
    .module-card:hover { border-color:color-mix(in srgb,var(--card-accent) 52%,rgba(255,255,255,.2)); box-shadow:0 20px 42px rgba(0,0,0,.26),0 0 25px color-mix(in srgb,var(--card-accent) 10%,transparent),inset 0 1px rgba(255,255,255,.2); transform:translateY(-3px); }
    .module-card.theme-currency { --card-accent:#e9ae70; --card-glow:rgba(233,174,112,.18); background:linear-gradient(145deg,rgba(111,75,45,.2),rgba(228,237,238,.04)); border-color:rgba(233,174,112,.24); }
    .module-card.theme-activity { --card-accent:#83cbd8; --card-glow:rgba(131,203,216,.17); background:linear-gradient(145deg,rgba(54,103,119,.2),rgba(228,237,238,.04)); border-color:rgba(131,203,216,.24); }
    .module-card.theme-scam { --card-accent:#83d5b5; --card-glow:rgba(131,213,181,.16); background:linear-gradient(145deg,rgba(51,105,82,.2),rgba(228,237,238,.04)); border-color:rgba(131,213,181,.23); }
    .module-card .module-icon { align-items:center; background:rgba(255,255,255,.075); border:1px solid rgba(255,255,255,.13); border-radius:13px; display:flex; font-size:1.12rem; height:40px; justify-content:center; position:relative; width:40px; z-index:1; }
    .module-title { color:var(--ink); font-size:1.12rem; font-weight:650; line-height:1.35; margin-top:.85rem; position:relative; z-index:1; }
    .module-tech { color:var(--card-accent); font-size:.62rem; letter-spacing:.04em; margin-top:.22rem; text-transform:uppercase; }
    .module-description { color:#aebbc0; flex:1; font-size:.79rem; line-height:1.55; margin:.8rem 0 .65rem; }
    .status-online,.status-planned { font-size:.61rem; letter-spacing:.06em; text-transform:uppercase; }
    .status-online { color:var(--card-accent); }
    .status-planned { color:#b6c5c7; }
    .module-meta { display:none; }
    .metric-card { min-height:78px; padding:.85rem .9rem; }
    .metric-label { color:var(--subtle); font-size:.58rem; letter-spacing:.07em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font-size:1.1rem; font-weight:650; margin-top:.28rem; overflow-wrap:anywhere; }
    .metric-value.purple { color:var(--cyan); }
    .metric-value.cyan { color:var(--cyan); }
    .feature-title { color:var(--ink); font-size:2.35rem; font-weight:700; line-height:1.08; margin:.42rem 0 .6rem; }
    .analysis-card { padding:1.2rem; }
    .card-kicker { color:var(--subtle); font-size:.62rem; letter-spacing:.1em; text-transform:uppercase; }
    .result-empty { align-items:center; background:linear-gradient(145deg,rgba(238,248,249,.075),rgba(121,157,163,.045)); border:1px solid rgba(230,243,245,.15); border-radius:24px; box-shadow:0 14px 36px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.14); display:flex; flex-direction:column; justify-content:center; min-height:238px; text-align:center; backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); }
    .result-empty.result-real { background:linear-gradient(145deg,rgba(76,132,105,.19),rgba(235,246,240,.035)); border-color:rgba(131,213,181,.32); box-shadow:0 14px 36px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.16),0 0 26px rgba(131,213,181,.07); }
    .result-empty.result-fake { background:linear-gradient(145deg,rgba(143,70,69,.2),rgba(247,235,234,.035)); border-color:rgba(233,137,132,.34); box-shadow:0 14px 36px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.16),0 0 26px rgba(233,137,132,.08); }
    .result-empty.result-unclear { background:linear-gradient(145deg,rgba(143,104,57,.19),rgba(248,241,229,.035)); border-color:rgba(233,174,112,.34); }
    .result-word { color:var(--mint); font-size:2.8rem; font-weight:700; }
    .result-word.fake { color:var(--red); }
    .result-word.unclear { color:var(--amber); }
    .confidence-number { color:var(--ink); font-family:'DM Mono',monospace; font-size:1.55rem; }
    .confidence-label { color:var(--muted); font-size:.65rem; letter-spacing:.09em; text-transform:uppercase; }
    .gauge { margin:.7rem auto .3rem; position:relative; width:148px; }
    .gauge svg { display:block; transform:rotate(-90deg); }
    .gauge-bg,.gauge-fill { fill:none; stroke-linecap:round; stroke-width:8; }
    .gauge-bg { stroke:rgba(230,242,245,.13); }
    .gauge-fill { filter:drop-shadow(0 1px 5px rgba(233,174,112,.25)); stroke:var(--amber); }
    .gauge-value { align-items:center; display:flex; flex-direction:column; inset:0; justify-content:center; position:absolute; }
    .pipeline { align-items:center; display:flex; gap:.35rem; justify-content:space-between; margin:.2rem auto 1.4rem; max-width:960px; width:100%; }
    .pipeline-step { align-items:center; display:flex; flex:1; gap:.32rem; min-width:0; }
    .pipeline-node { align-items:center; background:linear-gradient(145deg,rgba(237,249,250,.095),rgba(121,162,168,.045)); border:1px solid rgba(224,241,243,.16); border-radius:18px; box-shadow:inset 0 1px rgba(255,255,255,.16),0 8px 20px rgba(0,0,0,.12); color:#dce9eb; display:flex; flex-direction:column; font-family:'DM Mono',monospace; font-size:.65rem; gap:.35rem; justify-content:center; min-height:70px; min-width:0; padding:.55rem .35rem; text-align:center; width:100%; backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px); }
    .flow-label { color:#bac9cc; font-size:.55rem; letter-spacing:.08em; }
    .flow-detail { color:#ecf3f3; font-family:'Manrope',sans-serif; font-size:.75rem; font-weight:600; line-height:1.25; overflow-wrap:anywhere; }
    .pipeline-arrow { color:rgba(222,237,239,.55); flex:0 0 auto; font-size:.9rem; }
    .disclaimer { background:rgba(230,242,244,.035); border:1px solid rgba(230,242,244,.1); border-radius:16px; color:#9daeb2; font-size:.7rem; line-height:1.55; padding:.7rem .9rem; backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); }
    .planned-panel { min-height:260px; padding:1.8rem; text-align:center; }
    .planned-panel.theme-activity { border-color:rgba(131,203,216,.27); }
    .planned-panel.theme-scam { border-color:rgba(131,213,181,.25); }
    .planned-icon { font-size:2.25rem; margin-bottom:.7rem; }
    .planned-title { color:var(--ink); font-size:1.35rem; font-weight:700; }
    .planned-copy { color:var(--muted); line-height:1.65; margin:.65rem auto 1.2rem; max-width:580px; }
    .planned-chip { background:rgba(255,255,255,.04); border:1px solid rgba(233,174,112,.22); border-radius:999px; color:#d8b489; display:inline-block; font-family:'DM Mono',monospace; font-size:.61rem; letter-spacing:.08em; padding:.42rem .72rem; text-transform:uppercase; }
    .stButton > button { background:rgba(234,244,246,.055); border:1px solid rgba(229,241,243,.16); border-radius:999px; color:#e1ebed; font-family:'Manrope',sans-serif; font-size:.78rem; font-weight:600; min-height:2.55rem; transition:background .22s ease,border-color .22s ease,box-shadow .22s ease,transform .22s ease; }
    .stButton > button:hover { background:rgba(234,244,246,.1); border-color:rgba(229,241,243,.3); box-shadow:0 8px 20px rgba(0,0,0,.18); color:#fff; transform:translateY(-1px); }
    .stButton > button:focus-visible { box-shadow:0 0 0 3px rgba(152,207,216,.22); }
    [data-testid="stFileUploader"] { background:linear-gradient(145deg,rgba(241,249,250,.07),rgba(130,164,170,.035)); border:1px solid rgba(233,174,112,.24); border-radius:22px; padding:1rem; backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); }
    [data-testid="stFileUploaderDropzone"] { background:rgba(240,249,250,.025); border:1px dashed rgba(233,174,112,.32); border-radius:16px; }
    [data-testid="stFileUploaderDropzone"]:hover { background:rgba(233,174,112,.055); border-color:rgba(233,174,112,.52); }
    [data-testid="stFileUploaderDropzone"] button { background:rgba(240,249,250,.075); border-color:rgba(240,249,250,.16); border-radius:999px; }
    [data-testid="stImage"] img { border:1px solid rgba(230,242,244,.16); border-radius:20px; }
    [data-testid="stMetric"] { background:rgba(240,249,250,.045); border:1px solid rgba(230,242,244,.12); border-radius:16px; padding:.7rem; }
    [data-testid="stMetricLabel"] { color:var(--subtle); }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:'DM Mono',monospace; font-size:1rem; }
    @keyframes hero-arrive { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    @keyframes core-pulse { 50% { box-shadow:0 12px 34px rgba(0,0,0,.3),inset 0 1px rgba(255,255,255,.35),0 0 25px rgba(113,197,207,.23); } }
    @media (max-width:900px) { .block-container { padding-left:1.5rem; padding-right:1.5rem; } .hero-title { font-size:2.5rem; } .module-card { padding:1.15rem; } }
    @media (max-width:760px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero-title { font-size:2.2rem; } .hero-visual { height:126px; margin-top:.8rem; } .hero-hand { font-size:2.55rem; } .hero-core { height:60px; width:60px; } .hero-visual:after { height:138px; width:220px; } .module-card { min-height:220px; } .feature-title { font-size:1.9rem; } .pipeline { gap:.12rem; } .pipeline-step { gap:.12rem; } .pipeline-node { min-height:72px; padding:.45rem .2rem; } .flow-label { font-size:.5rem; } .flow-detail { font-size:.63rem; } .pipeline-arrow { font-size:.75rem; } }
    @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; } }
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
                f'<div style="text-align:right; padding-top:0.4rem; font-family:\'DM Mono\', monospace; font-size:0.75rem; color:#74d6ac;">'
                f'👤 <strong>{user_name}</strong> <span style="color:#79878e;">(@{username})</span>'
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
        st.markdown('<div class="eyebrow">Community Safety Center / Control Room</div><div class="hero-title">Detect. Analyze.<br><span style="color:#65ddff">Respond responsibly.</span></div><div class="hero-copy">A unified intelligence workspace for visual screening, activity awareness, and digital trust. <strong>Building safer communities through responsible AI.</strong></div>', unsafe_allow_html=True)
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
            st.markdown('<div class="result-empty result-pending"><div style="color:#e7eff0;font-weight:650">Awaiting image input</div></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="eyebrow">CNN / Computer Vision</div><div class="feature-title">Fake Currency<br><span style="color:#ffae58">Detector</span></div><div class="feature-copy">AI-powered Indian currency visual screening. Upload a note and receive a model-based Real/Fake signal.</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="eyebrow">Model output</div><div class="feature-title">AI <span style="color:#65ddff">Result</span></div><div class="feature-copy">The CNN evaluates the image independently of its denomination folder.</div>', unsafe_allow_html=True)
            if uploaded_file is None:
                st.markdown('<div class="result-empty result-pending"><div style="font-size:2rem;color:#87979c">◌</div><div style="color:#e7eff0;font-weight:650;margin-top:.55rem">Awaiting image input</div><div style="color:#8b999d;font-size:.72rem;margin-top:.35rem">Your prediction will appear here</div></div>', unsafe_allow_html=True)
            else:
                try:
                    prediction, confidence = predict_currency_image_ui(uploaded_file)
                    if prediction == "UNCLEAR":
                        st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#e9ae70;font-size:.75rem;margin-top:.45rem">Please scan the note again.</div></div>', unsafe_allow_html=True)
                    else:
                        word_class = "fake" if prediction == "FAKE" else ""
                        result_style = "result-fake" if prediction == "FAKE" else "result-real"
                        st.markdown(f'<div class="result-empty {result_style}"><div class="result-word {word_class}">{prediction}</div><div style="color:#98a7aa;font-size:.72rem;margin-top:.25rem">CNN classification</div></div>', unsafe_allow_html=True)
                    render_confidence_gauge(confidence)
                    for column, label, value, tone in zip(st.columns(4), ["Input", "Color", "Model", "Output"], ["224 × 224", "RGB", "CNN", "REAL / FAKE"], ["", "", "cyan", ""]):
                        with column:
                            render_metric_card(label, value, tone)
                except Exception as error:
                    st.error(f"Unable to analyze this image: {error}")
    else:
        st.markdown(
            """
            <style>
            .st-key-currency-front-upload,.st-key-currency-back-upload { background:linear-gradient(145deg,rgba(241,249,250,.07),rgba(130,164,170,.035)); border:1px solid rgba(233,174,112,.24); border-radius:22px; box-shadow:0 16px 38px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.12); margin-bottom:.9rem; padding:1rem 1.1rem; }
            .st-key-currency-front-result,.st-key-currency-back-result { background:linear-gradient(145deg,rgba(245,250,251,.075),rgba(153,183,189,.035)); border:1px solid rgba(232,242,246,.13); border-radius:22px; box-shadow:0 16px 38px rgba(0,0,0,.18),inset 0 1px rgba(255,255,255,.12); padding:1rem 1.1rem; }
            .scan-side-heading { color:#e1ebed; font-size:1rem; font-weight:650; margin-bottom:.8rem; }
            .st-key-currency-final-result { background:linear-gradient(145deg,rgba(245,250,251,.075),rgba(153,183,189,.035)); border:1px solid rgba(232,242,246,.15); border-radius:22px; box-shadow:0 16px 38px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.14); padding:1.2rem; }
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
                st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#e9ae70;font-size:.85rem;margin-top:.45rem">At least one side was unclear. Please scan both sides again clearly with good lighting and focus.</div></div>', unsafe_allow_html=True)
            elif front_result[0] != back_result[0]:
                st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#e9ae70;font-size:.85rem;margin-top:.45rem">The two scanned sides produced different predictions. Please scan both sides again with clear lighting and focus.</div></div>', unsafe_allow_html=True)
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
                '<div class="result-empty"><div style="color:#e7eff0;font-weight:650">'
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
st.stop()

# -----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #9aa4b2;
        margin-top: 0px;
    }

    .status {
        background-color: #10251b;
        border: 1px solid #1f7a4d;
        padding: 10px 16px;
        border-radius: 10px;
        color: #4ade80;
        font-weight: 600;
        text-align: center;
    }

    .module-card {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #30363d;
        background-color: #11161d;
        min-height: 210px;
    }

    .module-title {
        font-size: 23px;
        font-weight: 650;
    }

    .module-tech {
        font-size: 14px;
        color: #7dd3fc;
        font-weight: 600;
    }

    .module-description {
        color: #b8c0cc;
        font-size: 15px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Community Safety & Trust AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered platform for safety, fraud detection and digital trust'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# Status
status_col1, status_col2 = st.columns([5, 1])

with status_col2:
    st.markdown(
        '<div class="status">● SYSTEM ONLINE</div>',
        unsafe_allow_html=True
    )

st.divider()

# ----------------------------------------------------------
# INTRO
# ----------------------------------------------------------

st.subheader("Community Safety Center")

st.write(
    "Choose an AI module below to analyze currency, "
    "visual activity, or suspicious digital content."
)

st.write("")

# ----------------------------------------------------------
# THREE MODULES
# ----------------------------------------------------------

col1, col2, col3 = st.columns(3)

# ---------- Currency ----------

with col1:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        💵 Fake Currency Detector
        </div>

        <div class="module-tech">
        CNN • Computer Vision
        </div>

        <br>

        <div class="module-description">
        Analyze a currency image and predict whether
        the note is Real or Fake with a confidence score.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Currency Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "currency"


# ---------- Activity ----------

with col2:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        👁️ Suspicious Activity
        </div>

        <div class="module-tech">
        YOLO + Rule Engine • Computer Vision
        </div>

        <br>

        <div class="module-description">
        Detect people and objects from webcam/video
        and generate alerts using safety rules.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Activity Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "activity"


# ---------- Scam ----------

with col3:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        📰 Fake News / Scam Detector
        </div>

        <div class="module-tech">
        BERT + NLP • Text Analysis
        </div>

        <br>

        <div class="module-description">
        Analyze suspicious news or messages and
        predict Real/Fake with confidence.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Scam Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "scam"


# ----------------------------------------------------------
# PLATFORM WORKFLOW
# ----------------------------------------------------------

st.divider()

st.subheader("How the Platform Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 01")
    st.write("**Input**")
    st.caption("Image • Video • Text")

with step2:
    st.markdown("### 02")
    st.write("**AI Analysis**")
    st.caption("CNN • YOLO • BERT")

with step3:
    st.markdown("### 03")
    st.write("**Decision**")
    st.caption("Prediction + Confidence")

with step4:
    st.markdown("### 04")
    st.write("**Action**")
    st.caption("Result • Warning • Alert")


# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.divider()

st.caption(
    "Community Safety & Trust AI Platform | "
    "AI-assisted decision support system"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "currency_cnn_best.keras"


@st.cache_resource
def load_currency_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_currency_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


def render_currency_detector():
    st.subheader("Currency Detector")
    st.write("Upload a currency image for an AI-based Real or Fake prediction.")

    if st.button("Back to Dashboard", key="currency_back"):
        st.session_state["module"] = None
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload currency image",
        type=["jpg", "jpeg", "png", "avif", "webp"],
        key="currency_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin analysis.")
        return

    try:
        uploaded_file.seek(0)
        display_image = Image.open(uploaded_file).convert("RGB")
        st.image(display_image, caption="Uploaded currency image", use_container_width=True)
        model = load_currency_model()
        image_array = preprocess_currency_image(uploaded_file)
        fake_probability = float(model.predict(image_array, verbose=0)[0][0])
        predicted_label = 1 if fake_probability >= 0.5 else 0
        confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability

        if confidence < 0.60:
            st.warning("UNCLEAR — Please scan the note again.")
        else:
            prediction = "FAKE" if predicted_label == 1 else "REAL"
            st.success(f"Prediction: {prediction}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.caption(
            "This AI result is an image-based prediction and should not be treated "
            "as a definitive bank-grade counterfeit verification."
        )
    except Exception as error:
        st.error(f"Unable to analyze this image: {error}")


if st.session_state.get("module") == "currency":
    st.divider()
    render_currency_detector()