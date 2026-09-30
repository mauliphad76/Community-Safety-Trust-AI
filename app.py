from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ==========================================================
# COMMUNITY SAFETY & TRUST AI PLATFORM
# Main Dashboard
# ==========================================================

st.set_page_config(
    page_title="Community Safety & Trust AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "currency_cnn_best.keras"

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


def set_module(module):
    st.session_state["module"] = module
    st.rerun()


def render_topbar(active_module=None):
    st.markdown('<div class="topbar"><div class="brand"><div class="brand-mark">🛡️</div><div><div class="brand-name">COMMUNITY SAFETY<br>& TRUST AI</div><div class="brand-caption">AI-powered safety intelligence</div></div></div></div>', unsafe_allow_html=True)
    if active_module and st.button("←  Back to Dashboard", key=f"back_{active_module}", width="content"):
        set_module(None)


def render_metric_card(label, value, tone=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {tone}">{value}</div></div>', unsafe_allow_html=True)


def render_module_card(icon, title, technology, engine, description, module, planned):
    state = "COMING SOON" if planned else "ACTIVE"
    css_class = "status-planned" if planned else "status-online"
    theme = {"currency": "theme-currency", "activity": "theme-activity", "scam": "theme-scam"}[module]
    st.markdown(f'<div class="module-card {theme}"><div class="module-icon">{icon}</div><div class="module-title">{title}</div><div class="module-tech">{technology}</div><div class="module-description">{description}</div><div class="{css_class}">● {state}</div></div>', unsafe_allow_html=True)
    if st.button("View module →" if planned else "Open Currency Detector →", key=f"open_{module}", width="stretch"):
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
        render_module_card("👁️", "Suspicious Activity Detector", "YOLO • Computer Vision", "YOLO", "Planned video analysis for people, objects, and safety-related activity.", "activity", True)
    with col3:
        render_module_card("📰", "Fake News / Scam Detector", "BERT • NLP", "BERT", "Planned text analysis for suspicious news and scam signals.", "scam", True)


@st.cache_resource
def load_currency_model_ui():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


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


def render_currency_detector():
    render_topbar("currency")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<div class="eyebrow">CNN / Computer Vision</div><div class="feature-title">Fake Currency<br><span style="color:#ffae58">Detector</span></div><div class="feature-copy">AI-powered Indian currency visual screening. Upload a note and receive a model-based Real/Fake signal.</div><div class="section-label"><h2>Scan currency note</h2><span>Supported formats</span></div>', unsafe_allow_html=True)
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
                model = load_currency_model_ui()
                image_array = preprocess_currency_image_ui(uploaded_file)
                fake_probability = float(model.predict(image_array, verbose=0)[0][0])
                predicted_label = 1 if fake_probability >= .5 else 0
                confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability
                prediction = "FAKE" if predicted_label == 1 else "REAL"
                if confidence < .60:
                    st.markdown('<div class="result-empty result-unclear"><div class="result-word unclear">UNCLEAR</div><div style="color:#e9ae70;font-size:.75rem;margin-top:.45rem">Please scan the note again.</div></div>', unsafe_allow_html=True)
                else:
                    word_class = "fake" if prediction == "FAKE" else ""
                    result_style = "result-fake" if word_class else "result-real"
                    st.markdown(f'<div class="result-empty {result_style}"><div class="result-word {word_class}">{prediction}</div><div style="color:#98a7aa;font-size:.72rem;margin-top:.25rem">CNN classification</div></div>', unsafe_allow_html=True)
                render_confidence_gauge(confidence)
                for column, label, value, tone in zip(st.columns(4), ["Input", "Color", "Model", "Output"], ["224 × 224", "RGB", "CNN", "REAL / FAKE"], ["", "", "cyan", ""]):
                    with column:
                        render_metric_card(label, value, tone)
            except Exception as error:
                st.error(f"Unable to analyze this image: {error}")
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


if "module" not in st.session_state:
    st.session_state["module"] = None
active_module = st.session_state["module"]
if active_module is None:
    render_dashboard()
elif active_module == "currency":
    render_currency_detector()
else:
    render_planned_module(active_module)
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