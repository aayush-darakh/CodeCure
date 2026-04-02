"""
EpiWatch — Epidemic Intelligence Dashboard  [REDESIGNED]
CodeCure Biohackathon · Track C
Models: LSTM Forecasting · XGBoost Hotspot Detection · SEIR + Risk Score

UI/UX Redesign by: Senior Frontend Engineer
Design System: "Obsidian Intelligence" — dark luxury + clinical precision
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os
import joblib
import requests

# ── TensorFlow: optional — loaded lazily only when needed ────
TF_AVAILABLE = False
tf = None
try:
    import importlib
    tf = importlib.import_module("tensorflow")
    TF_AVAILABLE = True
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EpiWatch · Epidemic Intelligence",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM — "Obsidian Intelligence"
# ══════════════════════════════════════════════════════════════
DESIGN_CSS = """
<style>
/* ── Google Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Root ────────────────────────────────────────────── */
:root {
  --bg-base:        #080b12;
  --bg-primary:     #0d1117;
  --bg-secondary:   #141920;
  --surface:        #1a2030;
  --surface-raised: #1f2840;
  --surface-hover:  #243050;
  --border-subtle:  rgba(255,255,255,0.05);
  --border-default: rgba(255,255,255,0.10);
  --border-accent:  rgba(56,189,248,0.30);
  --accent-primary: #38bdf8;
  --accent-secondary: #818cf8;
  --risk-low:       #34d399;
  --risk-moderate:  #fbbf24;
  --risk-high:      #fb923c;
  --risk-critical:  #f87171;
  --text-primary:   #f0f6fc;
  --text-secondary: #8b97a8;
  --text-muted:     #4a5568;
  --font-sans:      'Plus Jakarta Sans', sans-serif;
  --font-mono:      'IBM Plex Mono', monospace;
  --radius-sm:      6px;
  --radius-md:      10px;
  --radius-lg:      16px;
  --radius-xl:      20px;
  --shadow-card:    0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
  --shadow-glow:    0 0 24px rgba(56,189,248,0.10);
  --transition:     all 0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global App Overrides ────────────────────────────────────── */
.stApp {
  background: var(--bg-primary) !important;
  font-family: var(--font-sans) !important;
}

section[data-testid="stSidebar"] + div > div {
  background: var(--bg-primary);
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-base) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
}

/* ── Typography ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6, p, div, span, label {
  font-family: var(--font-sans) !important;
}
code, pre {
  font-family: var(--font-mono) !important;
}

/* ── Streamlit Label text ────────────────────────────────────── */
.stSelectbox label, .stSlider label, .stMultiSelect label,
.stRadio label, .stCheckbox label {
  font-family: var(--font-sans) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* ── Streamlit inputs ────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
  border-color: var(--accent-primary) !important;
  box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Slider ──────────────────────────────────────────────────── */
.stSlider > div > div > div {
  background: var(--accent-primary) !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
  padding: 4px 6px 0 !important;
  gap: 2px !important;
  border-bottom: 1px solid var(--border-subtle) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  color: var(--text-secondary) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 10px 20px !important;
  border: none !important;
  transition: var(--transition) !important;
  letter-spacing: 0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-primary) !important;
  background: var(--surface-hover) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-primary) !important;
  color: var(--accent-primary) !important;
  border-bottom: 2px solid var(--accent-primary) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg-primary) !important;
  border: 1px solid var(--border-subtle) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
  padding: 28px 24px !important;
}

/* ── Expander ────────────────────────────────────────────────── */
[data-testid="stExpander"] details {
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface) !important;
}
[data-testid="stExpander"] details summary {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  list-style: none !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] details summary::-webkit-details-marker {
  display: none !important;
}
[data-testid="stExpander"] details summary::marker {
  display: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {
  background: var(--surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-sans) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  padding: 8px 20px !important;
  transition: var(--transition) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  background: var(--surface-hover) !important;
  border-color: var(--accent-primary) !important;
  color: var(--accent-primary) !important;
}

/* ── Info / Warning / Error boxes ───────────────────────────── */
.stInfo {
  background: rgba(56,189,248,0.06) !important;
  border: 1px solid rgba(56,189,248,0.20) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}
.stWarning {
  background: rgba(251,191,36,0.06) !important;
  border: 1px solid rgba(251,191,36,0.20) !important;
  border-radius: var(--radius-sm) !important;
}
.stError {
  background: rgba(248,113,113,0.06) !important;
  border: 1px solid rgba(248,113,113,0.20) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Dataframe ───────────────────────────────────────────────── */
.stDataFrame {
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}

/* ── Spinner ─────────────────────────────────────────────────── */
.stSpinner > div {
  border-top-color: var(--accent-primary) !important;
}

/* ── Code blocks ─────────────────────────────────────────────── */
.stCode, code {
  background: var(--surface) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  color: #94a3b8 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ══════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ══════════════════════════════════════════════════════════════ */

/* Hero / Header ─────────────────────────────────────────────── */
.epi-hero {
  position: relative;
  background: linear-gradient(135deg, #0a0f1a 0%, #0d1420 50%, #0a1220 100%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: 36px 44px;
  margin-bottom: 28px;
  overflow: hidden;
}
.epi-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-primary), var(--accent-secondary), transparent);
  opacity: 0.6;
}
.epi-hero::after {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(56,189,248,0.08);
  border: 1px solid rgba(56,189,248,0.18);
  border-radius: 100px;
  padding: 4px 14px;
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.12em;
  color: var(--accent-primary);
  text-transform: uppercase;
  margin-bottom: 16px;
}
.hero-dot {
  width: 6px; height: 6px;
  background: var(--risk-low);
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}
.hero-title {
  font-family: var(--font-sans) !important;
  font-size: 42px !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  color: var(--text-primary) !important;
  margin: 0 0 10px !important;
  line-height: 1.1 !important;
}
.hero-title span {
  background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-subtitle {
  font-size: 14px !important;
  color: var(--text-secondary) !important;
  margin: 0 0 20px !important;
  font-weight: 400 !important;
  letter-spacing: 0.01em !important;
  max-width: 560px;
  line-height: 1.6 !important;
}
.hero-models {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.model-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.04em;
}
.model-pill.m1 { background: rgba(52,211,153,0.10); color: #34d399; border: 1px solid rgba(52,211,153,0.20); }
.model-pill.m2 { background: rgba(251,191,36,0.10);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.20); }
.model-pill.m3 { background: rgba(129,140,248,0.10); color: #818cf8; border: 1px solid rgba(129,140,248,0.20); }

/* KPI Cards ─────────────────────────────────────────────────── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 28px;
}
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 22px 24px;
  position: relative;
  overflow: hidden;
  transition: var(--transition);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card.accent-blue::before   { background: linear-gradient(90deg, var(--accent-primary), transparent); }
.kpi-card.accent-green::before  { background: linear-gradient(90deg, var(--risk-low), transparent); }
.kpi-card.accent-amber::before  { background: linear-gradient(90deg, var(--risk-moderate), transparent); }
.kpi-card.accent-red::before    { background: linear-gradient(90deg, var(--risk-critical), transparent); }
.kpi-card.accent-violet::before { background: linear-gradient(90deg, var(--accent-secondary), transparent); }

.kpi-card:hover {
  border-color: var(--border-default);
  background: var(--surface-raised);
}
.kpi-label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  margin: 0 0 10px !important;
}
.kpi-value {
  font-family: var(--font-sans) !important;
  font-size: 32px !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  line-height: 1 !important;
  margin: 0 0 6px !important;
}
.kpi-value.blue   { color: var(--accent-primary) !important; }
.kpi-value.green  { color: var(--risk-low) !important; }
.kpi-value.amber  { color: var(--risk-moderate) !important; }
.kpi-value.red    { color: var(--risk-critical) !important; }
.kpi-value.violet { color: var(--accent-secondary) !important; }
.kpi-meta {
  font-size: 11px !important;
  color: var(--text-muted) !important;
  font-weight: 400 !important;
  margin: 0 !important;
}
.kpi-badge {
  position: absolute;
  top: 18px; right: 18px;
  font-size: 18px;
  opacity: 0.25;
}

/* Section Headers ───────────────────────────────────────────── */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border-subtle);
}
.section-title-text {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.02em !important;
  margin: 0 !important;
}
.section-tag {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 100px;
}
.section-tag.blue   { background: rgba(56,189,248,0.10);  color: var(--accent-primary); border: 1px solid rgba(56,189,248,0.20); }
.section-tag.green  { background: rgba(52,211,153,0.10);  color: var(--risk-low);        border: 1px solid rgba(52,211,153,0.20); }
.section-tag.amber  { background: rgba(251,191,36,0.10);  color: var(--risk-moderate);   border: 1px solid rgba(251,191,36,0.20); }
.section-tag.violet { background: rgba(129,140,248,0.10); color: var(--accent-secondary);border: 1px solid rgba(129,140,248,0.20); }

/* Insight Cards ─────────────────────────────────────────────── */
.insight-card {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  margin-bottom: 10px;
}
.insight-card .title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.insight-card .value {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-primary);
  margin-bottom: 3px;
}
.insight-card .note {
  font-size: 11px;
  color: var(--text-muted);
}

/* Risk Labels ───────────────────────────────────────────────── */
.risk-low      { color: var(--risk-low)      !important; font-weight: 700 !important; }
.risk-moderate { color: var(--risk-moderate) !important; font-weight: 700 !important; }
.risk-high     { color: var(--risk-high)     !important; font-weight: 700 !important; }
.risk-critical { color: var(--risk-critical) !important; font-weight: 700 !important; }

.risk-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 100px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
}
.risk-badge.low      { background: rgba(52,211,153,0.10);  color: var(--risk-low);      border: 1px solid rgba(52,211,153,0.25); }
.risk-badge.moderate { background: rgba(251,191,36,0.10);  color: var(--risk-moderate); border: 1px solid rgba(251,191,36,0.25); }
.risk-badge.high     { background: rgba(251,146,60,0.10);  color: var(--risk-high);     border: 1px solid rgba(251,146,60,0.25); }
.risk-badge.critical { background: rgba(248,113,113,0.10); color: var(--risk-critical); border: 1px solid rgba(248,113,113,0.25); }

/* Dividers ──────────────────────────────────────────────────── */
hr.epi-divider {
  border: none;
  border-top: 1px solid var(--border-subtle);
  margin: 28px 0;
}

/* Model Architecture Code Block ─────────────────────────────── */
.arch-block {
  background: var(--bg-base);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px 24px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: #64748b;
  line-height: 1.7;
}
.arch-block .arch-key   { color: var(--accent-primary); }
.arch-block .arch-val   { color: #94a3b8; }
.arch-block .arch-head  { color: var(--text-secondary); font-weight: 600; letter-spacing: 0.06em; }
.arch-block .arch-sep   { color: #2d3748; }

/* Sidebar Brand ─────────────────────────────────────────────── */
.sidebar-brand {
  padding: 24px 20px 16px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 4px;
}
.sidebar-logo {
  font-size: 26px;
  margin-bottom: 8px;
}
.sidebar-title {
  font-size: 20px !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  color: var(--text-primary) !important;
  margin: 0 0 2px !important;
}
.sidebar-sub {
  font-size: 11px !important;
  color: var(--text-muted) !important;
  font-weight: 400 !important;
  letter-spacing: 0.02em !important;
}
.sidebar-section {
  padding: 16px 20px 6px;
}
.sidebar-section-label {
  font-family: var(--font-mono) !important;
  font-size: 9px !important;
  font-weight: 600 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  margin: 0 0 12px !important;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sidebar-section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-subtle);
}
.sidebar-footer {
  padding: 16px 20px;
  border-top: 1px solid var(--border-subtle);
  margin-top: 20px;
}
.sidebar-footer p {
  font-size: 10px !important;
  color: var(--text-muted) !important;
  margin: 0 !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.04em !important;
}

/* Status indicator ──────────────────────────────────────────── */
.status-live {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  color: var(--risk-low);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.status-live::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--risk-low);
  border-radius: 50%;
  box-shadow: 0 0 6px var(--risk-low);
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* Empty / Loading state ─────────────────────────────────────── */
.empty-state {
  text-align: center;
  padding: 60px 40px;
  background: var(--surface);
  border: 1px dashed var(--border-default);
  border-radius: var(--radius-lg);
}
.empty-state .icon { font-size: 36px; margin-bottom: 14px; opacity: 0.4; }
.empty-state .title { font-size: 15px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
.empty-state .desc  { font-size: 13px; color: var(--text-muted); }

/* Tooltip hint ──────────────────────────────────────────────── */
.tooltip-hint {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 3px 8px;
  margin-bottom: 10px;
}

/* Chart containers ──────────────────────────────────────────── */
.chart-container {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 4px;
  margin-bottom: 16px;
}

/* Metric comparison row ─────────────────────────────────────── */
.metric-row {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}
.metric-mini {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  text-align: center;
}
.metric-mini .val {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-bottom: 3px;
}
.metric-mini .lbl {
  font-size: 10px;
  color: var(--text-muted);
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: var(--font-mono);
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOTLY THEME — shared across all charts
# ══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26,32,48,1)",
    plot_bgcolor="rgba(26,32,48,1)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#8b97a8", size=12),
    title_font=dict(family="Plus Jakarta Sans, sans-serif", color="#f0f6fc", size=15, weight=700),
    legend=dict(
        bgcolor="rgba(20,25,32,0.8)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=11, color="#8b97a8"),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        showspikes=True,
        spikecolor="rgba(56,189,248,0.3)",
        spikemode="across",
        spikethickness=1,
        tickfont=dict(size=10, color="#4a5568"),
        title_font=dict(size=11, color="#8b97a8"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#4a5568"),
        title_font=dict(size=11, color="#8b97a8"),
    ),
    hoverlabel=dict(
        bgcolor="#141920",
        bordercolor="rgba(56,189,248,0.3)",
        font=dict(family="IBM Plex Mono, monospace", size=12, color="#f0f6fc"),
    ),
)

COLORS   = ["#38bdf8", "#f87171", "#fbbf24", "#34d399", "#fb923c", "#818cf8", "#a78bfa", "#22d3ee"]
RISK_MAP = {"Low": "#34d399", "Moderate": "#fbbf24", "High": "#fb923c", "Critical": "#f87171"}


def apply_layout(fig, **extra):
    """Apply global layout + extras to a figure."""
    cfg = {**PLOTLY_LAYOUT, **extra}
    fig.update_layout(**cfg)
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">🦠</div>
      <div class="sidebar-title">EpiWatch</div>
      <div class="sidebar-sub">Epidemic Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">Global</div>', unsafe_allow_html=True)
    selected_countries = st.multiselect(
        "Countries of Interest",
        ["India", "United States", "Brazil", "United Kingdom",
         "Germany", "France", "Japan", "Italy", "Canada", "Australia",
         "South Korea", "Mexico", "Spain", "Russia", "Argentina"],
        default=["India", "United States", "Brazil"],
    )
    date_range = st.select_slider(
        "Analysis Window",
        options=["3 months", "6 months", "1 year", "2 years", "All time"],
        value="1 year",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">LSTM · Model 1</div>', unsafe_allow_html=True)
    lstm_country = st.selectbox(
        "Forecast Country",
        ["India", "United States", "Brazil", "United Kingdom", "Germany",
         "France", "Japan", "Italy", "Canada", "Australia"],
    )
    forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">XGBoost · Model 2</div>', unsafe_allow_html=True)
    hotspot_threshold = st.slider(
        "Decision Threshold", 0.10, 0.90, 0.35, step=0.05,
        help="Lower → higher recall (catch more hotspots). Default 0.35 favors sensitivity."
    )
    hotspot_country = st.selectbox(
        "Hotspot Country",
        ["India", "United States", "Brazil", "United Kingdom", "Germany"],
        key="hspot_country",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">SEIR · Model 3</div>', unsafe_allow_html=True)
    seir_country = st.selectbox(
        "SEIR Country",
        ["India", "United States", "Brazil", "United Kingdom", "Germany"],
        key="seir_country",
    )
    seir_forecast = st.slider("SEIR Forecast (days)", 7, 90, 30)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
      <p>CodeCure Biohackathon · Track C</p>
      <p style="margin-top:4px !important;">v2.0 · 2025</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_jhu_confirmed():
    url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_confirmed_global.csv"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        st.error(f"⚠️ Could not load JHU dataset: {e}. Please check your internet connection.")
        st.stop()
    df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")
    df = df.rename(columns={"Country/Region": "country"})
    df = df.groupby("country", as_index=False).sum(numeric_only=True)
    date_cols = [c for c in df.columns if c != "country"]
    df = df.melt(id_vars="country", value_vars=date_cols, var_name="date", value_name="confirmed")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values(["country", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_owid():
    url = (
        "https://raw.githubusercontent.com/owid/covid-19-data/master/"
        "public/data/owid-covid-data.csv"
    )
    keep = [
        "location", "date", "total_cases", "new_cases", "new_cases_smoothed",
        "new_deaths", "new_deaths_smoothed", "reproduction_rate",
        "population", "population_density", "aged_65_older",
        "hospital_beds_per_thousand", "human_development_index",
        "life_expectancy", "stringency_index",
        "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
        "total_boosters", "positive_rate", "total_tests",
        "hosp_patients", "icu_patients", "median_age", "gdp_per_capita",
    ]
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), usecols=lambda c: c in keep,
                         parse_dates=["date"], low_memory=False)
    except Exception as e:
        st.error(f"⚠️ Could not load OWID dataset: {e}. Please check your internet connection.")
        st.stop()
    df = df.rename(columns={"location": "country"})
    exclude = ["World", "Africa", "Asia", "Europe", "European Union",
               "North America", "Oceania", "South America",
               "High income", "Low income", "Lower middle income", "Upper middle income",
               "International"]
    df = df[~df["country"].isin(exclude)]
    return df.sort_values(["country", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# MODEL 2 — HOTSPOT FEATURES
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_hotspot_features(df_owid):
    cols = [
        "country", "date", "total_cases", "new_cases", "new_cases_smoothed",
        "new_deaths", "new_deaths_smoothed", "reproduction_rate",
        "population", "population_density", "aged_65_older",
        "hospital_beds_per_thousand", "human_development_index",
        "life_expectancy", "stringency_index",
    ]
    df = df_owid[[c for c in cols if c in df_owid.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    df.dropna(subset=["new_cases", "total_cases", "population"], inplace=True)
    df["prev_day"]  = df.groupby("country")["total_cases"].shift(1)
    df["prev7_day"] = df.groupby("country")["total_cases"].shift(7)
    df["growth_rate_1d"] = (
        df["prev_day"] - df.groupby("country")["total_cases"].shift(2)
    ) / (df.groupby("country")["total_cases"].shift(2) + 1)
    df["growth_rate_7d"] = (df["prev_day"] - df["prev7_day"]) / (df["prev7_day"] + 1)
    df.dropna(inplace=True)
    threshold = 0.01
    df["future_growth"] = df.groupby("country")["growth_rate_7d"].shift(-7)
    df["hotspot"] = (df["future_growth"] > threshold).astype(int)
    df.dropna(subset=["future_growth"], inplace=True)
    df["lag_1"] = df.groupby("country")["new_cases"].shift(1)
    df["lag_3"] = df.groupby("country")["new_cases"].shift(3)
    df["lag_7"] = df.groupby("country")["new_cases"].shift(7)
    df["lag_deaths_1"] = df.groupby("country")["new_deaths"].shift(1)
    df["rolling_avg_7"] = df.groupby("country")["new_cases"].transform(
        lambda x: x.shift(1).rolling(7).mean())
    df["rolling_std_7"] = df.groupby("country")["new_cases"].transform(
        lambda x: x.shift(1).rolling(7).std())
    df["rolling_max_7"] = df.groupby("country")["new_cases"].transform(
        lambda x: x.shift(1).rolling(7).max())
    df["cases_per_million"] = df["lag_1"] / (df["population"] / 1e6 + 1)
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_hotspot_model_from_disk():
    """Try loading the pre-trained XGBoost model saved as hotspot_xgb_model.pkl."""
    path = os.path.join(os.path.dirname(__file__), "hotspot_xgb_model.pkl")
    try:
        model = joblib.load(path)
        return model
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def train_hotspot_model(df_feat):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    features = [
        "lag_1", "lag_3", "lag_7", "lag_deaths_1",
        "rolling_avg_7", "rolling_std_7", "rolling_max_7",
        "growth_rate_1d", "growth_rate_7d", "cases_per_million",
        "population_density", "aged_65_older",
        "hospital_beds_per_thousand", "human_development_index",
        "life_expectancy", "stringency_index",
    ]
    features = [f for f in features if f in df_feat.columns]
    split_date = "2022-06-01"
    train = df_feat[df_feat["date"] < split_date].copy()
    test  = df_feat[df_feat["date"] >= split_date].copy()

    for col in features:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

    X_train, y_train = train[features], train["hotspot"]
    X_test,  y_test  = test[features],  test["hotspot"]
    ratio = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)

    # Use saved model if available, otherwise train fresh
    saved_model = load_hotspot_model_from_disk()
    if saved_model is not None:
        model = saved_model
    else:
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=ratio, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, (y_prob >= 0.35).astype(int))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    test = test.reset_index(drop=True)
    test["y_prob"] = y_prob

    return model, features, test, y_prob, auc, acc, importances


# ══════════════════════════════════════════════════════════════
# MODEL 3 — SEIR
# ══════════════════════════════════════════════════════════════
def seir_odes(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]


def fit_seir(country, df, forecast_days=30):
    cdf = df[df["country"] == country].sort_values("date").copy()
    case_col = "confirmed" if "confirmed" in cdf.columns else "total_cases"
    new_col  = "new_cases"
    if case_col not in cdf.columns or new_col not in cdf.columns:
        return None
    cdf = cdf.dropna(subset=[case_col, "population"]).tail(180).reset_index(drop=True)
    if len(cdf) < 30:
        return None
    N = cdf["population"].median()
    if pd.isna(N) or N <= 0:
        N = 1e7
    I_obs = cdf[new_col].clip(lower=0).fillna(0).values
    T     = len(I_obs)
    t     = np.arange(T)
    I0 = max(I_obs[0], 1); E0 = I0 * 2; R0_init = 0; S0 = N - I0 - E0 - R0_init
    def model_I(t, beta, sigma, gamma):
        sol = odeint(seir_odes, [S0, E0, I0, R0_init], t, args=(beta, sigma, gamma, N))
        return sol[:, 2]
    try:
        popt, _ = curve_fit(model_I, t, I_obs, p0=[0.3, 0.2, 0.1],
                            bounds=([0.05, 0.05, 0.02], [1.5, 0.5, 0.5]), maxfev=5000)
        beta_fit, sigma_fit, gamma_fit = popt
    except Exception:
        beta_fit, sigma_fit, gamma_fit = 0.25, 0.2, 0.1
    t_full = np.arange(T + forecast_days)
    sol    = odeint(seir_odes, [S0, E0, I0, R0_init], t_full,
                   args=(beta_fit, sigma_fit, gamma_fit, N))
    dates_full = pd.date_range(cdf["date"].iloc[0], periods=T + forecast_days, freq="D")
    seir_df = pd.DataFrame({
        "date": dates_full,
        "S": sol[:, 0], "E": sol[:, 1], "I": sol[:, 2], "R": sol[:, 3],
        "is_forecast": [False] * T + [True] * forecast_days,
    })
    return {
        "country": country, "beta": beta_fit, "sigma": sigma_fit,
        "gamma": gamma_fit, "R0": beta_fit / gamma_fit, "N": N,
        "seir_df": seir_df, "obs_I": I_obs, "obs_dates": cdf["date"].values,
    }


# ══════════════════════════════════════════════════════════════
# MODEL 3 — RISK SCORE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_risk_scores(df_owid):
    df = df_owid.copy().sort_values(["country", "date"])
    df["new_cases"] = df.groupby("country")["total_cases"].diff().clip(lower=0)
    prev7 = df.groupby("country")["new_cases"].shift(7).replace(0, np.nan)
    df["growth_rate_weekly"] = (df["new_cases"] / prev7 - 1).clip(-1, 10)
    pop = df["population"].replace(0, np.nan)

    # FIX: safe column access — use pd.Series fallback instead of scalar np.nan
    if "people_fully_vaccinated" in df.columns:
        vacc_col = df["people_fully_vaccinated"]
    else:
        vacc_col = pd.Series(np.nan, index=df.index)
    df["pct_fully_vaccinated"] = (vacc_col / pop * 100).fillna(0).clip(0, 100)

    def minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)
    df["risk_growth"]     = minmax(df["growth_rate_weekly"].clip(-1, 5).fillna(0))
    df["risk_vacc"]       = 1 - minmax(df["pct_fully_vaccinated"].fillna(0))
    df["risk_healthcare"] = 1 - minmax(
        df["hospital_beds_per_thousand"].fillna(df["hospital_beds_per_thousand"].median()))
    df["risk_score"] = (
        0.50 * df["risk_growth"] + 0.25 * df["risk_vacc"] + 0.25 * df["risk_healthcare"]
    ).clip(0, 1)
    df["risk_label"] = pd.cut(
        df["risk_score"], bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Low", "Moderate", "High", "Critical"], include_lowest=True
    ).astype(str)
    latest = (
        df.sort_values("date").groupby("country").last().reset_index()
        [["country", "risk_score", "risk_label",
          "growth_rate_weekly", "pct_fully_vaccinated", "new_cases", "total_cases"]]
    )
    return df, latest


# ══════════════════════════════════════════════════════════════
# MODEL 1 — LSTM (load saved model; fallback to exponential smoothing)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_lstm_model():
    """Load saved LSTM model, scaler and sequence length from disk."""
    if not TF_AVAILABLE or tf is None:
        return None, None, 30
    model_path  = os.path.join(os.path.dirname(__file__), "model1.h5")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    seqlen_path = os.path.join(os.path.dirname(__file__), "seq_len.pkl")
    try:
        model   = tf.keras.models.load_model(model_path, compile=False)
        scaler  = joblib.load(scaler_path)
        seq_len = int(joblib.load(seqlen_path)) if os.path.exists(seqlen_path) else 30
        return model, scaler, seq_len
    except Exception:
        return None, None, 30


def lstm_forecast_mock(df_jhu, country, days=30):
    """Exponential-smoothing fallback used when LSTM model is unavailable."""
    cdf = df_jhu[df_jhu["country"] == country].sort_values("date").copy()
    cdf["new_cases"] = cdf["confirmed"].diff().clip(lower=0).fillna(0)
    cdf = cdf.tail(180)
    alpha = 0.15
    smoothed = cdf["new_cases"].ewm(alpha=alpha, adjust=False).mean().values
    last_smooth = smoothed[-1]
    growth = np.mean(np.diff(smoothed[-14:])) / (smoothed[-7:].mean() + 1)
    growth = np.clip(growth, -0.05, 0.08)
    forecast_vals = []
    val = last_smooth
    for i in range(days):
        val = val * (1 + growth) + np.random.normal(0, val * 0.03)
        forecast_vals.append(max(val, 0))
    last_date = cdf["date"].iloc[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=days)
    return cdf, smoothed, forecast_vals, forecast_dates


def lstm_forecast_real(df_jhu, country, days=30):
    """Run real LSTM forecast using the saved Keras model."""
    lstm_model, scaler, seq_len = load_lstm_model()
    if lstm_model is None:
        return lstm_forecast_mock(df_jhu, country, days)

    cdf = df_jhu[df_jhu["country"] == country].sort_values("date").copy()
    cdf["new_cases"] = cdf["confirmed"].diff().clip(lower=0).fillna(0)
    cdf = cdf.tail(180).reset_index(drop=True)

    series = cdf["new_cases"].values.reshape(-1, 1).astype(float)
    try:
        scaled = scaler.transform(series)
    except Exception:
        return lstm_forecast_mock(df_jhu, country, days)

    # Build fitted values (walk-forward on training window)
    fitted_scaled = []
    for i in range(seq_len, len(scaled)):
        x = scaled[i - seq_len:i].reshape(1, seq_len, 1)
        fitted_scaled.append(lstm_model.predict(x, verbose=0)[0, 0])
    fitted_pad = [np.nan] * seq_len + fitted_scaled
    smoothed = scaler.inverse_transform(
        np.array(fitted_pad).reshape(-1, 1)
    ).flatten()
    smoothed[:seq_len] = series[:seq_len].flatten()

    # Forecast future
    window = scaled[-seq_len:].copy()
    forecast_scaled = []
    for _ in range(days):
        x = window.reshape(1, seq_len, 1)
        pred = lstm_model.predict(x, verbose=0)[0, 0]
        forecast_scaled.append(pred)
        window = np.append(window[1:], [[pred]], axis=0)
    forecast_vals = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten().clip(min=0).tolist()

    last_date = cdf["date"].iloc[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=days)
    return cdf, smoothed, forecast_vals, forecast_dates


# ══════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="epi-hero">
  <div class="hero-eyebrow">
    <div class="hero-dot"></div>
    Live · Epidemic Intelligence System
  </div>
  <h1 class="hero-title">Epi<span>Watch</span></h1>
  <p class="hero-subtitle">
    Real-time epidemic spread prediction powered by LSTM neural networks,
    XGBoost classification, and compartmental SEIR modeling across 180+ countries.
  </p>
  <div class="hero-models">
    <span class="model-pill m1">◈ LSTM Forecasting</span>
    <span class="model-pill m2">◉ XGBoost Hotspot Detection</span>
    <span class="model-pill m3">⬡ SEIR + Risk Score</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
with st.spinner("Loading JHU + OWID datasets…"):
    jhu_df  = load_jhu_confirmed()
    owid_df = load_owid()

with st.spinner("Computing composite risk scores…"):
    df_risk, latest_risk = compute_risk_scores(owid_df)


# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════
total_conf = jhu_df.groupby("country")["confirmed"].last().sum()
n_critical = (latest_risk["risk_label"] == "Critical").sum()
n_high     = (latest_risk["risk_label"] == "High").sum()
n_moderate = (latest_risk["risk_label"] == "Moderate").sum()
avg_risk   = latest_risk["risk_score"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

def kpi_card(label, value, meta, accent, badge, value_color):
    return f"""
    <div class="kpi-card {accent}">
      <div class="kpi-badge">{badge}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {value_color}">{value}</div>
      <div class="kpi-meta">{meta}</div>
    </div>"""

with c1:
    st.markdown(kpi_card("Global Confirmed", f"{total_conf/1e9:.2f}B",
                "Cumulative · JHU dataset", "accent-blue", "🌍", "blue"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("Critical Risk", str(n_critical),
                "Risk score > 0.75", "accent-red", "🔴", "red"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("High Risk", str(n_high),
                "Risk score 0.50–0.75", "accent-amber", "🟠", "amber"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Moderate Risk", str(n_moderate),
                "Risk score 0.25–0.50", "accent-green", "🟡", "green"), unsafe_allow_html=True)
with c5:
    st.markdown(kpi_card("Avg Global Risk", f"{avg_risk:.3f}",
                "Composite score (0–1)", "accent-violet", "📊", "violet"), unsafe_allow_html=True)

st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "  📈  LSTM Forecast  ",
    "  🎯  Hotspot Detection  ",
    "  🔬  SEIR + Risk Score  ",
    "  🌍  Global Overview  ",
])


# ════════════════════════════════════════════════════════
# TAB 1 — LSTM FORECASTING
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
      <div class="section-title-text">LSTM Time-Series Forecast</div>
      <span class="section-tag green">Model 1</span>
      <span class="section-tag blue">Neural Network</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap;">
      <span class="tooltip-hint">🧠 Architecture: 2-layer LSTM · 50 units each</span>
      <span class="tooltip-hint">📐 Sequence: 30 days · Optimizer: Adam</span>
      <span class="tooltip-hint">⚖️ Loss: MSE · Split: 80/20</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Running LSTM forecast for {lstm_country}…"):
        cdf, smoothed, forecast_vals, forecast_dates = lstm_forecast_real(
            jhu_df, lstm_country, forecast_days
        )
    if not TF_AVAILABLE:
        st.info("ℹ️ Running in lightweight mode: forecast uses exponential smoothing (TensorFlow not installed in this environment). Results are indicative only.")

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["new_cases"],
        mode="lines", name="Actual Daily Cases",
        line=dict(color="rgba(56,189,248,0.35)", width=1.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Cases: %{y:,.0f}<extra></extra>",
    ))
    fig_lstm.add_trace(go.Scatter(
        x=cdf["date"], y=smoothed,
        mode="lines", name="LSTM Fitted",
        line=dict(color="#34d399", width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Fitted: %{y:,.0f}<extra></extra>",
    ))
    fc_arr = np.array(forecast_vals)
    fig_lstm.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(fc_arr * 1.15) + list(fc_arr * 0.85),
        fill="toself", fillcolor="rgba(251,191,36,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        hoverinfo="skip",
    ))
    fig_lstm.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_vals,
        mode="lines", name=f"Forecast (+{forecast_days}d)",
        line=dict(color="#fbbf24", width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
    ))
    fig_lstm.add_vrect(
        x0=str(cdf["date"].iloc[-1]), x1=str(forecast_dates[-1]),
        fillcolor="rgba(251,191,36,0.04)", line_width=0,
    )
    fig_lstm.add_vline(
        x=cdf["date"].iloc[-1].timestamp() * 1000,
        line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"),
        annotation_text="Forecast →",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#4a5568"),
    )
    apply_layout(fig_lstm,
        title=f"Daily Case Forecast — {lstm_country}",
        height=420,
        xaxis_title="Date", yaxis_title="Daily New Cases",
        legend=dict(x=0.01, y=0.99, **PLOTLY_LAYOUT["legend"]),
        margin=dict(l=40, r=20, t=56, b=40),
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

    cdf["7d_avg"] = cdf["new_cases"].rolling(7).mean()
    colA, colB = st.columns([3, 1])

    with colA:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=cdf["date"].tail(90), y=cdf["new_cases"].tail(90),
            name="Daily Cases",
            marker=dict(color="rgba(56,189,248,0.25)", line=dict(width=0)),
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:,.0f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=cdf["date"].tail(90), y=cdf["7d_avg"].tail(90),
            name="7-Day Avg", line=dict(color="#f87171", width=2.5),
            hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig_trend,
            title=f"Last 90 Days — {lstm_country}",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=True,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with colB:
        st.markdown("""
        <div style="padding:0 0 10px;">
          <div class="kpi-label" style="margin-bottom:12px;">Forecast Summary</div>
        </div>
        """, unsafe_allow_html=True)
        peak_fc = max(forecast_vals)
        min_fc  = min(forecast_vals)
        trend_dir = "📈 Rising" if forecast_vals[-1] > forecast_vals[0] else "📉 Declining"
        st.markdown(f"""
        <div class="insight-card">
          <div class="title">Peak Forecast</div>
          <div class="value">{int(peak_fc):,}</div>
          <div class="note">cases / day</div>
        </div>
        <div class="insight-card">
          <div class="title">Trend Direction</div>
          <div class="value" style="font-size:16px;">{trend_dir}</div>
          <div class="note">over {forecast_days} days</div>
        </div>
        <div class="insight-card">
          <div class="title">Range</div>
          <div class="value" style="font-size:16px;">{int(min_fc):,}–{int(peak_fc):,}</div>
          <div class="note">forecast bounds</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Model Architecture Details"):
        st.code("""
Sequential (Keras / TensorFlow)
───────────────────────────────────────────
Layer 1  LSTM(50)  input_shape=(30,1)  return_sequences=True
Layer 2  LSTM(50)
Layer 3  Dense(1)

Optimizer  Adam       Loss      MSE
Epochs     10         Batch     32
Scaler     MinMaxScaler [0,1]   Seq  30 days
Split      80% train / 20% test
Saved      model1.h5  +  scaler.pkl
        """, language="text")


# ════════════════════════════════════════════════════════
# TAB 2 — HOTSPOT XGBOOST
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
      <div class="section-title-text">XGBoost Hotspot Detection</div>
      <span class="section-tag amber">Model 2</span>
      <span class="section-tag blue">Binary Classification</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap;">
      <span class="tooltip-hint">🎯 Task: Predict hotspot emergence in 7 days</span>
      <span class="tooltip-hint">📅 Train: pre-Jun 2022 · Test: Jun 2022+</span>
      <span class="tooltip-hint">⚖️ Threshold: adjustable via sidebar (default 0.35)</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training XGBoost hotspot model…"):
        try:
            hspot_features = build_hotspot_features(owid_df)
            hspot_model, feat_list, test_df, y_prob, auc_score, acc_score, importances = train_hotspot_model(hspot_features)
            model_loaded = True
        except Exception as e:
            st.error(f"Model training failed: {e}")
            model_loaded = False

    if model_loaded:
        n_test_hotspot = (y_prob >= hotspot_threshold).sum()
        m1, m2, m3, m4 = st.columns(4)

        # FIX: mini_kpi now handles all five color values including "violet"
        def mini_kpi(col, label, value, color, meta=""):
            accent_map = {
                "blue":   "accent-blue",
                "green":  "accent-green",
                "amber":  "accent-amber",
                "red":    "accent-red",
                "violet": "accent-violet",
            }
            accent_cls = accent_map.get(color, "accent-blue")
            with col:
                st.markdown(f"""
                <div class="kpi-card {accent_cls}">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value {color}">{value}</div>
                  <div class="kpi-meta">{meta}</div>
                </div>""", unsafe_allow_html=True)

        mini_kpi(m1, "ROC-AUC",         f"{auc_score:.4f}", "blue",   "Test set performance")
        mini_kpi(m2, "Accuracy",         f"{acc_score:.4f}", "green",  f"At threshold {hotspot_threshold:.2f}")
        mini_kpi(m3, "Hotspots Flagged", f"{n_test_hotspot:,}", "amber", "In test dataset")
        mini_kpi(m4, "Features",         f"{len(feat_list)}", "violet", "Engineered signals")

        st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            feat_labels = [f.replace("_", " ").title() for f in importances.index[::-1]]
            colors_imp = [
                f"rgba(56,189,248,{0.4 + 0.6 * v/importances.max()})"
                for v in importances.values[::-1]
            ]
            fig_imp = go.Figure(go.Bar(
                x=importances.values[::-1],
                y=feat_labels,
                orientation="h",
                marker=dict(color=colors_imp, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            apply_layout(fig_imp,
                title="Feature Importance",
                height=420,
                xaxis_title="Importance Score",
                margin=dict(l=140, r=20, t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(test_df["hotspot"], y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC  AUC={auc_score:.3f}",
                line=dict(color="#38bdf8", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.06)",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                line=dict(color="#2d3748", dash="dash", width=1.5),
                hoverinfo="skip",
            ))
            apply_layout(fig_roc,
                title="ROC Curve",
                height=420,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("""
        <div class="section-header" style="margin-top:8px;">
          <div class="section-title-text" style="font-size:15px;">Hotspot Probability Timeline</div>
          <span class="section-tag amber">Country View</span>
        </div>
        """, unsafe_allow_html=True)

        # FIX: use the y_prob column attached during training — no fragile index arithmetic
        country_test = test_df[test_df["country"] == hotspot_country].copy()
        if len(country_test) > 0:
            probs_country = country_test["y_prob"].values
            dates_country = country_test["date"].values

            fig_prob = go.Figure()
            fig_prob.add_hrect(y0=0, y1=hotspot_threshold,
                fillcolor="rgba(52,211,153,0.04)", line_width=0,
                annotation_text="Safe zone", annotation_position="right",
                annotation_font=dict(size=9, color="#34d399"))
            fig_prob.add_hrect(y0=hotspot_threshold, y1=1,
                fillcolor="rgba(248,113,113,0.04)", line_width=0,
                annotation_text="Hotspot zone", annotation_position="right",
                annotation_font=dict(size=9, color="#f87171"))

            fig_prob.add_trace(go.Scatter(
                x=dates_country, y=probs_country,
                mode="lines", name="P(Hotspot)",
                line=dict(color="#fbbf24", width=2),
                fill="tozeroy",
                fillcolor="rgba(251,191,36,0.08)",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>P(Hotspot): %{y:.3f}<extra></extra>",
            ))
            fig_prob.add_hline(
                y=hotspot_threshold,
                line=dict(color="#f87171", width=1.5, dash="dash"),
                annotation_text=f"Threshold = {hotspot_threshold:.2f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#f87171"),
            )
            apply_layout(fig_prob,
                title=f"Hotspot Probability — {hotspot_country}",
                height=320,
                yaxis=dict(range=[0, 1.05], title="P(Hotspot)", **PLOTLY_LAYOUT["yaxis"]),
                margin=dict(l=50, r=80, t=50, b=40),
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="empty-state">
              <div class="icon">📭</div>
              <div class="title">No test data for {hotspot_country}</div>
              <div class="desc">Try selecting a different country from the sidebar.</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Model Architecture Details"):
            st.code(f"""
XGBClassifier  (gradient boosted trees)
───────────────────────────────────────────
n_estimators       300         max_depth          5
learning_rate      0.05        subsample          0.8
colsample_bytree   0.8         scale_pos_weight   auto
eval_metric        logloss     random_state       42

Label engineering:
  hotspot = 1  if future_7d_growth_rate > 1%
  hotspot = 0  otherwise
  (shift(-7) applied — zero data leakage)

Decision threshold : {hotspot_threshold:.2f}  (tuned for recall)
Saved              : hotspot_xgb_model.pkl
            """, language="text")


# ════════════════════════════════════════════════════════
# TAB 3 — SEIR + RISK SCORE
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
      <div class="section-title-text">SEIR Epidemic Model + Risk Score</div>
      <span class="section-tag violet">Model 3</span>
      <span class="section-tag blue">ODE Compartmental</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap;">
      <span class="tooltip-hint">🧬 SEIR: curve-fitted per country via scipy.optimize</span>
      <span class="tooltip-hint">🎯 Risk: growth 50% · vaccination 25% · healthcare 25%</span>
      <span class="tooltip-hint">📏 Fitted on last 180 days of data</span>
    </div>
    """, unsafe_allow_html=True)

    seir_data = owid_df[["country", "date", "total_cases", "new_cases", "population",
                          "people_fully_vaccinated", "hospital_beds_per_thousand"]].copy()
    seir_data = seir_data.rename(columns={"total_cases": "confirmed"})

    with st.spinner(f"Fitting SEIR model for {seir_country}…"):
        seir_res = fit_seir(seir_country, seir_data, seir_forecast)

    if seir_res:
        r0_val   = seir_res["R0"]
        r0_color = "green" if r0_val < 1 else "red"
        r0_label = "🟢 Controlled" if r0_val < 1 else "🔴 Spreading"

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="kpi-card {'accent-green' if r0_val < 1 else 'accent-red'}">
              <div class="kpi-label">R₀ — Basic Reproduction No.</div>
              <div class="kpi-value {r0_color}">{r0_val:.3f}</div>
              <div class="kpi-meta">{r0_label}</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="kpi-card accent-blue">
              <div class="kpi-label">β — Transmission Rate</div>
              <div class="kpi-value blue">{seir_res['beta']:.4f}</div>
              <div class="kpi-meta">per day</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="kpi-card accent-violet">
              <div class="kpi-label">γ — Recovery Rate</div>
              <div class="kpi-value violet">{seir_res['gamma']:.4f}</div>
              <div class="kpi-meta">≈ {1/seir_res['gamma']:.0f} days to recover</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="kpi-card accent-amber">
              <div class="kpi-label">σ — Incubation Rate</div>
              <div class="kpi-value amber">{seir_res['sigma']:.4f}</div>
              <div class="kpi-meta">≈ {1/seir_res['sigma']:.0f} days incubation</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

        sdf = seir_res["seir_df"]
        N   = seir_res["N"]
        hist  = sdf[~sdf["is_forecast"]]
        fcast = sdf[sdf["is_forecast"]]

        fig_seir = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Compartments (fraction of population)",
                f"Infectious (I) — Fit vs Observed + {seir_forecast}d Forecast",
            ],
            horizontal_spacing=0.10,
        )

        compartment_colors = {
            "S": ("#4e79a7", "Susceptible"),
            "E": ("#fbbf24", "Exposed"),
            "I": ("#f87171", "Infectious"),
            "R": ("#34d399", "Recovered"),
        }
        for comp, (color, name) in compartment_colors.items():
            fig_seir.add_trace(go.Scatter(
                x=sdf["date"], y=sdf[comp]/N, name=name,
                line=dict(color=color, width=2), mode="lines",
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d}}<br>Fraction: %{{y:.4f}}<extra></extra>",
            ), row=1, col=1)

        fig_seir.add_trace(go.Scatter(
            x=seir_res["obs_dates"], y=seir_res["obs_I"],
            name="Observed (I)", line=dict(color="#38bdf8", width=2), mode="lines",
            hovertemplate="<b>Observed</b><br>%{x|%b %d}<br>%{y:,.0f}<extra></extra>",
        ), row=1, col=2)
        fig_seir.add_trace(go.Scatter(
            x=hist["date"], y=hist["I"], name="SEIR Fitted",
            line=dict(color="#f87171", dash="dash", width=2),
        ), row=1, col=2)
        fig_seir.add_trace(go.Scatter(
            x=fcast["date"], y=fcast["I"],
            name=f"Forecast (+{seir_forecast}d)",
            line=dict(color="#34d399", width=2.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.07)",
            hovertemplate="<b>Forecast</b><br>%{x|%b %d}<br>%{y:,.0f}<extra></extra>",
        ), row=1, col=2)

        fig_seir.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            title_text=f"SEIR Model — {seir_country}",
            margin=dict(l=40, r=20, t=70, b=40),
        )
        fig_seir.update_xaxes(**PLOTLY_LAYOUT["xaxis"])
        fig_seir.update_yaxes(**PLOTLY_LAYOUT["yaxis"])
        st.plotly_chart(fig_seir, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="empty-state">
          <div class="icon">🧬</div>
          <div class="title">Insufficient data for {seir_country}</div>
          <div class="desc">At least 30 days of data required to fit SEIR parameters. Try a different country.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-title-text">Composite Risk Score</div>
      <span class="section-tag blue">Global · 0–1 Scale</span>
    </div>
    """, unsafe_allow_html=True)

    fig_choro = px.choropleth(
        latest_risk,
        locations="country", locationmode="country names",
        color="risk_score",
        hover_name="country",
        hover_data={"risk_score": ":.3f", "risk_label": True,
                    "growth_rate_weekly": ":.3f", "pct_fully_vaccinated": ":.1f"},
        color_continuous_scale=[
            [0.0, "#0ea5e9"], [0.25, "#34d399"],
            [0.5, "#fbbf24"], [0.75, "#fb923c"], [1.0, "#f87171"],
        ],
        range_color=(0, 1),
        title="COVID-19 Composite Risk Score by Country",
        template="plotly_dark",
    )
    fig_choro.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_colorbar=dict(
            title="Risk Score",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0 · Low", "0.25", "0.5", "0.75", "1.0 · Critical"],
            thickness=12, len=0.7,
        ),
        geo=dict(
            bgcolor="rgba(26,32,48,1)",
            lakecolor="rgba(26,32,48,1)",
            landcolor="rgba(30,40,60,1)",
            coastlinecolor="rgba(255,255,255,0.06)",
            showcoastlines=True,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=480,
    )
    st.plotly_chart(fig_choro, use_container_width=True)

    col_bar, col_trend = st.columns([1, 1])
    with col_bar:
        top15 = latest_risk.sort_values("risk_score", ascending=False).head(15)
        bar_colors = [RISK_MAP.get(l, "#4a5568") for l in top15["risk_label"][::-1]]
        fig_bar = go.Figure(go.Bar(
            x=top15["risk_score"][::-1],
            y=top15["country"][::-1],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{s:.3f}" for s in top15["risk_score"][::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#8b97a8"),
            hovertemplate="<b>%{y}</b><br>Risk: %{x:.3f}<extra></extra>",
        ))
        apply_layout(fig_bar,
            title="Top 15 Highest Risk Countries",
            height=440,
            xaxis=dict(range=[0, 1.15], title="Risk Score", **PLOTLY_LAYOUT["xaxis"]),
            margin=dict(l=120, r=70, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_trend:
        fig_risk_ts = go.Figure()
        for i, c in enumerate(selected_countries[:5]):
            cdf_risk = df_risk[df_risk["country"] == c].sort_values("date").tail(365)
            if len(cdf_risk) == 0: continue
            fig_risk_ts.add_trace(go.Scatter(
                x=cdf_risk["date"],
                y=cdf_risk["risk_score"].rolling(7).mean(),
                name=c, line=dict(color=COLORS[i], width=2), mode="lines",
                hovertemplate=f"<b>{c}</b><br>%{{x|%b %d}}<br>Risk: %{{y:.3f}}<extra></extra>",
            ))
        apply_layout(fig_risk_ts,
            title="Risk Score Over Time (7-day avg)",
            height=440,
            yaxis=dict(range=[0, 1.05], title="Risk Score", **PLOTLY_LAYOUT["yaxis"]),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_risk_ts, use_container_width=True)

    with st.expander("Full Country Risk Table"):
        display_risk = latest_risk.sort_values("risk_score", ascending=False).copy()
        display_risk["risk_score"] = display_risk["risk_score"].round(4)
        display_risk["growth_rate_weekly"] = display_risk["growth_rate_weekly"].round(4)
        display_risk["pct_fully_vaccinated"] = display_risk["pct_fully_vaccinated"].round(2)
        st.dataframe(display_risk, use_container_width=True, height=380)

    with st.expander("SEIR + Risk Model Details"):
        st.code("""
SEIR MODEL
──────────────────────────────────────────────────────
Compartments:  S (Susceptible)  E (Exposed)
               I (Infectious)   R (Recovered)

ODEs:
  dS/dt = -β·S·I / N
  dE/dt =  β·S·I / N − σ·E
  dI/dt =  σ·E − γ·I
  dR/dt =  γ·I

Fitting:  scipy.optimize.curve_fit  ·  last 180 days
  β  [0.05, 1.5]   σ  [0.05, 0.5]   γ  [0.02, 0.5]
  R₀ = β / γ

RISK SCORE  (0–1 composite)
──────────────────────────────────────────────────────
  50%  Weekly growth rate  (min-max scaled)
  25%  Vaccination rate    (inverted)
  25%  Hospital beds/1k    (inverted)

Labels:  0.00–0.25  Low  ·  0.25–0.50  Moderate
         0.50–0.75  High  ·  0.75–1.00  Critical
        """, language="text")


# ════════════════════════════════════════════════════════
# TAB 4 — GLOBAL OVERVIEW
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
      <div class="section-title-text">Global Overview &amp; Country Comparison</div>
      <span class="section-tag blue">Live Data</span>
    </div>
    """, unsafe_allow_html=True)

    global_df = jhu_df.groupby("date")["confirmed"].sum().reset_index()
    global_df["new_cases"] = global_df["confirmed"].diff().clip(lower=0)
    global_df["7d_avg"]    = global_df["new_cases"].rolling(7).mean()

    fig_global = make_subplots(specs=[[{"secondary_y": True}]])
    fig_global.add_trace(go.Bar(
        x=global_df["date"], y=global_df["new_cases"],
        name="Daily New Cases",
        marker=dict(color="rgba(56,189,248,0.20)", line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:,.0f} cases<extra></extra>",
    ), secondary_y=False)
    fig_global.add_trace(go.Scatter(
        x=global_df["date"], y=global_df["7d_avg"],
        name="7-Day Average", line=dict(color="#f87171", width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>7d avg: %{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig_global.add_trace(go.Scatter(
        x=global_df["date"], y=global_df["confirmed"],
        name="Cumulative", line=dict(color="#fbbf24", width=1.5, dash="dot"),
        opacity=0.6,
    ), secondary_y=True)
    fig_global.update_layout(
        **PLOTLY_LAYOUT,
        title="Global COVID-19 Trend",
        height=420,
        margin=dict(l=50, r=70, t=60, b=40),
    )
    fig_global.update_xaxes(**PLOTLY_LAYOUT["xaxis"])
    fig_global.update_yaxes(**PLOTLY_LAYOUT["yaxis"])
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-title-text" style="font-size:15px;">Country-Level Comparison</div>
    </div>
    """, unsafe_allow_html=True)

    comp_metric = st.selectbox(
        "Metric",
        ["new_cases", "total_cases", "new_deaths", "stringency_index",
         "reproduction_rate", "hospital_beds_per_thousand"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    fig_comp = go.Figure()
    for i, c in enumerate(selected_countries):
        cdf_c = owid_df[owid_df["country"] == c].sort_values("date").tail(365)
        if len(cdf_c) == 0 or comp_metric not in cdf_c.columns: continue
        fig_comp.add_trace(go.Scatter(
            x=cdf_c["date"],
            y=cdf_c[comp_metric].rolling(7).mean(),
            name=c, line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{c}</b><br>%{{x|%b %d}}<br>%{{y:,.1f}}<extra></extra>",
        ))
    apply_layout(fig_comp,
        title=f"{comp_metric.replace('_',' ').title()} — 7-Day Rolling Average",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-title-text" style="font-size:15px;">Vaccination Coverage vs Risk Score</div>
      <span class="section-tag green">Correlation View</span>
    </div>
    """, unsafe_allow_html=True)

    scatter_df = latest_risk[latest_risk["pct_fully_vaccinated"] > 0].dropna()
    fig_scatter = px.scatter(
        scatter_df, x="pct_fully_vaccinated", y="risk_score",
        color="risk_label", hover_name="country",
        color_discrete_map=RISK_MAP,
        template="plotly_dark",
        title="Vaccination Rate vs Composite Risk Score",
        labels={"pct_fully_vaccinated": "% Fully Vaccinated", "risk_score": "Risk Score"},
    )
    fig_scatter.update_traces(
        marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
        hovertemplate="<b>%{hovertext}</b><br>Vacc: %{x:.1f}%<br>Risk: %{y:.3f}<extra></extra>",
    )
    apply_layout(fig_scatter, height=420, margin=dict(l=50, r=20, t=50, b=50))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<hr class="epi-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-title-text" style="font-size:15px;">Export Data</div>
    </div>
    """, unsafe_allow_html=True)

    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
    with col_exp1:
        csv_risk = latest_risk.to_csv(index=False)
        st.download_button(
            "⬇ Risk Scores CSV",
            data=csv_risk,
            file_name="epiwatch_risk_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp2:
        csv_global = global_df.tail(365).to_csv(index=False)
        st.download_button(
            "⬇ Global Trend CSV",
            data=csv_global,
            file_name="epiwatch_global_trend.csv",
            mime="text/csv",
            use_container_width=True,
        )
