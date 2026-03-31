# ================================================================
# VBL STOCK INTELLIGENCE PLATFORM v4 — "Snow & Slate" Edition
# Sentiment · Macro · Prophet · ARIMA · Monte Carlo
# Momentum · Seasonality · Value · Growth Trading Strategies
# ================================================================
# INSTALL:
# pip install streamlit plotly yfinance prophet pandas numpy
#             statsmodels scikit-learn openpyxl vaderSentiment requests
#
# RUN: streamlit run vbl_streamlit_2.py
# ================================================================

import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="VBL Intelligence v4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CSS — Snow & Slate (clean modern light theme)
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #ffffff;
  --bg-alt:     #f8f9fc;
  --surface:    #f1f3f8;
  --surface-2:  #e8ebf2;
  --border:     #e2e5ee;
  --border-l:   #eff1f5;
  --ink:        #1a1d27;
  --ink-2:      #2d3142;
  --ink-3:      #4a4f65;
  --muted:      #6b7189;
  --hint:       #9398ac;
  --accent:     #4361ee;
  --accent-h:   #3651d4;
  --accent-l:   #6980f5;
  --accent-bg:  rgba(67,97,238,0.06);
  --accent-bg2: rgba(67,97,238,0.10);
  --green:      #10b981;
  --green-d:    #059669;
  --green-bg:   rgba(16,185,129,0.08);
  --red:        #ef4444;
  --red-d:      #dc2626;
  --red-bg:     rgba(239,68,68,0.08);
  --amber:      #f59e0b;
  --amber-bg:   rgba(245,158,11,0.08);
  --cyan:       #06b6d4;
  --purple:     #8b5cf6;
  --shadow-sm:  0 1px 2px rgba(0,0,0,0.04);
  --shadow:     0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:  0 4px 6px -1px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
  --radius:     10px;
  --radius-sm:  6px;
  --radius-lg:  14px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background-color: var(--bg-alt) !important;
  color: var(--ink) !important;
}
.main { background-color: var(--bg-alt) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background-color: var(--bg-alt) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--ink) !important;
  font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stDateInput input {
  background: var(--bg-alt) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 13px !important;
}
section[data-testid="stSidebar"] .stSlider > div > div {
  background: var(--surface-2) !important;
}
section[data-testid="stSidebar"] button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.5px !important;
  border-radius: var(--radius-sm) !important;
  transition: background .15s, transform .1s;
}
section[data-testid="stSidebar"] button:hover {
  background: var(--accent-h) !important;
  transform: translateY(-1px);
}

/* ── Banner ── */
.banner {
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 24px 36px 20px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  position: relative;
}
.b-tick {
  font-family: 'DM Sans', sans-serif;
  font-size: 42px;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -1px;
  line-height: 1;
  margin: 0;
}
.b-name {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: var(--muted);
  letter-spacing: 0.3px;
  margin-top: 6px;
}
.b-price {
  font-family: 'DM Sans', sans-serif;
  font-size: 36px;
  font-weight: 700;
  color: var(--ink);
  text-align: right;
}
.b-chg {
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  text-align: right;
  font-weight: 500;
}
.b-meta {
  color: var(--hint);
  font-size: 11px;
  font-family: 'Inter', sans-serif;
  text-align: right;
  margin-top: 4px;
  letter-spacing: 0.2px;
}

/* ── KPI strip ── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(8,1fr);
  gap: 0;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
}
.kpi-cell {
  padding: 16px 18px;
  border-right: 1px solid var(--border-l);
  transition: background .15s;
}
.kpi-cell:last-child { border-right: none; }
.kpi-cell:hover { background: var(--bg-alt); }
.kpi-lbl {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--hint);
  margin-bottom: 6px;
}
.kpi-val {
  font-family: 'DM Sans', sans-serif;
  font-size: 16px;
  font-weight: 700;
  color: var(--ink);
}
.kpi-sub {
  font-size: 10px;
  color: var(--hint);
  margin-top: 2px;
}

/* ── Section header ── */
.sh {
  font-family: 'DM Sans', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.2px;
  padding-left: 14px;
  margin: 24px 0 14px;
  position: relative;
}
.sh::before {
  content: '';
  position: absolute;
  left: 0;
  top: 2px;
  bottom: 2px;
  width: 3px;
  background: var(--accent);
  border-radius: 2px;
}

/* ── Cards ── */
.card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 12px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .15s, border-color .15s;
}
.card:hover {
  box-shadow: var(--shadow);
  border-color: var(--surface-2);
}
.card-t {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--hint);
  margin-bottom: 8px;
}

/* ── Fundamentals grid ── */
.fg {
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 18px;
}
.fc {
  background: var(--bg);
  padding: 18px 20px;
  transition: background .15s;
}
.fc:hover { background: var(--bg-alt); }
.fl {
  font-size: 10px;
  font-weight: 600;
  color: var(--hint);
  letter-spacing: 0.4px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.fv {
  font-family: 'DM Sans', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--ink);
}
.fs {
  font-size: 11px;
  color: var(--hint);
  margin-top: 3px;
}

/* ── Strategy boxes ── */
.strat-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 14px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .2s, transform .1s;
  position: relative;
  overflow: hidden;
}
.strat-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
.strat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; bottom: 0;
  width: 3px;
  background: var(--accent);
}
.strat-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.2px;
  margin-bottom: 12px;
}
.strat-sig {
  font-size: 13px;
  color: var(--ink-3);
  line-height: 2;
}

/* ── News card ── */
.news-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  margin-bottom: 10px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .15s;
}
.news-card:hover { box-shadow: var(--shadow); }
.news-head {
  font-size: 13px;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.6;
  margin-bottom: 6px;
}
.news-meta {
  font-size: 10px;
  color: var(--hint);
}

/* ── Info box ── */
.ib {
  background: var(--accent-bg);
  border: 1px solid rgba(67,97,238,0.12);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  font-size: 12px;
  color: var(--ink-3);
  margin-top: 12px;
  line-height: 1.8;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 28px !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  color: var(--muted) !important;
  padding: 14px 16px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color .15s;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--ink-2) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding: 24px 30px !important;
  background: var(--bg-alt) !important;
}

/* ── Streamlit metrics ── */
[data-testid="metric-container"] {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 18px !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="metric-container"] label {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.5px !important;
  text-transform: uppercase !important;
  color: var(--hint) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
}
[data-testid="stMetricDelta"] {
  font-size: 11px !important;
  font-weight: 500;
}

/* ── Table ── */
.ft {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.ft th {
  background: var(--surface);
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.ft td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-l);
  color: var(--ink-2);
  background: var(--bg);
}
.ft tr:last-child td { border-bottom: none; }
.ft tr:hover td { background: var(--bg-alt); }
.g  { color: var(--green-d); font-weight: 600; }
.r  { color: var(--red-d); font-weight: 600; }
.mu { color: var(--hint); }

/* ── Sentiment bar ── */
.sent-bar {
  height: 4px;
  border-radius: 2px;
  background: var(--surface-2);
  margin: 6px 0;
  overflow: hidden;
}
.sent-fill {
  height: 100%;
  border-radius: 2px;
  transition: width .5s;
}

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  border-radius: 4px;
}
.b-buy  { background: var(--green-bg); color: var(--green-d); }
.b-sell { background: var(--red-bg); color: var(--red); }
.b-hold { background: var(--amber-bg); color: #d97706; }
.b-watch{ background: var(--surface); color: var(--muted); }

/* ── Spinner / loading ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-alt); }
::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--hint); }

/* ── Download button ── */
.stDownloadButton button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.3px !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 22px !important;
  transition: background .15s, transform .1s;
}
.stDownloadButton button:hover {
  background: var(--accent-h) !important;
  transform: translateY(-1px);
}

/* ── Warning / info / success boxes ── */
.stAlert {
  border-radius: var(--radius-sm) !important;
  font-size: 13px !important;
}
</style>
"", unsafe_allow_html=True)

# ================================================================
# COLOUR PALETTE (Plotly / Python usage)
# ================================================================
C = {
    'bg':       '#ffffff',
    'bg_alt':   '#f8f9fc',
    'surface':  '#f1f3f8',
    'surface_2':'#e8ebf2',
    'border':   '#e2e5ee',
    'ink':      '#1a1d27',
    'ink2':     '#2d3142',
    'ink3':     '#4a4f65',
    'muted':    '#6b7189',
    'hint':     '#9398ac',
    'accent':   '#4361ee',
    'green':    '#10b981',
    'red':      '#ef4444',
    'amber':    '#f59e0b',
    'cyan':     '#06b6d4',
    'purple':   '#8b5cf6',
}

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="VBL Intelligence v4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# LOAD DATA
# ================================================================
with st.spinner("Fetching price data…"):
    df = load_prices(ticker, str(start_date))
    info = load_info(ticker)

with st.spinner("Fetching & scoring news sentiment…"):
    news_items = fetch_sentiment(ticker)
    raw_sent = build_sentiment_series(ticker, len(df))
    sentiment_series = _align_sentiment(raw_sent, df)
    avg_sentiment = float(np.mean(sentiment_series)) if VADER_OK else 0.0
    recent_sentiment = float(np.mean(sentiment_series[-5:])) if VADER_OK else 0.0

with st.spinner("Training Prophet (sentiment + macro)…"):
    pm, pfc, used_sent = fit_prophet_enhanced(df, forecast_days, sentiment_series, macro if use_macro else None)

# ================================================================
# STRATEGY ENGINES
# ================================================================
def momentum_strategy(df):
    sig = pd.DataFrame(index=df.index)
    sig['date'] = df['Date']
    sig['rsi_sig'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    sig['macd_sig'] = np.where(df['MACD'] > df['MACD_sig'], 1, -1)
    sig['combined'] = sig['rsi_sig'] + sig['macd_sig']
    sig['action'] = np.where(sig['combined'] >= 1, 'BUY', np.where(sig['combined'] <= -1, 'SELL', 'HOLD'))

    return sig

# ================================================================
# VISUALIZATION
# ================================================================
with st.spinner('Preparing the metric…'):
    st.metric(label='Recent Sentiment', value=f'{recent_sentiment:.3f}', delta="NA")
