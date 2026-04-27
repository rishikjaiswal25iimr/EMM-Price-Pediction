"""
EMM 97% Hybrid Forecasting — Streamlit Dashboard  ·  v3.0
==========================================================
LIGHTWEIGHT web app — loads pre-computed CSVs only.
No model training inside this file.

New in v3.0
-----------
  • Dual display: Index value + Real price (Rs/Kg)
  • Real market price overlay (from Excel-ingested data)
  • Fixed forecast horizon: 4 wk / 12 wk / 6 mo / 1 yr / 2 yr
  • Steel Production Value Calculator (Effective Mn + Real Mn Value)
  • Dynamic Real Mn Value time-series graph
  • Full-horizon future forecast display (no silent truncation)

Run:
    streamlit run app.py

Prerequisites:
    Run pipeline.py first → ./outputs/
    pip install streamlit plotly pandas numpy scipy
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EMM Price Forecasting · v3.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

C_ACTUAL   = "#E8E8F0"
C_HYBRID   = "#40E090"
C_LGB      = "#4A9EDA"
C_FUTURE   = "#F0A030"
C_MARKET   = "#FF6B9D"
C_MN_VAL   = "#C084FC"
C_REG1     = "rgba(140,30,30,0.20)"
C_FLOOR    = "rgba(50,120,220,0.10)"
C_CEILING  = "rgba(220,50,50,0.08)"
C_GRID     = "rgba(255,255,255,0.06)"
C_CI       = "rgba(240,160,48,0.12)"

OUTPUT_DIR = "./outputs"

HORIZON_OPTIONS = {
    "4 weeks (1 month)":     4,
    "12 weeks (3 months)":  12,
    "26 weeks (6 months)":  26,
    "52 weeks (1 year)":    52,
    "104 weeks (2 years)": 104,
}

SHOCK_ANNOTATIONS = {
    "2018-07-06": "US 301 Tariff",
    "2021-03-01": "Cartel Cut",
    "2021-09-15": "Env Audit",
    "2025-01-15": "Env Audit '25",
    "2025-04-01": "Cartel '25",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_historical(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


@st.cache_data(show_spinner=False)
def load_future(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    # Ensure column name compatibility with both v2 and v3
    if "predicted_price" in df.columns and "predicted_index" not in df.columns:
        df = df.rename(columns={"predicted_price": "predicted_index"})
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if "feature" not in df.columns:
        df.columns = ["feature", "importance"]
    return df.sort_values("importance", ascending=False).head(20)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def files_exist(out_dir: str) -> tuple[bool, list]:
    needed  = ["historical_predictions.csv", "future_forecast.csv", "model_metadata.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(out_dir, f))]
    return not missing, missing


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_shapes(dates: pd.DatetimeIndex, probs: np.ndarray,
                        threshold: float = 0.5) -> list:
    labels = (probs > threshold).astype(int)
    shapes, in_block, t0 = [], False, None
    for d, lbl in zip(dates, labels):
        if lbl == 1 and not in_block:
            in_block, t0 = True, d
        elif lbl == 0 and in_block:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=str(t0), x1=str(d), y0=0, y1=1,
                               fillcolor=C_REG1, line_width=0, layer="below"))
            in_block = False
    if in_block:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=str(t0), x1=str(dates[-1]), y0=0, y1=1,
                           fillcolor=C_REG1, line_width=0, layer="below"))
    return shapes


def _layout(title: str, y_title: str = "Price", height: int = 460) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=11, color="#b0b0c8"),
        title=dict(text=title, font=dict(size=14, color="#e0e0f0"), x=0.01),
        legend=dict(bgcolor="rgba(10,10,25,0.7)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=55, r=20, t=55, b=40),
    )


def _add_shock_vlines(fig: go.Figure, date_min: pd.Timestamp):
    for ds, label in SHOCK_ANNOTATIONS.items():
        d = pd.Timestamp(ds)
        if d >= date_min:
            fig.add_vline(x=d.timestamp()*1000,
                          line=dict(color="rgba(255,80,80,0.4)", width=1, dash="dot"),
                          annotation_text=label,
                          annotation_font=dict(size=8.5, color="rgba(255,120,120,0.8)"),
                          annotation_position="top")


def _confidence_band(fig: go.Figure, dates, values: np.ndarray):
    n     = len(values)
    idx   = np.arange(1, n+1)
    sigma = np.std(values) * 0.015 * idx
    upper = values + 1.96*sigma
    lower = values - 1.96*sigma
    fig.add_trace(go.Scatter(
        x=list(dates)+list(dates[::-1]),
        y=list(upper)+list(lower[::-1]),
        fill="toself", fillcolor=C_CI, line=dict(width=0),
        name="95% CI", showlegend=True, hoverinfo="skip",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_price_comparison(
    hist: pd.DataFrame,
    future: pd.DataFrame,
    display_window: int,
    price_mode: str,          # "index" | "real"
    show_lgb: bool,
    show_regime: bool,
    show_market: bool,
) -> go.Figure:
    """
    Main chart: Actual vs Hybrid (historical) + Future forecast.
    Supports both index and real Rs/Kg display modes.
    """
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    if price_mode == "real":
        actual_col  = "real_price"
        pred_col    = "real_price"
        future_col  = "real_price"
        y_title     = "Price (Rs/Kg)"
        actual_vals = h["real_price"] if "real_price" in h else h["actual"]
        hybrid_vals = h["real_price"] if "real_price" in h else h.get("hybrid_prediction", h["lgb_prediction"])
    else:
        actual_col  = "actual"
        pred_col    = "hybrid_prediction" if "hybrid_prediction" in h.columns else "lgb_prediction"
        future_col  = "predicted_index"
        y_title     = "Price Index"
        actual_vals = h["actual"]
        hybrid_vals = h[pred_col]

    fig = go.Figure()

    # Regime shading
    if show_regime and "regime_probability" in h.columns:
        for s in build_regime_shapes(h.index, h["regime_probability"].values):
            fig.add_shape(**s)

    # Actual
    fig.add_trace(go.Scatter(
        x=h.index, y=actual_vals,
        name=f"Actual {'(Rs/Kg)' if price_mode=='real' else '(Index)'}",
        mode="lines", line=dict(color=C_ACTUAL, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.2f}<extra></extra>",
    ))

    # LightGBM base (index mode only)
    if show_lgb and price_mode == "index" and "lgb_prediction" in h.columns:
        fig.add_trace(go.Scatter(
            x=h.index, y=h["lgb_prediction"],
            name="LightGBM base", mode="lines",
            line=dict(color=C_LGB, width=1.0, dash="dot"), opacity=0.65,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>LGB: %{y:.3f}<extra></extra>",
        ))

    # Hybrid prediction
    fig.add_trace(go.Scatter(
        x=h.index, y=hybrid_vals,
        name=f"Hybrid Forecast {'(Rs/Kg)' if price_mode=='real' else '(Index)'}",
        mode="lines", line=dict(color=C_HYBRID, width=2.0),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Hybrid: %{y:.2f}<extra></extra>",
    ))

    # Real market price overlay
    if show_market and "market_price" in h.columns:
        mkt = h["market_price"].dropna()
        if not mkt.empty:
            fig.add_trace(go.Scatter(
                x=mkt.index, y=mkt.values,
                name="Market Price (Excel)", mode="lines+markers",
                line=dict(color=C_MARKET, width=2.0),
                marker=dict(size=4, color=C_MARKET),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Market: ₹%{y:.2f}/Kg<extra></extra>",
            ))

    # Bridge connector
    if not future.empty and future_col in future.columns:
        bridge_y0 = float(hybrid_vals.iloc[-1])
        bridge_y1 = float(future[future_col].iloc[0])
        fig.add_trace(go.Scatter(
            x=[h.index[-1], future.index[0]], y=[bridge_y0, bridge_y1],
            mode="lines", line=dict(color=C_FUTURE, width=1.2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # Future forecast
    if not future.empty and future_col in future.columns:
        fp = future[future_col]
        _confidence_band(fig, future.index, fp.values)
        fig.add_trace(go.Scatter(
            x=future.index, y=fp,
            name=f"Future Forecast {'(Rs/Kg)' if price_mode=='real' else '(Index)'}",
            mode="lines+markers",
            line=dict(color=C_FUTURE, width=2.2, dash="dash"),
            marker=dict(size=4.5, color=C_FUTURE),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.2f}<extra></extra>",
        ))

    _add_shock_vlines(fig, h.index[0])
    fig.update_layout(**_layout(
        f"EMM 97% Hybrid Price Forecast — {'Real Price (Rs/Kg)' if price_mode=='real' else 'Index Mode'}",
        y_title=y_title, height=470,
    ))
    return fig


def chart_dual_axis(
    hist: pd.DataFrame,
    future: pd.DataFrame,
    display_window: int,
) -> go.Figure:
    """
    Dual Y-axis chart: Index (left) + Real Rs/Kg (right).
    Shows both values simultaneously for model interpretation.
    """
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=h.index, y=h["hybrid_prediction"] if "hybrid_prediction" in h else h["lgb_prediction"],
        name="Hybrid Index (L-axis)", mode="lines",
        line=dict(color=C_HYBRID, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Index: %{y:.3f}<extra></extra>",
    ), secondary_y=False)

    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(
            x=h.index, y=h["real_price"],
            name="Real Price Rs/Kg (R-axis)", mode="lines",
            line=dict(color=C_FUTURE, width=1.8, dash="dot"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f}/Kg<extra></extra>",
        ), secondary_y=True)

    if not future.empty:
        if "predicted_index" in future.columns:
            fig.add_trace(go.Scatter(
                x=future.index, y=future["predicted_index"],
                name="Future Index (L)", mode="lines",
                line=dict(color=C_HYBRID, width=1.5, dash="dash"), opacity=0.7,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Idx: %{y:.3f}<extra></extra>",
            ), secondary_y=False)
        if "real_price" in future.columns:
            fig.add_trace(go.Scatter(
                x=future.index, y=future["real_price"],
                name="Future Rs/Kg (R)", mode="lines",
                line=dict(color=C_FUTURE, width=1.5, dash="dash"), opacity=0.7,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f}/Kg<extra></extra>",
            ), secondary_y=True)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=11, color="#b0b0c8"),
        title=dict(text="Index vs Real Price (Rs/Kg) — Dual Axis",
                   font=dict(size=14, color="#e0e0f0"), x=0.01),
        legend=dict(bgcolor="rgba(10,10,25,0.7)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, font=dict(size=10)),
        hovermode="x unified", height=400,
        margin=dict(l=55, r=65, t=55, b=40),
    )
    fig.update_yaxes(title_text="Price Index", secondary_y=False,
                     gridcolor=C_GRID, showgrid=True)
    fig.update_yaxes(title_text="Price (Rs/Kg)", secondary_y=True,
                     showgrid=False)
    return fig


def chart_regime(hist: pd.DataFrame, display_window: int) -> go.Figure:
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff]
    if "regime_probability" not in h.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h.index, y=h["regime_probability"],
        name="P(Supply Squeeze)", mode="lines",
        fill="tozeroy", fillcolor="rgba(180,40,40,0.22)",
        line=dict(color="#cc4444", width=1.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P(Squeeze): %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color="rgba(200,200,220,0.5)", width=1, dash="dash"))
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(140,30,30,0.07)", line_width=0)
    fig.update_layout(**_layout(
        "Markov Regime: P(Supply Squeeze / Cartel Intervention)",
        y_title="Probability", height=260,
    ))
    fig.update_layout(yaxis=dict(range=[0,1], tickformat=".1f",
                                 showgrid=True, gridcolor=C_GRID,
                                 title="Probability"), showlegend=False)
    return fig


def chart_feature_importance(fi: pd.DataFrame) -> go.Figure:
    CAT_COLOR = {
        "usd_cny":"#5577dd","crude":"#ff9933","ore":"#cc6633",
        "freight":"#9966cc","steel":"#55aacc","nickel":"#55aa55",
        "cartel":"#ff4444","env_audit":"#ff5555","tariff":"#ff6655",
        "chn":"#ffcc44",
    }
    def gc(name):
        for k,c in CAT_COLOR.items():
            if k in name.lower(): return c
        return "#8080a0"

    top    = fi.head(15).iloc[::-1]
    colors = [gc(f) for f in top["feature"]]
    fig    = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"], orientation="h",
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(**_layout("Feature Importance (LightGBM)", "Score", 400))
    fig.update_layout(yaxis=dict(tickfont=dict(size=9), showgrid=False),
                      margin=dict(l=195, r=20, t=55, b=40))
    return fig


def chart_market_comparison(hist: pd.DataFrame, display_window: int) -> go.Figure | None:
    """Dedicated chart: real_price (model) vs market_price (Excel)."""
    if "market_price" not in hist.columns:
        return None
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff]
    mkt = h["market_price"].dropna()
    if mkt.empty:
        return None

    fig = go.Figure()
    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(
            x=h.index, y=h["real_price"],
            name="Model (Rs/Kg)", mode="lines",
            line=dict(color=C_HYBRID, width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Model: ₹%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=mkt.index, y=mkt.values,
        name="Market — Avg Mn Briq (Rs/Kg)", mode="lines+markers",
        line=dict(color=C_MARKET, width=2),
        marker=dict(size=4.5, color=C_MARKET),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Market: ₹%{y:.2f}<extra></extra>",
    ))
    # Compute error band
    common = h[["real_price"]].join(mkt, how="inner").dropna()
    if len(common) > 2:
        diff = common["real_price"] - common["market_price"]
        fig.add_trace(go.Scatter(
            x=common.index, y=diff,
            name="Model − Market error", mode="lines",
            line=dict(color="#ffcc44", width=1.2, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Error: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            yaxis2=dict(title="Error (Rs/Kg)", overlaying="y", side="right",
                        showgrid=False, zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.2)")
        )

    fig.update_layout(**_layout(
        "Real Price Comparison: Model vs Market Data (Avg Mn Briq)",
        y_title="Rs/Kg", height=380,
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# STEEL PRODUCTION VALUE CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════

def render_steel_calculator(
    hist: pd.DataFrame,
    future: pd.DataFrame,
    display_window: int,
) -> None:
    """
    Section: Steel Production Value Calculator

    Inputs  : Mn %  (user slider)
              Recovery %  (user slider)

    Formulas:
      Effective Mn  = Mn% × Recovery%                (fraction)
      Real Mn Value = Real Price (Rs/Kg) / Effective Mn   (Rs/Kg of effective Mn)

    Display :
      • KPI cards: Effective Mn %, Real Mn Value at latest prediction
      • Time-series: Real Mn Value across full timeline (historical + future)
        overlaid with real price and market price
    """
    st.markdown("---")
    st.markdown("### 🏭 Steel Production Value Calculator")
    st.caption(
        "Computes the **economic value of Mn** per kg of effective manganese delivered "
        "to the steelmaking process. Updates dynamically as model prices change."
    )

    col_mn, col_rec, col_spacer = st.columns([1, 1, 3])
    with col_mn:
        mn_pct = st.slider(
            "Mn content (%)", min_value=1.0, max_value=99.0,
            value=97.0, step=0.5,
            help="Manganese % in the EMM product (97% for EMM 97%)",
        )
    with col_rec:
        rec_pct = st.slider(
            "Recovery (%)", min_value=50.0, max_value=100.0,
            value=92.0, step=0.5,
            help="Metallurgical recovery of Mn in the furnace/steel process",
        )

    eff_mn_frac = (mn_pct / 100.0) * (rec_pct / 100.0)   # e.g. 0.97 × 0.92
    eff_mn_pct  = eff_mn_frac * 100.0                      # display as %

    # Latest real price (prefer last historical, fall back to first future)
    if "real_price" in hist.columns:
        latest_real = float(hist["real_price"].iloc[-1])
    elif "real_price" in future.columns:
        latest_real = float(future["real_price"].iloc[0])
    else:
        latest_real = 0.0

    real_mn_value = latest_real / eff_mn_frac if eff_mn_frac > 0 else 0.0

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Mn Content",      f"{mn_pct:.1f}%")
    kc2.metric("Recovery",        f"{rec_pct:.1f}%")
    kc3.metric("Effective Mn",    f"{eff_mn_pct:.2f}%")
    kc4.metric("Real Mn Value",   f"₹{real_mn_value:.2f}/Kg",
               help="Rs per kg of effective Mn delivered")

    st.markdown(
        f"> **Formula:** &nbsp; Effective Mn = {mn_pct:.1f}% × {rec_pct:.1f}% = **{eff_mn_pct:.2f}%**"
        f" &nbsp;|&nbsp; Real Mn Value = ₹{latest_real:.2f} ÷ {eff_mn_frac:.4f} = **₹{real_mn_value:.2f}/Kg of eff. Mn**"
    )

    # ── Build combined timeline: historical + future ───────────────────────────
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    # Compute Real Mn Value for historical
    if "real_price" in h.columns:
        h["mn_value"] = h["real_price"] / eff_mn_frac
    else:
        h["mn_value"] = np.nan

    # Compute Real Mn Value for future
    fut = future.copy()
    if "real_price" in fut.columns:
        fut["mn_value"] = fut["real_price"] / eff_mn_frac
    else:
        fut["mn_value"] = np.nan

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Real price (Rs/Kg) — left axis
    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(
            x=h.index, y=h["real_price"],
            name="Real Price (Rs/Kg)", mode="lines",
            line=dict(color=C_HYBRID, width=1.8),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f}/Kg<extra></extra>",
        ))
    if "real_price" in fut.columns:
        fig.add_trace(go.Scatter(
            x=fut.index, y=fut["real_price"],
            name="Future Price (Rs/Kg)", mode="lines",
            line=dict(color=C_FUTURE, width=1.8, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f}/Kg<extra></extra>",
        ))

    # Market price overlay
    if "market_price" in h.columns:
        mkt = h["market_price"].dropna()
        if not mkt.empty:
            fig.add_trace(go.Scatter(
                x=mkt.index, y=mkt.values,
                name="Market Price (Rs/Kg)", mode="lines+markers",
                line=dict(color=C_MARKET, width=1.6),
                marker=dict(size=3.5, color=C_MARKET),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Market: ₹%{y:.2f}<extra></extra>",
            ))

    # Real Mn Value — right axis
    if not h["mn_value"].isna().all():
        fig.add_trace(go.Scatter(
            x=h.index, y=h["mn_value"],
            name=f"Mn Value ({mn_pct:.0f}%×{rec_pct:.0f}% rec) — Rs/Kg eff.Mn",
            mode="lines", line=dict(color=C_MN_VAL, width=2.0),
            yaxis="y2",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Mn Val: ₹%{y:.2f}/Kg eff.<extra></extra>",
        ))
    if not fut["mn_value"].isna().all():
        fig.add_trace(go.Scatter(
            x=fut.index, y=fut["mn_value"],
            name="Future Mn Value", mode="lines",
            line=dict(color=C_MN_VAL, width=1.8, dash="dash"),
            yaxis="y2",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Mn Val: ₹%{y:.2f}/Kg eff.<extra></extra>",
        ))

    # Vertical line at today (transition from historical to future)
    if not future.empty:
        fig.add_vline(
            x=hist.index[-1].timestamp()*1000,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            annotation_text="Today", annotation_position="top right",
            annotation_font=dict(size=9, color="rgba(200,200,220,0.8)"),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=11, color="#b0b0c8"),
        title=dict(
            text=f"Real Mn Value (Mn={mn_pct:.1f}%, Recovery={rec_pct:.1f}%) — Full Timeline",
            font=dict(size=13, color="#e0e0f0"), x=0.01,
        ),
        legend=dict(bgcolor="rgba(10,10,25,0.7)",
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                    font=dict(size=10)),
        yaxis=dict(title="Price (Rs/Kg)", showgrid=True, gridcolor=C_GRID),
        yaxis2=dict(title="Mn Value (Rs/Kg eff.Mn)", overlaying="y",
                    side="right", showgrid=False,
                    tickfont=dict(color=C_MN_VAL)),
        hovermode="x unified",
        height=430,
        margin=dict(l=55, r=70, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Summary table
    with st.expander("📋 Steel Value Table (sampled — every 4 weeks)"):
        all_dates = list(h.index[::4]) + list(fut.index[::4])
        all_real  = (list(h["real_price"].iloc[::4])
                     if "real_price" in h.columns else [np.nan]*len(h.index[::4]))
        all_mn    = (list(h["mn_value"].iloc[::4])
                     if "mn_value" in h.columns else [np.nan]*len(h.index[::4]))
        fut_real  = (list(fut["real_price"].iloc[::4])
                     if "real_price" in fut.columns else [np.nan]*len(fut.index[::4]))
        fut_mn    = (list(fut["mn_value"].iloc[::4])
                     if "mn_value" in fut.columns else [np.nan]*len(fut.index[::4]))

        tbl = pd.DataFrame({
            "Date":              [d.strftime("%Y-%m-%d") for d in all_dates],
            "Real Price Rs/Kg":  all_real + fut_real,
            "Mn Value Rs/Kg":    all_mn + fut_mn,
            "Period":            ["Historical"]*len(h.index[::4]) + ["Forecast"]*len(fut.index[::4]),
        })
        st.dataframe(
    tbl.style.format({
        "Real Price Rs/Kg": "{:.2f}",
        "Mn Value Rs/Kg":   "{:.2f}",
    }),
    use_container_width=True, hide_index=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(meta: dict) -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.divider()

        output_dir = st.text_input("Output directory", value=OUTPUT_DIR)

        st.markdown("### 📅 Display Window")
        display_window = st.slider(
            "History to show (weeks)", 52, 520, 260, step=26,
        )

        st.markdown("### 🔭 Forecast Horizon")
        horizon_label = st.selectbox(
            "Forecast horizon",
            list(HORIZON_OPTIONS.keys()),
            index=2,   # default: 26 weeks (6 months)
        )
        horizon = HORIZON_OPTIONS[horizon_label]

        st.markdown("### 📊 Price Mode")
        price_mode = st.radio(
            "Display prices as",
            ["Real price (Rs/Kg)", "Index value"],
            index=0,
        )
        price_mode_key = "real" if "Real" in price_mode else "index"

        st.markdown("### 🎨 Chart Options")
        show_lgb    = st.checkbox("Show LightGBM base",      value=False)
        show_regime = st.checkbox("Show regime shading",     value=True)
        show_market = st.checkbox("Show market price (Excel)", value=True)
        show_dual   = st.checkbox("Show dual-axis chart",    value=False)

        st.divider()
        st.markdown("### 🧠 Model Info")
        if meta:
            cal = meta.get("calibration", {})
            st.markdown(f"- **Train:** {meta.get('train_start','-')} → {meta.get('train_end','-')}")
            st.markdown(f"- **Obs:** {meta.get('n_observations',0):,}  |  **Feats:** {meta.get('n_features',0)}")
            st.markdown(f"- **LSTM:** {'✅' if meta.get('lstm_enabled') else '❌'}  "
                        f"**Markov:** {'✅' if meta.get('markov_enabled') else '❌'}")
            st.markdown(f"- **Scaling factor:** `{cal.get('scaling_factor', 'N/A')}`")
            st.markdown(f"- **Anchor:** {cal.get('reference_date','?')} = "
                        f"₹{cal.get('reference_price','?')}/Kg")
            if meta.get("excel_market_loaded"):
                st.markdown("- **Excel market data:** ✅ loaded")

        st.divider()
        st.caption(
            "⚠️ Proxy: MXI ETF → calibrated to INR/Kg via anchor price.  \n"
            "`real_price = index × scaling_factor`  \n"
            "Recalibrate by updating anchor in pipeline.py."
        )

    return dict(
        output_dir    = output_dir,
        display_window= display_window,
        horizon       = horizon,
        price_mode    = price_mode_key,
        show_lgb      = show_lgb,
        show_regime   = show_regime,
        show_market   = show_market,
        show_dual     = show_dual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

def render_metrics(hist: pd.DataFrame, future: pd.DataFrame, meta: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # Last actual index + real
    last_idx  = float(hist["actual"].iloc[-1])
    last_real = float(hist["real_price"].iloc[-1]) if "real_price" in hist.columns else 0.0

    # Latest hybrid real
    pred_col = "real_price" if "real_price" in hist.columns else "lgb_prediction"
    last_pred_real = float(hist[pred_col].iloc[-1])

    # Next week forecast
    nw_real = float(future["real_price"].iloc[0]) if "real_price" in future.columns else 0.0
    nw_idx  = float(future["predicted_index"].iloc[0]) if "predicted_index" in future.columns else 0.0
    end_real= float(future["real_price"].iloc[-1]) if "real_price" in future.columns else 0.0

    pct_chg = (nw_real - last_real) / (last_real + 1e-9) * 100 if last_real else 0.0

    # MAPE
    pred_for_mape = hist.get("hybrid_prediction", hist.get("lgb_prediction", hist["actual"]))
    mape = float(np.mean(np.abs((hist["actual"] - pred_for_mape)
                                / (hist["actual"] + 1e-9))) * 100)

    c1.metric("Last Actual (Index)",  f"{last_idx:.3f}")
    c2.metric("Last Actual (Rs/Kg)",  f"₹{last_real:.2f}")
    c3.metric("Next-Wk Forecast",     f"₹{nw_real:.2f}",
              delta=f"{pct_chg:+.1f}%")
    c4.metric(f"End Forecast (Rs/Kg)",f"₹{end_real:.2f}",
              delta=f"{len(future)} wk ahead", delta_color="off")
    c5.metric("In-Sample MAPE",       f"{mape:.2f}%")
    cal = meta.get("calibration", {})
    c6.metric("Scaling Factor",       f"{cal.get('scaling_factor', 0):.4f}",
              delta="1 idx = X Rs/Kg", delta_color="off")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:12px 0 6px;'>
      <h1 style='font-size:25px;margin:0;color:#e8e8f8;'>
        📈 EMM 97% Hybrid Price Forecasting &nbsp;
        <span style='font-size:14px;color:#6666aa;font-weight:400;'>v3.0</span>
      </h1>
      <p style='color:#8888aa;font-size:12px;margin:3px 0 0;font-family:monospace;'>
        LightGBM + Markov Regime-Switching + LSTM Residuals + Sigmoid Damping
        &nbsp;|&nbsp; Index → INR/Kg Calibration &nbsp;|&nbsp; Steel Value Calculator
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pre-load meta for sidebar ──────────────────────────────────────────────
    meta = {}
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        meta = load_metadata(meta_path)

    opts      = render_sidebar(meta)
    out_dir   = opts["output_dir"]
    horizon   = opts["horizon"]

    # ── File check ─────────────────────────────────────────────────────────────
    ok, missing = files_exist(out_dir)
    if not ok:
        st.error(
            f"⛔ Missing outputs in `{out_dir}`: **{', '.join(missing)}**\n\n"
            "Run the pipeline first:\n"
            "```bash\n"
            "python pipeline.py --horizon 104 --no-lstm --output-dir ./outputs\n"
            "```"
        )
        st.info(
            "**Google Colab quickstart**\n"
            "1. Upload `pipeline.py`\n"
            "2. `!pip install lightgbm yfinance scikit-learn statsmodels scipy openpyxl`\n"
            "3. `!python pipeline.py --horizon 104 --no-lstm`\n"
            "4. Download `outputs/` folder\n"
            "5. Place next to `app.py` and run `streamlit run app.py`"
        )
        st.stop()

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data …"):
        hist        = load_historical(os.path.join(out_dir, "historical_predictions.csv"))
        future_full = load_future(os.path.join(out_dir, "future_forecast.csv"))
        if os.path.exists(meta_path):
            meta = load_metadata(meta_path)

    # Trim future to selected horizon — critical fix for horizon bug
    # future_full always contains up to 104 weeks; we slice here for display
    future = future_full.iloc[:horizon]

    fi_path = os.path.join(out_dir, "feature_importance.csv")
    fi      = load_feature_importance(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("---")
    render_metrics(hist, future, meta)
    st.markdown("---")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Forecast",
        "📊 Market Comparison",
        "🔀 Regime & Features",
        "🏭 Steel Calculator",
    ])

    # ══ Tab 1: Price Forecast ══════════════════════════════════════════════════
    with tab1:
        st.plotly_chart(
            chart_price_comparison(
                hist, future,
                display_window = opts["display_window"],
                price_mode     = opts["price_mode"],
                show_lgb       = opts["show_lgb"],
                show_regime    = opts["show_regime"],
                show_market    = opts["show_market"],
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

        if opts["show_dual"]:
            st.plotly_chart(
                chart_dual_axis(hist, future, opts["display_window"]),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Future forecast table
        st.markdown("##### 🔮 Future Forecast Detail")
        disp_future = future.copy().reset_index()
        disp_future["date"] = disp_future["date"].dt.strftime("%Y-%m-%d")
        if "real_price" in disp_future.columns:
            disp_future["real_price_fmt"] = disp_future["real_price"].apply(
                lambda x: f"₹{x:.2f}/Kg"
            )
        if "regime_probability" in disp_future.columns:
            disp_future["regime"] = disp_future["regime_probability"].apply(
                lambda p: "🔴 Squeeze" if p > 0.5 else "🟢 Normal"
            )
        cols_show = [c for c in ["date","predicted_index","real_price_fmt",
                                 "regime_probability","regime"]
                     if c in disp_future.columns]
        st.dataframe(disp_future[cols_show], use_container_width=True,
                     hide_index=True, height=280)

        # Downloads
        dlc1, dlc2 = st.columns(2)
        with dlc1:
            st.download_button("⬇️ Download Historical CSV",
                               hist.to_csv(), "historical_predictions.csv", "text/csv")
        with dlc2:
            st.download_button("⬇️ Download Forecast CSV",
                               future_full.to_csv(), "future_forecast.csv", "text/csv")

    # ══ Tab 2: Market Comparison ═══════════════════════════════════════════════
    with tab2:
        mkt_chart = chart_market_comparison(hist, opts["display_window"])
        if mkt_chart:
            st.plotly_chart(mkt_chart, use_container_width=True,
                            config={"displayModeBar": False})
            if "market_price" in hist.columns and "real_price" in hist.columns:
                common = hist[["real_price","market_price"]].dropna()
                if not common.empty:
                    err   = common["real_price"] - common["market_price"]
                    rmse  = float(np.sqrt((err**2).mean()))
                    mae   = float(err.abs().mean())
                    bias  = float(err.mean())
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("RMSE vs Market", f"₹{rmse:.2f}")
                    ec2.metric("MAE vs Market",  f"₹{mae:.2f}")
                    ec3.metric("Bias (Model−Mkt)",f"₹{bias:+.2f}")
        else:
            st.info(
                "📂 No market price data found.  \n"
                "Re-run pipeline with `--excel-path ./market_prices.xlsx` to enable this tab.\n\n"
                "**Excel format required:**\n"
                "| Date | Avg Mn Briq price (Rs/Kg) |\n"
                "|------|---------------------------|\n"
                "| 2023-01-06 | 152.30 |"
            )

    # ══ Tab 3: Regime & Features ═══════════════════════════════════════════════
    with tab3:
        if "regime_probability" in hist.columns:
            st.plotly_chart(
                chart_regime(hist, opts["display_window"]),
                use_container_width=True, config={"displayModeBar": False},
            )
            r1c, r2c = st.columns(2)
            pct1 = float((hist["regime_probability"] > 0.5).mean() * 100)
            pct0 = 100 - pct1
            r1c.metric("🔴 Regime 1 — Supply Squeeze", f"{pct1:.1f}% of time")
            r2c.metric("🟢 Regime 0 — Oversupply / Normal", f"{pct0:.1f}% of time")

        if not fi.empty:
            st.plotly_chart(
                chart_feature_importance(fi),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ══ Tab 4: Steel Calculator ════════════════════════════════════════════════
    with tab4:
        render_steel_calculator(hist, future, opts["display_window"])

    # ── Metadata expander ──────────────────────────────────────────────────────
    with st.expander("🧾 Model Metadata & Calibration Details"):
        st.json(meta)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "**Calibration Note** — Scaling factor = ₹154.46585 ÷ index(2024-01-02).  "
        "To recalibrate: update `REFERENCE_DATE` and `REFERENCE_PRICE` in pipeline.py and re-run.  \n"
        "EMM 97% Hybrid Forecasting v3.0 | LightGBM + Markov + LSTM + Sigmoid Damping"
    )


if __name__ == "__main__":
    main()
