"""
EMM 97% Hybrid Forecasting — Streamlit Dashboard
==================================================
LIGHTWEIGHT web app — loads pre-computed CSVs only.
NO model training or heavy computation inside this file.

Run:
    streamlit run app.py

Prerequisites:
    Run pipeline.py first to generate ./outputs/ files.
    pip install streamlit plotly pandas numpy
"""

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
    page_title="EMM Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COLOR_ACTUAL      = "#E8E8F0"
COLOR_LGB         = "#4A9EDA"
COLOR_HYBRID      = "#40E090"
COLOR_FUTURE      = "#F0A030"
COLOR_REGIME_0    = "rgba(26,58,92,0.18)"   # navy  → normal
COLOR_REGIME_1    = "rgba(140,30,30,0.22)"  # crimson → squeeze
COLOR_FLOOR       = "rgba(50,120,220,0.12)"
COLOR_CEILING     = "rgba(220,50,50,0.10)"
COLOR_GRID        = "rgba(255,255,255,0.06)"

OUTPUT_DIR = "./outputs"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached — no reloads on widget interaction)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_historical(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


@st.cache_data(show_spinner=False)
def load_future(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Normalise column names
    df.columns = ["feature", "importance"] if len(df.columns) == 2 else df.columns
    return df.sort_values("importance", ascending=False).head(20)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def files_exist(output_dir: str) -> tuple[bool, list]:
    required = [
        "historical_predictions.csv",
        "future_forecast.csv",
        "model_metadata.json",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    return len(missing) == 0, missing


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: regime shading shapes for Plotly
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_shapes(dates: pd.DatetimeIndex, regime_probs: np.ndarray,
                        threshold: float = 0.5) -> list:
    """Return a list of Plotly vrect-style shape dicts for regime shading."""
    shapes = []
    labels = (regime_probs > threshold).astype(int)

    in_block = False
    block_start = None

    for i, (d, lbl) in enumerate(zip(dates, labels)):
        if lbl == 1 and not in_block:
            in_block = True
            block_start = d
        elif lbl == 0 and in_block:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=str(block_start), x1=str(d),
                y0=0, y1=1,
                fillcolor=COLOR_REGIME_1,
                line_width=0, layer="below"
            ))
            in_block = False

    if in_block:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=str(block_start), x1=str(dates[-1]),
            y0=0, y1=1,
            fillcolor=COLOR_REGIME_1,
            line_width=0, layer="below"
        ))
    return shapes


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

SHOCK_ANNOTATIONS = {
    "2018-07-06": "US 301 Tariff",
    "2021-03-01": "Cartel Cut",
    "2021-09-15": "Env Audit",
    "2025-01-15": "Env Audit '25",
    "2025-04-01": "Cartel '25",
}


def chart_main(hist: pd.DataFrame, future: pd.DataFrame,
               show_lgb: bool, show_regime: bool,
               display_window: int) -> go.Figure:
    """
    Primary chart: Actual vs Hybrid Prediction vs Future Forecast.
    """
    # Slice history to display window
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff]

    fig = go.Figure()

    # ── Regime shading ────────────────────────────────────────────────────────
    if show_regime and "regime_probability" in h.columns:
        shapes = build_regime_shapes(h.index, h["regime_probability"].values)
        for s in shapes:
            fig.add_shape(**s)

    # ── Actual ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=h.index, y=h["actual"],
        name="Actual (EMM proxy)", mode="lines",
        line=dict(color=COLOR_ACTUAL, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.3f}<extra></extra>",
    ))

    # ── LightGBM base (optional) ──────────────────────────────────────────────
    if show_lgb and "lgb_prediction" in h.columns:
        fig.add_trace(go.Scatter(
            x=h.index, y=h["lgb_prediction"],
            name="LightGBM base", mode="lines",
            line=dict(color=COLOR_LGB, width=1.0, dash="dot"),
            opacity=0.65,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>LGB: %{y:.3f}<extra></extra>",
        ))

    # ── Hybrid prediction ─────────────────────────────────────────────────────
    pred_col = "hybrid_prediction" if "hybrid_prediction" in h.columns else "lgb_prediction"
    fig.add_trace(go.Scatter(
        x=h.index, y=h[pred_col],
        name="Hybrid Forecast (in-sample)", mode="lines",
        line=dict(color=COLOR_HYBRID, width=2.0),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Hybrid: %{y:.3f}<extra></extra>",
    ))

    # ── Bridge connector (last actual → first future) ─────────────────────────
    if not future.empty:
        bridge_x = [h.index[-1], future.index[0]]
        bridge_y = [float(h[pred_col].iloc[-1]),
                    float(future["predicted_price"].iloc[0])]
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            name="_bridge", mode="lines",
            line=dict(color=COLOR_FUTURE, width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Future forecast + confidence band ────────────────────────────────────
    if not future.empty:
        fp = future["predicted_price"]
        # Simple uncertainty band: ±σ growing with horizon
        n   = len(fp)
        idx = np.arange(1, n + 1)
        sigma = fp.std() * 0.015 * idx          # grows linearly
        upper = fp.values + sigma * 1.96
        lower = fp.values - sigma * 1.96

        fig.add_trace(go.Scatter(
            x=list(future.index) + list(future.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(240,160,48,0.12)",
            line=dict(width=0),
            name="95% CI (future)",
            showlegend=True,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future.index, y=fp,
            name="Future Forecast", mode="lines+markers",
            line=dict(color=COLOR_FUTURE, width=2.2, dash="dash"),
            marker=dict(size=5, color=COLOR_FUTURE),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.3f}<extra></extra>",
        ))

    # ── Shock event lines ─────────────────────────────────────────────────────
    for date_str, label in SHOCK_ANNOTATIONS.items():
        d = pd.Timestamp(date_str)
        if d >= h.index[0]:
            fig.add_vline(
                x=d.timestamp() * 1000,
                line=dict(color="rgba(255,80,80,0.45)", width=1, dash="dot"),
                annotation_text=label,
                annotation_position="top",
                annotation_font=dict(size=9, color="rgba(255,120,120,0.8)"),
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=11, color="#b0b0c8"),
        title=dict(
            text="EMM 97% — Hybrid Price Forecast with Markov Regimes",
            font=dict(size=15, color="#e0e0f0"),
            x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(10,10,25,0.7)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(
            showgrid=True, gridcolor=COLOR_GRID,
            zeroline=False, tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=COLOR_GRID,
            zeroline=False, tickfont=dict(size=10),
            title="Price Index (proxy)",
        ),
        hovermode="x unified",
        height=480,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def chart_regime(hist: pd.DataFrame, display_window: int) -> go.Figure:
    """Regime probability area chart."""
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff]

    if "regime_probability" not in h.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h.index, y=h["regime_probability"],
        name="P(Supply Squeeze)",
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(180,40,40,0.22)",
        line=dict(color="#cc4444", width=1.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P(Squeeze): %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color="rgba(180,180,200,0.5)", width=1, dash="dash"))
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(140,30,30,0.08)", line_width=0)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=11, color="#b0b0c8"),
        title=dict(
            text="Markov Regime: P(Supply Squeeze / Cartel Intervention)",
            font=dict(size=13, color="#e0e0f0"), x=0.01,
        ),
        yaxis=dict(range=[0, 1], tickformat=".1f",
                   showgrid=True, gridcolor=COLOR_GRID,
                   title="Probability"),
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID),
        height=280,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def chart_feature_importance(fi: pd.DataFrame) -> go.Figure:
    """Horizontal bar: top feature importances."""
    top = fi.head(15).iloc[::-1]  # reverse for horizontal bar (top = highest)

    CATEGORY_COLORS = {
        "usd_cny":   "#5577dd",
        "crude":     "#ff9933",
        "ore":       "#cc6633",
        "freight":   "#9966cc",
        "steel":     "#55aacc",
        "nickel":    "#55aa55",
        "cartel":    "#ff4444",
        "env_audit": "#ff5555",
        "tariff":    "#ff6655",
        "chn_elec":  "#ffcc44",
    }

    def get_color(name: str) -> str:
        for key, col in CATEGORY_COLORS.items():
            if key in name.lower():
                return col
        return "#8080a0"

    colors = [get_color(f) for f in top["feature"]]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0.4, color="white")),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        font=dict(family="monospace", size=10, color="#b0b0c8"),
        title=dict(
            text="Feature Importance (LightGBM)",
            font=dict(size=13, color="#e0e0f0"), x=0.01,
        ),
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title="Importance Score"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9)),
        height=420,
        margin=dict(l=200, r=20, t=50, b=40),
    )
    return fig


def chart_forecast_table(future: pd.DataFrame) -> go.Figure:
    """Future forecast as a styled table."""
    df = future.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["predicted_price"] = df["predicted_price"].round(3)
    df["regime_probability"] = df["regime_probability"].round(3)
    df["regime_state"] = df["regime_probability"].apply(
        lambda p: "🔴 Squeeze" if p > 0.5 else "🟢 Normal"
    )

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Date</b>", "<b>Forecast Price</b>",
                    "<b>Regime Prob</b>", "<b>Regime State</b>"],
            fill_color="#1a1a2e",
            align="center",
            font=dict(color="#c0c0e0", size=11, family="monospace"),
            line_color="rgba(255,255,255,0.1)",
            height=32,
        ),
        cells=dict(
            values=[df["date"], df["predicted_price"],
                    df["regime_probability"], df["regime_state"]],
            fill_color=[
                ["#0d0d1a", "#111120"] * len(df),
            ],
            align="center",
            font=dict(color="#a0a0c0", size=10, family="monospace"),
            line_color="rgba(255,255,255,0.06)",
            height=28,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
        height=min(40 * len(df) + 60, 500),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(meta: dict) -> dict:
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.divider()

        output_dir = st.text_input(
            "Output directory", value=OUTPUT_DIR,
            help="Folder containing pipeline output CSVs"
        )

        st.markdown("### 📅 Display Window")
        display_window = st.slider(
            "History to show (weeks)", min_value=52, max_value=520,
            value=260, step=26,
            help="How many weeks of historical data to display"
        )

        st.markdown("### 🔭 Forecast Horizon")
        horizon_options = {
            "4 weeks (1 month)":   4,
            "8 weeks (2 months)":  8,
            "12 weeks (3 months)": 12,
            "24 weeks (6 months)": 24,
        }
        horizon_label = st.selectbox(
            "Forecast horizon", list(horizon_options.keys()), index=2
        )
        horizon = horizon_options[horizon_label]

        st.markdown("### 🎨 Chart Options")
        show_lgb    = st.checkbox("Show LightGBM base line", value=False)
        show_regime = st.checkbox("Show regime shading",     value=True)
        show_fi     = st.checkbox("Show feature importance", value=True)

        st.divider()
        st.markdown("### 🧠 Model Info")
        if meta:
            st.markdown(f"- **Train period:** {meta.get('train_start','-')} → {meta.get('train_end','-')}")
            st.markdown(f"- **Observations:** {meta.get('n_observations','-'):,}")
            st.markdown(f"- **Features:** {meta.get('n_features','-')}")
            lstm_flag = "✅" if meta.get("lstm_enabled") else "❌"
            mrs_flag  = "✅" if meta.get("markov_enabled") else "❌"
            st.markdown(f"- **LSTM:** {lstm_flag}  **Markov:** {mrs_flag}")

        st.divider()
        st.caption(
            "⚠️ EMM proxy = MXI ETF (iShares Global Materials).  \n"
            "Scale to USD/tonne: `price × calibration_factor`  \n"
            "~1 index pt ≈ $25–35 USD/tonne (indicative)."
        )

    return dict(
        output_dir=output_dir,
        display_window=display_window,
        horizon=horizon,
        show_lgb=show_lgb,
        show_regime=show_regime,
        show_fi=show_fi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_metrics(hist: pd.DataFrame, future: pd.DataFrame, meta: dict):
    """Top KPI row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    last_actual = float(hist["actual"].iloc[-1])
    pred_col = "hybrid_prediction" if "hybrid_prediction" in hist.columns else "lgb_prediction"
    last_pred  = float(hist[pred_col].iloc[-1])
    next_pred  = float(future["predicted_price"].iloc[0]) if not future.empty else 0.0
    end_pred   = float(future["predicted_price"].iloc[-1]) if not future.empty else 0.0
    regime_now = float(hist["regime_probability"].iloc[-1]) if "regime_probability" in hist.columns else 0.0

    # MAPE
    actual = hist["actual"].values
    preds  = hist[pred_col].values
    mape   = np.mean(np.abs((actual - preds) / (actual + 1e-9))) * 100

    pct_chg = (next_pred - last_actual) / (last_actual + 1e-9) * 100
    trend   = "↑" if pct_chg > 0 else "↓"
    regime_label = "🔴 Squeeze" if regime_now > 0.5 else "🟢 Normal"

    col1.metric("Last Actual",       f"{last_actual:.3f}")
    col2.metric("Latest Prediction", f"{last_pred:.3f}",
                delta=f"{last_pred - last_actual:+.3f} vs actual")
    col3.metric("Next-Week Forecast", f"{next_pred:.3f}",
                delta=f"{pct_chg:+.1f}% {trend}")
    col4.metric(f"Week+{len(future)} Forecast", f"{end_pred:.3f}")
    col5.metric("In-Sample MAPE",    f"{mape:.2f}%",
                delta=regime_label, delta_color="off")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <h1 style='font-size:26px;margin:0;color:#e8e8f8;'>
            📈 EMM 97% Hybrid Price Forecasting
        </h1>
        <p style='color:#8888aa;font-size:13px;margin:4px 0 0;font-family:monospace;'>
            LightGBM + Markov Regime-Switching + LSTM Residuals + Sigmoid Damping
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load metadata first (for sidebar) ─────────────────────────────────────
    meta = {}
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        meta = load_metadata(meta_path)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    opts = render_sidebar(meta)
    out_dir = opts["output_dir"]

    # ── Check files ────────────────────────────────────────────────────────────
    ok, missing = files_exist(out_dir)
    if not ok:
        st.error(
            f"⛔ Missing output files in `{out_dir}`: **{', '.join(missing)}**\n\n"
            "Run the pipeline first:\n"
            "```bash\n"
            "python pipeline.py --horizon 12 --use-lstm --output-dir ./outputs\n"
            "```"
        )
        st.info(
            "💡 **Quick start (Google Colab)**\n"
            "1. Upload `pipeline.py` to Colab\n"
            "2. `!pip install lightgbm yfinance tensorflow scikit-learn statsmodels`\n"
            "3. `!python pipeline.py --horizon 12 --no-lstm` *(no-lstm for speed)*\n"
            "4. Download the `outputs/` folder\n"
            "5. Place alongside `app.py` and run `streamlit run app.py`"
        )
        st.stop()

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading forecast data..."):
        hist   = load_historical(os.path.join(out_dir, "historical_predictions.csv"))
        future_full = load_future(os.path.join(out_dir, "future_forecast.csv"))

    # Trim future to selected horizon
    future = future_full.iloc[:opts["horizon"]]

    fi_path = os.path.join(out_dir, "feature_importance.csv")
    fi = load_feature_importance(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

    if os.path.exists(meta_path):
        meta = load_metadata(meta_path)

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("---")
    render_metrics(hist, future, meta)
    st.markdown("---")

    # ── Main forecast chart ────────────────────────────────────────────────────
    st.plotly_chart(
        chart_main(hist, future,
                   show_lgb=opts["show_lgb"],
                   show_regime=opts["show_regime"],
                   display_window=opts["display_window"]),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ── Regime chart ───────────────────────────────────────────────────────────
    if "regime_probability" in hist.columns:
        st.plotly_chart(
            chart_regime(hist, opts["display_window"]),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Bottom row: Feature importance + Forecast table ────────────────────────
    col_fi, col_tbl = st.columns([1.1, 0.9])

    with col_fi:
        if opts["show_fi"] and not fi.empty:
            st.plotly_chart(
                chart_feature_importance(fi),
                use_container_width=True, config={"displayModeBar": False},
            )
        elif opts["show_fi"]:
            st.info("feature_importance.csv not found — re-run pipeline to generate it.")

    with col_tbl:
        st.markdown("##### 🔭 Future Forecast Detail")
        if not future.empty:
            st.plotly_chart(
                chart_forecast_table(future),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ── Raw data expanders ─────────────────────────────────────────────────────
    with st.expander("📂 Historical Predictions (raw data)"):
        st.dataframe(
            hist.style.format({
                "actual":              "{:.4f}",
                "lgb_prediction":      "{:.4f}",
                "hybrid_prediction":   "{:.4f}",
                "regime_probability":  "{:.3f}",
            }),
            use_container_width=True, height=320,
        )
        st.download_button(
            "⬇️ Download CSV",
            data=hist.to_csv(),
            file_name="historical_predictions.csv",
            mime="text/csv",
        )

    with st.expander("🔮 Future Forecast (raw data)"):
        st.dataframe(
            future_full.style.format({
                "predicted_price":    "{:.4f}",
                "regime_probability": "{:.3f}",
            }),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download CSV",
            data=future_full.to_csv(),
            file_name="future_forecast.csv",
            mime="text/csv",
        )

    with st.expander("🧾 Model Metadata"):
        st.json(meta)

    # ── Scaling note ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "**Proxy Scaling Note** — This model uses the iShares Global Materials ETF (MXI) "
        "as the EMM price proxy. To convert to USD/tonne: "
        "`EMM_USD_tonne = proxy_value × calibration_factor` where "
        "`calibration_factor = mean(actual_EMM_USD) / mean(MXI_index)` "
        "over a known overlap period. Contact your data provider (Asian Metal, "
        "Fastmarkets, Argus) for real EMM spot data to calibrate this factor."
    )

    st.markdown(
        "<div style='text-align:center;color:#444466;font-size:11px;padding-top:8px;'>"
        "EMM Hybrid Forecasting v2.0 | LightGBM + Markov + LSTM + Sigmoid Damping"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
