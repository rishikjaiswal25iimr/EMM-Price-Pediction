import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA FETCH
# ==========================================
@st.cache_data
def fetch_data():
    start_date = "2010-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    tickers = {
        'CNY=X': 'USD_CNY',
        'CL=F': 'Energy',
        'BDRY': 'Freight',
        'SLX': 'Steel',
        '^GSPC': 'Market'
    }

    raw = yf.download(list(tickers.keys()), start=start_date, end=end_date, auto_adjust=True)

    # Robust extraction
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Close' in raw.columns.levels[0]:
            df = raw['Close']
        else:
            df = raw.xs('Close', level=1, axis=1)
    else:
        df = raw[['Close']]

    df.columns = [tickers.get(col, col) for col in df.columns]

    df = df.resample('W').ffill()

    # Better proxy (normalized weighted)
    df_norm = (df - df.mean()) / df.std()
    df['EMM_Price_Proxy'] = (
        0.35 * df_norm['Energy'] +
        0.25 * df_norm['Steel'] +
        0.20 * df_norm['Freight'] +
        0.20 * df_norm['Market']
    )

    df = df.ffill().bfill()
    return df


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()

    # Returns & volatility
    for col in df.columns:
        df[f"{col}_ret"] = df[col].pct_change()
        df[f"{col}_vol"] = df[col].rolling(4).std()

    # Shock indicators
    df['covid_dummy'] = ((df.index >= '2020-01-01') & (df.index <= '2020-12-31')).astype(int)
    df['energy_crisis_dummy'] = ((df.index >= '2021-06-01') & (df.index <= '2022-12-31')).astype(int)

    df.dropna(inplace=True)
    return df


# ==========================================
# 3. RANDOM FOREST (Feature importance)
# ==========================================
def run_random_forest(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.nlargest(10)


# ==========================================
# 4. ARIMAX MODEL
# ==========================================
def run_arimax(train_y, test_y, train_exog, test_exog):
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=(2, 1, 2),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.predict(
        start=test_y.index[0],
        end=test_y.index[-1],
        exog=test_exog
    )

    return forecast


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def run_pipeline():
    st.title("EMM Price Forecast (ARIMAX Model - Version 2)")
    st.write("Model uses macro + commodity drivers for realistic forecasting")

    df = fetch_data()
    df = engineer_features(df)

    target = "EMM_Price_Proxy"

    # Exogenous variables
    exog_vars = ['Energy', 'Freight', 'Steel', 'USD_CNY']

    train = df[df.index < '2021-01-01']
    test = df[df.index >= '2021-01-01']

    # Random Forest importance
    importance = run_random_forest(train, target)
    st.subheader("Top Drivers (Random Forest)")
    st.dataframe(importance.to_frame("Importance"))

    # ARIMAX
    forecast = run_arimax(
        train[target],
        test[target],
        train[exog_vars],
        test[exog_vars]
    )

    test['Prediction'] = forecast

    # Metrics
    rmse = np.sqrt(mean_squared_error(test[target], test['Prediction']))
    mae = mean_absolute_error(test[target], test['Prediction'])
    mape = mean_absolute_percentage_error(test[target], test['Prediction'])

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MAPE", f"{mape:.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train[target], label="Train", alpha=0.6)
    ax.plot(test.index, test[target], label="Actual", color="black")
    ax.plot(test.index, test['Prediction'], label="ARIMAX Prediction", linestyle="--")

    ax.legend()
    ax.set_title("Actual vs ARIMAX Prediction")

    st.pyplot(fig)


if __name__ == "__main__":
    run_pipeline()
