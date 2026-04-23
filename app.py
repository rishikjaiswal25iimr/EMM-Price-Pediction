import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA FETCH (NO pandas_datareader)
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

    raw = yf.download(
        list(tickers.keys()),
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    # 🔥 Handle ALL cases safely
    if isinstance(raw.columns, pd.MultiIndex):
        # Case 1: ('Close', ticker)
        if 'Close' in raw.columns.levels[0]:
            df = raw['Close']
        # Case 2: (ticker, 'Close')
        elif 'Close' in raw.columns.levels[1]:
            df = raw.xs('Close', level=1, axis=1)
        else:
            # fallback
            df = raw.copy()
    else:
        # Single ticker case
        df = raw[['Close']].copy()

    # Rename columns safely
    df.columns = [tickers.get(col, col) for col in df.columns]

    # Weekly frequency
    df = df.resample('W').ffill()

    # Create proxy
    df['EMM_Price_Proxy'] = (
        0.4 * df['Energy'] +
        0.2 * df['Freight'] +
        0.2 * df['Steel'] +
        0.2 * df['Market']
    )

    df = df.ffill().bfill()

    return df
# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()

    cols = df.columns

    # Lag features
    for col in cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_4"] = df[col].shift(4)

    # Rolling features
    for col in cols:
        df[f"{col}_ma_4"] = df[col].rolling(4).mean()
        df[f"{col}_std_4"] = df[col].rolling(4).std()

    # Shock indicators
    df['covid_dummy'] = ((df.index >= '2020-01-01') & (df.index <= '2020-12-31')).astype(int)
    df['energy_crisis_dummy'] = ((df.index >= '2021-06-01') & (df.index <= '2022-12-31')).astype(int)

    df.dropna(inplace=True)
    return df


# ==========================================
# 3. RANDOM FOREST
# ==========================================
def run_random_forest(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10)

    return top_features


# ==========================================
# 4. ARIMA MODEL
# ==========================================
def run_arima(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index

    return forecast


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def run_pipeline():
    st.title("EMM Price Forecast (RF + ARIMA Model)")
    st.write("Forecasting proxy EMM prices using macro & commodity drivers")

    df = fetch_data()
    df = engineer_features(df)

    target = "EMM_Price_Proxy"

    train = df[df.index < '2021-01-01']
    test = df[df.index >= '2021-01-01']

    # Random Forest
    top_features = run_random_forest(train, target)

    st.subheader("Top Feature Importance")
    st.dataframe(top_features.to_frame("Importance"))

    # ARIMA Forecast
    forecast = run_arima(train[target], test[target])
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
    ax.plot(test.index, test['Prediction'], label="Prediction", linestyle="--")

    ax.legend()
    ax.set_title("Actual vs Predicted EMM Price Proxy")

    st.pyplot(fig)


if __name__ == "__main__":
    run_pipeline()
