import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA REQUIREMENTS & AUTO-FETCH
# ==========================================
@st.cache_data
def fetch_data():
    """Fetches proxy data for EMM forecasting from public APIs."""
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch FRED Data (Target Proxy & Macro)
    # PCU331110331110 is the PPI for Iron, Steel, and Ferroalloy Manufacturing
    fred_series =
    df_fred = web.DataReader(fred_series, 'fred', start_date, end_date)
    df_fred = df_fred.resample('W').ffill()
    df_fred.rename(columns={'PCU331110331110': 'Target_EMM_Proxy', 'FEDFUNDS': 'Interest_Rates'}, inplace=True)

    # Fetch Yahoo Finance Data (Supply, Demand, Macro Proxies)
    tickers = {
        'CNY=X': 'USD_CNY',            # Exchange Rate
        'CL=F': 'Energy_Proxy',        # Crude Oil (Cost floor proxy)
        'BDRY': 'Freight_Proxy',       # Baltic Dry Index ETF Proxy
        'SLX': 'Steel_Demand_Proxy'    # Steel Industry ETF
    }
    
    df_yf = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Adj Close']
    df_yf.rename(columns=tickers, inplace=True)
    df_yf = df_yf.resample('W').ffill()

    # Merge datasets
    df = pd.merge(df_fred, df_yf, left_index=True, right_index=True, how='inner')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True) # Handle initial NaNs
    
    return df

# ==========================================
# 2. FEATURE ENGINEERING & SHOCK CREATION
# ==========================================
def engineer_features(df):
    """Creates lagged features, rolling stats, and exogenous shock indicators."""
    df_feat = df.copy()
    
    # Lag Features (Captures the 30-60 day transit delay)
    lags = [1, 2, 3] # Weekly lags (~1 week, ~1 month, ~2 months)
    cols_to_lag =
    
    for col in cols_to_lag:
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
            
    # Rolling Features (Volatility & Momentum)
    for col in cols_to_lag +:
        df_feat[f'{col}_MA_4'] = df_feat[col].rolling(window=4).mean()
        df_feat = df_feat[col].rolling(window=4).std()
    
    # Exogenous Shock / Event Indicators (Binary)
    df_feat = np.where((df_feat.index >= '2020-01-01') & (df_feat.index <= '2020-12-31'), 1, 0)
    df_feat = np.where((df_feat.index >= '2021-06-01') & (df_feat.index <= '2022-12-31'), 1, 0)
    
    # Regime Indicators (Cost push vs Demand destruction)
    energy_75th = df_feat['Energy_Proxy'].quantile(0.75)
    steel_25th = df_feat.quantile(0.25)
    df_feat = np.where((df_feat['Energy_Proxy'] > energy_75th) & 
                                             (df_feat < steel_25th), 1, 0)
    
    df_feat.dropna(inplace=True)
    return df_feat

# ==========================================
# 3. RANDOM FOREST (FEATURE SELECTION)
# ==========================================
def select_features_rf(df, target_col):
    """Uses Random Forest to identify the most critical lagged and macro drivers."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    
    st.write("**Top 10 Features selected by Random Forest:**")
    st.dataframe(importances.nlargest(10).to_frame(name="Importance Score"))
    
    return top_features

# ==========================================
# 4. ARIMA BASELINE (TREND EXTRACTION)
# ==========================================
def fit_arima_extract_residuals(train_y, test_y):
    """Fits ARIMA to extract linear trend and returns the non-linear residuals."""
    model = ARIMA(train_y, order=(5, 1, 0))
    model_fit = model.fit()
    
    train_predictions = model_fit.predict(start=train_y.index, end=train_y.index[-1])
    train_residuals = train_y - train_predictions
    
    test_predictions = model_fit.forecast(steps=len(test_y))
    test_predictions.index = test_y.index
    
    return train_predictions, train_residuals, test_predictions

# ==========================================
# 5. LSTM MODEL (NON-LINEAR SHOCKS)
# ==========================================
def build_and_train_lstm(X_train, y_train_res):
    """Trains LSTM on the residuals left by the ARIMA model."""
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape, 1, X_train_scaled.shape[1]))
    
    model = Sequential()),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train_res, epochs=50, batch_size=16, verbose=0)
    
    return model, scaler_X

# ==========================================
# 6 & 7. REGIME-AWARE ADJUSTMENTS
# ==========================================
def apply_soft_regime_constraints(base_pred, row):
    """
    Applies soft economic boundaries without hard clipping.
    Dampens runaway spikes if steel demand is weak, but allows cost-push spikes.
    """
    adjusted_pred = base_pred
    
    # Soft constraint 1: Demand Destruction Ceiling
    if row < row * 0.8:
        adjusted_pred = adjusted_pred * 0.95 
        
    # Soft constraint 2: Cost Floor Defense
    if row['Energy_Proxy'] > row['Energy_Proxy_mean'] * 1.2:
        adjusted_pred = adjusted_pred * 1.05
        
    return adjusted_pred

# ==========================================
# 8. PIPELINE EXECUTION & BACKTESTING
# ==========================================
def run_pipeline():
    st.title("EMM 97% Hybrid Price Forecasting Pipeline")
    st.markdown("An end-to-end Machine Learning model utilizing Random Forest, ARIMA, and LSTM networks to predict Electrolytic Manganese Metal proxies.")

    with st.spinner("Fetching and engineering data from public APIs..."):
        df = fetch_data()
        df = engineer_features(df)
    
    # Calculate expanding means to prevent forward-looking bias
    df = df.expanding().mean()
    df['Energy_Proxy_mean'] = df['Energy_Proxy'].expanding().mean()
    
    # Chronological Split: Train (2010-2020), Val/Test (2021-Present)
    train_df = df[df.index < '2021-01-01']
    test_df = df[df.index >= '2021-01-01']
    target = 'Target_EMM_Proxy'
    
    with st.spinner("Selecting optimal features using Random Forest..."):
        top_features = select_features_rf(train_df, target)
    
    with st.spinner("Training ARIMA for linear trends..."):
        train_pred_arima, train_residuals, test_pred_arima = fit_arima_extract_residuals(train_df[target], test_df[target])
    
    with st.spinner("Training LSTM network on non-linear residuals..."):
        X_train = train_df[top_features]
        lstm_model, scaler_X = build_and_train_lstm(X_train, train_residuals)
        
        # Predict Test Residuals
        X_test_scaled = scaler_X.transform(test_df[top_features])
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape, 1, X_test_scaled.shape[1]))
        test_pred_residuals = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    with st.spinner("Synthesizing predictions and applying regime constraints..."):
        final_predictions =
        for i in range(len(test_df)):
            raw_hybrid_pred = test_pred_arima.iloc[i] + test_pred_residuals[i]
            adj_pred = apply_soft_regime_constraints(raw_hybrid_pred, test_df.iloc[i])
            final_predictions.append(adj_pred)
            
        test_df['Hybrid_Prediction'] = final_predictions
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(test_df[target], test_df['Hybrid_Prediction']))
    mae = mean_absolute_error(test_df[target], test_df['Hybrid_Prediction'])
    mape = mean_absolute_percentage_error(test_df[target], test_df['Hybrid_Prediction'])
    
    st.subheader("Out-of-Sample Performance Metrics (2021-Present)")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MAPE", f"{mape:.2%}")
    
    # Visualization
    st.subheader("Hybrid Model vs. Actual Baseline")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(train_df.index, train_df[target], label='Historical Train (Actual)', color='grey', alpha=0.6)
    ax.plot(test_df.index, test_df[target], label='Out-of-Sample Actual', color='black', linewidth=2)
    ax.plot(test_df.index, test_df['Hybrid_Prediction'], label='Hybrid ARIMA-LSTM Forecast', color='red', linestyle='--')
    ax.axvline(pd.to_datetime('2021-01-01'), color='blue', linestyle=':', label='Train/Test Split')
    ax.set_title('ARIMA-LSTM-RF Pipeline: EMM Price Proxy')
    ax.set_ylabel('Ferroalloy PPI (Proxy)')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    run_pipeline()
