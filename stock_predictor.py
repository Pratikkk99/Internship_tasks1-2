import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Market Price Predictor")
st.markdown("Predict closing prices using Machine Learning")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'INTC', 'IBM']
stock_choice = st.selectbox(" Choose a stock:", tickers)

if not os.path.exists("data"):
    os.makedirs("data")

@st.cache_data
def load_data(ticker):
    try:
        # Try downloading from Yahoo Finance
        df = yf.download(ticker, period='5y')
        if df.empty:
            raise ValueError("Empty online data")
        df.dropna(inplace=True)
        df.to_csv(f"data/{ticker}.csv")  # Save backup
    except:
        st.warning(f"⚠️ Could not fetch online data for {ticker}. Trying local backup...")
        try:
            df = pd.read_csv(f"data/{ticker}.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except FileNotFoundError:
            st.error(f"❌ No local backup found. Please connect to the internet to download stock data.")
            st.stop()
    return df

data = load_data(stock_choice)

st.subheader("📊 Latest Stock Data (5 Years)")
st.dataframe(data.tail(), use_container_width=True)

if len(data) < 10:
    st.error("Not enough data to train/test models.")
    st.stop()

features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

lr_rmse, lr_r2 = evaluate(y_test, lr_preds)
rf_rmse, rf_r2 = evaluate(y_test, rf_preds)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Linear Regression RMSE", f"{lr_rmse:.2f}")
    st.metric("Linear Regression R²", f"{lr_r2:.2f}")
with col2:
    st.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
    st.metric("Random Forest R²", f"{rf_r2:.2f}")

st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label="Actual Prices", marker='o')
ax.plot(lr_preds, label="Linear Regression", linestyle='--')
ax.plot(rf_preds, label="Random Forest", linestyle='--')
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("Stock Price (USD)")
ax.set_title(f"{stock_choice} - Closing Price Predictions")
ax.legend()
st.pyplot(fig)
