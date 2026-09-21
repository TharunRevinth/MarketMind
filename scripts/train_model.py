import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def prepare_ml_data(ticker="TSLA", period="60d", interval="5m"):
    print(f"Fetching intraday data for {ticker}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    if data.empty:
        return None
    
    # Feature Engineering
    data['Return'] = data['Close'].pct_change()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    
    # RSI calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Target: 1 if next candle close > current close, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop NaNs
    data = data.dropna()
    
    features = ['Return', 'MA5', 'MA10', 'RSI']
    X = data[features]
    y = data['Target']
    
    return X, y

def train_and_save(ticker="TSLA"):
    X, y = prepare_ml_data(ticker)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy for {ticker}: {accuracy:.2%}")
    
    os.makedirs("models", exist_ok=True)
    with open(f"models/{ticker}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"models/{ticker}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model and scaler saved to models/")

if __name__ == "__main__":
    train_and_save("TSLA")
