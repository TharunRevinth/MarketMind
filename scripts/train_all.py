import yfinance as yf
import pandas as pd
import numpy as np
import os
from scripts.train_model import train_and_save

# NIFTY 50 and Top NSE Stocks
INDIAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "BAJFINANCE.NS", "SUNPHARMA.NS", "ADANIENT.NS", "TATASTEEL.NS", "M&M.NS",
    "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS", "ONGC.NS", "ADANIPORTS.NS",
    "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS", "ULTRACEMCO.NS", "PAYTM.NS", "NYKAA.NS"
]

def main():
    print(f"🚀 Starting Batch Training for {len(INDIAN_TICKERS)} stocks...")
    
    os.makedirs("models", exist_ok=True)
    
    for ticker in INDIAN_TICKERS:
        try:
            print(f"\n--- Training {ticker} ---")
            train_and_save(ticker)
        except Exception as e:
            print(f"❌ Failed to train {ticker}: {e}")

    print("\n✅ All models trained and saved in /models folder.")

if __name__ == "__main__":
    main()
