import yfinance as yf
import pandas as pd
import numpy as np
import os

# Comprehensive list of NIFTY 100 + Key Midcaps
INDIAN_TICKERS = [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "ADANIENT.NS", "TATASTEEL.NS", "M&M.NS",
    "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS", "ONGC.NS", "ADANIPORTS.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "LTIM.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "NESTLEIND.NS", "TECHM.NS", "SBILIFE.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HDFCLIFE.NS", "ADANIPOWER.NS",
    "INDUSINDBK.NS", "TATARELI.NS", "TATACONSUM.NS", "CIPLA.NS", "BPCL.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    # Next 50 / Large Midcaps
    "ZOMATO.NS", "HAL.NS", "BEL.NS", "JIOFIN.NS", "PFC.NS", "RECLTD.NS", "IRFC.NS", "VBL.NS", "TRENT.NS", "DLF.NS",
    "CHOLAFIN.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "TVSMOTOR.NS", "PNB.NS", "CANBK.NS", "IDFCFIRSTB.NS", "YESBANK.NS", "FEDERALBNK.NS",
    "AUBNK.NS", "BANDHANBNK.NS", "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "PAGEIND.NS", "TATAELXSI.NS", "KPITTECH.NS", "PERSISTENT.NS", "MPHASIS.NS",
    "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS", "SHREECEM.NS", "POLYCAB.NS", "KEI.NS", "CUMMINSIND.NS", "BHEL.NS", "RVNL.NS", "IRCON.NS",
    "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "DELHIVERY.NS", "CARADE.NS", "IDEA.NS", "GMRINFRA.NS", "SUZLON.NS", "NHPC.NS", "SJVN.NS"
]

def fetch_historical_data(ticker):
    print(f"Analyzing {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            return None
        return data
    except:
        return None

def calculate_advanced_risk(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    rolling_max = data['Close'].cummax()
    drawdown = (data['Close'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return volatility, max_drawdown

def get_detailed_category(vol, mdd):
    if vol < 0.18:
        return "Conservative (Blue Chip)", "Very Low", "High"
    elif vol < 0.30:
        return "Growth (Moderate)", "Medium", "Medium"
    elif vol < 0.50:
        return "Aggressive (Volatile)", "High", "Low"
    else:
        return "Speculative (High Risk)", "Extreme", "Very Low"

def main():
    results = []
    for ticker in INDIAN_TICKERS:
        data = fetch_historical_data(ticker)
        if data is not None:
            vol, mdd = calculate_advanced_risk(data)
            cat, risk_lvl, capital_safety = get_detailed_category(vol, mdd)
            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Volatility": round(vol, 4),
                "Max_Drawdown": round(mdd, 4),
                "Category": cat,
                "Risk_Level": risk_lvl,
                "Capital_Safety": capital_safety
            })
    
    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/indian_stock_categories.csv", index=False)
    print(f"\n✅ Categorization Complete for {len(results)} Indian Stocks.")

if __name__ == "__main__":
    main()
