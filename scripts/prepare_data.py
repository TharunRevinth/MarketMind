"""
MarketMind - Prepare Stock Categories for Nifty 500 Universe
============================================================
Rebuilds data/indian_stock_categories.csv dynamically using the live Nifty 500
universe from basket_universe.py and cached price data in data/cache/.
Skips dead or invalid tickers.
Ensures Company_Name and Industry metadata are preserved alongside:
Ticker, Volatility, Max_Drawdown, Category, Risk_Level, Capital_Safety.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import basket_universe as bu

OUTPUT_CSV_PATH = os.path.join(REPO_ROOT, "data", "indian_stock_categories.csv")


def calculate_advanced_risk(data: pd.DataFrame) -> tuple:
    """Calculates 1-year annualized volatility and maximum drawdown from Close prices."""
    if data.empty or len(data) < 30:
        return np.nan, np.nan
    # Use most recent 252 trading days (1 year)
    recent = data.tail(252).copy()
    returns = recent["Close"].pct_change().dropna()
    if len(returns) < 20:
        return np.nan, np.nan
    volatility = float(returns.std() * np.sqrt(252))
    rolling_max = recent["Close"].cummax()
    drawdown = (recent["Close"] - rolling_max) / (rolling_max + 1e-9)
    max_drawdown = float(drawdown.min())
    return volatility, max_drawdown


def get_detailed_category(vol: float, mdd: float) -> tuple:
    """Classifies stock into standard risk categories based on historical volatility."""
    if np.isnan(vol) or vol <= 0:
        return "Growth (Moderate)", "Medium", "Medium"
    if vol < 0.20:
        return "Conservative (Blue Chip)", "Very Low", "High"
    elif vol < 0.32:
        return "Growth (Moderate)", "Medium", "Medium"
    elif vol < 0.50:
        return "Aggressive (Volatile)", "High", "Low"
    else:
        return "Speculative (High Risk)", "Extreme", "Very Low"


def main():
    print("=" * 80)
    print("REBUILDING INDIAN STOCK CATEGORIES FROM NIFTY 500 UNIVERSE")
    print("=" * 80)

    # 1. Fetch universe constituents
    symbols_df = bu.get_nifty500_symbols()
    print(f"Total constituents from universe: {len(symbols_df)}")

    # 2. Iterate through constituents and compute metrics from cached/fetched prices
    results = []
    skipped_count = 0

    for idx, row in symbols_df.iterrows():
        raw_symbol = str(row["Symbol"]).strip()
        y_symbol = str(row["Yahoo_Symbol"]).strip()
        company_name = str(row.get("Company Name", raw_symbol)).strip()
        industry = str(row.get("Industry", "General")).strip()

        # Check for known typo or dead tickers like TATARELI
        if "TATARELI" in raw_symbol or "DUMMY" in raw_symbol:
            skipped_count += 1
            continue

        # Load price data from cache or yfinance
        df = bu.fetch_stock_history(y_symbol, period="2y")
        if df.empty or len(df) < 30:
            skipped_count += 1
            continue

        vol, mdd = calculate_advanced_risk(df)
        if np.isnan(vol):
            skipped_count += 1
            continue

        cat, risk_lvl, capital_safety = get_detailed_category(vol, mdd)

        results.append({
            "Ticker": raw_symbol,
            "Company_Name": company_name,
            "Industry": industry,
            "Volatility": round(vol, 4),
            "Max_Drawdown": round(mdd, 4),
            "Category": cat,
            "Risk_Level": risk_lvl,
            "Capital_Safety": capital_safety
        })

    if not results:
        print("❌ Error: No valid stocks could be categorized.")
        sys.exit(1)

    df_out = pd.DataFrame(results)
    # Sort alphabetically by Ticker
    df_out.sort_values(by="Ticker", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n✅ Categorization Complete for {len(df_out)} valid Nifty 500 stocks (Skipped/Dead: {skipped_count}).")
    print(f"Saved to: {OUTPUT_CSV_PATH}")
    print(f"Category breakdown:\n{df_out['Category'].value_counts().to_string()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
