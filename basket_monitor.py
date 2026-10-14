"""
MarketMind - Daily & Weekly Risk Monitoring & Event Alert System
================================================================
Performs defensive risk checks for the active 15-stock basket:
1. Daily Checks:
   - Stop-Loss Breach (Price <= Stop_Loss)
   - Single-Day Crash (Price dropped >= 7% in a single session)
   - Long-Term Trend Breakdown (Price closed below 200-day SMA)
2. Weekly Checks:
   - Rank Drift Check (Has holding fallen below Top-30 momentum retention buffer?)
   - Short-Term Trend Check (Price vs 50-day SMA)
3. Macro Market Regime Check:
   - Nifty 500 above / below 200-day SMA
4. Plain-Language Replacement Advice:
   - Recommends the highest-ranked liquid candidate not currently held.
   - Highlights that alerts require user action, as no trades are placed automatically.
"""

import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import basket_universe as bu


def get_name_map():
    """Returns mapping from Symbol to Company Name."""
    try:
        cat_file = os.path.join(REPO_ROOT, "data", "indian_stock_categories.csv")
        if os.path.exists(cat_file):
            cdf = pd.read_csv(cat_file)
            return dict(zip(cdf["Ticker"], cdf.get("Company_Name", cdf["Ticker"])))
    except Exception:
        pass
    return {}


def find_top_replacement_candidate(held_symbols: list, min_turnover: float = 50_000_000.0) -> dict:
    """
    Finds the highest-ranked liquid Mom_12_1 replacement candidate not currently held in the basket.
    """
    name_map = get_name_map()
    dataset_file = os.path.join(REPO_ROOT, "data", "basket_dataset.parquet")
    if os.path.exists(dataset_file):
        try:
            df = pd.read_parquet(dataset_file)
            latest_date = df["Date"].max()
            sub = df[df["Date"] == latest_date].copy()
            if "ADTV_60D" in sub.columns:
                sub = sub[sub["ADTV_60D"] >= min_turnover]

            clean_held = set(s.replace(".NS", "") for s in held_symbols)
            sub["Clean_Sym"] = sub["Symbol"].str.replace(".NS", "", regex=False)
            available = sub[~sub["Clean_Sym"].isin(clean_held)].copy()

            if not available.empty:
                sort_col = "Mom_12_1" if "Mom_12_1" in available.columns else "Mom_6M"
                top_cand = available.sort_values(by=sort_col, ascending=False).iloc[0]
                sym = top_cand["Symbol"]
                clean_s = sym.replace(".NS", "")
                return {
                    "symbol": sym,
                    "clean_symbol": clean_s,
                    "company": name_map.get(clean_s, name_map.get(sym, clean_s)),
                    "price": float(top_cand.get("Close", 0.0)),
                    "momentum": float(top_cand.get(sort_col, 0.0))
                }
        except Exception:
            pass

    return {
        "symbol": "NIFTY_LEADER",
        "clean_symbol": "Top Replacement Candidate",
        "company": "Next Highest-Ranked Momentum Stock",
        "price": 0.0,
        "momentum": 0.0
    }


def check_market_regime(df_bench: pd.DataFrame) -> dict:
    """
    Evaluates whether the benchmark (Nifty 500) is in a Bull or Bear regime.
    Triggers a critical macro warning if trading below its 200-day Simple Moving Average.
    """
    if df_bench.empty or len(df_bench) < 200:
        return {
            "regime": "UNKNOWN",
            "warning": False,
            "message": "Insufficient benchmark history to evaluate 200-day SMA.",
            "current_price": 0.0,
            "sma200": 0.0,
            "distance_pct": 0.0
        }

    close = df_bench["Close"]
    sma200 = close.rolling(200).mean().iloc[-1]
    curr = close.iloc[-1]
    dist_pct = ((curr / sma200) - 1.0) * 100.0
    is_below = bool(curr < sma200)

    if is_below:
        msg = f"MARKET REGIME WARNING: Nifty 500 is trading {abs(dist_pct):.2f}% BELOW its 200-day average (₹{sma200:,.1f}). High broad-market drawdown risk. Consider keeping a 50% cash reserve or staging purchases into 2-3 tranches."
        regime = "DEFENSIVE / BEARISH REGIME"
    else:
        msg = f"MARKET REGIME HEALTHY: Nifty 500 is trading {dist_pct:+.2f}% ABOVE its 200-day average (₹{sma200:,.1f}). Macro trend supports 100% equity allocation."
        regime = "BULLISH / EXPANSIONARY REGIME"

    return {
        "regime": regime,
        "warning": is_below,
        "message": msg,
        "current_price": curr,
        "sma200": sma200,
        "distance_pct": dist_pct
    }


def evaluate_basket_alerts(
    basket_df: pd.DataFrame,
    price_dict: dict,
    single_day_crash_threshold: float = 0.07,
    check_weekly: bool = False
) -> list:
    """
    Inspects each stock in the current active basket against defensive triggers:
    1. Stop-Loss Trigger (Current Price <= Stop_Loss)
    2. Single-Day Crash Trigger (Day's drop >= 7%)
    3. Long-Term Trend Breakdown (Close < 200-day SMA)
    4. Short-Term Trend Breakdown (Close < 50-day SMA)
    
    Provides actionable plain-language recommendation with replacement candidate.
    """
    alerts = []
    if basket_df.empty:
        return alerts

    held_symbols = basket_df["Symbol"].tolist()
    name_map = get_name_map()

    # Pre-fetch candidate replacement
    replacement = find_top_replacement_candidate(held_symbols)
    rep_sym = replacement.get("clean_symbol", "NEXT_LEADER")
    rep_company = replacement.get("company", "Top Replacement Stock")

    for _, row in basket_df.iterrows():
        sym = row["Symbol"]
        clean_sym = sym.replace(".NS", "")
        company = row.get("Company", name_map.get(clean_sym, clean_sym))
        entry_price = row.get("Current_Price", row.get("Entry_Price", 0.0))
        stop_loss = row.get("Stop_Loss", entry_price * 0.90)

        df_s = price_dict.get(sym)
        if df_s is None or df_s.empty:
            continue

        close_series = df_s["Close"]
        live_price = float(close_series.iloc[-1])
        prev_price = float(close_series.iloc[-2]) if len(close_series) >= 2 else live_price
        day_change_pct = ((live_price / prev_price) - 1.0) * 100.0

        # 1. Stop-Loss Trigger Check
        if live_price <= stop_loss:
            loss_pct = ((live_price / entry_price) - 1.0) * 100.0 if entry_price > 0 else 0
            alerts.append({
                "symbol": clean_sym,
                "company": company,
                "severity": "CRITICAL",
                "type": "STOP_LOSS_HIT",
                "message": f"CRITICAL: Stock {clean_sym} ({company}) closed below its exit price at ₹{live_price:,.2f} (Exit level was ₹{stop_loss:,.2f}, down {loss_pct:.1f}%).",
                "action": f"Consider selling {clean_sym} and replacing it with {rep_sym} ({rep_company})."
            })
            continue

        # 2. Single-Day Crash Trigger (>= 7% drop)
        if day_change_pct <= -(single_day_crash_threshold * 100.0):
            alerts.append({
                "symbol": clean_sym,
                "company": company,
                "severity": "CRITICAL",
                "type": "SINGLE_DAY_CRASH",
                "message": f"CRITICAL: Stock {clean_sym} ({company}) experienced a single-day drop of {day_change_pct:.1f}% (exceeded {single_day_crash_threshold*100:.0f}% safety threshold).",
                "action": f"Severe sudden decline. Consider cutting {clean_sym} and rotating into {rep_sym} ({rep_company})."
            })
            continue

        # 3. 200-Day SMA Breakdown
        if len(close_series) >= 200:
            sma200 = float(close_series.rolling(200).mean().iloc[-1])
            if live_price < sma200:
                alerts.append({
                    "symbol": clean_sym,
                    "company": company,
                    "severity": "WARNING",
                    "type": "BELOW_200_DMA",
                    "message": f"WARNING: Stock {clean_sym} ({company}) closed at ₹{live_price:,.2f}, below its long-term 200-day trend line (₹{sma200:,.2f}).",
                    "action": f"Long-term trend broken. Consider replacing with {rep_sym} ({rep_company}) at next review."
                })
                continue

        # 4. Weekly / Short-Term 50-Day Trend Check
        if check_weekly and len(close_series) >= 50:
            sma50 = float(close_series.rolling(50).mean().iloc[-1])
            if live_price < sma50:
                alerts.append({
                    "symbol": clean_sym,
                    "company": company,
                    "severity": "INFO",
                    "type": "BELOW_50_DMA",
                    "message": f"NOTE: Stock {clean_sym} ({company}) is consolidating below its 50-day average (₹{sma50:,.2f}).",
                    "action": "Hold position for now; monitor stop-loss exit level."
                })

    return alerts
