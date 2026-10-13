"""
MarketMind - Daily Monitoring & Risk Alert System
=================================================
Performs daily safety monitoring for the active 15-stock basket:
- Stop-loss trigger detection (Price <= Entry - 2*ATR)
- Technical moving average breakdowns (Below 50-day / 200-day SMA)
- Macro market regime alert (Nifty 500 below 200-day SMA)
No daily re-prediction is performed; strictly defensive risk controls.
"""

import pandas as pd
import numpy as np


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
        msg = f"MARKET REGIME WARNING: Nifty 500 is trading {abs(dist_pct):.2f}% BELOW its 200-day SMA (₹{sma200:,.1f}). High broad-market drawdown risk. Consider defensive capital allocation or tighter trailing stops."
        regime = "DEFENSIVE / BEARISH REGIME ⚠️"
    else:
        msg = f"MARKET REGIME HEALTHY: Nifty 500 is trading {dist_pct:+.2f}% ABOVE its 200-day SMA (₹{sma200:,.1f}). Macro trend supports equity baskets."
        regime = "BULLISH / EXPANSIONARY REGIME 🚀"

    return {
        "regime": regime,
        "warning": is_below,
        "message": msg,
        "current_price": curr,
        "sma200": sma200,
        "distance_pct": dist_pct
    }


def evaluate_basket_alerts(basket_df: pd.DataFrame, price_dict: dict) -> list:
    """
    Inspects each stock in the current active basket against defensive triggers:
    1. Stop-Loss Trigger (Current <= Stop_Loss)
    2. Breakdown below 50-day SMA
    3. Breakdown below 200-day SMA
    
    Returns:
        list of alert dicts with keys: ['symbol', 'severity', 'type', 'message', 'action']
    """
    alerts = []

    if basket_df.empty:
        return alerts

    for _, row in basket_df.iterrows():
        sym = row["Symbol"]
        entry_price = row.get("Current_Price", row.get("Entry_Price", 0.0))
        stop_loss = row.get("Stop_Loss", entry_price * 0.95)

        df_s = price_dict.get(sym)
        if df_s is None or df_s.empty:
            continue

        close_series = df_s["Close"]
        live_price = close_series.iloc[-1]

        # 1. Stop-Loss Trigger Check
        if live_price <= stop_loss:
            loss_pct = ((live_price / entry_price) - 1.0) * 100.0 if entry_price > 0 else 0
            alerts.append({
                "symbol": sym,
                "severity": "CRITICAL",
                "type": "STOP_LOSS_HIT",
                "message": f"🚨 {sym}: Stop-Loss Hit at ₹{live_price:,.2f} (Stop Level: ₹{stop_loss:,.2f}, P&L: {loss_pct:.2f}%).",
                "action": "Trigger defensive exit / cut position to protect capital."
            })

        # 2. Moving Average Breakdowns
        if len(close_series) >= 50:
            sma50 = close_series.rolling(50).mean().iloc[-1]
            if live_price < sma50:
                alerts.append({
                    "symbol": sym,
                    "severity": "WARNING",
                    "type": "BELOW_50_DMA",
                    "message": f"⚠️ {sym}: Price (₹{live_price:,.2f}) slipped below 50-day SMA (₹{sma50:,.2f}).",
                    "action": "Monitor closely for potential trend exhaustion."
                })

        if len(close_series) >= 200:
            sma200 = close_series.rolling(200).mean().iloc[-1]
            if live_price < sma200:
                alerts.append({
                    "symbol": sym,
                    "severity": "WARNING",
                    "type": "BELOW_200_DMA",
                    "message": f"⚠️ {sym}: Price (₹{live_price:,.2f}) slipped below long-term 200-day SMA (₹{sma200:,.2f}).",
                    "action": "Macro support violated. Reconsider holding through monthly rebalance."
                })

    return alerts
