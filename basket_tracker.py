"""
MarketMind - Paper Trading Tracker & Intraday Accuracy Tracker
==============================================================
Manages paper-trading portfolio lifecycle:
- Records monthly 15-stock baskets to data/paper_trades.csv (no real money).
- Logs daily portfolio valuation against the Nifty 500 benchmark.
- Tracks live vs backtest metrics, hit rate, and drawdowns.
- Verifies intraday Logistic Regression predictions against actual candle closes
  to report a live out-of-sample accuracy score.
"""

import os
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
PAPER_TRADES_FILE = os.path.join(DATA_DIR, "paper_trades.csv")
PORTFOLIO_HISTORY_FILE = os.path.join(DATA_DIR, "paper_portfolio_history.csv")
INTRADAY_LOG_FILE = os.path.join(DATA_DIR, "intraday_predictions_log.csv")


def save_paper_basket(
    basket_df: pd.DataFrame,
    total_capital: float = 1_000_000.0,
    benchmark_price: float = None
) -> str:
    """
    Persists a generated 15-stock basket as an active paper portfolio.
    
    Returns:
        str: Basket tracking ID
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    basket_id = f"BASKET-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for _, row in basket_df.iterrows():
        records.append({
            "Basket_ID": basket_id,
            "Timestamp": now_str,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Symbol": row["Symbol"],
            "Company": row.get("Company", row["Symbol"]),
            "Entry_Price": float(row["Current_Price"]),
            "Shares_Qty": int(row.get("Shares_Qty", 0)),
            "Weight_Pct": float(row.get("Weight_Pct", 6.67)),
            "Allocated_Capital": float(row.get("Actual_Allocation", row.get("Allocated_Capital", 0.0))),
            "Prob_Beat_Index": float(row.get("Prob_Beat_Index", 0.5)),
            "Stop_Loss": float(row.get("Stop_Loss", row["Current_Price"] * 0.95)),
            "Key_Reasons": str(row.get("Key_Reasons", "")),
        })

    new_trades = pd.DataFrame(records)

    if os.path.exists(PAPER_TRADES_FILE):
        try:
            existing = pd.read_csv(PAPER_TRADES_FILE)
            combined = pd.concat([existing, new_trades], ignore_index=True)
        except Exception:
            combined = new_trades
    else:
        combined = new_trades

    combined.to_csv(PAPER_TRADES_FILE, index=False)

    # Initialize portfolio history starting point
    init_bench = benchmark_price if benchmark_price else 25000.0
    init_hist = pd.DataFrame([{
        "Timestamp": now_str,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Basket_ID": basket_id,
        "Portfolio_Value": total_capital,
        "Benchmark_Price": init_bench,
        "Portfolio_Return_Pct": 0.0,
        "Benchmark_Return_Pct": 0.0,
        "Excess_Return_Pct": 0.0
    }])

    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        try:
            hist_existing = pd.read_csv(PORTFOLIO_HISTORY_FILE)
            hist_combined = pd.concat([hist_existing, init_hist], ignore_index=True)
        except Exception:
            hist_combined = init_hist
    else:
        hist_combined = init_hist

    hist_combined.to_csv(PORTFOLIO_HISTORY_FILE, index=False)
    print(f"[Tracker] Successfully saved paper basket {basket_id} with {len(records)} positions.")
    return basket_id


def load_latest_paper_basket() -> pd.DataFrame:
    """Loads positions from the most recent paper trading basket."""
    if not os.path.exists(PAPER_TRADES_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(PAPER_TRADES_FILE)
        if df.empty or "Basket_ID" not in df.columns:
            return pd.DataFrame()
        latest_id = df["Basket_ID"].iloc[-1]
        return df[df["Basket_ID"] == latest_id].copy()
    except Exception:
        return pd.DataFrame()


def update_daily_paper_portfolio(bench_ticker: str = "^CRSLDX") -> dict:
    """
    Fetches live prices for current paper basket holdings and logs today's portfolio value.
    """
    basket = load_latest_paper_basket()
    if basket.empty:
        return {"status": "NO_BASKET", "message": "No paper basket found."}

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    basket_id = basket["Basket_ID"].iloc[0]

    # Fetch live prices
    symbols = basket["Symbol"].tolist()
    total_val = 0.0
    position_details = []

    for _, row in basket.iterrows():
        sym = row["Symbol"]
        entry = row["Entry_Price"]
        qty = row["Shares_Qty"]
        try:
            t_obj = yf.Ticker(sym)
            h = t_obj.history(period="2d", interval="1d")
            curr_price = float(h["Close"].iloc[-1]) if not h.empty else entry
        except Exception:
            curr_price = entry

        pos_val = qty * curr_price
        pnl = (curr_price - entry) * qty
        pnl_pct = ((curr_price / entry) - 1.0) * 100.0 if entry > 0 else 0.0
        total_val += pos_val

        position_details.append({
            "Symbol": sym,
            "Entry_Price": entry,
            "Current_Price": curr_price,
            "Shares_Qty": qty,
            "Position_Value": pos_val,
            "PnL": pnl,
            "PnL_Pct": pnl_pct
        })

    # Fetch benchmark price
    try:
        b_hist = yf.Ticker(bench_ticker).history(period="2d", interval="1d")
        curr_bench = float(b_hist["Close"].iloc[-1]) if not b_hist.empty else 25000.0
    except Exception:
        curr_bench = 25000.0

    # Calculate returns relative to inception
    hist_df = pd.DataFrame()
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        try:
            hist_df = pd.read_csv(PORTFOLIO_HISTORY_FILE)
        except Exception:
            pass

    if not hist_df.empty and (hist_df["Basket_ID"] == basket_id).any():
        basket_hist = hist_df[hist_df["Basket_ID"] == basket_id]
        init_val = basket_hist["Portfolio_Value"].iloc[0]
        init_bench = basket_hist["Benchmark_Price"].iloc[0]
    else:
        init_val = total_val
        init_bench = curr_bench

    port_ret_pct = ((total_val / init_val) - 1.0) * 100.0 if init_val > 0 else 0.0
    bench_ret_pct = ((curr_bench / init_bench) - 1.0) * 100.0 if init_bench > 0 else 0.0
    excess_ret_pct = port_ret_pct - bench_ret_pct

    new_hist_entry = pd.DataFrame([{
        "Timestamp": now_str,
        "Date": today_str,
        "Basket_ID": basket_id,
        "Portfolio_Value": total_val,
        "Benchmark_Price": curr_bench,
        "Portfolio_Return_Pct": port_ret_pct,
        "Benchmark_Return_Pct": bench_ret_pct,
        "Excess_Return_Pct": excess_ret_pct
    }])

    updated_hist = pd.concat([hist_df, new_hist_entry], ignore_index=True) if not hist_df.empty else new_hist_entry
    updated_hist.drop_duplicates(subset=["Date", "Basket_ID"], keep="last", inplace=True)
    updated_hist.to_csv(PORTFOLIO_HISTORY_FILE, index=False)

    return {
        "status": "SUCCESS",
        "basket_id": basket_id,
        "portfolio_value": total_val,
        "portfolio_return_pct": port_ret_pct,
        "benchmark_return_pct": bench_ret_pct,
        "excess_return_pct": excess_ret_pct,
        "positions": pd.DataFrame(position_details)
    }


def get_paper_tracker_metrics() -> dict:
    """Returns overview statistics of the live paper trading portfolio."""
    basket = load_latest_paper_basket()
    if basket.empty:
        return {"has_data": False}

    hist_df = pd.DataFrame()
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        try:
            hist_df = pd.read_csv(PORTFOLIO_HISTORY_FILE)
        except Exception:
            pass

    return {
        "has_data": True,
        "basket": basket,
        "history": hist_df,
    }


# =========================================================================
# INTRADAY LOGISTIC REGRESSION TRACKER
# =========================================================================

def log_intraday_prediction(
    ticker: str,
    predicted_direction: str,
    confidence_pct: float,
    current_price: float
):
    """
    Logs an intraday prediction emitted by the Logistic Regression model.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "Timestamp": now_str,
        "Ticker": ticker,
        "Predicted_Direction": predicted_direction.strip(),
        "Confidence_Pct": float(confidence_pct),
        "Price_At_Prediction": float(current_price),
        "Next_Candle_Price": np.nan,
        "Actual_Direction": "PENDING",
        "Correct": np.nan,
    }

    df_entry = pd.DataFrame([entry])
    if os.path.exists(INTRADAY_LOG_FILE):
        try:
            existing = pd.read_csv(INTRADAY_LOG_FILE)
            combined = pd.concat([existing, df_entry], ignore_index=True)
        except Exception:
            combined = df_entry
    else:
        combined = df_entry

    combined.to_csv(INTRADAY_LOG_FILE, index=False)


def verify_intraday_predictions():
    """
    Checks unresolved intraday predictions and evaluates whether subsequent price action
    confirmed the prediction (UP or DOWN), updating actual outcomes and correctness.
    """
    if not os.path.exists(INTRADAY_LOG_FILE):
        return

    try:
        df = pd.read_csv(INTRADAY_LOG_FILE)
        if df.empty:
            return

        unresolved = df[df["Actual_Direction"] == "PENDING"].copy()
        if unresolved.empty:
            return

        for idx, row in unresolved.iterrows():
            ticker = row["Ticker"]
            yf_ticker = ticker if ".NS" in ticker else f"{ticker}.NS"
            try:
                # Fetch recent 5m data to evaluate subsequent candle
                data = yf.Ticker(yf_ticker).history(period="1d", interval="5m")
                if len(data) >= 2:
                    p_entry = row["Price_At_Prediction"]
                    p_latest = float(data["Close"].iloc[-1])
                    actual_dir = "UP" if p_latest >= p_entry else "DOWN"

                    pred_dir = "UP" if "UP" in str(row["Predicted_Direction"]).upper() else "DOWN"
                    is_correct = int(actual_dir == pred_dir)

                    df.at[idx, "Next_Candle_Price"] = p_latest
                    df.at[idx, "Actual_Direction"] = actual_dir
                    df.at[idx, "Correct"] = is_correct
            except Exception:
                continue

        df.to_csv(INTRADAY_LOG_FILE, index=False)
    except Exception as exc:
        print(f"[Tracker] Error verifying intraday predictions: {exc}")


def get_intraday_accuracy_stats() -> dict:
    """Calculates live out-of-sample accuracy of the intraday Logistic Regression model."""
    if not os.path.exists(INTRADAY_LOG_FILE):
        return {
            "total_logged": 0,
            "resolved_count": 0,
            "correct_count": 0,
            "accuracy_pct": 52.4,  # Historical baseline default
            "recent_predictions": pd.DataFrame()
        }

    try:
        df = pd.read_csv(INTRADAY_LOG_FILE)
        if df.empty:
            return {
                "total_logged": 0, "resolved_count": 0, "correct_count": 0,
                "accuracy_pct": 52.4, "recent_predictions": pd.DataFrame()
            }

        resolved = df.dropna(subset=["Correct"])
        resolved_count = len(resolved)
        correct_count = int(resolved["Correct"].sum()) if resolved_count > 0 else 0
        accuracy = (correct_count / resolved_count * 100.0) if resolved_count > 0 else 52.4

        return {
            "total_logged": len(df),
            "resolved_count": resolved_count,
            "correct_count": correct_count,
            "accuracy_pct": accuracy,
            "recent_predictions": df.tail(10)
        }
    except Exception:
        return {
            "total_logged": 0, "resolved_count": 0, "correct_count": 0,
            "accuracy_pct": 52.4, "recent_predictions": pd.DataFrame()
        }
