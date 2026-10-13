"""
MarketMind - Technical Features & Panel Dataset Generator
=========================================================
Computes daily technical features strictly from past data (no lookahead),
and creates a monthly panel dataset with forward 20-day benchmark outperformance labels.
"""

import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "RS_1M",           # Relative strength vs Nifty 500 (1-month / 21 trading days)
    "RS_3M",           # Relative strength vs Nifty 500 (3-month / 63 trading days)
    "RS_6M",           # Relative strength vs Nifty 500 (6-month / 126 trading days)
    "Mom_3M",          # 3-Month return (63 trading days)
    "Mom_6M",          # 6-Month return (126 trading days)
    "Mom_9M",          # 9-Month return (189 trading days)
    "Mom_12M",         # 12-Month return (252 trading days)
    "Mom_12_1",        # 12-1 Month return (12M skipping latest 1M: Close.shift(21)/Close.shift(252) - 1)
    "Dist_52W_High",   # Distance from 52-week (252-day) high (percentage)
    "Vol_Surge_20D",   # Volume surge ratio (today's volume / 20-day average volume)
    "RSI_14",          # 14-day Relative Strength Index
    "MACD_Hist",       # MACD histogram (12/26 EMA line minus 9 EMA signal)
    "Price_vs_SMA50",  # Percentage distance from 50-day Simple Moving Average
    "Price_vs_SMA200", # Percentage distance from 200-day Simple Moving Average
    "ATR_Pct",         # ATR(14) normalized as % of price (volatility)
    "Beta_63D",        # 63-day rolling beta vs Nifty 500
    "Sharpe_63D",      # 63-day rolling annualized Sharpe ratio
]


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window, min_periods=window).mean()
    loss = ((-delta).where(delta < 0, 0.0)).rolling(window=window, min_periods=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_stock_features(df_stock: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical features for a single stock aligned with benchmark.
    STRICTLY USES PAST DATA ONLY (rolling windows and shifts; no forward references).
    
    Args:
        df_stock: OHLCV dataframe for stock
        df_bench: OHLCV dataframe for benchmark (Nifty 500)
        
    Returns:
        pd.DataFrame with original price columns plus all computed feature columns.
    """
    if df_stock.empty or len(df_stock) < 60:
        return pd.DataFrame()

    # Align dates
    stock_close = df_stock["Close"].copy()
    stock_vol = df_stock["Volume"].copy()
    
    # Common date index
    idx = stock_close.index.intersection(df_bench.index)
    if len(idx) < 60:
        return pd.DataFrame()

    stock_df = df_stock.loc[idx].copy()
    bench_close = df_bench.loc[idx, "Close"].copy()

    # Daily returns
    stock_ret = stock_df["Close"].pct_change()
    bench_ret = bench_close.pct_change()

    feat = pd.DataFrame(index=idx)
    feat["Close"] = stock_df["Close"]
    feat["Volume"] = stock_df["Volume"]
    feat["High"] = stock_df["High"]
    feat["Low"] = stock_df["Low"]

    # 1. Relative Strength vs Benchmark (1M=21d, 3M=63d, 6M=126d)
    stock_ret_21 = stock_df["Close"].pct_change(21)
    bench_ret_21 = bench_close.pct_change(21)
    feat["RS_1M"] = stock_ret_21 - bench_ret_21

    stock_ret_63 = stock_df["Close"].pct_change(63)
    bench_ret_63 = bench_close.pct_change(63)
    feat["RS_3M"] = stock_ret_63 - bench_ret_63

    stock_ret_126 = stock_df["Close"].pct_change(126)
    bench_ret_126 = bench_close.pct_change(126)
    feat["RS_6M"] = stock_ret_126 - bench_ret_126

    # 2. Momentum Variants (3M, 6M, 9M, 12M, 12-1)
    feat["Mom_3M"] = stock_ret_63
    feat["Mom_6M"] = stock_ret_126
    feat["Mom_9M"] = stock_df["Close"].pct_change(189)
    feat["Mom_12M"] = stock_df["Close"].pct_change(252)
    # 12-1 Momentum: Return from 12 months ago to 1 month ago (skips latest 21 trading days)
    # (P_{t-21} / P_{t-252}) - 1.0
    p_t21 = stock_df["Close"].shift(21)
    p_t252 = stock_df["Close"].shift(252)
    feat["Mom_12_1"] = (p_t21 / (p_t252 + 1e-9)) - 1.0

    # 3. Distance from 52-week (252-day) High
    rolling_52w_high = stock_df["High"].rolling(window=252, min_periods=60).max()
    feat["Dist_52W_High"] = (stock_df["Close"] - rolling_52w_high) / rolling_52w_high

    # 4. Volume Surge Ratio (today's volume / 20-day average volume)
    vol_sma20 = stock_vol.rolling(window=20, min_periods=5).mean()
    feat["Vol_Surge_20D"] = (stock_vol / (vol_sma20 + 1e-6)).clip(upper=10.0)

    # 5. RSI(14)
    feat["RSI_14"] = calculate_rsi(stock_df["Close"], window=14)

    # 6. MACD (12, 26, 9)
    ema12 = stock_df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = stock_df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["MACD_Hist"] = (macd_line - macd_signal) / stock_df["Close"]  # Normalized by price

    # 7. Price vs 50-day and 200-day Moving Averages
    sma50 = stock_df["Close"].rolling(window=50, min_periods=20).mean()
    sma200 = stock_df["Close"].rolling(window=200, min_periods=50).mean()
    feat["Price_vs_SMA50"] = (stock_df["Close"] / sma50) - 1.0
    feat["Price_vs_SMA200"] = (stock_df["Close"] / sma200) - 1.0

    # 8. ATR(14) as % of price
    atr14 = calculate_atr(stock_df, window=14)
    feat["ATR"] = atr14
    feat["ATR_Pct"] = (atr14 / stock_df["Close"]).clip(upper=0.3)

    # 9. Rolling Beta vs Nifty 500 (63 trading days)
    cov_63 = stock_ret.rolling(63, min_periods=30).cov(bench_ret)
    var_bench_63 = bench_ret.rolling(63, min_periods=30).var()
    feat["Beta_63D"] = (cov_63 / (var_bench_63 + 1e-9)).clip(-3.0, 5.0)

    # 10. Rolling Sharpe Ratio (63 trading days)
    mean_ret = stock_ret.rolling(63, min_periods=30).mean()
    std_ret = stock_ret.rolling(63, min_periods=30).std()
    feat["Sharpe_63D"] = ((mean_ret / (std_ret + 1e-9)) * np.sqrt(252)).clip(-5.0, 5.0)

    # 11. Average Daily Traded Value (ADTV 60-day turnover: Close * Volume)
    daily_turnover = stock_df["Close"] * stock_df["Volume"]
    feat["ADTV_60D"] = daily_turnover.rolling(window=60, min_periods=20).mean()

    return feat


def build_monthly_panel_dataset(
    price_dict: dict,
    df_bench: pd.DataFrame,
    forward_days: int = 20,
    min_date_points: int = 120
) -> pd.DataFrame:
    """
    Constructs a walk-forward monthly panel dataset across the liquid universe.
    
    For each month-end date T and each stock:
    - Features: Computed strictly using data up to date T.
    - Label: 1 if the stock's forward return from T to T+20 exceeds the Nifty 500
             return over the exact same period, else 0.
    
    Returns:
        pd.DataFrame containing columns:
        ['Date', 'Symbol', ...FEATURE_COLUMNS..., 'Fwd_Return', 'Bench_Fwd_Return', 'Excess_Return', 'Label']
    """
    bench_close = df_bench["Close"].copy()
    all_dates = bench_close.index.sort_values()

    # Identify month-end trading dates from the benchmark calendar
    dates_df = pd.DataFrame({"Date": all_dates})
    dates_df["YearMonth"] = dates_df["Date"].dt.to_period("M")
    # Take the last trading day of each month
    month_end_dates = dates_df.groupby("YearMonth")["Date"].max().values

    # Pre-compute features for all stocks
    stock_features = {}
    print(f"[Panel] Pre-computing technical features for {len(price_dict)} stocks...")
    for sym, df_s in price_dict.items():
        try:
            f = compute_stock_features(df_s, df_bench)
            if not f.empty:
                stock_features[sym] = f
        except Exception:
            continue

    rows = []
    print(f"[Panel] Building monthly panel across {len(month_end_dates)} rebalance periods...")

    for i, t_raw in enumerate(month_end_dates):
        t_date = pd.Timestamp(t_raw)
        if t_date not in all_dates:
            continue
            
        try:
            idx_pos = all_dates.get_loc(t_date)
        except Exception:
            continue
            
        fwd_idx = idx_pos + forward_days
        has_future = (fwd_idx < len(all_dates))

        if has_future:
            t_fwd_date = all_dates[fwd_idx]
            bench_p0 = bench_close.loc[t_date]
            bench_p1 = bench_close.loc[t_fwd_date]
            bench_fwd_ret = (bench_p1 / bench_p0) - 1.0
        else:
            t_fwd_date = None
            bench_fwd_ret = np.nan

        # Collect features for all stocks at t_date
        for sym, feat_df in stock_features.items():
            if t_date not in feat_df.index:
                continue

            stock_row = feat_df.loc[t_date]
            
            # Check if all required features are present and non-null
            if stock_row[FEATURE_COLUMNS].isna().any():
                continue

            p0 = stock_row["Close"]
            if p0 <= 0:
                continue

            # Forward return computation (ONLY used as target label, NEVER as feature)
            if has_future and (t_fwd_date in feat_df.index):
                p1 = feat_df.loc[t_fwd_date, "Close"]
                fwd_ret = (p1 / p0) - 1.0
                excess_ret = fwd_ret - bench_fwd_ret
                label = int(excess_ret > 0.0)
            else:
                fwd_ret = np.nan
                excess_ret = np.nan
                label = np.nan

            row_data = {
                "Date": t_date,
                "Symbol": sym,
                "Close": p0,
                "ATR": stock_row.get("ATR", np.nan),
                "ADTV_60D": stock_row.get("ADTV_60D", np.nan),
                "Fwd_Return": fwd_ret,
                "Bench_Fwd_Return": bench_fwd_ret,
                "Excess_Return": excess_ret,
                "Label": label
            }
            # Append features
            for col in FEATURE_COLUMNS:
                row_data[col] = stock_row[col]

            rows.append(row_data)

    if not rows:
        print("[Panel] Warning: No valid panel rows generated.")
        return pd.DataFrame(columns=["Date", "Symbol", "Close", "ATR", "Fwd_Return", "Bench_Fwd_Return", "Excess_Return", "Label"] + FEATURE_COLUMNS)

    panel_df = pd.DataFrame(rows)
    panel_df.sort_values(by=["Date", "Symbol"], inplace=True)
    panel_df.reset_index(drop=True, inplace=True)
    print(f"[Panel] Generated panel dataset with {len(panel_df)} rows and {len(FEATURE_COLUMNS)} features.")
    return panel_df
