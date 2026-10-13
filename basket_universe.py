"""
MarketMind - Universe and Data Management
=========================================
Handles downloading and caching the Nifty 500 universe, benchmark index,
and historical daily prices with liquidity filtering.
"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

NSE_NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FALLBACK_CSV_PATH = os.path.join(REPO_ROOT, "data", "nifty500.csv")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")

BENCHMARK_TICKER = "^CRSLDX"      # Nifty 500 Index on Yahoo Finance
BENCHMARK_FALLBACK = "^NSEI"     # Fallback: Nifty 50 Index

DEFAULT_LIQUIDITY_MIN_TURNOVER = 50_000_000  # Default 5 Crore INR daily traded value


def get_nifty500_symbols(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the Nifty 500 constituents table.
    
    Tries to fetch the live constituent list from NSE archives.
    If the request fails or is blocked, falls back to data/nifty500.csv.
    Adds a 'Yahoo_Symbol' column formatted with the '.NS' suffix for yfinance.
    
    Returns:
        pd.DataFrame: Contains columns ['Company Name', 'Industry', 'Symbol', 'Yahoo_Symbol']
    """
    os.makedirs(os.path.dirname(FALLBACK_CSV_PATH), exist_ok=True)
    df = None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(NSE_NIFTY500_URL, headers=headers, timeout=12)
        if response.status_code == 200 and len(response.text) > 1000:
            df = pd.read_csv(io.StringIO(response.text))
            # Save fresh copy to fallback path
            df.to_csv(FALLBACK_CSV_PATH, index=False)
            print(f"[Universe] Fetched {len(df)} constituents live from NSE.")
    except Exception as exc:
        print(f"[Universe] Live NSE fetch failed ({exc}). Trying local fallback.")

    if df is None or df.empty:
        if os.path.exists(FALLBACK_CSV_PATH):
            df = pd.read_csv(FALLBACK_CSV_PATH)
            print(f"[Universe] Loaded {len(df)} constituents from local fallback: {FALLBACK_CSV_PATH}")
        else:
            raise RuntimeError(
                f"Could not load Nifty 500 constituents from NSE or fallback at {FALLBACK_CSV_PATH}."
            )

    # Standardize column names if needed
    if "Symbol" not in df.columns:
        for col in df.columns:
            if "symbol" in col.lower():
                df.rename(columns={col: "Symbol"}, inplace=True)
                break

    # Add Yahoo symbol format
    df["Yahoo_Symbol"] = df["Symbol"].apply(lambda s: f"{str(s).strip()}.NS")
    return df


def get_benchmark_history(period: str = "10y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads and caches historical daily data for the benchmark (Nifty 500).
    Uses ^CRSLDX, with fallback to ^NSEI if ^CRSLDX returns no data.
    
    Returns:
        pd.DataFrame: Clean daily OHLCV dataframe with DatetimeIndex.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_ticker = BENCHMARK_TICKER.replace("^", "INDEX_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_ticker}.parquet")

    if not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            min_req = 1200 if ("10y" in period or "max" in period) else 50
            if not df.empty and len(df) >= min_req:
                return df
        except Exception:
            pass

    # Attempt primary ticker
    ticker_to_try = BENCHMARK_TICKER
    print(f"[Universe] Downloading benchmark history for {ticker_to_try} (period={period})...")
    df = yf.Ticker(ticker_to_try).history(period=period, interval="1d", auto_adjust=True)

    if df.empty or len(df) < 50:
        print(f"[Universe] Benchmark {BENCHMARK_TICKER} returned empty/low data. Falling back to {BENCHMARK_FALLBACK}.")
        ticker_to_try = BENCHMARK_FALLBACK
        df = yf.Ticker(ticker_to_try).history(period=period, interval="1d", auto_adjust=True)

    if df.empty:
        raise RuntimeError("Failed to download benchmark history from yfinance.")

    # Standardize columns and index
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    # Cache to disk
    df.to_parquet(cache_file)
    return df


def fetch_stock_history(
    symbol: str,
    period: str = "10y",
    force_refresh: bool = False,
    cache_dir: str = CACHE_DIR
) -> pd.DataFrame:
    """
    Fetches daily price history for a single stock from yfinance, caching to disk.
    
    Args:
        symbol: Yahoo ticker symbol (e.g. 'RELIANCE.NS')
        period: History period (e.g. '10y')
        force_refresh: If True, bypasses disk cache
        cache_dir: Directory where parquet files are saved
        
    Returns:
        pd.DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume]
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = symbol.replace("/", "_").replace("^", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            min_req = 1200 if ("10y" in period or "max" in period) else 30
            if not df.empty and len(df) >= min_req:
                return df
        except Exception:
            pass

    try:
        data = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if data.empty or len(data) < 30:
            return pd.DataFrame()
        
        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data.sort_index(inplace=True)
        data = data[~data.index.duplicated(keep="last")]
        
        # Save cache
        data.to_parquet(cache_path)
        return data
    except Exception as exc:
        return pd.DataFrame()


def download_universe_prices(
    symbols: list,
    period: str = "10y",
    max_workers: int = 16,
    force_refresh: bool = False
) -> dict:
    """
    Downloads historical prices for a list of symbols in parallel with disk caching.
    
    Returns:
        dict: {symbol: pd.DataFrame}
    """
    results = {}
    print(f"[Universe] Downloading/loading price histories for {len(symbols)} tickers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(fetch_stock_history, sym, period, force_refresh): sym
            for sym in symbols
        }
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if not df.empty and len(df) >= 60:
                    results[sym] = df
            except Exception as e:
                pass

    print(f"[Universe] Successfully loaded {len(results)}/{len(symbols)} stocks with valid data.")
    return results


def apply_liquidity_filter(
    price_dict: dict,
    min_avg_turnover: float = DEFAULT_LIQUIDITY_MIN_TURNOVER,
    lookback_days: int = 60
) -> tuple:
    """
    Filters out stocks with low average daily traded value (ADTV = Close * Volume).
    
    Args:
        price_dict: dict of {symbol: pd.DataFrame}
        min_avg_turnover: Minimum average daily turnover in INR (e.g. 5 Crore = 50,000,000)
        lookback_days: Number of recent trading days to average over
        
    Returns:
        tuple: (filtered_dict, liquidity_stats_df)
    """
    filtered = {}
    records = []

    for sym, df in price_dict.items():
        if df.empty or len(df) < lookback_days:
            continue
        recent = df.tail(lookback_days)
        turnover = (recent["Close"] * recent["Volume"]).mean()
        avg_volume = recent["Volume"].mean()
        last_price = recent["Close"].iloc[-1]
        
        passes = bool(turnover >= min_avg_turnover)
        records.append({
            "Symbol": sym,
            "Last_Price": last_price,
            "Avg_Volume": avg_volume,
            "Avg_Daily_Turnover": turnover,
            "Passes_Liquidity": passes
        })
        
        if passes:
            filtered[sym] = df

    stats_df = pd.DataFrame(records)
    print(f"[Universe] Liquidity filter (>= ₹{min_avg_turnover:,.0f}/day): "
          f"{len(filtered)}/{len(price_dict)} stocks passed.")
    return filtered, stats_df
