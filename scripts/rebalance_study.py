"""
MarketMind - Rebalance Frequency & Event-Driven Replacement Study
================================================================
Comprehensive simulation engine comparing:
1. (a) Monthly Rebalance
2. (b) Weekly Rebalance
3. (c) Monthly + Event-Driven Replacement (Stop-loss ATR, single-day crash, 200 SMA break)
4. (d) Monthly + Event-Driven Replacement with 50% Cash Market Regime Filter

Key simulation principles:
- Strict no-lookahead: Signals use data up to previous close only.
- Gap-down execution: Exits triggered at close t are executed at Open t+1.
- Round-trip cost: 0.4% base commission/taxes + 0.1% slippage = 0.5% total friction (0.25% per side).
- Grid robustness: Single-day drop [5%, 7%, 10%] x ATR stop [1.5x, 2x, 3x].
- Chronological split: Development (Oct 2017 - Sep 2024) vs Holdout (Oct 2024 - Oct 2026).
"""

import glob
import os
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def load_universe_and_precompute():
    """Loads all 5 daily price matrices (Open, High, Low, Close, Volume) and precomputes indicators."""
    print("Loading daily price histories for Nifty 500 universe...")
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "cache", "*.NS.parquet")))
    if not files:
        raise FileNotFoundError("No parquet price files found in data/cache/")

    open_d, high_d, low_d, close_d, vol_d = {}, {}, {}, {}, {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        open_d[sym] = df["Open"]
        high_d[sym] = df["High"]
        low_d[sym] = df["Low"]
        close_d[sym] = df["Close"]
        vol_d[sym] = df["Volume"]

    open_df = pd.DataFrame(open_d)
    high_df = pd.DataFrame(high_d)
    low_df = pd.DataFrame(low_d)
    close_df = pd.DataFrame(close_d)
    vol_df = pd.DataFrame(vol_d)

    # Benchmark
    bench_file = os.path.join(REPO_ROOT, "data", "cache", "INDEX_CRSLDX.parquet")
    bench_df = pd.read_parquet(bench_file)
    bench_close = bench_df["Close"].reindex(close_df.index).ffill()
    bench_sma200 = bench_close.rolling(200, min_periods=50).mean()

    # Precompute indicators
    turnover = close_df * vol_df
    adtv_60 = turnover.rolling(60, min_periods=20).mean()
    mom_12_1 = (close_df.shift(21) / close_df.shift(252)) - 1.0
    sma_200 = close_df.rolling(200, min_periods=50).mean()

    tr1 = high_df - low_df
    tr2 = (high_df - close_df.shift(1)).abs()
    tr3 = (low_df - close_df.shift(1)).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = tr.rolling(14, min_periods=10).mean()

    print(f"Loaded {len(files)} stocks across {len(close_df)} dates in {time.time()-t0:.2f}s")

    return {
        "open": open_df,
        "high": high_df,
        "low": low_df,
        "close": close_df,
        "vol": vol_df,
        "adtv_60": adtv_60,
        "mom_12_1": mom_12_1,
        "sma_200": sma_200,
        "atr_14": atr_14,
        "bench_close": bench_close,
        "bench_sma200": bench_sma200,
        "dates": close_df.index
    }


def run_portfolio_simulation(
    data: dict,
    mode: str = "monthly",  # 'monthly', 'weekly', 'event', 'event_regime'
    min_adtv: float = 50_000_000.0,
    basket_size: int = 15,
    buffer_rank: int = 30,
    roundtrip_cost: float = 0.005,  # 0.4% fees + 0.1% slippage
    stop_atr_mult: float = 2.0,
    drop_threshold: float = 0.07,
    check_200sma: bool = True,
    use_regime_cash: bool = False,
    regime_cash_fraction: float = 0.50,
    start_date: str = "2017-10-31",
    end_date: str = "2026-10-13",
):
    """
    Executes accurate daily portfolio simulation with zero lookahead.
    Signals computed at close t; trades executed at Open t+1.
    """
    half_cost = roundtrip_cost / 2.0
    cash_rate_daily = 0.065 / 252.0  # 6.5% annual risk-free rate on cash

    dates = data["dates"]
    
    # Locate date indices safely
    valid_dates = [d for d in dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]
    if not valid_dates:
        raise ValueError(f"No dates found between {start_date} and {end_date}")
    start_idx = dates.get_loc(valid_dates[0])
    end_idx = dates.get_loc(valid_dates[-1])

    open_mat = data["open"].values
    close_mat = data["close"].values
    adtv_mat = data["adtv_60"].values
    mom_mat = data["mom_12_1"].values
    sma_mat = data["sma_200"].values
    atr_mat = data["atr_14"].values
    bench_close = data["bench_close"].values
    bench_sma200 = data["bench_sma200"].values
    tickers = list(data["close"].columns)

    # Pre-identify monthly rebalance trigger dates
    all_dates_series = pd.Series(dates)
    month_groups = all_dates_series.groupby([all_dates_series.dt.year, all_dates_series.dt.month]).apply(lambda x: x.index[-1])
    month_end_indices = set(month_groups.values)

    # Weekly rebalance every 5 trading days
    weekly_indices = set(range(start_idx, end_idx + 1, 5))

    # Portfolio state
    holdings = {}
    cash = 1_000_000.0
    portfolio_history = []
    bench_history = []
    event_replacements = 0
    total_costs_paid = 0.0
    all_trade_losses = []
    turnover_dollars = 0.0

    pending_rebalance = True  # Trigger initial construction at first day's open
    pending_exits = []

    for t in range(start_idx, end_idx + 1):
        dt = dates[t]

        # -----------------------------------------------------------------
        # 1. EXECUTE PENDING TRADES AT OPEN t (Honest Gap-Down Capture)
        # -----------------------------------------------------------------
        # A. Process Event Exits triggered at close t-1
        if pending_exits:
            for s_idx in pending_exits:
                if s_idx in holdings:
                    exit_open = open_mat[t, s_idx]
                    if np.isnan(exit_open) or exit_open <= 0:
                        exit_open = close_mat[t - 1, s_idx]
                    shares = holdings[s_idx]["shares"]
                    trade_val = shares * exit_open
                    cost_paid = trade_val * half_cost
                    proceeds = trade_val - cost_paid
                    total_costs_paid += cost_paid
                    turnover_dollars += trade_val
                    cash += proceeds

                    # Record single stock trade return
                    entry_p = holdings[s_idx]["entry_price"]
                    pos_return = (exit_open / entry_p) - 1.0
                    all_trade_losses.append(pos_return)
                    del holdings[s_idx]

                    # Event replacement: Buy highest-ranked candidate not currently held (using data up to close t-1)
                    valid_mask = (adtv_mat[t - 1, :] >= min_adtv) & (~np.isnan(mom_mat[t - 1, :]))
                    for held_s in holdings.keys():
                        valid_mask[held_s] = False
                    valid_mask[s_idx] = False

                    candidate_indices = np.where(valid_mask)[0]
                    if len(candidate_indices) > 0:
                        best_cand = candidate_indices[np.argmax(mom_mat[t - 1, candidate_indices])]
                        buy_open = open_mat[t, best_cand]
                        if not np.isnan(buy_open) and buy_open > 0:
                            current_est_val = cash + sum(h["shares"] * (open_mat[t, s] if not np.isnan(open_mat[t, s]) else close_mat[t - 1, s]) for s, h in holdings.items())
                            alloc = min(cash * 0.95, current_est_val / basket_size)
                            if alloc > 10_000:
                                cost_paid = alloc * half_cost
                                net_alloc = alloc - cost_paid
                                total_costs_paid += cost_paid
                                turnover_dollars += alloc
                                buy_shares = net_alloc / buy_open
                                cash -= alloc
                                cand_atr = atr_mat[t - 1, best_cand]
                                if np.isnan(cand_atr) or cand_atr <= 0:
                                    cand_atr = buy_open * 0.02
                                stop_p = buy_open - (stop_atr_mult * cand_atr)
                                holdings[best_cand] = {
                                    "shares": buy_shares,
                                    "entry_price": buy_open,
                                    "stop_loss": stop_p,
                                    "entry_atr": cand_atr
                                }
                                event_replacements += 1
            pending_exits = []

        # B. Process Scheduled Rebalance (Monthly or Weekly)
        if pending_rebalance:
            sig_idx = max(0, t - 1)
            valid_mask = (adtv_mat[sig_idx, :] >= min_adtv) & (~np.isnan(mom_mat[sig_idx, :]))
            ranked_indices = np.where(valid_mask)[0]
            ranked_indices = ranked_indices[np.argsort(-mom_mat[sig_idx, ranked_indices])]

            current_syms = list(holdings.keys())
            new_syms = []
            top_buffer_syms = set(ranked_indices[:buffer_rank])
            for s in current_syms:
                if s in top_buffer_syms:
                    new_syms.append(s)

            for s in ranked_indices:
                if len(new_syms) >= basket_size:
                    break
                if s not in new_syms:
                    new_syms.append(s)

            # Liquidate stocks not in new_syms
            for s in current_syms:
                if s not in new_syms:
                    exit_open = open_mat[t, s]
                    if np.isnan(exit_open) or exit_open <= 0:
                        exit_open = close_mat[t - 1, s]
                    shares = holdings[s]["shares"]
                    trade_val = shares * exit_open
                    cost_paid = trade_val * half_cost
                    total_costs_paid += cost_paid
                    turnover_dollars += trade_val
                    cash += (trade_val - cost_paid)
                    pos_return = (exit_open / holdings[s]["entry_price"]) - 1.0
                    all_trade_losses.append(pos_return)
                    del holdings[s]

            # Rebalance target allocations
            total_equity_val = cash + sum(h["shares"] * (open_mat[t, s] if not np.isnan(open_mat[t, s]) else close_mat[sig_idx, s]) for s, h in holdings.items())

            is_bear_now = False
            if use_regime_cash:
                b_c = bench_close[sig_idx]
                b_s = bench_sma200[sig_idx]
                if not np.isnan(b_s) and b_c < b_s:
                    is_bear_now = True

            effective_cap = total_equity_val * (1.0 - regime_cash_fraction) if is_bear_now else total_equity_val
            target_per_stock = effective_cap / basket_size

            # Buy new positions
            for s in new_syms:
                buy_open = open_mat[t, s]
                if np.isnan(buy_open) or buy_open <= 0:
                    buy_open = close_mat[sig_idx, s]

                if s not in holdings:
                    alloc = min(cash * 0.98, target_per_stock)
                    if alloc > 10_000:
                        cost_paid = alloc * half_cost
                        net_alloc = alloc - cost_paid
                        total_costs_paid += cost_paid
                        turnover_dollars += alloc
                        shares = net_alloc / buy_open
                        cash -= alloc
                        s_atr = atr_mat[sig_idx, s]
                        if np.isnan(s_atr) or s_atr <= 0:
                            s_atr = buy_open * 0.02
                        holdings[s] = {
                            "shares": shares,
                            "entry_price": buy_open,
                            "stop_loss": buy_open - (stop_atr_mult * s_atr),
                            "entry_atr": s_atr
                        }
            pending_rebalance = False

        # -----------------------------------------------------------------
        # 2. MARK-TO-MARKET AT CLOSE t
        # -----------------------------------------------------------------
        cash *= (1.0 + cash_rate_daily)
        stock_val = sum(h["shares"] * (close_mat[t, s] if not np.isnan(close_mat[t, s]) else close_mat[t - 1, s]) for s, h in holdings.items())
        total_val = cash + stock_val
        portfolio_history.append({"Date": dt, "Value": total_val})
        bench_history.append({"Date": dt, "Value": bench_close[t]})

        # -----------------------------------------------------------------
        # 3. EVALUATE EXIT CONDITIONS & SCHEDULE REBALANCE AT CLOSE t
        # -----------------------------------------------------------------
        if t < end_idx:
            if mode == "monthly" or mode.startswith("event"):
                if t in month_end_indices:
                    pending_rebalance = True
            elif mode == "weekly":
                if t in weekly_indices:
                    pending_rebalance = True

            if mode.startswith("event") and not pending_rebalance:
                for s, h in holdings.items():
                    c_now = close_mat[t, s]
                    c_prev = close_mat[t - 1, s]
                    if np.isnan(c_now) or np.isnan(c_prev):
                        continue

                    triggered = False
                    # Rule 1: Hit ATR stop-loss
                    if c_now <= h["stop_loss"]:
                        triggered = True
                    # Rule 2: Single-day crash
                    elif (c_now / c_prev) - 1.0 <= -drop_threshold:
                        triggered = True
                    # Rule 3: Close below 200 SMA
                    elif check_200sma:
                        s_sma = sma_mat[t, s]
                        if not np.isnan(s_sma) and c_now < s_sma:
                            triggered = True

                    if triggered and s not in pending_exits:
                        pending_exits.append(s)

    # Performance Analytics
    p_df = pd.DataFrame(portfolio_history)
    b_df = pd.DataFrame(bench_history)
    p_df["Daily_Return"] = p_df["Value"].pct_change().fillna(0.0)
    b_df["Daily_Return"] = b_df["Value"].pct_change().fillna(0.0)

    n_days = len(p_df)
    years = n_days / 252.0

    start_val = p_df["Value"].iloc[0]
    final_val = p_df["Value"].iloc[-1]
    cagr = (final_val / start_val) ** (1.0 / years) - 1.0 if years > 0 and final_val > 0 else 0.0

    b_start = b_df["Value"].iloc[0]
    b_final = b_df["Value"].iloc[-1]
    b_cagr = (b_final / b_start) ** (1.0 / years) - 1.0 if years > 0 and b_final > 0 else 0.0
    excess_cagr = cagr - b_cagr

    rf_daily = 0.065 / 252.0
    excess_daily = p_df["Daily_Return"] - rf_daily
    sharpe = float(excess_daily.mean() / (excess_daily.std() + 1e-9) * np.sqrt(252.0))

    cummax = p_df["Value"].cummax()
    drawdown = (p_df["Value"] - cummax) / cummax
    max_dd = float(drawdown.min())

    worst_trade_loss = float(min(all_trade_losses)) * 100.0 if all_trade_losses else 0.0

    p_df["Year"] = p_df["Date"].dt.year
    b_df["Year"] = b_df["Date"].dt.year
    yearly_port = p_df.groupby("Year").apply(lambda x: (x["Value"].iloc[-1] / x["Value"].iloc[0]) - 1.0)
    yearly_bench = b_df.groupby("Year").apply(lambda x: (x["Value"].iloc[-1] / x["Value"].iloc[0]) - 1.0)
    excess_years = yearly_port - yearly_bench
    pct_years_beating = float((excess_years > 0).mean()) * 100.0

    avg_port_val = p_df["Value"].mean()
    monthly_turnover_pct = (turnover_dollars / avg_port_val) / (years * 12.0) * 100.0 if years > 0 else 0.0
    total_cost_pct = (total_costs_paid / avg_port_val) * 100.0

    return {
        "mode": mode,
        "CAGR": cagr * 100.0,
        "Excess_CAGR": excess_cagr * 100.0,
        "Benchmark_CAGR": b_cagr * 100.0,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd * 100.0,
        "Worst_Trade_Loss": worst_trade_loss,
        "Avg_Monthly_Turnover_Pct": monthly_turnover_pct,
        "Total_Costs_Paid": total_costs_paid,
        "Total_Cost_Pct": total_cost_pct,
        "Event_Replacements": event_replacements,
        "Pct_Years_Beating": pct_years_beating,
        "Final_Value": final_val,
        "Years": years,
        "all_trade_losses": all_trade_losses,
        "yearly_excess": excess_years.to_dict()
    }


def run_full_study():
    """Runs the 3x3 robustness grid on Development Split, picks defaults, and tests all 4 setups."""
    data = load_universe_and_precompute()

    dev_start = "2017-10-31"
    dev_end = "2024-09-30"    # ~7 years (84 months) development period
    holdout_start = "2024-10-01"
    holdout_end = "2026-10-13"  # 24 months holdout
    full_end = "2026-10-13"

    print("\n" + "="*80)
    print("STEP 1: 3x3 EVENT THRESHOLD ROBUSTNESS GRID (DEVELOPMENT PERIOD: 2017-10 TO 2024-09)")
    print("DO NOT TUNE ON THE 24-MONTH HOLDOUT!")
    print("="*80)

    drops = [0.05, 0.07, 0.10]
    stops = [1.5, 2.0, 3.0]

    grid_results = []
    for d in drops:
        for s in stops:
            res = run_portfolio_simulation(
                data,
                mode="event",
                drop_threshold=d,
                stop_atr_mult=s,
                start_date=dev_start,
                end_date=dev_end
            )
            grid_results.append({
                "Crash_Drop_%": f"{d*100:.0f}%",
                "ATR_Stop": f"{s:.1f}x",
                "CAGR": f"{res['CAGR']:.2f}%",
                "Excess_CAGR": f"{res['Excess_CAGR']:.2f}%",
                "Sharpe": f"{res['Sharpe']:.2f}",
                "Max_DD": f"{res['Max_Drawdown']:.2f}%",
                "Worst_Trade": f"{res['Worst_Trade_Loss']:.2f}%",
                "Replacements": res["Event_Replacements"],
                "Costs_Paid": f"₹{res['Total_Costs_Paid']:,.0f}",
                "Win_Years_%": f"{res['Pct_Years_Beating']:.1f}%",
                "_raw_sharpe": res["Sharpe"],
                "_raw_cagr": res["CAGR"],
                "_raw_dd": res["Max_Drawdown"]
            })

    grid_df = pd.DataFrame(grid_results)
    print("\n--- 3x3 Grid Results (Development Period: Oct 2017 to Sep 2024) ---")
    print(grid_df.drop(columns=["_raw_sharpe", "_raw_cagr", "_raw_dd"]).to_string(index=False))

    # Evaluate grid stability
    print("\nGrid Summary Analysis across 9 combinations:")
    mean_sharpe = grid_df["_raw_sharpe"].mean()
    std_sharpe = grid_df["_raw_sharpe"].std()
    print(f"Mean Sharpe across grid: {mean_sharpe:.3f} (Std: {std_sharpe:.3f})")
    print(f"CAGR range across grid: {grid_df['_raw_cagr'].min():.2f}% to {grid_df['_raw_cagr'].max():.2f}%")
    print(f"Max DD range across grid: {grid_df['_raw_dd'].min():.2f}% to {grid_df['_raw_dd'].max():.2f}%")

    # Chosen robust default based on central stability across grid
    chosen_drop = 0.07  # 7% single-day crash
    chosen_stop = 2.0   # 2.0x ATR stop-loss
    print(f"\n=> Chosen Robust Default: Single-Day Crash = {chosen_drop*100:.0f}%, Stop-Loss = {chosen_stop:.1f}x ATR")

    print("\n" + "="*80)
    print("STEP 2: FULL-HISTORY COMPARISON OF ALL 4 SETUPS (2017-10 TO 2026-10, NET OF 0.5% COSTS)")
    print("="*80)

    setups = [
        ("Setup (a): Monthly Rebalance", "monthly", False),
        ("Setup (b): Weekly Rebalance", "weekly", False),
        ("Setup (c): Monthly + Event Replacement", "event", False),
        ("Setup (d): Monthly + Event + 50% Cash Regime Filter", "event_regime", True),
    ]

    full_results = []
    for label, mode, use_regime in setups:
        res = run_portfolio_simulation(
            data,
            mode="event" if mode.startswith("event") else mode,
            drop_threshold=chosen_drop,
            stop_atr_mult=chosen_stop,
            use_regime_cash=use_regime,
            start_date=dev_start,
            end_date=full_end
        )
        full_results.append({
            "Setup": label,
            "CAGR": f"{res['CAGR']:.2f}%",
            "Excess_CAGR": f"{res['Excess_CAGR']:.2f}%",
            "Sharpe": f"{res['Sharpe']:.2f}",
            "Max_DD": f"{res['Max_Drawdown']:.2f}%",
            "Worst_Trade": f"{res['Worst_Trade_Loss']:.2f}%",
            "Avg_Turnover/Mo": f"{res['Avg_Monthly_Turnover_Pct']:.1f}%",
            "Total_Costs": f"₹{res['Total_Costs_Paid']:,.0f}",
            "Replacements": res["Event_Replacements"],
            "Beating_Years_%": f"{res['Pct_Years_Beating']:.1f}%"
        })

    full_df = pd.DataFrame(full_results)
    print("\n--- Full History Results (All ~9 Years / 107 Months) ---")
    print(full_df.to_string(index=False))

    print("\n" + "="*80)
    print("STEP 3: UNTOUCHED 24-MONTH FINAL HOLDOUT EVALUATION (OCT 2024 TO OCT 2026)")
    print("="*80)

    holdout_results = []
    for label, mode, use_regime in setups:
        res = run_portfolio_simulation(
            data,
            mode="event" if mode.startswith("event") else mode,
            drop_threshold=chosen_drop,
            stop_atr_mult=chosen_stop,
            use_regime_cash=use_regime,
            start_date=holdout_start,
            end_date=holdout_end
        )
        holdout_results.append({
            "Setup": label,
            "CAGR": f"{res['CAGR']:.2f}%",
            "Excess_CAGR": f"{res['Excess_CAGR']:.2f}%",
            "Sharpe": f"{res['Sharpe']:.2f}",
            "Max_DD": f"{res['Max_Drawdown']:.2f}%",
            "Worst_Trade": f"{res['Worst_Trade_Loss']:.2f}%",
            "Avg_Turnover/Mo": f"{res['Avg_Monthly_Turnover_Pct']:.1f}%",
            "Total_Costs": f"₹{res['Total_Costs_Paid']:,.0f}",
            "Replacements": res["Event_Replacements"],
            "Beating_Years_%": f"{res['Pct_Years_Beating']:.1f}%"
        })

    holdout_df = pd.DataFrame(holdout_results)
    print("\n--- Final 24-Month Holdout Results (Oct 2024 to Oct 2026) ---")
    print(holdout_df.to_string(index=False))


if __name__ == "__main__":
    run_full_study()
