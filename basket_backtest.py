"""
MarketMind - Strategy Walk-Forward Backtester
=============================================
Simulates monthly walk-forward rebalancing of the 15-stock basket
with top-30 rank retention buffer and round-trip transaction costs (0.4%).
Compares performance directly against the Nifty 500 index.
"""

import pandas as pd
import numpy as np


def run_walk_forward_backtest(
    panel_df: pd.DataFrame,
    model_bundle: dict = None,
    basket_size: int = 15,
    buffer_rank: int = 30,
    tx_cost_roundtrip: float = 0.004,
    test_dates: list = None,
    strategy_name: str = None,
    min_adtv: float = 50_000_000,
    use_regime_filter: bool = False,
    regime_cash_fraction: float = 0.5,
    bench_df: pd.DataFrame = None
) -> dict:
    """
    Executes walk-forward backtesting with turnover control and transaction costs.
    
    Args:
        panel_df: Monthly panel dataset.
        model_bundle: Trained model bundle with model, scaler, and feature list (optional).
        basket_size: Target portfolio size (15 stocks).
        buffer_rank: Rank threshold to retain existing holdings (e.g. 30).
        tx_cost_roundtrip: Round-trip trading friction (default 0.4% = 0.004).
        test_dates: Optional list of dates to restrict backtest to out-of-sample period.
        strategy_name: Strategy identifier:
            'MOMENTUM_12_1' (default robust), 'MOMENTUM_6M', 'MOMENTUM_3M', 'MOMENTUM_9M',
            'MOMENTUM_12M', 'RELATIVE_STRENGTH_6M', or 'ML_MODEL'.
        min_adtv: Minimum average daily turnover in INR (default ₹5 Cr = 50,000,000).
        use_regime_filter: If True, moves regime_cash_fraction into cash when Nifty 500 < 200 SMA.
        regime_cash_fraction: Fraction allocated to cash when below 200 SMA (default 0.50).
        bench_df: Benchmark daily OHLCV dataframe for 200-SMA regime checking.
        
    Returns:
        dict: Performance metrics, equity curves, yearly breakdown, and period-by-period logs.
    """
    if model_bundle is None:
        model_bundle = {}
    model = model_bundle.get("model")
    scaler = model_bundle.get("scaler")
    feature_cols = model_bundle.get("feature_names", [])

    df = panel_df.dropna(subset=["Fwd_Return", "Bench_Fwd_Return"]).copy()

    # Track how many candidates liquidity filter removes
    total_candidates_before_liq = len(df)
    if "ADTV_60D" in df.columns and min_adtv > 0:
        df = df[df["ADTV_60D"] >= min_adtv].copy()
    liq_removed_candidates = total_candidates_before_liq - len(df)

    all_dates = sorted(df["Date"].unique())

    if test_dates is not None:
        eval_dates = [d for d in all_dates if d in test_dates]
    else:
        eval_dates = all_dates

    if len(eval_dates) < 3:
        eval_dates = all_dates

    # Pre-calculate Benchmark 200 SMA if regime filter requested
    if use_regime_filter:
        if bench_df is None or bench_df.empty:
            import basket_universe as bu
            try:
                bench_df = bu.get_benchmark_history()
            except Exception:
                bench_df = pd.DataFrame()
        if not bench_df.empty and "Close" in bench_df.columns:
            bench_df = bench_df.copy()
            bench_df["SMA200"] = bench_df["Close"].rolling(200, min_periods=50).mean()

    current_holdings = []
    portfolio_equity = [100.0]
    benchmark_equity = [100.0]
    monthly_records = []

    active_strategy = strategy_name or model_bundle.get("active_strategy", "MOMENTUM_12_1")

    for i, dt in enumerate(eval_dates):
        month_data = df[df["Date"] == dt].copy()
        if month_data.empty or len(month_data) < basket_size:
            continue

        # Score all candidates for this month based on active strategy
        if active_strategy == "ML_MODEL" and model is not None and scaler is not None:
            X_scaled = scaler.transform(month_data[feature_cols].values)
            month_data["Score"] = model.predict_proba(X_scaled)[:, 1]
        elif active_strategy in ["MOMENTUM_12_1", "Mom_12_1"]:
            month_data["Score"] = month_data["Mom_12_1"]
        elif active_strategy in ["MOMENTUM_6M", "Mom_6M"]:
            month_data["Score"] = month_data["Mom_6M"]
        elif active_strategy in ["MOMENTUM_3M", "Mom_3M"]:
            month_data["Score"] = month_data["Mom_3M"]
        elif active_strategy in ["MOMENTUM_9M", "Mom_9M"]:
            month_data["Score"] = month_data["Mom_9M"]
        elif active_strategy in ["MOMENTUM_12M", "Mom_12M"]:
            month_data["Score"] = month_data["Mom_12M"]
        elif active_strategy in ["RELATIVE_STRENGTH_6M", "RS_6M"]:
            month_data["Score"] = month_data["RS_6M"]
        elif active_strategy in ["RANK_HISTGB"]:
            # If rank regressor passed
            X_scaled = scaler.transform(month_data[feature_cols].values)
            month_data["Score"] = model.predict(X_scaled)
        else:
            # Fallback to Mom_12_1 or Mom_6M
            month_data["Score"] = month_data.get("Mom_12_1", month_data.get("Mom_6M"))

        ranked_df = month_data.sort_values(by="Score", ascending=False).reset_index(drop=True)
        ranked_df["Rank"] = ranked_df.index + 1

        # Rebalance selection with Top-30 buffer rule
        new_holdings = []
        if not current_holdings:
            # Initial portfolio construction
            new_holdings = ranked_df.head(basket_size)["Symbol"].tolist()
            turnover_fraction = 1.0
        else:
            # 1. Keep existing holdings if their new rank is within buffer_rank (30)
            eligible_keepers = ranked_df[
                (ranked_df["Symbol"].isin(current_holdings)) & (ranked_df["Rank"] <= buffer_rank)
            ]["Symbol"].tolist()
            new_holdings.extend(eligible_keepers)

            # 2. Fill remaining slots from highest ranked candidates not currently held
            slots_to_fill = basket_size - len(new_holdings)
            if slots_to_fill > 0:
                candidates = ranked_df[~ranked_df["Symbol"].isin(new_holdings)]
                top_fillers = candidates.head(slots_to_fill)["Symbol"].tolist()
                new_holdings.extend(top_fillers)

            # Calculate portfolio turnover
            replaced_count = len(set(current_holdings) - set(new_holdings))
            turnover_fraction = replaced_count / float(basket_size)

        # Apply transaction costs on turnover (0.4% per replaced stock)
        month_tx_cost = turnover_fraction * tx_cost_roundtrip

        # Measure forward 20-day returns of the selected basket
        picks_data = ranked_df[ranked_df["Symbol"].isin(new_holdings)]
        gross_return = picks_data["Fwd_Return"].mean()
        net_stock_return = gross_return - month_tx_cost
        bench_return = picks_data["Bench_Fwd_Return"].iloc[0]

        # Check market regime (200 SMA on Nifty 500)
        dt_ts = pd.to_datetime(dt)
        is_bear_regime = False
        if use_regime_filter and bench_df is not None and not bench_df.empty and "SMA200" in bench_df.columns:
            sub_b = bench_df.loc[:dt_ts]
            if not sub_b.empty:
                c_bench = sub_b["Close"].iloc[-1]
                s_bench = sub_b["SMA200"].iloc[-1]
                if pd.notna(s_bench) and c_bench < s_bench:
                    is_bear_regime = True

        if use_regime_filter and is_bear_regime:
            # Shift fraction to cash (earning risk-free cash rate 6.5%/12)
            cash_return = 0.065 / 12.0
            net_return = (1.0 - regime_cash_fraction) * net_stock_return + regime_cash_fraction * cash_return
        else:
            net_return = net_stock_return

        excess_return = net_return - bench_return

        # Update compounding equities
        new_port_val = portfolio_equity[-1] * (1.0 + net_return)
        new_bench_val = benchmark_equity[-1] * (1.0 + bench_return)
        portfolio_equity.append(new_port_val)
        benchmark_equity.append(new_bench_val)

        monthly_records.append({
            "Date": dt_ts,
            "Date_Str": dt_ts.strftime("%Y-%m"),
            "Year": dt_ts.year,
            "Gross_Return": gross_return,
            "Net_Stock_Return": net_stock_return,
            "Net_Return": net_return,
            "Net_Return_Pct": net_return * 100.0,
            "Benchmark_Return": bench_return,
            "Benchmark_Return_Pct": bench_return * 100.0,
            "Excess_Return": excess_return,
            "Excess_Return_Pct": excess_return * 100.0,
            "Turnover_Pct": turnover_fraction * 100.0,
            "Tx_Cost_Pct": month_tx_cost * 100.0,
            "Is_Bear_Regime": is_bear_regime,
            "Portfolio_Value": new_port_val,
            "Benchmark_Value": new_bench_val,
            "Holdings": new_holdings,
        })

        current_holdings = new_holdings

    perf_df = pd.DataFrame(monthly_records)
    if perf_df.empty:
        return {"status": "ERROR", "message": "Not enough periods for backtest."}

    # Summary Statistics Calculation
    n_months = len(perf_df)
    years = max(n_months / 12.0, 0.25)

    final_port = perf_df["Portfolio_Value"].iloc[-1]
    final_bench = perf_df["Benchmark_Value"].iloc[-1]

    port_total_ret = (final_port / 100.0) - 1.0
    bench_total_ret = (final_bench / 100.0) - 1.0

    port_cagr = ((final_port / 100.0) ** (1.0 / years)) - 1.0
    bench_cagr = ((final_bench / 100.0) ** (1.0 / years)) - 1.0

    port_vol = perf_df["Net_Return"].std() * np.sqrt(12)
    bench_vol = perf_df["Benchmark_Return"].std() * np.sqrt(12)

    risk_free_rate = 0.065  # 6.5% typical RBI repo rate benchmark
    port_sharpe = (port_cagr - risk_free_rate) / (port_vol + 1e-9)
    bench_sharpe = (bench_cagr - risk_free_rate) / (bench_vol + 1e-9)

    # Max Drawdown
    port_series = pd.Series(portfolio_equity)
    port_peaks = port_series.cummax()
    port_dd = (port_series - port_peaks) / port_peaks
    max_dd_port = float(port_dd.min())

    bench_series = pd.Series(benchmark_equity)
    bench_peaks = bench_series.cummax()
    bench_dd = (bench_series - bench_peaks) / bench_peaks
    max_dd_bench = float(bench_dd.min())

    # Hit Rate (% of months with positive excess return over benchmark)
    hit_rate = (perf_df["Excess_Return"] > 0.0).mean()
    avg_turnover = perf_df["Turnover_Pct"].mean()

    metrics = {
        "Total_Months": n_months,
        "Portfolio_Total_Return_Pct": port_total_ret * 100.0,
        "Benchmark_Total_Return_Pct": bench_total_ret * 100.0,
        "Portfolio_CAGR_Pct": port_cagr * 100.0,
        "Benchmark_CAGR_Pct": bench_cagr * 100.0,
        "Portfolio_Vol_Pct": port_vol * 100.0,
        "Benchmark_Vol_Pct": bench_vol * 100.0,
        "Portfolio_Sharpe": port_sharpe,
        "Benchmark_Sharpe": bench_sharpe,
        "Portfolio_Max_Drawdown_Pct": max_dd_port * 100.0,
        "Benchmark_Max_Drawdown_Pct": max_dd_bench * 100.0,
        "Monthly_Hit_Rate_Pct": hit_rate * 100.0,
        "Avg_Monthly_Turnover_Pct": avg_turnover,
        "Transaction_Cost_Per_Turnover_Pct": tx_cost_roundtrip * 100.0,
    }

    # Yearly Breakdown
    yearly_records = []
    for yr, ydf in perf_df.groupby("Year"):
        y_port = float(np.prod(1.0 + ydf["Net_Return"]) - 1.0)
        y_bench = float(np.prod(1.0 + ydf["Benchmark_Return"]) - 1.0)
        y_excess = y_port - y_bench
        yearly_records.append({
            "Year": int(yr),
            "Portfolio_Return_Pct": y_port * 100.0,
            "Benchmark_Return_Pct": y_bench * 100.0,
            "Excess_Return_Pct": y_excess * 100.0,
            "Months": len(ydf),
            "Beat_Benchmark": bool(y_excess > 0)
        })
    yearly_df = pd.DataFrame(yearly_records)

    # Rolling 12-Month Windows
    rolling_excess_list = []
    if len(perf_df) >= 12:
        for i in range(12, len(perf_df) + 1):
            w = perf_df.iloc[i-12:i]
            p_12 = np.prod(1.0 + w["Net_Return"]) - 1.0
            b_12 = np.prod(1.0 + w["Benchmark_Return"]) - 1.0
            rolling_excess_list.append(p_12 - b_12)
    rolling_excess_arr = np.array(rolling_excess_list) if rolling_excess_list else np.array([])
    rolling_pos_pct = float((rolling_excess_arr > 0).mean() * 100.0) if len(rolling_excess_arr) > 0 else 0.0
    rolling_mean_pct = float(np.mean(rolling_excess_arr) * 100.0) if len(rolling_excess_arr) > 0 else 0.0

    metrics["Liquidity_Removed_Stock_Months"] = liq_removed_candidates
    metrics["Rolling_12M_Windows_Count"] = len(rolling_excess_arr)
    metrics["Rolling_12M_Positive_Excess_Pct"] = rolling_pos_pct
    metrics["Rolling_12M_Mean_Excess_Pct"] = rolling_mean_pct

    # Limitations & Disclaimers
    limitations = [
        "Survivorship Bias: Current Nifty 500 constituents are used historically, omitting delisted or demoted stocks.",
        "Overfitting & Regime Shifts: Statistical models fit historical regimes that may shift during unprecedented macro events.",
        "Slippage & Market Impact: Real execution may experience price impact beyond the modeled 0.4% round-trip friction.",
        "Educational Tool Only: Past performance is simulated and is not indicative of future returns.",
    ]

    return {
        "status": "SUCCESS",
        "metrics": metrics,
        "monthly_performance": perf_df,
        "yearly_performance": yearly_df,
        "limitations": limitations,
        "portfolio_equity": portfolio_equity,
        "benchmark_equity": benchmark_equity,
    }
