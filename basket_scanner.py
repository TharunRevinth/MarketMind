"""
MarketMind - Basket Scanner Engine
==================================
Scans the liquid Nifty 500 universe, computes latest technical features,
scores each stock using the trained walk-forward model, applies risk tiering,
and builds an equal-weighted 15-stock basket with stop-loss and position sizing.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf

import basket_universe as bu
import basket_features as bf
import basket_risk as br
from basket_model import load_basket_model


def build_live_basket(
    total_capital: float = 1_000_000.0,
    weight_cap_pct: float = 6.67,
    risk_tier: str = "Balanced",
    min_turnover: float = bu.DEFAULT_LIQUIDITY_MIN_TURNOVER,
    basket_size: int = 15,
    stop_loss_atr_mult: float = 2.0,
    symbols_subset: list = None,
    momentum_variant: str = "Mom_12_1",
    use_regime_filter: bool = False,
    regime_cash_fraction: float = 0.5
) -> dict:
    """
    Executes the complete basket construction pipeline:
    1. Loads Nifty 500 universe and benchmark.
    2. Downloads / updates price history from local cache.
    3. Filters for liquidity (min turnover).
    4. Computes latest technical indicators.
    5. Scores stocks with the chosen strategy / momentum variant.
    6. Filters by chosen risk tier and ranks top 15.
    7. Computes position sizes, stop losses, and regime-based cash controls.
    
    Returns:
        dict: Basket DataFrame, allocation summary, scored candidate records, and regime alert.
    """
    model_bundle = load_basket_model()
    if model_bundle is None:
        model_bundle = {
            "active_strategy": "MOMENTUM_12_1",
            "active_strategy_name": "12-1 Month Momentum Baseline (Robust Default)",
            "feature_names": bf.FEATURE_COLUMNS,
            "model_name": "Momentum Baseline"
        }

    model = model_bundle.get("model")
    scaler = model_bundle.get("scaler")
    feature_names = model_bundle.get("feature_names", bf.FEATURE_COLUMNS)
    model_name = model_bundle.get("model_name", "Momentum Baseline")

    # 1. Universe & Benchmark
    try:
        symbols_df = bu.get_nifty500_symbols()
        bench_df = bu.get_benchmark_history(period="2y")
    except Exception as exc:
        return {"status": "ERROR", "message": f"Failed to load universe or benchmark: {exc}"}

    if symbols_subset:
        target_symbols = [s if ".NS" in s else f"{s}.NS" for s in symbols_subset]
    else:
        target_symbols = symbols_df["Yahoo_Symbol"].tolist()

    # Create symbol -> company name mapping
    name_map = {}
    industry_map = {}
    for _, row in symbols_df.iterrows():
        y_sym = row.get("Yahoo_Symbol", "")
        name_map[y_sym] = row.get("Company Name", y_sym)
        industry_map[y_sym] = row.get("Industry", "Diversified")

    # 2. Download / Cache Prices
    price_dict = bu.download_universe_prices(target_symbols, period="2y")

    # 3. Liquidity Filter
    liquid_prices, liq_stats = bu.apply_liquidity_filter(price_dict, min_avg_turnover=min_turnover)
    if len(liquid_prices) < basket_size:
        liquid_prices = price_dict

    # Check Market Regime (Nifty 500 vs 200-day SMA)
    bench_df = bench_df.copy()
    bench_df["SMA200"] = bench_df["Close"].rolling(200, min_periods=50).mean()
    latest_bench_close = float(bench_df["Close"].iloc[-1])
    latest_bench_sma200 = float(bench_df["SMA200"].iloc[-1]) if pd.notna(bench_df["SMA200"].iloc[-1]) else latest_bench_close
    is_bear_regime = bool(latest_bench_close < latest_bench_sma200)

    # 4. Feature Computation (Latest available bar)
    scored_rows = []

    for sym, df_s in liquid_prices.items():
        try:
            feat_df = bf.compute_stock_features(df_s, bench_df)
            if feat_df.empty or len(feat_df) < 50:
                continue

            latest_row = feat_df.iloc[-1]
            if latest_row[feature_names].isna().any():
                continue

            last_close = latest_row["Close"]
            atr_val = latest_row.get("ATR", last_close * 0.02)
            if last_close <= 0 or np.isnan(last_close):
                continue

            # Feature vector for model scoring
            prob_beat = 0.50
            if model is not None and scaler is not None:
                try:
                    X_raw = latest_row[feature_names].values.reshape(1, -1)
                    X_scaled = scaler.transform(X_raw)
                    prob_beat = float(model.predict_proba(X_scaled)[0, 1])
                except Exception:
                    prob_beat = 0.50

            record = {
                "Symbol": sym,
                "Company": name_map.get(sym, sym),
                "Industry": industry_map.get(sym, "General"),
                "Current_Price": last_close,
                "ATR": atr_val,
                "Prob_Beat_Index": prob_beat,
                "Prob_Beat_Pct": prob_beat * 100.0,
            }
            # Append all computed indicator columns for risk tiering, sorting, and display
            for col in feat_df.columns:
                if col not in record:
                    record[col] = latest_row[col]

            scored_rows.append(record)
        except Exception:
            continue

    if not scored_rows:
        return {"status": "ERROR", "message": "No valid stocks could be scored with features."}

    all_scored_df = pd.DataFrame(scored_rows)

    # 5. Risk Tier Filtering
    cohort_df = br.filter_by_risk_tier(all_scored_df, risk_tier=risk_tier)

    active_strategy = model_bundle.get("active_strategy", "MOMENTUM_12_1")
    active_strategy_name = model_bundle.get("active_strategy_name", "12-1 Month Momentum Baseline (Robust Default)")
    active_rationale = model_bundle.get("active_rationale", "")
    ml_beats_momentum = model_bundle.get("ml_beats_momentum", False)

    # 6. Rank Top N Stocks based on active strategy / selected momentum variant
    if active_strategy == "ML_MODEL":
        ranking_col = "Prob_Beat_Index"
    else:
        # Use selected variant if present in columns, else default Mom_12_1
        ranking_col = momentum_variant if momentum_variant in cohort_df.columns else "Mom_12_1"

    cohort_df.sort_values(by=ranking_col, ascending=False, inplace=True)
    top_picks = cohort_df.head(basket_size).copy().reset_index(drop=True)

    # 7. Sizing & Risk Controls
    # If Market Regime Filter is active and index is below 200 SMA, shift regime_cash_fraction to cash
    effective_capital = total_capital
    forced_cash_weight = 0.0
    if use_regime_filter and is_bear_regime:
        forced_cash_weight = regime_cash_fraction
        effective_capital = total_capital * (1.0 - forced_cash_weight)

    # Equal weight capped at weight_cap_pct across the equity allocation
    cap_weight = min(weight_cap_pct / 100.0, 1.0 / len(top_picks))
    actual_weights = [cap_weight * (1.0 - forced_cash_weight)] * len(top_picks)

    top_picks["Weight_Pct"] = [w * 100.0 for w in actual_weights]
    top_picks["Allocated_Capital"] = (top_picks["Weight_Pct"] / 100.0) * total_capital
    top_picks["Shares_Qty"] = (top_picks["Allocated_Capital"] / top_picks["Current_Price"]).astype(int)
    top_picks["Actual_Allocation"] = top_picks["Shares_Qty"] * top_picks["Current_Price"]

    # Suggested Stop Loss: Entry - 2 * ATR
    top_picks["Stop_Loss"] = top_picks["Current_Price"] - (stop_loss_atr_mult * top_picks["ATR"])
    top_picks["Stop_Loss_Pct"] = ((top_picks["Stop_Loss"] / top_picks["Current_Price"]) - 1.0) * 100.0

    # Human-readable reasons
    top_picks["Key_Reasons"] = top_picks.apply(br.generate_technical_reasons, axis=1)

    # Basket level summary statistics
    basket_beta = float(top_picks["Beta_63D"].mean())
    basket_vol = float(top_picks["ATR_Pct"].mean() * 100.0)
    avg_prob = float(top_picks["Prob_Beat_Pct"].mean())
    actual_capital_invested = float(top_picks["Actual_Allocation"].sum())
    cash_reserve = float(total_capital - actual_capital_invested)

    summary = {
        "Risk_Tier": risk_tier,
        "Total_Capital": total_capital,
        "Invested_Capital": actual_capital_invested,
        "Cash_Reserve": cash_reserve,
        "Cash_Reserve_Pct": (cash_reserve / total_capital) * 100.0,
        "Basket_Beta": basket_beta,
        "Basket_Avg_ATR_Pct": basket_vol,
        "Avg_Model_Probability": avg_prob,
        "Model_Name": model_name,
        "Active_Strategy": active_strategy,
        "Active_Strategy_Name": active_strategy_name,
        "Active_Rationale": active_rationale,
        "ML_Beats_Momentum": ml_beats_momentum,
        "Benchmark_Price": latest_bench_close,
        "Benchmark_SMA200": latest_bench_sma200,
        "Is_Bear_Regime": is_bear_regime,
        "Regime_Filter_Active": use_regime_filter,
        "Forced_Cash_Fraction": forced_cash_weight,
        "Stock_Count": len(top_picks),
    }

    return {
        "status": "SUCCESS",
        "basket_df": top_picks,
        "summary": summary,
        "all_scored_df": all_scored_df,
        "benchmark_price": latest_bench_close,
        "model_name": model_name,
        "active_strategy_name": active_strategy_name,
    }
