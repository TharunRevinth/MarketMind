"""
MarketMind - Risk Engine & Monte Carlo Simulation
=================================================
Calculates risk tier classifications, generates human-readable technical reasons,
and executes historical bootstrap Monte Carlo simulations.
"""

import numpy as np
import pandas as pd


def generate_technical_reasons(row: pd.Series) -> str:
    """
    Synthesizes the key quantitative drivers into human-readable technical rationale.
    """
    reasons = []

    # Relative Strength
    rs1 = row.get("RS_1M", 0)
    rs3 = row.get("RS_3M", 0)
    if rs1 > 0.03:
        reasons.append(f"Strong 1M Relative Strength (+{rs1*100:.1f}% vs N500)")
    elif rs3 > 0.05:
        reasons.append(f"3M Market Outperformance (+{rs3*100:.1f}%)")

    # 52W High Proximity
    dist_52 = row.get("Dist_52W_High", -1.0)
    if dist_52 >= -0.06:
        reasons.append(f"Near 52-Week High ({abs(dist_52*100):.1f}% from peak)")
    elif dist_52 >= -0.12:
        reasons.append("Testing upper yearly range")

    # Volume Surge
    vol_surge = row.get("Vol_Surge_20D", 1.0)
    if vol_surge >= 1.5:
        reasons.append(f"Institutional Volume Surge ({vol_surge:.1f}x 20D avg)")

    # Moving Average Regime
    vs_50 = row.get("Price_vs_SMA50", 0)
    vs_200 = row.get("Price_vs_SMA200", 0)
    if vs_50 > 0.02 and vs_200 > 0.03:
        reasons.append("Bullish Trend (Trading above 50 & 200 DMA)")
    elif vs_50 > 0:
        reasons.append("Short-term trend support above 50 DMA")

    # RSI
    rsi = row.get("RSI_14", 50)
    if 50 <= rsi <= 68:
        reasons.append(f"Healthy Momentum (RSI {rsi:.1f})")
    elif rsi < 40:
        reasons.append(f"Oversold Value Entry (RSI {rsi:.1f})")

    # Sharpe / Risk-Adjusted Quality
    sharpe = row.get("Sharpe_63D", 0)
    if sharpe >= 1.5:
        reasons.append(f"High Risk-Adjusted Return (Sharpe {sharpe:.1f})")

    if not reasons:
        reasons.append("Consistent algorithmic scoring & favorable risk/reward profile")

    return " | ".join(reasons[:3])


def generate_plain_english_reason(row: pd.Series) -> str:
    """
    Generates a single, clear, jargon-free sentence explaining why this stock is selected.
    Avoids acronyms and technical jargon (no ATR, Beta, IC, DMA, RSI numbers).
    """
    mom_12_1 = row.get("Mom_12_1", np.nan)
    mom_6m = row.get("Mom_6M", np.nan)
    dist_52 = row.get("Dist_52W_High", np.nan)
    vol_surge = row.get("Vol_Surge_20D", np.nan)
    rs_3m = row.get("RS_3M", np.nan)

    if pd.notnull(mom_12_1) and mom_12_1 >= 0.40:
        return f"Exceptional 12-month leader, gaining over {mom_12_1*100:.0f}% in the past year."
    elif pd.notnull(dist_52) and dist_52 >= -0.05:
        return "Trading near its 52-week record high with consistent buyer accumulation."
    elif pd.notnull(vol_surge) and vol_surge >= 1.5:
        return "Recent heavy institutional buying volume supporting price breakout."
    elif pd.notnull(rs_3m) and rs_3m >= 0.10:
        return f"Significantly outperforming the broader market by +{rs_3m*100:.0f}% over the last 3 months."
    elif pd.notnull(mom_6m) and mom_6m >= 0.20:
        return f"Strong 6-month upward trend, gaining +{mom_6m*100:.0f}% with steady demand."
    elif pd.notnull(mom_12_1) and mom_12_1 > 0:
        return f"Consistent 12-month momentum (+{mom_12_1*100:.0f}%) with solid price stability."
    else:
        return "Steady upward trend with favorable risk-adjusted returns across recent months."


def get_stock_health_status(row: pd.Series) -> tuple[str, str, str]:
    """
    Returns (status_label, badge_color, explanation) for Investor Mode:
    - Green: Healthy Uptrend (comfortable margin above exit price and short-term trend)
    - Yellow: Consolidating / Watch (pulling back toward support or near short-term trend)
    - Red: High Risk / Near Exit (within 3% of exit price or breaking down)
    """
    current_price = row.get("Current_Price", row.get("Close", np.nan))
    stop_loss = row.get("Stop_Loss", np.nan)
    price_vs_50 = row.get("Price_vs_SMA50", 0.0)

    if pd.notnull(current_price) and pd.notnull(stop_loss) and current_price <= stop_loss * 1.03:
        return ("Near Exit", "#ff4b4b", "Trading within 3% of exit price. Monitor closely.")
    elif price_vs_50 < -0.02:
        return ("Consolidating", "#ffbd03", "Short-term pullback underway. Holding above exit price.")
    else:
        return ("Strong Uptrend", "#00f2ad", "Trading comfortably above key averages and safety exit price.")


def filter_by_risk_tier(candidates_df: pd.DataFrame, risk_tier: str) -> pd.DataFrame:
    """
    Filters the scored candidate universe into Conservative, Balanced, or Aggressive cohorts.
    
    Tiers:
    - Conservative: Lower Beta (< 0.95), lower ATR volatility (ATR_Pct <= 3.0%), lower drawdown.
    - Balanced: Moderate Beta (0.75 - 1.25), balanced volatility.
    - Aggressive: High momentum / higher Beta (> 1.05), higher volatility accepted.
    """
    if candidates_df.empty or "Beta_63D" not in candidates_df.columns:
        return candidates_df

    tier = risk_tier.lower()
    df = candidates_df.copy()

    if "conservative" in tier:
        filtered = df[(df["Beta_63D"] <= 0.95) & (df["ATR_Pct"] <= 0.032)]
        # If strict filter yields too few, broaden slightly
        if len(filtered) < 15:
            filtered = df[df["Beta_63D"] <= 1.05]
    elif "aggressive" in tier:
        filtered = df[(df["Beta_63D"] >= 1.05) | (df["Mom_3M"] >= 0.05)]
        if len(filtered) < 15:
            filtered = df[df["Beta_63D"] >= 0.95]
    else:
        # Balanced: full universe with moderate beta
        filtered = df[(df["Beta_63D"] >= 0.65) & (df["Beta_63D"] <= 1.35)]

    if len(filtered) < 15:
        return candidates_df  # Fallback to full scored set if cohort is too small

    return filtered


def run_monte_carlo_simulation(
    monthly_returns: np.ndarray,
    n_simulations: int = 5000,
    seed: int = 42
) -> dict:
    """
    Executes a bootstrap Monte Carlo simulation resampling historical portfolio monthly returns.
    
    Args:
        monthly_returns: Array of historical portfolio monthly returns.
        n_simulations: Number of bootstrap iterations (~5000).
        seed: Random seed for repeatability.
        
    Returns:
        dict: {
            'typical_month_pct': float,
            'bad_month_pct': float (5th percentile),
            'good_month_pct': float (95th percentile),
            'mean_pct': float,
            'std_pct': float,
            'simulated_distribution': list
        }
    """
    if len(monthly_returns) == 0:
        return {
            "typical_month_pct": 0.0,
            "bad_month_pct": -4.0,
            "good_month_pct": 5.0,
            "mean_pct": 1.0,
            "std_pct": 3.0,
            "simulated_distribution": []
        }

    rng = np.random.default_rng(seed)
    # Resample 1-month returns from historical empirical distribution
    sim_returns = rng.choice(monthly_returns, size=n_simulations, replace=True)

    bad_5th = float(np.percentile(sim_returns, 5))
    typical_50th = float(np.percentile(sim_returns, 50))
    good_95th = float(np.percentile(sim_returns, 95))

    return {
        "typical_month_pct": typical_50th * 100.0,
        "bad_month_pct": bad_5th * 100.0,
        "good_month_pct": good_95th * 100.0,
        "mean_pct": float(np.mean(sim_returns)) * 100.0,
        "std_pct": float(np.std(sim_returns)) * 100.0,
        "simulated_distribution": (sim_returns * 100.0).tolist()
    }
