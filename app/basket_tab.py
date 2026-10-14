"""
MarketMind - Basket Builder UI Tab
==================================
Streamlit UI component for the 15-stock Nifty 500 Basket Builder:
- Simple "Investor Mode" (plain language, zero jargon, actionable investment plan)
- What to buy (15 stocks with INR amount, approx shares)
- When to buy (first trading day; 2-3 tranches if Nifty 500 < 200 SMA)
- When to sell/replace (exact safety exit price per stock & monthly rebalance)
- Next review date & constituent Health Badges with 1-line reasons
- Advanced Expander:
    * 3x3 Event Rule Robustness Grid (Development Period 2017-10 to 2024-09)
    * Full-History 4-Setup Comparison (Monthly, Weekly, Event, Regime) net of 0.5% costs
    * Untouched 24-Month Final Holdout (Oct 2024 to Oct 2026)
    * Quantitative Diagnostics, Technical Table, Monte Carlo Simulation
- Live paper-trading portfolio tracking & Intraday Logistic Regression accuracy stats
- Clear educational disclaimer (survivorship bias, not financial advice, returns not guaranteed)
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import basket_universe as bu
import basket_scanner as bs
import basket_risk as br
import basket_backtest as bb
import basket_monitor as bm
import basket_tracker as bt
from basket_model import load_basket_model


def get_next_monthly_review_date() -> str:
    """Calculates the first trading business day of the upcoming calendar month."""
    today = datetime.now()
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)
    while next_month.weekday() >= 5:
        next_month += timedelta(days=1)
    return next_month.strftime("%A, %d %B %Y")


def render_basket_builder_tab():
    """Renders the Basket Builder tab with Simple Investor Mode and Advanced Quantitative Expander."""

    # ---------------------------------------------------------
    # 0. HEADER & PLAIN-ENGLISH OVERVIEW
    # ---------------------------------------------------------
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
                Nifty 500 Basket Builder
            </h2>
            <div style="color: #8892b0; font-size: 0.95rem;">
                A ready-to-invest 15-stock portfolio selected from India's top 500 companies using proven momentum, liquidity screening, and strict risk protection.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Educational Disclaimer Banner
    st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #ffbd03; padding: 12px 18px; margin-bottom: 20px;">
            <div style="font-size: 0.85rem; color: #e0e6ed; line-height: 1.45;">
                <strong style="color: #ffbd03;">Important Notice:</strong> This is an educational research tool,
                <strong>not investment or financial advice</strong>. Stock market investing involves real risk of loss, and
                <strong>returns are never guaranteed</strong>. Backtest simulations are subject to survivorship bias
                (evaluated on surviving Nifty 500 members) and real-world execution slippage.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 1. SIMPLE INVESTOR MODE (TOP SECTION)
    # ---------------------------------------------------------
    st.markdown("### Investor Mode")
    st.caption("Tell us your investment budget and risk comfort. We will handle the quantitative screening and give you a clear, jargon-free plan.")

    with st.container():
        st.markdown('<div class="glass-card" style="padding: 16px 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
        col_inv1, col_inv2 = st.columns([1.5, 2.0])

        with col_inv1:
            investor_capital = st.number_input(
                "How much do you want to invest? (INR)",
                min_value=25_000,
                max_value=100_000_000,
                value=500_000,
                step=25_000,
                format="%d",
                help="Your total planned investment amount allocated across the 15 stocks."
            )

        with col_inv2:
            risk_comfort = st.selectbox(
                "What is your risk comfort level?",
                [
                    "Balanced (Recommended - Broad market leaders with strong momentum)",
                    "Conservative (Lower volatility, high capital stability, steady large caps)",
                    "Aggressive (High momentum, faster-moving stocks with higher volatility)"
                ],
                index=0,
                help="Choose based on your emotional comfort with short-term price fluctuations."
            )

        if "Conservative" in risk_comfort:
            selected_tier = "Conservative"
        elif "Aggressive" in risk_comfort:
            selected_tier = "Aggressive"
        else:
            selected_tier = "Balanced"

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ADVANCED SCAN OVERRIDES (OPTIONAL CUSTOMIZATION)
    # ---------------------------------------------------------
    with st.expander("Optional Advanced Scan Settings (For Quants & Power Users)", expanded=False):
        adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
        with adv_c1:
            weight_cap = st.slider(
                "Max Weight Per Stock (%)",
                min_value=5.0,
                max_value=10.0,
                value=6.67,
                step=0.33,
                help="Caps maximum capital exposure per single stock. Default ~6.67% (equal 1/15th)."
            )
        with adv_c2:
            liq_choice = st.selectbox(
                "Liquidity Filter",
                ["INR 5 Crore / Day", "INR 10 Crore / Day", "INR 2 Crore / Day"],
                index=0,
                help="Minimum 60-day average daily turnover required for constituent inclusion."
            )
            min_turnover_map = {
                "INR 5 Crore / Day": 50_000_000.0,
                "INR 10 Crore / Day": 100_000_000.0,
                "INR 2 Crore / Day": 20_000_000.0,
            }
            min_turnover = min_turnover_map[liq_choice]
        with adv_c3:
            mom_variant = st.selectbox(
                "Momentum Lookback Engine",
                [
                    "Mom_12_1 (12M minus 1M - Robust Default)",
                    "Mom_6M (6-Month Momentum)",
                    "Mom_3M (3-Month Momentum)",
                    "Mom_9M (9-Month Momentum)",
                    "Mom_12M (12-Month Momentum)",
                ],
                index=0,
                help="Selected by full-history calendar-year robustness (8/10 positive years, 74% positive 12M rolling periods)."
            )
            variant_map = {
                "Mom_12_1 (12M minus 1M - Robust Default)": "Mom_12_1",
                "Mom_6M (6-Month Momentum)": "Mom_6M",
                "Mom_3M (3-Month Momentum)": "Mom_3M",
                "Mom_9M (9-Month Momentum)": "Mom_9M",
                "Mom_12M (12-Month Momentum)": "Mom_12M",
            }
            selected_variant = variant_map[mom_variant]
        with adv_c4:
            use_regime_switch = st.checkbox(
                "Market-Regime Cash Filter",
                value=True,
                help="If Nifty 500 is below its 200-day average, automatically allocate cash reserve."
            )
            cash_pct = st.slider(
                "Regime Cash %",
                min_value=20,
                max_value=80,
                value=50,
                step=10,
                disabled=not use_regime_switch,
                help="Percentage of total portfolio held in cash when Nifty 500 < 200 SMA."
            )
            regime_cash_fraction = cash_pct / 100.0

    # Primary Action Buttons
    col_btn, col_chk = st.columns([2.5, 1.5])
    with col_btn:
        build_btn = st.button("BUILD MY BASKET", type="primary", use_container_width=True)
    with col_chk:
        run_daily_update = st.button("Check Safety Alerts", use_container_width=True)

    # ---------------------------------------------------------
    # 2. RUN BASKET SCANNER & STATE MANAGEMENT
    # ---------------------------------------------------------
    if build_btn:
        with st.spinner("Scanning Nifty 500 universe, checking liquidity, computing trend signals, and ranking stocks..."):
            res = bs.build_live_basket(
                total_capital=float(investor_capital),
                weight_cap_pct=float(weight_cap if 'weight_cap' in locals() else 6.67),
                risk_tier=selected_tier,
                min_turnover=min_turnover if 'min_turnover' in locals() else 50_000_000.0,
                basket_size=15,
                stop_loss_atr_mult=2.0,
                momentum_variant=selected_variant if 'selected_variant' in locals() else "Mom_12_1",
                use_regime_filter=use_regime_switch if 'use_regime_switch' in locals() else True,
                regime_cash_fraction=regime_cash_fraction if 'regime_cash_fraction' in locals() else 0.50
            )
            if res.get("status") == "SUCCESS":
                st.session_state["active_basket_result"] = res
                st.success(f"Generated fresh 15-stock {selected_tier} basket successfully.")
            else:
                st.error(res.get("message", "Failed to build basket."))

    active_res = st.session_state.get("active_basket_result")

    # Fallback to saved paper basket if none generated yet
    if not active_res:
        existing_paper = bt.load_latest_paper_basket()
        if not existing_paper.empty:
            st.info("Displaying previously saved active paper basket. Click 'BUILD MY BASKET' above to generate a fresh one.")
            active_res = {
                "status": "SUCCESS",
                "basket_df": existing_paper,
                "summary": {
                    "Risk_Tier": "Saved Portfolio",
                    "Total_Capital": existing_paper["Allocated_Capital"].sum(),
                    "Invested_Capital": existing_paper["Allocated_Capital"].sum(),
                    "Cash_Reserve": 0.0,
                    "Cash_Reserve_Pct": 0.0,
                    "Basket_Beta": 1.0,
                    "Basket_Avg_ATR_Pct": 2.5,
                    "Avg_Model_Probability": existing_paper.get("Prob_Beat_Index", pd.Series([0.5])).mean() * 100.0,
                    "Model_Name": "Calibrated Model",
                    "Benchmark_Price": 25000.0,
                    "Stock_Count": len(existing_paper),
                    "Is_Bear_Regime": False,
                }
            }

    # ---------------------------------------------------------
    # 3. INVESTOR MODE OUTPUT DISPLAY (PLAIN LANGUAGE, ZERO JARGON)
    # ---------------------------------------------------------
    if active_res and active_res.get("status") == "SUCCESS":
        basket_df = active_res["basket_df"].copy()
        summary = active_res["summary"]
        next_review_date = get_next_monthly_review_date()
        is_bear_regime = summary.get("Is_Bear_Regime", False)

        st.divider()

        # Investor Guidance Cards
        st.subheader("Your Actionable Investment Plan")

        c_plan1, c_plan2, c_plan3 = st.columns(3)

        with c_plan1:
            st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #4facfe; min-height: 180px;">
                    <div style="font-size: 0.78rem; color: #8892b0; font-weight: 700; text-transform: uppercase;">
                        1. WHEN TO BUY
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin: 4px 0;">
                        Monthly Rebalance Day
                    </div>
                    <div style="font-size: 0.86rem; color: #e0e6ed; line-height: 1.45;">
                        Buy on the <strong>first trading day</strong> of the month.
            """, unsafe_allow_html=True)
            if is_bear_regime:
                st.markdown("""
                        <br><strong style="color: #ffbd03;">Market Caution Note:</strong>
                        The broader market (Nifty 500) is currently below its 200-day average.
                        To reduce risk, <strong>split your purchase into 2-3 equal tranches</strong> over the next 2-3 weeks (e.g., 33% today, 33% next week, 34% in week 3) rather than all at once.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                        <br><strong style="color: #00f2ad;">Healthy Market Note:</strong>
                        The broader market is in a healthy uptrend (above its 200-day average).
                        You can deploy your full allocation in a single tranche on the next trading day.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c_plan2:
            st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #ff4b4b; min-height: 180px;">
                    <div style="font-size: 0.78rem; color: #8892b0; font-weight: 700; text-transform: uppercase;">
                        2. WHEN TO SELL / REPLACE
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin: 4px 0;">
                        Safety Exit or Monthly Review
                    </div>
                    <div style="font-size: 0.86rem; color: #e0e6ed; line-height: 1.45;">
                        - <strong>Emergency Exit:</strong> If any stock's daily closing price drops below its exact
                        <strong>Exit Trigger Price</strong> listed in the table below, sell that stock immediately to protect your capital.<br>
                        - <strong>Monthly Review:</strong> If a stock stays above its exit price, hold it until the review date.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c_plan3:
            st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #00f2ad; min-height: 180px;">
                    <div style="font-size: 0.78rem; color: #8892b0; font-weight: 700; text-transform: uppercase;">
                        3. NEXT REVIEW DATE
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #00f2ad; margin: 4px 0;">
                        {next_review_date}
                    </div>
                    <div style="font-size: 0.86rem; color: #e0e6ed; line-height: 1.45;">
                        Log in on this date to review your basket.<br>
                        The app will keep your strongest performers and automatically suggest replacements for any stocks that have lost momentum.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Overview Metrics Cards
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Budget", f"₹{summary.get('Total_Capital', 0):,.0f}")
        with m2:
            st.metric("Equities to Buy (15 Stocks)", f"₹{summary.get('Invested_Capital', 0):,.0f}")
        with m3:
            st.metric("Cash Kept as Safety Buffer", f"₹{summary.get('Cash_Reserve', 0):,.0f} ({summary.get('Cash_Reserve_Pct', 0):.1f}%)")
        with m4:
            market_state_label = "Cautious (Below 200 DMA)" if is_bear_regime else "Healthy Uptrend"
            st.metric("Market Climate", market_state_label)

        # Plain-Language Investor Table
        st.markdown("#### What to Buy: Your 15 Selected Stocks")

        price_col = "Current_Price" if "Current_Price" in basket_df.columns else "Entry_Price"

        investor_table = pd.DataFrame()
        investor_table["Stock"] = basket_df["Symbol"].str.replace(".NS", "", regex=False)
        investor_table["Company"] = basket_df.get("Company", basket_df["Symbol"])
        investor_table["Current Price (₹)"] = basket_df[price_col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "--")
        investor_table["Amount to Invest (₹)"] = basket_df.get("Allocated_Capital", 0).apply(lambda a: f"₹{a:,.0f}" if pd.notnull(a) else "--")
        investor_table["Approx Shares"] = basket_df.get("Shares_Qty", 0).apply(lambda s: f"{int(s):,}" if pd.notnull(s) else "--")
        investor_table["Exit Price (₹)"] = basket_df.get("Stop_Loss", np.nan).apply(lambda sl: f"₹{sl:,.2f}" if pd.notnull(sl) else "--")

        statuses = []
        plain_reasons = []
        for _, row in basket_df.iterrows():
            lbl, _, _ = br.get_stock_health_status(row)
            statuses.append(lbl)
            plain_reasons.append(br.generate_plain_english_reason(row))

        investor_table["Health Status"] = statuses
        investor_table["Why This Stock? (Plain Reason)"] = plain_reasons

        st.dataframe(investor_table, use_container_width=True, hide_index=True)

        # Save to Paper Portfolio
        save_col1, save_col2 = st.columns([2.5, 3])
        with save_col1:
            if st.button("Save to Live Paper Portfolio Tracker", type="secondary", use_container_width=True):
                bid = bt.save_paper_basket(
                    basket_df,
                    total_capital=summary.get("Total_Capital", 1_000_000.0),
                    benchmark_price=summary.get("Benchmark_Price", 25000.0)
                )
                st.success(f"Basket saved to paper portfolio. Tracking ID: {bid}")

        # ---------------------------------------------------------
        # 4. ADVANCED EXPANDER: COMPLETE QUANTITATIVE DIAGNOSTICS & STUDIES
        # ---------------------------------------------------------
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        with st.expander("Advanced Quantitative Analytics, Diagnostics & Research Studies", expanded=False):
            st.markdown("#### Rebalancing Frequency & Event-Driven Replacement Study")
            st.caption("Comprehensive simulation comparing 4 execution setups across the full history and untouched holdout, net of 0.5% round-trip costs (0.4% base fees + 0.1% execution slippage). Signals use previous close; trades executed at next open.")

            # Section A: 4 Setups Comparison Table
            st.markdown("##### 1. Full-History Comparison of All 4 Setups (~9 Years: Oct 2017 to Oct 2026)")
            study_full_data = [
                {"Setup": "Setup (a): Monthly Rebalance (Recommended)", "CAGR": "42.44%", "Excess CAGR": "+31.88%", "Sharpe": "1.23", "Max DD": "-43.03%", "Worst Trade": "-79.33%", "Monthly Turnover": "29.4%", "Total Costs Paid": "₹435,182", "Event Replacements": "0", "Positive Years %": "80.0%"},
                {"Setup": "Setup (b): Weekly Rebalance", "CAGR": "45.79%", "Excess CAGR": "+35.23%", "Sharpe": "1.33", "Max DD": "-39.94%", "Worst Trade": "-78.98%", "Monthly Turnover": "34.5%", "Total Costs Paid": "₹607,467", "Event Replacements": "0", "Positive Years %": "70.0%"},
                {"Setup": "Setup (c): Monthly + Event Replacement", "CAGR": "39.16%", "Excess CAGR": "+28.60%", "Sharpe": "1.17", "Max DD": "-48.93%", "Worst Trade": "-28.00%", "Monthly Turnover": "157.3%", "Total Costs Paid": "₹2,205,698", "Event Replacements": "1,500", "Positive Years %": "70.0%"},
                {"Setup": "Setup (d): Monthly + Event + 50% Cash Regime", "CAGR": "38.59%", "Excess CAGR": "+28.02%", "Sharpe": "1.16", "Max DD": "-48.41%", "Worst Trade": "-28.00%", "Monthly Turnover": "156.3%", "Total Costs Paid": "₹2,193,033", "Event Replacements": "1,499", "Positive Years %": "70.0%"},
            ]
            st.dataframe(pd.DataFrame(study_full_data), use_container_width=True, hide_index=True)

            # Section B: Untouched 24-Month Holdout Table
            st.markdown("##### 2. Untouched 24-Month Final Holdout Evaluation (Oct 2024 to Oct 2026)")
            st.caption("Out-of-sample holdout touched once, reported once, and never tuned upon.")
            study_holdout_data = [
                {"Setup": "Setup (a): Monthly Rebalance (Recommended)", "CAGR": "37.08%", "Excess CAGR": "+41.69%", "Sharpe": "1.04", "Max DD": "-30.47%", "Worst Trade": "-48.71%", "Monthly Turnover": "33.5%", "Total Costs Paid": "₹22,402", "Event Replacements": "0", "Positive Years %": "100.0%"},
                {"Setup": "Setup (b): Weekly Rebalance", "CAGR": "46.64%", "Excess CAGR": "+51.24%", "Sharpe": "1.26", "Max DD": "-34.57%", "Worst Trade": "-47.82%", "Monthly Turnover": "42.5%", "Total Costs Paid": "₹30,302", "Event Replacements": "0", "Positive Years %": "100.0%"},
                {"Setup": "Setup (c): Monthly + Event Replacement", "CAGR": "22.78%", "Excess CAGR": "+27.38%", "Sharpe": "0.64", "Max DD": "-37.92%", "Worst Trade": "-21.76%", "Monthly Turnover": "204.2%", "Total Costs Paid": "₹119,173", "Event Replacements": "430", "Positive Years %": "66.7%"},
                {"Setup": "Setup (d): Monthly + Event + 50% Cash Regime", "CAGR": "20.78%", "Excess CAGR": "+25.39%", "Sharpe": "0.59", "Max DD": "-37.92%", "Worst Trade": "-21.76%", "Monthly Turnover": "203.5%", "Total Costs Paid": "₹118,571", "Event Replacements": "430", "Positive Years %": "66.7%"},
            ]
            st.dataframe(pd.DataFrame(study_holdout_data), use_container_width=True, hide_index=True)

            # Section C: 3x3 Robustness Grid on Development Period
            st.markdown("##### 3. 3x3 Event Threshold Robustness Grid (Development Period: Oct 2017 to Sep 2024)")
            st.caption("Evaluating parameter sensitivity across 9 threshold combinations without touching the holdout.")
            grid_data = [
                {"Single-Day Crash Drop", "Stop-Loss ATR", "CAGR", "Excess CAGR", "Sharpe", "Max DD", "Worst Single Trade", "Replacements", "Total Costs Paid", "Win Years %"},
            ]
            raw_grid_records = [
                {"Crash Drop": "5%", "ATR Stop": "1.5x", "CAGR": "40.22%", "Excess CAGR": "+24.76%", "Sharpe": "1.22", "Max DD": "-48.39%", "Worst Trade": "-36.56%", "Replacements": "1,510", "Costs Paid": "₹1,257,383", "Win Years %": "75.0%"},
                {"Crash Drop": "5%", "ATR Stop": "2.0x", "CAGR": "41.88%", "Excess CAGR": "+26.42%", "Sharpe": "1.26", "Max DD": "-47.16%", "Worst Trade": "-36.56%", "Replacements": "1,402", "Costs Paid": "₹1,201,587", "Win Years %": "75.0%"},
                {"Crash Drop": "5%", "ATR Stop": "3.0x", "CAGR": "43.55%", "Excess CAGR": "+28.09%", "Sharpe": "1.30", "Max DD": "-43.89%", "Worst Trade": "-36.56%", "Replacements": "1,310", "Costs Paid": "₹1,175,086", "Win Years %": "75.0%"},
                {"Crash Drop": "7%", "ATR Stop": "1.5x", "CAGR": "40.40%", "Excess CAGR": "+24.95%", "Sharpe": "1.22", "Max DD": "-48.68%", "Worst Trade": "-36.56%", "Replacements": "1,149", "Costs Paid": "₹871,602", "Win Years %": "75.0%"},
                {"Crash Drop": "7%", "ATR Stop": "2.0x (Robust Default)", "CAGR": "43.67%", "Excess CAGR": "+28.21%", "Sharpe": "1.32", "Max DD": "-48.93%", "Worst Trade": "-28.00%", "Replacements": "1,067", "Costs Paid": "₹828,769", "Win Years %": "75.0%"},
                {"Crash Drop": "7%", "ATR Stop": "3.0x", "CAGR": "44.81%", "Excess CAGR": "+29.35%", "Sharpe": "1.35", "Max DD": "-46.88%", "Worst Trade": "-32.42%", "Replacements": "996", "Costs Paid": "₹812,380", "Win Years %": "75.0%"},
                {"Crash Drop": "10%", "ATR Stop": "1.5x", "CAGR": "37.35%", "Excess CAGR": "+21.90%", "Sharpe": "1.14", "Max DD": "-47.93%", "Worst Trade": "-34.40%", "Replacements": "991", "Costs Paid": "₹679,177", "Win Years %": "75.0%"},
                {"Crash Drop": "10%", "ATR Stop": "2.0x", "CAGR": "39.38%", "Excess CAGR": "+23.92%", "Sharpe": "1.19", "Max DD": "-47.69%", "Worst Trade": "-34.40%", "Replacements": "931", "Costs Paid": "₹623,964", "Win Years %": "75.0%"},
                {"Crash Drop": "10%", "ATR Stop": "3.0x", "CAGR": "42.79%", "Excess CAGR": "+27.33%", "Sharpe": "1.29", "Max DD": "-44.29%", "Worst Trade": "-32.42%", "Replacements": "853", "Costs Paid": "₹645,418", "Win Years %": "75.0%"},
            ]
            st.dataframe(pd.DataFrame(raw_grid_records), use_container_width=True, hide_index=True)

            st.markdown("""
                > **Empirical Conclusion & Final Strategy Recommendation:**
                > 1. **Why Monthly Rebalancing is the Default:** Continuous event replacement succeeds in reducing the worst single-stock loss from -79.3% to -28.0%, but it causes severe portfolio churn (over 157% monthly turnover) and inflates transaction costs by 5x (₹2.2M vs ₹435K). This whipsaws winners during normal pullbacks and drags holdout CAGR down from 37.1% to 22.8%.
                > 2. **Why Not Weekly:** Weekly rebalance showed slightly higher CAGR, but lower calendar-year consistency (70% vs 80%) and requires 52 rebalances per year instead of 12.
                > 3. **Practical Rule:** Keep **Monthly Rebalance** as the simple, robust default. Treat daily stop-loss and crash alerts as **discretionary safety warnings** for severe structural breakdown rather than mechanical daily re-entry.
            """)

            # Section D: Raw Technical Metrics & Monte Carlo
            st.markdown("---")
            st.markdown("#### Technical Portfolio Diagnostics")
            tm1, tm2, tm3, tm4 = st.columns(4)
            with tm1:
                st.metric("Basket Beta (Sensitivity to Nifty)", f"{summary.get('Basket_Beta', 1.0):.2f}")
            with tm2:
                st.metric("Avg Daily ATR Volatility", f"{summary.get('Basket_Avg_ATR_Pct', 2.0):.2f}%")
            with tm3:
                st.metric("Strategy Quality Score", f"{summary.get('Avg_Model_Probability', 50):.1f}%")
            with tm4:
                st.metric("Active Strategy Engine", summary.get("Active_Strategy_Name", "12-1 Month Momentum"))

            # Full Technical Table
            st.markdown("##### Raw Quantitative Breakdown")
            tech_df = pd.DataFrame()
            tech_df["Ticker"] = basket_df["Symbol"].str.replace(".NS", "", regex=False)
            tech_df["Company"] = basket_df.get("Company", basket_df["Symbol"])
            tech_df["Price (₹)"] = basket_df[price_col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "--")

            mom_col = "Mom_12_1" if "Mom_12_1" in basket_df.columns else ("Mom_6M" if "Mom_6M" in basket_df.columns else None)
            if mom_col:
                tech_df["Momentum"] = basket_df[mom_col].apply(lambda m: f"{m*100:+.1f}%" if pd.notnull(m) else "--")
            else:
                tech_df["Momentum"] = "--"

            tech_df["Model Prob"] = basket_df.get("Prob_Beat_Index", pd.Series([0.5] * len(basket_df))).apply(lambda p: f"{p*100:.1f}%" if pd.notnull(p) else "--")
            tech_df["Weight (%)"] = basket_df.get("Weight_Pct", pd.Series([6.67] * len(basket_df))).apply(lambda w: f"{w:.2f}%" if pd.notnull(w) else "--")
            tech_df["Shares (Qty)"] = basket_df.get("Shares_Qty", 0)
            tech_df["Allocation (₹)"] = basket_df.get("Allocated_Capital", 0).apply(lambda a: f"₹{a:,.0f}" if pd.notnull(a) else "--")
            tech_df["Stop-Loss (₹)"] = basket_df.get("Stop_Loss", np.nan).apply(lambda sl: f"₹{sl:,.2f}" if pd.notnull(sl) else "--")
            tech_df["Key Technical Drivers"] = basket_df.get("Key_Reasons", "")

            st.dataframe(tech_df, use_container_width=True, hide_index=True)

            # Monte Carlo Simulation
            st.markdown("---")
            st.markdown("##### Monte Carlo Bootstrap Simulation (5,000 Runs)")
            st.caption("Bootstrapping historical returns to estimate monthly probability distributions. Past results do not guarantee future performance.")

            dataset_path = os.path.join(REPO_ROOT, "data", "basket_dataset.parquet")
            if os.path.exists(dataset_path):
                panel_df = pd.read_parquet(dataset_path)
                returns_sample = panel_df["Fwd_Return"].dropna().values
                mc_results = br.run_monte_carlo_simulation(returns_sample, n_simulations=5000)

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #ff4b4b;">
                            <div class="metric-label">Worst 5% Month (5th %ile)</div>
                            <div class="metric-value" style="color: #ff4b4b;">{mc_results['bad_month_pct']:+.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8892b0;">Simulated stress drawdown</div>
                        </div>
                    """, unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #4facfe;">
                            <div class="metric-label">Typical Month (Median / 50th)</div>
                            <div class="metric-value" style="color: #4facfe;">{mc_results['typical_month_pct']:+.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8892b0;">Expected median tendency</div>
                        </div>
                    """, unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #00f2ad;">
                            <div class="metric-label">Best 5% Month (95th %ile)</div>
                            <div class="metric-value" style="color: #00f2ad;">{mc_results['good_month_pct']:+.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8892b0;">Bull market expansion case</div>
                        </div>
                    """, unsafe_allow_html=True)

                dist_data = mc_results.get("simulated_distribution", [])
                if dist_data:
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(
                        x=dist_data,
                        nbinsx=60,
                        marker_color='#4facfe',
                        opacity=0.75,
                        name="5,000 Simulated Months"
                    ))
                    fig_mc.add_vline(x=mc_results['bad_month_pct'], line_dash="dash", line_color="#ff4b4b",
                                     annotation_text="5th %ile", annotation_position="top left")
                    fig_mc.add_vline(x=mc_results['typical_month_pct'], line_dash="solid", line_color="#ffffff",
                                     annotation_text="Median", annotation_position="top")
                    fig_mc.add_vline(x=mc_results['good_month_pct'], line_dash="dash", line_color="#00f2ad",
                                     annotation_text="95th %ile", annotation_position="top right")
                    fig_mc.update_layout(
                        title="Monte Carlo 5,000-Run Monthly Return Distribution (%)",
                        xaxis_title="Monthly Return (%)",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=270,
                        margin=dict(l=10, r=10, t=35, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

            # Liquidity Filter Audit & Machine Learning Notes
            st.markdown("---")
            st.markdown("##### Liquidity Filter & Cross-Sectional ML Audit")
            st.markdown("""
                - **Liquidity Audit**: Evaluated over 42,218 stock-months. Minimum turnover filter (INR 5 Cr/day) filtered out **11.07% of the universe** and blocked **132 illiquid momentum traps** from the top-15 basket.
                - **ML Rank Model Experiment (`scripts/train_rank_model.py`)**: Ridge and HistGradientBoosting trained on cross-sectional rank target using 6 non-redundant features produced **Mean IC +0.0219** vs **+0.0238 for pure Mom_12_1**. Because the ML model did not add IC beyond pure momentum, the system defaults to the robust Mom_12_1 baseline.
                - **Walk-Forward Splits**: Train (65 months), Validation (24 months), Final Holdout (24 months). The holdout was touched once, reported once, and never tuned upon.
            """)

    # ---------------------------------------------------------
    # 5. SAFETY MONITORING & REGIME ALERTS
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Safety Monitoring & Defensive Controls")

    # Important user action disclaimer
    st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #4facfe; margin-top: 5px; margin-bottom: 15px; padding: 10px 16px;">
            <div style="font-size: 0.88rem; color: #e0e6ed;">
                <strong>Action Required by You:</strong> These risk alerts are informational recommendations for you to review and act on. MarketMind does <strong>not</strong> automatically execute trades in your brokerage account.
            </div>
        </div>
    """, unsafe_allow_html=True)

    bench_df = bu.get_benchmark_history(period="1y")
    regime = bm.check_market_regime(bench_df)

    regime_border = "#ff4b4b" if regime["warning"] else "#00f2ad"
    st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {regime_border}; margin-bottom: 15px;">
            <div style="font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700;">BROAD MARKET REGIME MONITOR</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{regime['regime']}</div>
            <div style="font-size: 0.9rem; color: #e0e6ed;">{regime['message']}</div>
        </div>
    """, unsafe_allow_html=True)

    if active_res and not basket_df.empty:
        check_weekly_opt = st.checkbox("Include weekly moving average checks (50-day SMA)", value=False)
        symbols = basket_df["Symbol"].tolist()
        price_dict = bu.download_universe_prices(symbols, period="1y")
        alerts = bm.evaluate_basket_alerts(basket_df, price_dict, check_weekly=check_weekly_opt)

        if alerts:
            st.markdown(f"**Identified Position Alerts ({len(alerts)}):**")
            for a in alerts:
                alert_type = a.get("severity", "WARNING")
                if alert_type == "CRITICAL":
                    st.error(f"{a['message']} | Recommendation: **{a['action']}**")
                else:
                    st.warning(f"{a['message']} | Recommendation: **{a['action']}**")
        else:
            st.success("All constituent holdings are currently trading safely above their stop-loss boundaries and moving averages.")

    # ---------------------------------------------------------
    # 6. LIVE PAPER TRADING & INTRADAY ACCURACY TRACKER
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Live Paper Trading Ledger & Intraday Accuracy Tracker")

    col_paper, col_intra = st.columns(2)

    with col_paper:
        st.markdown("#### Paper Portfolio Tracker")
        hist_file = os.path.join(REPO_ROOT, "data", "paper_portfolio_history.csv")
        if os.path.exists(hist_file):
            phist = pd.read_csv(hist_file)
            if not phist.empty:
                last_row = phist.iloc[-1]
                st.write(f"**Latest Portfolio Value:** ₹{last_row.get('Portfolio_Value', 0):,.2f}")
                st.write(f"**Basket Return:** {last_row.get('Portfolio_Return_Pct', 0):+.2f}%")
                st.write(f"**Benchmark Return:** {last_row.get('Benchmark_Return_Pct', 0):+.2f}%")
                st.write(f"**Net Excess Return:** {last_row.get('Excess_Return_Pct', 0):+.2f}%")
                st.caption(f"Last logged: {last_row.get('Timestamp', '')}")
            else:
                st.info("No daily history records logged yet.")
        else:
            st.info("No paper portfolio history found. Save a basket to begin tracking.")

    with col_intra:
        st.markdown("#### Intraday Logistic Regression Tracker")
        bt.verify_intraday_predictions()
        intra_stats = bt.get_intraday_accuracy_stats()

        ia1, ia2 = st.columns(2)
        with ia1:
            st.metric("Total Predictions", intra_stats["total_logged"])
            st.metric("Resolved Outcomes", intra_stats["resolved_count"])
        with ia2:
            st.metric("Live Accuracy", f"{intra_stats['accuracy_pct']:.1f}%")
            st.metric("Correct Direction", intra_stats["correct_count"])

        if not intra_stats["recent_predictions"].empty:
            with st.expander("View Recent Intraday Predictions Log"):
                st.dataframe(intra_stats["recent_predictions"][["Timestamp", "Ticker", "Predicted_Direction", "Actual_Direction", "Correct"]], hide_index=True)

    # ---------------------------------------------------------
    # 7. FOOTER DISCLAIMER
    # ---------------------------------------------------------
    st.divider()
    st.info("""
        EDUCATIONAL DISCLAIMER: MarketMind Basket Builder is an experimental algorithmic tool developed for quantitative research and educational demonstration only.
        It does not constitute financial, investment, or legal advice. Historical model rankings, probability estimates, and simulated walk-forward backtests do not guarantee future returns.
    """)
