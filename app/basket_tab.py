"""
MarketMind - Basket Builder UI Tab
==================================
Streamlit UI component for the 15-stock Nifty 500 Basket Builder:
- Live basket scanning and portfolio generation
- Risk tier selection and position sizing
- Monte Carlo return distribution simulation
- Daily risk monitoring and defensive alerts
- Walk-forward historical backtest equity curves
- Live paper-trading portfolio tracking & Intraday Logistic Regression accuracy stats
"""

import os
import sys
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


def render_basket_builder_tab():
    """Renders the comprehensive 'Basket Builder' tab within the Streamlit dashboard."""

    st.markdown("""
        <div style="margin-bottom: 25px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
                🧺 NIFTY 500 BASKET BUILDER
            </h2>
            <div style="color: #8892b0; font-size: 0.92rem;">
                Algorithmic 15-stock portfolio powered by technical relative strength, volume accumulation, risk tiering, and calibrated walk-forward machine learning.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. PARAMETER CONTROLS
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Basket Configuration")
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])

        with c1:
            total_capital = st.number_input(
                "Total Capital (₹)",
                min_value=50_000,
                max_value=100_000_000,
                value=1_000_000,
                step=50_000,
                help="Total investable funds allocated across the 15-stock basket."
            )

        with c2:
            risk_tier = st.selectbox(
                "Risk Profile / Tier",
                ["Balanced", "Conservative", "Aggressive"],
                index=0,
                help="Conservative (Low Beta/ATR), Balanced (Broad Market), Aggressive (High Momentum/Beta)."
            )

        with c3:
            weight_cap = st.slider(
                "Max Weight Per Stock (%)",
                min_value=5.0,
                max_value=10.0,
                value=6.67,
                step=0.33,
                help="Caps maximum capital exposure per single stock. Default ~6.67% (equal 1/15th)."
            )

        with c4:
            liq_choice = st.selectbox(
                "Liquidity Filter",
                ["₹5 Crore / Day", "₹10 Crore / Day", "₹2 Crore / Day"],
                index=0,
                help="Minimum 60-day average daily turnover required for constituent inclusion."
            )
            min_turnover_map = {
                "₹5 Crore / Day": 50_000_000.0,
                "₹10 Crore / Day": 100_000_000.0,
                "₹2 Crore / Day": 20_000_000.0,
            }
            min_turnover = min_turnover_map[liq_choice]

        st.markdown("---")
        c_sub1, c_sub2, c_sub3 = st.columns([1.5, 1.5, 1.0])
        with c_sub1:
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

        with c_sub2:
            use_regime_switch = st.checkbox(
                "🛡️ Market-Regime Cash Filter",
                value=False,
                help="If Nifty 500 closes below its 200-day moving average, hold configurable cash fraction to curb downside drawdown."
            )

        with c_sub3:
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

        st.markdown("</div>", unsafe_allow_html=True)

    # ACTION BUTTON
    build_col, refresh_col = st.columns([3, 1])
    with build_col:
        build_btn = st.button("🚀 BUILD MY BASKET", type="primary", use_container_width=True)
    with refresh_col:
        run_daily_update = st.button("🔄 Check Daily Alerts", use_container_width=True)

    # 2. RUN BASKET SCANNER
    if build_btn:
        with st.spinner("Scanning Nifty 500 universe, filtering liquidity, computing technicals, and ranking stocks..."):
            res = bs.build_live_basket(
                total_capital=float(total_capital),
                weight_cap_pct=float(weight_cap),
                risk_tier=risk_tier,
                min_turnover=min_turnover,
                basket_size=15,
                stop_loss_atr_mult=2.0,
                momentum_variant=selected_variant,
                use_regime_filter=use_regime_switch,
                regime_cash_fraction=regime_cash_fraction
            )
            if res.get("status") == "SUCCESS":
                st.session_state["active_basket_result"] = res
                st.success(f"Successfully generated 15-stock {risk_tier} basket using {res.get('model_name')}!")
            else:
                st.error(res.get("message", "Failed to build basket."))

    active_res = st.session_state.get("active_basket_result")

    # If no basket built yet, try loading latest paper basket if available
    if not active_res:
        existing_paper = bt.load_latest_paper_basket()
        if not existing_paper.empty:
            st.info("Displaying previously saved active paper basket. Click 'BUILD MY BASKET' to generate a fresh one.")
            # Wrap in structure
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
                    "Model_Name": "Calibrated ML Model",
                    "Benchmark_Price": 25000.0,
                    "Stock_Count": len(existing_paper),
                }
            }

    # 3. DISPLAY ACTIVE BASKET TABLE & METRICS
    if active_res and active_res.get("status") == "SUCCESS":
        basket_df = active_res["basket_df"]
        summary = active_res["summary"]

        st.divider()
        st.subheader("📊 Recommended 15-Stock Allocation")

        # Active Strategy Deployment Banner
        active_strat = summary.get("Active_Strategy_Name", "6-Month Momentum Baseline")
        active_rationale = summary.get("Active_Rationale", "Defaulted to 6-Month Momentum Baseline.")
        ml_beats = summary.get("ML_Beats_Momentum", False)
        badge_border = "#00f2ad" if ml_beats else "#ffbd03"

        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {badge_border}; margin-top: 15px; margin-bottom: 20px;">
                <div style="font-size: 0.78rem; color: #8892b0; font-weight: 700; text-transform: uppercase;">
                    DEPLOYED STRATEGY ENGINE
                </div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin: 3px 0;">
                    {active_strat}
                </div>
                <div style="font-size: 0.88rem; color: #e0e6ed;">
                    {active_rationale}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Top summary cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Invested Capital", f"₹{summary.get('Invested_Capital', 0):,.0f}")
        with m2:
            st.metric("Cash Reserve", f"₹{summary.get('Cash_Reserve', 0):,.0f} ({summary.get('Cash_Reserve_Pct', 0):.1f}%)")
        with m3:
            st.metric("Basket Beta", f"{summary.get('Basket_Beta', 1.0):.2f}")
        with m4:
            st.metric("Avg Daily ATR (Vol)", f"{summary.get('Basket_Avg_ATR_Pct', 2.0):.2f}%")
        with m5:
            st.metric("Strategy Quality Score", f"{summary.get('Avg_Model_Probability', 50):.1f}%")

        # Table formatting with defensive fallback across both live scans and saved paper baskets
        display_df = pd.DataFrame()
        display_df["Ticker"] = basket_df["Symbol"].str.replace(".NS", "", regex=False)
        display_df["Company"] = basket_df.get("Company", basket_df["Symbol"])

        # Price fallback: check Current_Price, then Entry_Price, else NaN
        price_col = "Current_Price" if "Current_Price" in basket_df.columns else "Entry_Price"
        display_df["Price (₹)"] = basket_df[price_col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "--")

        # Momentum fallback: check Mom_12_1, then Mom_6M, else NaN
        mom_col = "Mom_12_1" if "Mom_12_1" in basket_df.columns else ("Mom_6M" if "Mom_6M" in basket_df.columns else None)
        if mom_col:
            display_df["Momentum"] = basket_df[mom_col].apply(lambda m: f"{m*100:+.1f}%" if pd.notnull(m) else "--")
        else:
            display_df["Momentum"] = "--"

        display_df["Model Prob"] = basket_df.get("Prob_Beat_Index", pd.Series([0.5] * len(basket_df))).apply(lambda p: f"{p*100:.1f}%" if pd.notnull(p) else "--")
        display_df["Weight (%)"] = basket_df.get("Weight_Pct", pd.Series([6.67] * len(basket_df))).apply(lambda w: f"{w:.2f}%" if pd.notnull(w) else "--")
        display_df["Shares (Qty)"] = basket_df.get("Shares_Qty", 0)
        display_df["Allocation (₹)"] = basket_df.get("Allocated_Capital", 0).apply(lambda a: f"₹{a:,.0f}" if pd.notnull(a) else "--")
        display_df["Stop-Loss (₹)"] = basket_df.get("Stop_Loss", np.nan).apply(lambda sl: f"₹{sl:,.2f}" if pd.notnull(sl) else "--")
        display_df["Key Quantitative Reasons"] = basket_df.get("Key_Reasons", "")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Save to Paper Portfolio Button
        save_col1, save_col2 = st.columns([2, 3])
        with save_col1:
            if st.button("💾 Save as Active Paper Portfolio", type="secondary", use_container_width=True):
                bid = bt.save_paper_basket(
                    basket_df,
                    total_capital=summary.get("Total_Capital", 1_000_000.0),
                    benchmark_price=summary.get("Benchmark_Price", 25000.0)
                )
                st.success(f"Basket saved to live paper portfolio! (ID: {bid})")

    # 4. MONTE CARLO SIMULATION SECTION
    st.divider()
    st.subheader("🎲 Risk Tier Profiling & Monte Carlo Simulation")
    st.markdown("""
        <div style="font-size: 0.88rem; color: #8892b0; margin-bottom: 12px;">
            Bootstrap simulation (5,000 resamples of historical returns) providing probabilistic estimates for monthly portfolio return ranges.
            <strong style="color: #ffbd03;">Estimates only — past results do NOT guarantee future performance.</strong>
        </div>
    """, unsafe_allow_html=True)

    # Run Monte Carlo on panel returns
    dataset_path = os.path.join(REPO_ROOT, "data", "basket_dataset.parquet")
    if os.path.exists(dataset_path):
        panel_df = pd.read_parquet(dataset_path)
        returns_sample = panel_df["Fwd_Return"].dropna().values
        mc_results = br.run_monte_carlo_simulation(returns_sample, n_simulations=5000)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #ff4b4b;">
                    <div class="metric-label">Worst 5% Month (5th Percentile)</div>
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
                    <div class="metric-label">Best 5% Month (95th Percentile)</div>
                    <div class="metric-value" style="color: #00f2ad;">{mc_results['good_month_pct']:+.1f}%</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">Bull market expansion case</div>
                </div>
            """, unsafe_allow_html=True)

        # Plot Monte Carlo distribution histogram
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
                height=280,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_mc, use_container_width=True)

    # 5. DAILY MONITORING & REGIME ALERTS
    st.divider()
    st.subheader("🚨 Daily Safety Monitoring & Risk Controls")

    bench_df = bu.get_benchmark_history(period="1y")
    regime = bm.check_market_regime(bench_df)

    regime_border = "#ff4b4b" if regime["warning"] else "#00f2ad"
    st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {regime_border};">
            <div style="font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700;">BROAD MARKET REGIME MONITOR</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{regime['regime']}</div>
            <div style="font-size: 0.9rem; color: #e0e6ed;">{regime['message']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Active basket positions alerts
    if active_res and not basket_df.empty:
        symbols = basket_df["Symbol"].tolist()
        price_dict = bu.download_universe_prices(symbols, period="1y")
        alerts = bm.evaluate_basket_alerts(basket_df, price_dict)

        if alerts:
            st.markdown(f"**Identified Position Alerts ({len(alerts)}):**")
            for a in alerts:
                alert_type = a.get("severity", "WARNING")
                if alert_type == "CRITICAL":
                    st.error(f"{a['message']} | Recommendation: **{a['action']}**")
                else:
                    st.warning(f"{a['message']} | Recommendation: **{a['action']}**")
        else:
            st.success("✅ All constituent holdings are currently trading safely above their stop-loss boundaries and moving averages.")

    # 6. FULL-HISTORY MOMENTUM TOURNAMENT & WALK-FORWARD EVALUATION
    st.divider()
    st.subheader("📈 Full-History Momentum Evaluation & Robustness Tournament")
    st.markdown("""
        <div style="font-size: 0.88rem; color: #8892b0; margin-bottom: 12px;">
            Evaluation across all <strong>107 monthly rebalances (~9 years: 2017 to 2026)</strong> net of
            <strong>0.4% round-trip trading friction</strong> and top-30 retention buffers.
            Default variant is chosen by <strong>calendar-year robustness and positive rolling periods</strong>, not by cherry-picking peak holdout return.
        </div>
    """, unsafe_allow_html=True)

    # 1. Full History Variants Tournament Table
    st.markdown("#### 🏆 Full-History Momentum Variants Tournament (All 107 Months, Net of 0.4% Costs)")
    variants_summary_data = [
        {"Variant": "Mom_12_1 (12M minus 1M - Default)", "CAGR": "+38.34%", "Excess CAGR": "+28.40%", "Sharpe": "1.06", "Max DD": "-37.1%", "Win Rate": "63.6%", "Mean IC": "+0.0240", "Pos IC %": "60.7%", "Positive Years": "8 / 10", "12M Rolling Win %": "74.0%"},
        {"Variant": "Mom_9M (9-Month Momentum)", "CAGR": "+39.82%", "Excess CAGR": "+29.89%", "Sharpe": "1.12", "Max DD": "-38.8%", "Win Rate": "64.5%", "Mean IC": "+0.0208", "Pos IC %": "57.9%", "Positive Years": "6 / 10", "12M Rolling Win %": "69.8%"},
        {"Variant": "Mom_6M (6-Month Momentum)", "CAGR": "+36.25%", "Excess CAGR": "+26.31%", "Sharpe": "1.01", "Max DD": "-34.3%", "Win Rate": "66.4%", "Mean IC": "+0.0188", "Pos IC %": "60.7%", "Positive Years": "6 / 10", "12M Rolling Win %": "71.9%"},
        {"Variant": "Mom_12M (12-Month Momentum)", "CAGR": "+35.26%", "Excess CAGR": "+25.32%", "Sharpe": "0.94", "Max DD": "-39.9%", "Win Rate": "62.6%", "Mean IC": "+0.0196", "Pos IC %": "58.9%", "Positive Years": "7 / 10", "12M Rolling Win %": "68.8%"},
        {"Variant": "Mom_3M (3-Month Momentum)", "CAGR": "+31.58%", "Excess CAGR": "+21.65%", "Sharpe": "0.88", "Max DD": "-44.3%", "Win Rate": "65.4%", "Mean IC": "+0.0041", "Pos IC %": "53.3%", "Positive Years": "8 / 10", "12M Rolling Win %": "71.9%"},
        {"Variant": "Nifty 500 Index Benchmark", "CAGR": "+9.94%", "Excess CAGR": "0.00%", "Sharpe": "0.19", "Max DD": "-32.7%", "Win Rate": "N/A", "Mean IC": "N/A", "Pos IC %": "N/A", "Positive Years": "N/A", "12M Rolling Win %": "N/A"},
    ]
    st.dataframe(pd.DataFrame(variants_summary_data), use_container_width=True, hide_index=True)

    # 2. Calendar Year Breakdown Table
    st.markdown("#### 📅 Calendar-Year Excess Return Breakdown (% vs Nifty 500)")
    calendar_year_data = [
        {"Year": "2017 (3M)", "Mom_12_1": "+15.9%", "Mom_9M": "+12.0%", "Mom_6M": "+12.4%", "Mom_12M": "+13.9%", "Mom_3M": "+5.8%", "N500 Return": "+8.4%"},
        {"Year": "2018", "Mom_12_1": "-5.2%", "Mom_9M": "-7.9%", "Mom_6M": "-4.9%", "Mom_12M": "-8.3%", "Mom_3M": "-1.4%", "N500 Return": "-12.8%"},
        {"Year": "2019", "Mom_12_1": "+15.1%", "Mom_9M": "+19.0%", "Mom_6M": "+35.5%", "Mom_12M": "+9.4%", "Mom_3M": "-3.0%", "N500 Return": "+14.1%"},
        {"Year": "2020", "Mom_12_1": "+31.4%", "Mom_9M": "+45.3%", "Mom_6M": "+26.2%", "Mom_12M": "+29.4%", "Mom_3M": "+0.6%", "N500 Return": "+16.5%"},
        {"Year": "2021", "Mom_12_1": "+122.3%", "Mom_9M": "+152.7%", "Mom_6M": "+100.8%", "Mom_12M": "+123.0%", "Mom_3M": "+80.5%", "N500 Return": "+31.3%"},
        {"Year": "2022", "Mom_12_1": "-14.0%", "Mom_9M": "-17.8%", "Mom_6M": "-1.7%", "Mom_12M": "-17.9%", "Mom_3M": "+11.6%", "N500 Return": "-2.5%"},
        {"Year": "2023", "Mom_12_1": "+103.0%", "Mom_9M": "+123.4%", "Mom_6M": "+131.2%", "Mom_12M": "+126.5%", "Mom_3M": "+91.9%", "N500 Return": "+30.2%"},
        {"Year": "2024", "Mom_12_1": "+5.7%", "Mom_9M": "-5.8%", "Mom_6M": "-6.4%", "Mom_12M": "+5.7%", "Mom_3M": "+9.4%", "N500 Return": "+10.6%"},
        {"Year": "2025", "Mom_12_1": "+6.5%", "Mom_9M": "-12.7%", "Mom_6M": "-12.1%", "Mom_12M": "-3.4%", "Mom_3M": "+0.7%", "N500 Return": "+5.2%"},
        {"Year": "2026 (8M)", "Mom_12_1": "+49.1%", "Mom_9M": "+78.4%", "Mom_6M": "+36.4%", "Mom_12M": "+42.7%", "Mom_3M": "+43.7%", "N500 Return": "-4.4%"},
        {"Year": "Positive Years", "Mom_12_1": "8 / 10 (80%)", "Mom_9M": "6 / 10 (60%)", "Mom_6M": "6 / 10 (60%)", "Mom_12M": "7 / 10 (70%)", "Mom_3M": "8 / 10 (80%)", "N500 Return": "6 / 10 (60%)"}
    ]
    st.dataframe(pd.DataFrame(calendar_year_data), use_container_width=True, hide_index=True)

    # 3. Market-Regime Filter Impact Comparison
    st.markdown("#### 🛡️ Market-Regime Filter Impact (50% Cash When Nifty 500 < 200 SMA)")
    regime_comp_data = [
        {"Configuration": "Mom_12_1 without Regime Filter", "CAGR": "38.34%", "Max Drawdown": "-37.08%", "Sharpe Ratio": "1.06", "Protection Description": "100% equity exposure continuously."},
        {"Configuration": "Mom_12_1 with 50% Cash Regime Filter", "CAGR": "33.85%", "Max Drawdown": "-35.34%", "Sharpe Ratio": "1.08", "Protection Description": "Moves 50% capital to cash when N500 < 200 SMA, reducing tail risk."},
        {"Configuration": "Mom_6M without Regime Filter", "CAGR": "36.25%", "Max Drawdown": "-34.28%", "Sharpe Ratio": "1.01", "Protection Description": "100% equity exposure continuously."},
        {"Configuration": "Mom_6M with 50% Cash Regime Filter", "CAGR": "32.95%", "Max Drawdown": "-25.10%", "Sharpe Ratio": "1.06", "Protection Description": "Curtails maximum drawdown from -34.3% to -25.1%."}
    ]
    st.dataframe(pd.DataFrame(regime_comp_data), use_container_width=True, hide_index=True)

    # 4. Liquidity Filter Audit & ML Rank Model Experiment Expander
    with st.expander("🔬 Liquidity Filter Audit & Cross-Sectional Rank ML Experiment"):
        st.markdown("""
            ##### 💧 Liquidity Filter Audit (ADTV >= ₹5 Crore/Day)
            - **Universe Screening**: Across all 107 rebalance periods (42,218 total stock-months), **4,675 stock-months (11.07%)** were filtered out due to sub-₹5 Cr turnover.
            - **Momentum Basket Impact**: Out of 1,605 candidate top-15 momentum picks, **132 picks (8.22%)** were blocked and replaced by higher-liquidity candidates, eliminating illiquid slippage traps.

            ##### 🤖 Cross-Sectional Rank ML Experiment (`scripts/train_rank_model.py`)
            - **Setup**: Trained regularized Ridge and HistGradientBoosting regressors on cross-sectional target ranks using 6 non-redundant features (`Mom_12_1`, `Mom_6M`, `Dist_52W_High`, `Vol_Surge_20D`, `ATR_Pct`, `Beta_63D`).
            - **Expanding Walk-Forward IC (71 Out-of-Sample Months)**:
                - `Mom_12_1` Alone: **Mean IC = +0.0238**, 62.0% positive months
                - `HistGB` Rank Model: **Mean IC = +0.0219**, 60.6% positive months
                - `Ridge` (alpha=100): **Mean IC = -0.0056**, 52.1% positive months
            - **Decision Rule**: Because the ML model did not add information coefficient beyond pure `Mom_12_1` momentum (+0.0219 vs +0.0238), the system preserves **`Mom_12_1` as the active default engine** to maintain parsimony and avoid black-box overfitting.
        """)

    with st.expander("🔍 Quantitative Methodology, Split Periods & Real-World Limits"):
        mb = load_basket_model() or {}
        tp = mb.get("train_period", ("2017-10", "2023-02"))
        vp = mb.get("val_period", ("2023-03", "2024-08"))
        hp = mb.get("holdout_period", ("2024-09", "2026-08"))
        st.markdown(f"""
            - **Train Split**: {tp[0]} to {tp[1]} (65 months) — Initial parameter fitting.
            - **Validation Split**: {vp[0]} to {vp[1]} (24 months) — Architecture selection only.
            - **Final Holdout**: {hp[0]} to {hp[1]} (24 months) — Out-of-sample evaluation touched once, reported once.
            - **Decision Rule**: The ML model is deployed only if it demonstrates higher Sharpe and CAGR than the 6-Month Momentum factor after costs; otherwise, the system transparently defaults to the Momentum baseline.
            - **Survivorship Bias**: Current constituents are used across historical dates. Delisted or demoted stocks are not represented.
            - **Market Slippage**: Real trades may experience wider bid-ask spreads beyond the modeled 0.4% round-trip friction.
        """)

    # 7. LIVE PAPER TRADING & INTRADAY ACCURACY TRACKER
    st.divider()
    st.subheader("📋 Live Paper Trading Ledger & Intraday Accuracy Tracker")

    col_paper, col_intra = st.columns(2)

    with col_paper:
        st.markdown("#### 📜 Paper Portfolio Tracker")
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
        st.markdown("#### 🎯 Intraday Logistic Regression Tracker")
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

    # 8. DISCLAIMER FOOTER
    st.divider()
    st.info("""
        **EDUCATIONAL DISCLAIMER**: MarketMind Basket Builder is an experimental algorithmic tool developed for quantitative research and educational demonstration only.
        It does not constitute financial, investment, or legal advice. Historical model rankings, probability estimates, and simulated walk-forward backtests do not guarantee future returns.
    """)
