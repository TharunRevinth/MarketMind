# MarketMind Terminal 📈

MarketMind is an algorithmic trading dashboard, quantitative basket builder, and stock analysis suite tailored for Indian equities (NSE/NIFTY) and global benchmarks. It combines machine learning ranking models, walk-forward backtesting, real-time technical indicators, risk-tiering, and live paper-trading tracking into a high-performance Streamlit terminal.

---

## Core Systems

### 1. NIFTY 500 Basket Builder (New 🧺)
An end-to-end quantitative portfolio engine that scans the liquid Nifty 500 universe and builds an optimized 15-stock equity basket:
- **Universe & Liquidity**: Fetches live constituents from the National Stock Exchange (NSE) archives with local fallback. Automatically applies a 60-day turnover liquidity filter (default $\ge$ ₹5 Crore/day) and benchmark alignment (`^CRSLDX` Nifty 500 index).
- **Strict Anti-Leakage Features**: Daily indicators computed strictly from past data (1M/3M/6M relative strength vs Nifty 500, momentum, distance from 52-week high, 20-day volume surge, RSI-14, normalized MACD histogram, distance from 50/200 DMA, ATR volatility %, rolling beta, and rolling Sharpe).
- **Walk-Forward Model Tournament**: Chronological walk-forward splits compare Logistic Regression, Gradient Boosting (`HistGradientBoosting`), and a Multi-Layer Perceptron (MLP) Neural Network on out-of-sample test months. Probability calibration ensures realistic confidence percentages.
- **Position Sizing & Risk Controls**: Equal-weighted allocations (capped at ~6.67% per stock; residual held in cash) with automatic 2 $\times$ ATR defensive stop-loss levels and transparent technical reasoning for every pick.
- **Risk Tiers & Monte Carlo Simulation**: Segregates selections into Conservative (low beta/volatility), Balanced (core market), and Aggressive (high momentum/beta) cohorts. Simulates 5,000 bootstrap iterations to project typical, bad (5th percentile), and good (95th percentile) monthly return ranges.
- **Monthly Rebalancing with Turnover Buffer**: Retains existing basket holdings if they remain within the top 30 ranks, suppressing unnecessary turnover. Deducts realistic round-trip transaction costs (0.4%) on all portfolio transitions.
- **Daily Monitoring & Regime Alerts**: Tracks daily stop-loss hits, 50/200-day moving average breakdowns, and triggers a macro market warning when the Nifty 500 drops below its 200-day SMA.
- **Live Paper Trading & Intraday Accuracy Tracker**: Maintains an ongoing ledger (`data/paper_trades.csv`) tracking live portfolio valuation against the benchmark, and continuously audits out-of-sample intraday Logistic Regression predictions for an honest live accuracy score.

### 2. Single Stock Terminal (Intraday & Macro 📊)
- **AI Trend Signals**: 5-minute candle next-bar direction predictions powered by per-ticker Logistic Regression models (`model.pkl` + `scaler.pkl`).
- **Volume & Volatility Suite**: Real-time calculations for VWAP (Volume-Weighted Average Price), OBV (On-Balance Volume money flow), ATR (Average True Range volatility & risk stop-loss), and Bollinger Bands (20-period, 2σ).
- **Multi-Timeframe Analysis**: Cross-examines macro daily trends against short-term intraday momentum (15M) for directional confluence.
- **Trader's Intelligence**: Automated news sentiment scanner identifying management instability, regulatory hurdles, or earnings surprises.
- **Investor Profiling**: Dynamic risk assessment matching user income capacity with categorized asset profiles.

### 3. Algorithmic Scanner (CLI 🖥️)
- Multi-threaded scanning engine using Python's `concurrent.futures.ThreadPoolExecutor` for batch technical screening across dozens of tickers.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd MarketMind
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Workflow (Step-by-Step Order)

Follow this order to run the data pipeline, train models, launch the dashboard, and manage daily monitoring:

### Step 1: Prepare Nifty 500 Panel Dataset
Fetches Nifty 500 constituents, downloads historical daily prices with disk caching, computes technical indicators without lookahead, and builds the walk-forward monthly dataset (`data/basket_dataset.parquet`).

- **Quick sample run (~50 stocks)**:
  ```bash
  python3 scripts/prepare_basket_data.py --sample 50
  ```
- **Full Nifty 500 universe run**:
  ```bash
  python3 scripts/prepare_basket_data.py
  ```

### Step 2: Full-History Momentum Robustness & Model Tournament
Runs the comprehensive evaluation across all historical dates and splits:
- **Default Strategy**: `Mom_12_1` (12-month return skipping the most recent 1-month reversal period). Selected strictly by calendar-year robustness (8/10 positive excess return years, 74.0% positive 12M rolling periods, +0.0240 Spearman IC across all 107 rebalance months).
- **Three-Way Chronological Split**: 65 months Train -> 24 months Validation -> 24 months Final Holdout (touched once, reported once).
- **ML Decision Gate**: The classification ML model is only deployed if it beats the momentum baseline on the final holdout after 0.4% costs. Since classification models showed no out-of-sample edge (AUC 0.475, IC -0.045), the system transparently defaults to `Mom_12_1`.

```bash
python3 scripts/train_basket_model.py
```

### Optional Step 2b: Cross-Sectional Rank ML Experiment
To run the specialized cross-sectional rank machine learning tournament (Ridge vs HistGradientBoosting on 6 non-redundant features using expanding-window walk-forward IC):

```bash
python3 scripts/train_rank_model.py
```

### Step 3: Launch the Interactive Dashboard
Opens the Streamlit terminal featuring both the **Single Stock Terminal** and the **Basket Builder** tabs:

```bash
streamlit run app/dashboard.py
```

### Step 4: Run Daily Tracking & Risk Auditing
Executes daily maintenance: updates paper trading portfolio valuation against the Nifty 500, scans for defensive stop-loss / moving average alerts, and verifies intraday predictions against actual candles:

```bash
python3 scripts/update_tracker.py
```

### Optional: Train Intraday Logistic Regression Models (Existing Feature)
To retrain the 5-minute single-stock intraday models:
```bash
python3 -m scripts.train_all
```

### Optional: Run CLI Batch Scanner
```bash
python3 main.py RELIANCE.NS TCS.NS INFY.NS
```

---

## Empirical Tournament Results (Full 107 Months, Net of 0.4% Costs)

### Momentum Variants Comparison (2017 to 2026)
| Variant | Lookback Horizon | Full-History CAGR | Excess CAGR vs N500 | Sharpe | Max DD | Win Rate | Mean IC | Positive Years | 12M Rolling Win % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Mom_12_1`** | **12M skipping 1M (Default)** | **+38.34%** | **+28.40%** | **1.06** | **-37.1%** | **63.6%** | **+0.0240** | **8 / 10** | **74.0%** |
| `Mom_9M` | 9-Month Return | +39.82% | +29.89% | 1.12 | -38.8% | 64.5% | +0.0208 | 6 / 10 | 69.8% |
| `Mom_6M` | 6-Month Return | +36.25% | +26.31% | 1.01 | -34.3% | 66.4% | +0.0188 | 6 / 10 | 71.9% |
| `Mom_12M` | 12-Month Return | +35.26% | +25.32% | 0.94 | -39.9% | 62.6% | +0.0196 | 7 / 10 | 68.8% |
| `Mom_3M` | 3-Month Return | +31.58% | +21.65% | 0.88 | -44.3% | 65.4% | +0.0041 | 8 / 10 | 71.9% |
| **Nifty 500** | Benchmark (`^CRSLDX`) | +9.94% | 0.00% | 0.19 | -32.7% | N/A | N/A | 6 / 10 | N/A |

### Calendar-Year Excess Return (% vs Nifty 500 Index)
| Year | `Mom_12_1` (Default) | `Mom_9M` | `Mom_6M` | `Mom_12M` | `Mom_3M` | Nifty 500 Return |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2017 (3M)** | +15.9% | +12.0% | +12.4% | +13.9% | +5.8% | +8.4% |
| **2018** | **-5.2%** | -7.9% | -4.9% | -8.3% | -1.4% | -12.8% |
| **2019** | +15.1% | +19.0% | +35.5% | +9.4% | -3.0% | +14.1% |
| **2020** | +31.4% | +45.3% | +26.2% | +29.4% | +0.6% | +16.5% |
| **2021** | +122.3% | +152.7% | +100.8% | +123.0% | +80.5% | +31.3% |
| **2022** | **-14.0%** | -17.8% | -1.7% | -17.9% | +11.6% | -2.5% |
| **2023** | +103.0% | +123.4% | +131.2% | +126.5% | +91.9% | +30.2% |
| **2024** | +5.7% | -5.8% | -6.4% | +5.7% | +9.4% | +10.6% |
| **2025** | +6.5% | -12.7% | -12.1% | -3.4% | +0.7% | +5.2% |
| **2026 (8M)**| +49.1% | +78.4% | +36.4% | +42.7% | +43.7% | -4.4% |
| **Pos Yrs** | **8 / 10 (80%)** | 6 / 10 (60%) | 6 / 10 (60%) | 7 / 10 (70%) | 8 / 10 (80%) | 6 / 10 (60%) |

> **Honest Assessment on Robustness**: Momentum is **not uniform across every single year**. In severe market bear transitions and choppy mean-reverting regimes (notably **2018** and **2022**), momentum experienced negative excess returns (-5.2% and -14.0% respectively) as market leaders rotated abruptly. However, over multi-year horizons, `Mom_12_1` demonstrated the highest calendar consistency (positive excess returns in 8 of 10 years and positive excess returns in 74.0% of rolling 12-month periods).

### Market-Regime Filter Impact (50% Cash When Nifty 500 < 200 SMA)
| Strategy Configuration | CAGR | Max Drawdown | Sharpe Ratio | Tail Risk Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `Mom_12_1` without Regime Filter | **38.34%** | -37.08% | 1.06 | Full market participation during all pullbacks. |
| `Mom_12_1` with 50% Cash Regime Filter | **33.85%** | **-35.34%** | **1.08** | Improves risk-adjusted return (Sharpe 1.08) while mitigating bear drawdowns. |
| `Mom_6M` without Regime Filter | **36.25%** | -34.28% | 1.01 | Higher volatility during sharp corrections. |
| `Mom_6M` with 50% Cash Regime Filter | **32.95%** | **-25.10%** | **1.06** | Curbs maximum peak-to-trough drawdown from -34.3% down to -25.1%. |

### Liquidity Filter Audit (ADTV >= ₹5 Crore/Day)
- **Universe Filter**: Filtered out **4,675 / 42,218 (11.07%)** stock-months across the 107 rebalance periods due to low trading turnover.
- **Basket Filter**: Removed and replaced **132 / 1,605 (8.22%)** raw top-15 momentum picks that fell below the ₹5 Cr/day threshold, preventing illiquid slippage in live trading.

---

## Project Structure

```
MarketMind/
├── app/
│   ├── dashboard.py           # Main Streamlit dashboard (Single Stock + Basket Builder tabs)
│   └── basket_tab.py          # Basket Builder UI component (with variants, regime filter, and tracking)
├── basket_universe.py         # NSE Nifty 500 constituents, price caching, and liquidity filter
├── basket_features.py         # Indicator calculations (past-only) and monthly panel dataset generator
├── basket_model.py            # Walk-forward model tournament, calibration, and persistence
├── basket_scanner.py          # Live basket builder, equal-weight allocation, stop losses, and regime sizing
├── basket_risk.py             # Risk tier filtering, technical reasoning, and Monte Carlo engine
├── basket_backtest.py         # Walk-forward backtester with turnover buffer and 0.4% friction
├── basket_monitor.py          # Daily defensive monitoring (stops, 50/200 DMA, market regime)
├── basket_tracker.py          # Paper trading portfolio ledger and intraday prediction auditor
├── analyzer.py                # Single-stock technical indicator analyzer (VWAP, OBV, ATR, BB)
├── scanner.py                 # Multi-threaded batch stock scanner
├── main.py                    # CLI entry point for batch scanner
├── scripts/
│   ├── prepare_basket_data.py # Script: Downloads data and creates data/basket_dataset.parquet
│   ├── train_basket_model.py  # Script: Trains and selects the best basket model
│   ├── train_rank_model.py    # Script: Cross-sectional rank ML experiment (Ridge vs HistGB)
│   ├── update_tracker.py      # Script: Updates paper trading valuations and audits predictions
│   ├── train_model.py         # Script: Trains intraday 5m Logistic Regression model
│   └── train_all.py           # Script: Batch trains intraday models for top Indian stocks
├── data/                      # Cached datasets, paper trade ledgers, and categories
├── models/                    # Saved model bundles (.pkl) and scalers
└── requirements.txt           # Python dependencies
```

---

## Quantitative Methodology & Risk Disclaimers

### Anti-Leakage Protocol
All feature engineering is strictly computed using information available at or before rebalance time $T$. Forward price returns from $T$ to $T+20$ are used exclusively as target labels during model evaluation and are never fed into feature transformations or scalers. Scalers are fitted solely on historical training splits.

### Walk-Forward Model Selection Rule
Candidate models are evaluated across chronological out-of-sample test months. Models are judged on:
1. Top-15 Precision (% of top picks outperforming the Nifty 500).
2. Top-15 Average Excess Return over the benchmark.
3. Test Accuracy, ROC-AUC, and Spearman Rank Correlation (IC).
If the classification or regression ML model does not clearly add IC or Sharpe beyond simpler transparent baselines after costs, the simpler baseline (`Mom_12_1`) is selected to eliminate overfitting risk.

### Known Limitations
- **Survivorship Bias**: Historical backtests use current Nifty 500 constituents and do not include historical constituents that were delisted or relegated.
- **Execution Slippage**: Real executions may experience wider spreads or slippage beyond the 0.4% modeled round-trip friction.
- **Regime Shifts**: Momentum factors can suffer severe whipsaws during sudden market inflection points and sharp macro drawdowns.

---

## Disclaimer
**MarketMind is strictly for educational and research purposes.** Stock trading involves significant risk of capital loss. No signals or backtests generated by this system represent guaranteed returns or financial advice. Always consult a certified financial advisor before committing capital.

---
*System Version: 2.1.0 | Built with Python, Streamlit, Scikit-Learn, and Plotly*
