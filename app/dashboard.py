import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# --- LOAD DATA ---
@st.cache_data
def load_categories():
    try:
        return pd.read_csv("data/indian_stock_categories.csv")
    except:
        return pd.DataFrame(columns=["Ticker", "Category", "Risk_Level", "Capital_Safety"])

@st.cache_data(ttl=3600)
def get_trader_intelligence(ticker):
    """Professional-grade news analysis for bearish risks and trade moves."""
    try:
        yf_ticker = ticker if ".NS" in ticker else ticker + ".NS"
        news = yf.Ticker(yf_ticker).news
        if not news:
            return "No recent intelligence available.", "NEUTRAL", "Wait for news."
        
        bearish_triggers = {
            'regulatory': 'Regulatory hurdles or legal issues detected.',
            'miss': 'Earnings miss or growth slowdown concerns.',
            'exit': 'Key leadership exit or management instability.',
            'debt': 'Concerns regarding debt levels or liquidity.',
            'downgrade': 'Brokerage downgrade or lowered price targets.',
            'fraud': 'Allegations of misconduct or audit issues.',
            'slump': 'Sector-wide slump or macro headwinds.',
            'selloff': 'Heavy insider selling or institutional exit.'
        }
        
        found_risks = []
        sentiment_score = 0
        
        for item in news:
            # Handle nested yfinance news structure
            content = item.get('content', {})
            title = content.get('title', '').lower()
            
            for trigger, desc in bearish_triggers.items():
                if trigger in title:
                    if desc not in found_risks:
                        found_risks.append(desc)
                    sentiment_score -= 2 # Heavy weight on bearish news
            
            # General positive keywords to balance
            if any(w in title for w in ['growth', 'profit', 'deal', 'expansion']):
                sentiment_score += 1

        if sentiment_score <= -3:
            verdict = "BEARISH 📉"
            move = "SHORT / AVOID"
            reason = "Significant bearish triggers detected in recent headlines. High risk of correction."
        elif sentiment_score < 0:
            verdict = "CAUTIOUS ⚠️"
            move = "WAIT / PROTECT"
            reason = "Mixed sentiment with some negative undertones. Monitor support levels."
        else:
            verdict = "BULLISH 🚀"
            move = "LONG / ACCUMULATE"
            reason = "News sentiment is largely positive or stable. No immediate red flags."

        if not found_risks:
            risk_summary = "No specific bearish triggers identified in recent news."
        else:
            risk_summary = "\n".join([f"• {r}" for r in found_risks])

        return risk_summary, verdict, move, reason
    except:
        return "Intelligence fetch failed.", "NEUTRAL", "HOLD", "System error."

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Simple keyword-based sentiment analysis on ticker news."""
    try:
        # Standardize for yfinance
        yf_ticker = ticker if ".NS" in ticker else ticker + ".NS"
        news = yf.Ticker(yf_ticker).news
        if not news:
            return 0, "Neutral"
        
        pos_words = ['growth', 'profit', 'up', 'surge', 'buy', 'positive', 'win', 'deal', 'expansion', 'success']
        neg_words = ['loss', 'down', 'fall', 'drop', 'sell', 'negative', 'decline', 'debt', 'cut', 'slump']
        
        score = 0
        for item in news:
            # Handle nested yfinance news structure
            content = item.get('content', {})
            title = content.get('title', '').lower()
            
            for pw in pos_words:
                if pw in title: score += 1
            for nw in neg_words:
                if nw in title: score -= 1
        
        sentiment = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
        return score, sentiment
    except:
        return 0, "Neutral"

@st.cache_data(ttl=3600)
def get_prediction_of_the_day():
    """Scans all models and returns the one with the highest confidence prediction."""
    import os
    model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]
    best_ticker = None
    best_conf = -1
    best_dir = ""
    
    for mf in model_files:
        ticker = mf.replace(".NS_model.pkl", "").replace("_model.pkl", "")
        # Only process NSE stocks for this feature to keep it focused
        if ".NS" not in mf and not any(t in mf for t in ["BTC", "TSLA", "MSFT"]): 
            # This is a bit hacky, but let's assume we want NSE stocks primarily
            # or just process whatever is there. Let's just process all.
            pass
            
        m, s = load_model(ticker)
        if m and s:
            try:
                yf_ticker = ticker if ".NS" in ticker else ticker + ".NS"
                d = yf.Ticker(yf_ticker).history(period="1d", interval="1m").tail(20).copy()
                if len(d) < 20: continue
                
                d['Return'] = d['Close'].pct_change()
                d['MA5'] = d['Close'].rolling(window=5).mean()
                d['MA10'] = d['Close'].rolling(window=10).mean()
                delta = d['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-9)
                d['RSI'] = 100 - (100 / (1 + rs))
                
                features = d[['Return', 'MA5', 'MA10', 'RSI']].tail(1)
                X_scaled = s.transform(features)
                prediction = m.predict(X_scaled)[0]
                prob = m.predict_proba(X_scaled)[0]
                conf = max(prob) * 100
                
                if conf > best_conf:
                    best_conf = conf
                    best_ticker = ticker
                    best_dir = "UP 📈" if prediction == 1 else "DOWN 📉"
            except:
                continue
    return best_ticker, best_dir, best_conf

def load_model(ticker):
    try:
        # Standardizing ticker to match the training script output (Ticker.NS)
        path_model = f"models/{ticker}.NS_model.pkl"
        path_scaler = f"models/{ticker}.NS_scaler.pkl"
        
        with open(path_model, "rb") as f:
            model = pickle.load(f)
        with open(path_scaler, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

# --- UI CONFIG ---
st.set_page_config(page_title="MarketMind", layout="wide", page_icon="📈")

# Premium CSS Theme Injection
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Montserrat:wght@700;800&display=swap" rel="stylesheet">

<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c24, #0b0d11);
        font-family: 'Inter', sans-serif;
        color: #e0e6ed;
    }
    
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
    }

    /* Glassmorphism Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(74, 144, 226, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Custom Metric Styling */
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .delta-up { color: #00f2ad; }
    .delta-down { color: #ff4b4b; }

    /* Premium Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 30px;
    }
    .brand-title {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .market-status {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(0, 242, 173, 0.1);
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid rgba(0, 242, 173, 0.2);
    }
    .status-dot {
        width: 10px;
        height: 10px;
        background-color: #00f2ad;
        border-radius: 50%;
        box-shadow: 0 0 10px #00f2ad;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #0d0f14;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 20px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        border: none;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown(f"""
    <div class="header-container">
        <div>
            <h1 class="brand-title">MARKETMIND TERMINAL</h1>
            <div style="color: #8892b0; font-size: 0.9rem; font-weight: 500;">MARKETMIND ALGORITHMIC TRADING SYSTEM</div>
        </div>
        <div class="market-status">
            <div class="status-dot"></div>
            <div style="color: #00f2ad; font-size: 0.85rem; font-weight: 700;">LIVE MARKET FEED: {datetime.now().strftime("%H:%M:%S")}</div>
        </div>
    </div>
""",unsafe_allow_html=True)

# --- SIDEBAR: INVESTOR PROFILING ---
st.sidebar.header("👤 Investor Profile")
age = st.sidebar.slider("Your Age", 18, 80, 25)
income_level = st.sidebar.selectbox("Monthly Income", ["Low (< 30k)", "Medium (30k-1L)", "High (> 1L)"])

def get_risk_tolerance(age, income_str):
    # Professional Scoring System (0-10)
    score = 0
    
    # Age Factor (Inverse relationship with risk)
    if age <= 25: score += 5  # Very Aggressive phase
    elif age <= 35: score += 4
    elif age <= 45: score += 3
    elif age <= 60: score += 2
    else: score += 1 # Preservation phase
    
    # Income Factor (Capacity to absorb loss)
    if "High" in income_str: score += 5
    elif "Medium" in income_str: score += 3
    else: score += 1
    
    # Final Categorization with strict levels
    if score >= 9:
        return "Aggressive (Volatile)", "High", "High Risk, High Reward"
    elif score >= 7:
        return "Growth (Moderate)", "Medium", "Balanced Portfolio"
    elif score >= 5:
        return "Conservative (Blue Chip)", "Low", "Capital Preservation"
    else:
        # Very low income or very high age
        return "Conservative (Blue Chip)", "Very Low", "Fixed Income Focus"

user_risk_cat, user_risk_level, risk_desc = get_risk_tolerance(age, income_level)
st.sidebar.info(f"Risk Profile: **{user_risk_cat}**\nRisk Level: **{user_risk_level}**\n\nStrategy: *{risk_desc}*")

# --- MANUAL OVERRIDE (For Low Probabilities) ---
st.sidebar.divider()
st.sidebar.header("🛠️ Manual Analysis Mode")
manual_mode = st.sidebar.toggle("Enable Manual Override", help="Use this when the AI signal is WEAK or for news-driven events.")
manual_trend = "NEUTRAL"
manual_move = 0.0

if manual_mode:
    manual_trend = st.sidebar.radio("Your Trend Analysis", ["UP 📈", "DOWN 📉"])
    manual_move = st.sidebar.slider("Manual Expected Move (%)", 0.1, 5.0, 0.5)
    st.sidebar.warning("Manual Override Active: AI signals will be ignored.")

# --- SUGGESTOR MODULE ---
st.sidebar.divider()
st.sidebar.header("💡 Top Picks for You")
cats = load_categories()

# Strict filtering based ONLY on the user's specific age/income profile
user_matches = cats[
    (cats['Category'] == user_risk_cat) & 
    (cats['Risk_Level'] == user_risk_level)
].copy()

# Fallback: if no perfect matches for strict level, just use the category
if user_matches.empty:
    user_matches = cats[cats['Category'] == user_risk_cat].copy()

if not user_matches.empty:
    # Shuffle or select from a broader range to ensure variety when profile changes
    candidates = user_matches.sample(min(len(user_matches), 15)).copy()
    sentiments = []
    for t in candidates['Ticker']:
        score, sent = get_news_sentiment(t)
        sentiments.append((score, sent))
    
    candidates['Sentiment_Score'] = [s[0] for s in sentiments]
    candidates['Sentiment_Label'] = [s[1] for s in sentiments]
    
    # Prioritize sentiment but within the strict risk group
    top_3 = candidates.sort_values(by='Sentiment_Score', ascending=False).head(3)
    
    for _, row in top_3.iterrows():
        emoji = "🚀" if row['Sentiment_Score'] > 0 else "📉" if row['Sentiment_Score'] < 0 else "⚖️"
        st.sidebar.write(f"{emoji} **{row['Ticker']}**")
        st.sidebar.caption(f"News: {row['Sentiment_Label']} | Risk: {row['Risk_Level']}")
else:
    st.sidebar.write("Scan stocks to see picks")

# --- MAIN CONTENT ---
ticker_list = cats['Ticker'].tolist()
if not ticker_list:
    ticker_list = ["RELIANCE", "TCS", "INFY"] # Fallback

ticker_col, info_col = st.columns([1, 3])
with ticker_col:
    ticker_input = st.selectbox("Select Asset", ticker_list)
full_ticker = ticker_input + ".NS"

# --- TIMEFRAME SELECTOR ---
st.markdown("### 📊 Market Momentum")
timeframe = st.radio(
    "Select Timeframe",
    ["1D", "5D", "1M", "6M", "1Y", "MAX"],
    horizontal=True,
    label_visibility="collapsed"
)

# Mapping timeframe to yfinance period/interval
tf_map = {
    "1D": ("1d", "1m"),
    "5D": ("5d", "5m"),
    "1M": ("1mo", "15m"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "MAX": ("max", "1wk")
}
period, interval = tf_map[timeframe]

# Fetch Data
with st.spinner("FETCHING MARKET DATA..."):
    # Chart Data (Dynamic)
    chart_data = yf.Ticker(full_ticker).history(period=period, interval=interval)
    # Analysis Data (Always 1D/1M for short-term AI signals)
    data = yf.Ticker(full_ticker).history(period="1d", interval="1m")
    # Macro Data
    lt_data = yf.Ticker(full_ticker).history(period="1y", interval="1d")

if not chart_data.empty:
    # 1. LIVE CHART
    st.markdown('<div class="glass-card" style="padding: 0;">', unsafe_allow_html=True)
    fig = go.Figure(data=[go.Candlestick(x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Market Price")])
    fig.update_layout(
        height=450, 
        template="plotly_dark", 
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_gridcolor='rgba(255,255,255,0.05)',
        yaxis_gridcolor='rgba(255,255,255,0.05)',
        title=f"{ticker_input} - {timeframe} Perspective"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. METRICS & PREDICTION
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_diff = current_price - prev_price
    price_pct = (price_diff / prev_price) * 100
    delta_class = "delta-up" if price_diff >= 0 else "delta-down"
    delta_icon = "▴" if price_diff >= 0 else "▾"

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label"><i class="fas fa-chart-line"></i> Live Price</div>
                <div class="metric-value">₹{current_price:,.2f}</div>
                <div class="metric-delta {delta_class}">{delta_icon} {abs(price_diff):.2f} ({price_pct:+.2f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    model, scaler = load_model(ticker_input)
    
    # Pre-define signals and fallbacks to avoid NameErrors
    prediction_val = None
    is_up = None
    prob_val = [0.5, 0.5] # Neutral fallback

    if model:
        try:
            latest = data.tail(20).copy()
            latest['Return'] = latest['Close'].pct_change()
            latest['MA5'] = latest['Close'].rolling(window=5).mean()
            latest['MA10'] = latest['Close'].rolling(window=10).mean()
            delta = latest['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            latest['RSI'] = 100 - (100 / (1 + rs))
            
            features = latest[['Return', 'MA5', 'MA10', 'RSI']].tail(1)
            X_scaled = scaler.transform(features)
            prediction_val = model.predict(X_scaled)[0]
            prob_val = model.predict_proba(X_scaled)[0]
            is_up = (prediction_val == 1)
        except:
            model = None # Fallback to technicals if prediction fails

    if not model:
        avg_5 = data['Close'].tail(5).mean()
        avg_10 = data['Close'].tail(10).mean()
        is_up = (avg_5 > avg_10)

    with m_col2:
        if manual_mode:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label"><i class="fas fa-user-edit"></i> Manual Signal</div>
                    <div class="metric-value" style="color: #4facfe;">{manual_trend}</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">User Override Active</div>
                </div>
            """, unsafe_allow_html=True)
        elif model:
            direction = "BULLISH 🚀" if is_up else "BEARISH 📉"
            dir_color = "#00f2ad" if is_up else "#ff4b4b"
            confidence = max(prob_val) * 100
            
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label"><i class="fas fa-brain"></i> AI Trend Signal</div>
                    <div class="metric-value" style="color: {dir_color};">{direction}</div>
                    <div class="metric-delta" style="color: #ffffff;">{confidence:.1f}% Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            pulse_dir = "BULLISH 🚀" if is_up else "BEARISH 📉"
            pulse_color = "#00f2ad" if is_up else "#ff4b4b"
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label"><i class="fas fa-wave-square"></i> Pulse Signal</div>
                    <div class="metric-value" style="color: {pulse_color};">{pulse_dir}</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">Technical Edge Only</div>
                </div>
            """, unsafe_allow_html=True)

    with m_col3:
        if not manual_mode:
            # Improved Target Price Logic
            volatility = data['Close'].pct_change().std()
            conf_factor = max(prob_val) 
            expected_change = current_price * volatility * conf_factor
            expected_price = current_price + expected_change if is_up else current_price - expected_change
            target_pct = ((expected_price/current_price)-1)*100
            target_class = "delta-up" if target_pct > 0 else "delta-down"

            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label"><i class="fas fa-crosshairs"></i> Algorithm Target</div>
                    <div class="metric-value">₹{expected_price:,.2f}</div>
                    <div class="metric-delta {target_class}">{target_pct:+.2f}% Est. Move</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label"><i class="fas fa-crosshairs"></i> Manual Target</div>
                    <div class="metric-value">--</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">Override Mode</div>
                </div>
            """, unsafe_allow_html=True)

    with m_col4:
        # Determine Holding Strategy
        if not lt_data.empty and len(lt_data) > 200:
            lt_data['SMA200'] = lt_data['Close'].rolling(window=200).mean()
            macro_bullish = lt_data['Close'].iloc[-1] > lt_data['SMA200'].iloc[-1]
        else:
            macro_bullish = None

        if is_up is not None:
            if macro_bullish is not None and is_up == macro_bullish:
                h_period = "SWING (3-10D)"
                h_color = "#00f2ad"
            else:
                h_period = "SCALP (5-30M)"
                h_color = "#ffbd03"
        else:
            h_period = "WAITING"
            h_color = "#8892b0"

        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label"><i class="fas fa-shield-alt"></i> Trade Strategy</div>
                <div class="metric-value" style="color: {h_color};">{h_period}</div>
                <div style="font-size: 0.8rem; color: #8892b0;">Risk-Adjusted Suggestion</div>
            </div>
        """, unsafe_allow_html=True)

    # --- PROFILE SAFETY (Moved to Sidebar/Info) ---
    st.sidebar.divider()
    if not cats[cats['Ticker'] == ticker_input].empty:
        stock_info = cats[cats['Ticker'] == ticker_input].iloc[0]
        st.sidebar.write(f"Asset Safety: **{stock_info['Capital_Safety']}**")
        if stock_info['Category'] == user_risk_cat:
            st.sidebar.success("Profile Match ✅")
        else:
            st.sidebar.warning("Profile Mismatch ⚠️")

    # 3. LONG TERM ANALYSIS
    st.divider()
    st.subheader("📅 Long-Term Analysis (Macro Trend)")
    
    with st.expander("Show Macro Trend Details", expanded=True):
        if not lt_data.empty:
            lt_data['SMA50'] = lt_data['Close'].rolling(window=50).mean()
            lt_data['SMA200'] = lt_data['Close'].rolling(window=200).mean()
            
            curr_sma50 = lt_data['SMA50'].iloc[-1]
            curr_sma200 = lt_data['SMA200'].iloc[-1]
            curr_price_lt = lt_data['Close'].iloc[-1]
            
            l_col1, l_col2, l_col3 = st.columns(3)
            
            with l_col1:
                trend_status = "BULLISH 🐂" if curr_price_lt > curr_sma200 else "BEARISH 🐻"
                st.metric("Macro Sentiment", trend_status)
                st.caption("Price vs 200-Day SMA")
            
            with l_col2:
                cross_status = "Golden Cross ✨" if curr_sma50 > curr_sma200 else "Death Cross 💀"
                st.metric("Trend Phase", cross_status)
                st.caption("50-Day vs 200-Day SMA")
                
            with l_col3:
                dist_200 = ((curr_price_lt / curr_sma200) - 1) * 100
                st.metric("Distance to 200 SMA", f"{dist_200:.2f}%")
                st.caption("Value Relative to Long-term Mean")

            # Mini Chart for Long Term
            fig_lt = go.Figure()
            fig_lt.add_trace(go.Scatter(x=lt_data.index, y=lt_data['Close'], name="Price", line=dict(color='white', width=1)))
            fig_lt.add_trace(go.Scatter(x=lt_data.index, y=lt_data['SMA50'], name="50 SMA", line=dict(color='orange', width=1)))
            fig_lt.add_trace(go.Scatter(x=lt_data.index, y=lt_data['SMA200'], name="200 SMA", line=dict(color='cyan', width=1)))
            fig_lt.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig_lt, use_container_width=True)
        else:
            st.warning("Insufficient data for long-term analysis.")

    # 4. TRADER'S INTELLIGENCE (News Analysis)
    st.divider()
    st.subheader("🕵️ Trader's Intelligence (Critical News Analysis)")
    
    with st.container():
        risk_sum, verdict, move, reason = get_trader_intelligence(ticker_input)
        
        i_col1, i_col2 = st.columns([1, 2])
        
        with i_col1:
            st.markdown(f"### Verdict: {verdict}")
            st.markdown(f"### Move: `{move}`")
            
            if "BEARISH" in verdict:
                st.error("High Risk Detected")
            elif "CAUTIOUS" in verdict:
                st.warning("Monitor Closely")
            else:
                st.success("Safe Sentiment")
        
        with i_col2:
            st.info(f"**Professional Rationale:**\n\n{reason}")
            with st.expander("View Identified Bearish Risks"):
                st.write(risk_sum)

    # 5. LIVE NEWS FEED
    st.divider()
    st.subheader("📰 Live Market Intelligence Feed")
    
    try:
        yf_ticker = ticker_input if ".NS" in ticker_input else ticker_input + ".NS"
        news_list = yf.Ticker(yf_ticker).news
        
        if news_list:
            for item in news_list[:5]: # Show top 5 news items
                # Handle nested yfinance news structure
                content = item.get('content', {})
                title = content.get('title', 'No Headline Available')
                
                # Try to get link from multiple possible keys
                link = content.get('clickThroughUrl', {}).get('url', '#')
                if link == '#':
                    link = content.get('canonicalUrl', {}).get('url', '#')
                
                provider_info = content.get('provider', {})
                publisher = provider_info.get('displayName', 'Unknown Source')
                n_type = content.get('contentType', 'News')
                
                with st.container():
                    col_icon, col_content = st.columns([1, 15])
                    with col_icon:
                        st.markdown("### 📰")
                    with col_content:
                        st.markdown(f"**[{title}]({link})**")
                        st.caption(f"Source: {publisher} | Type: {n_type}")
                    st.divider()
        else:
            st.info("No recent news found for this asset.")
    except Exception as e:
        st.error(f"Could not fetch live news: {e}")

# --- SAFETY DISCLAIMER ---
st.divider()
st.error("⚠️ **FINANCIAL SAFETY DISCLAIMER**")
st.markdown("""
1. **Educational Purpose Only**: This tool is an experimental project for learning Machine Learning and technical analysis. It is **NOT** a financial advisory tool.
2. **High Risk**: Stock trading involves significant risk of capital loss. The Indian market is highly volatile.
3. **No Guarantees**: Predictions are based on historical patterns (Logistic Regression) and do not account for news, dividends, or global events.
4. **Consult a Professional**: Always consult a SEBI-registered financial advisor before making any investment decisions.
""")

st.caption(f"System Version: 1.0.4 | Last Market Pulse: {datetime.now().strftime('%H:%M:%S')}")
