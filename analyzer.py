import pandas as pd
import numpy as np
import yfinance as yf

class StockAnalyzer:
    def __init__(self, data):
        self.data = data.copy() if data is not None else pd.DataFrame()

    def calculate_rsi(self, window=14):
        """Calculates Relative Strength Index (RSI)."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        self.data['RSI'] = rsi
        return rsi

    def calculate_ema(self, span_short=20, span_long=50):
        """Calculates short-term and long-term Exponential Moving Averages."""
        self.data['EMA_short'] = self.data['Close'].ewm(span=span_short, adjust=False).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=span_long, adjust=False).mean()
        return self.data['EMA_short'], self.data['EMA_long']

    # --- VOLUME INDICATORS ---
    def calculate_vwap(self):
        """
        Calculates Volume-Weighted Average Price (VWAP).
        For intraday charts, VWAP resets daily; for daily/macro data, computes cumulative session VWAP.
        """
        if 'Volume' not in self.data or self.data['Volume'].sum() == 0:
            self.data['VWAP'] = self.data['Close']
            return self.data['VWAP']

        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3.0
        tp_vol = typical_price * self.data['Volume']

        # If data has DatetimeIndex, reset VWAP daily for intraday intervals
        if isinstance(self.data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dates = self.data.index.date
            cum_vol = self.data['Volume'].groupby(dates).cumsum()
            cum_tp_vol = tp_vol.groupby(dates).cumsum()
            vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        else:
            cum_vol = self.data['Volume'].cumsum()
            cum_tp_vol = tp_vol.cumsum()
            vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        self.data['VWAP'] = vwap.bfill().ffill()
        return self.data['VWAP']

    def calculate_obv(self):
        """
        Calculates On-Balance Volume (OBV).
        Running total of volume: adds volume on up days and subtracts on down days.
        Shows if institutional capital is accumulating (flowing in) or distributing (flowing out).
        """
        if 'Volume' not in self.data or self.data.empty:
            self.data['OBV'] = 0
            self.data['OBV_EMA'] = 0
            return self.data['OBV']

        close_diff = self.data['Close'].diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        obv = (direction * self.data['Volume']).cumsum()
        self.data['OBV'] = obv
        self.data['OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()
        return self.data['OBV']

    # --- VOLATILITY INDICATORS ---
    def calculate_atr(self, window=14):
        """
        Calculates Average True Range (ATR).
        Measures market volatility so traders can set safe risk-adjusted stop-loss limits.
        """
        if self.data.empty:
            self.data['ATR'] = 0
            return self.data['ATR']

        high = self.data['High']
        low = self.data['Low']
        close_prev = self.data['Close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=window, min_periods=1).mean()
        self.data['ATR'] = atr
        return atr

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """
        Calculates Bollinger Bands:
        - Middle Band = 20-period Simple Moving Average
        - Upper Band = Middle Band + 2 * standard deviation
        - Lower Band = Middle Band - 2 * standard deviation
        Shows whether price is historically high (overbought) or low (oversold) and volatility squeeze.
        """
        sma = self.data['Close'].rolling(window=window, min_periods=1).mean()
        rolling_std = self.data['Close'].rolling(window=window, min_periods=1).std().fillna(0)
        
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)

        self.data['BB_Middle'] = sma
        self.data['BB_Upper'] = upper_band
        self.data['BB_Lower'] = lower_band
        self.data['BB_Bandwidth'] = ((upper_band - lower_band) / sma.replace(0, np.nan)).fillna(0)
        band_range = (upper_band - lower_band).replace(0, np.nan)
        self.data['BB_PctB'] = ((self.data['Close'] - lower_band) / band_range).fillna(0.5)

        return upper_band, sma, lower_band

    def calculate_all(self):
        """Calculates all volume, volatility, and trend indicators on self.data."""
        self.calculate_rsi()
        self.calculate_ema()
        self.calculate_vwap()
        self.calculate_obv()
        self.calculate_atr()
        self.calculate_bollinger_bands()
        return self.data

    def get_signals(self):
        """
        Original entrypoint maintained for backward compatibility.
        Combines EMA crossover and RSI checks.
        """
        self.calculate_rsi()
        self.calculate_ema()
        
        if len(self.data) < 2:
            return "NEUTRAL", "Insufficient data"

        last_row = self.data.iloc[-1]
        prev_row = self.data.iloc[-2]
        
        signal = "NEUTRAL"
        reasons = []
        
        # EMA Crossover
        if prev_row['EMA_short'] <= prev_row['EMA_long'] and last_row['EMA_short'] > last_row['EMA_long']:
            signal = "BUY"
            reasons.append("EMA Crossover (Golden Cross)")
        elif prev_row['EMA_short'] >= prev_row['EMA_long'] and last_row['EMA_short'] < last_row['EMA_long']:
            signal = "SELL"
            reasons.append("EMA Crossover (Death Cross)")
            
        # RSI Check
        if last_row['RSI'] < 30:
            if signal == "NEUTRAL" or signal == "BUY":
                signal = "BUY"
                reasons.append("Oversold (RSI < 30)")
        elif last_row['RSI'] > 70:
            if signal == "NEUTRAL" or signal == "SELL":
                signal = "SELL"
                reasons.append("Overbought (RSI > 70)")
                
        reason_str = " | ".join(reasons) if reasons else "No trigger criteria met"
        return signal, reason_str

    def get_comprehensive_analysis(self):
        """
        Full technical analysis incorporating Volume (VWAP, OBV) and Volatility (ATR, Bollinger Bands).
        Returns a rich dictionary of signals, indicator values, and risk parameters.
        """
        self.calculate_all()
        if len(self.data) < 2:
            return {"signal": "NEUTRAL", "reason": "Insufficient data"}

        last = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        price = last['Close']

        bullish_votes = 0
        bearish_votes = 0
        reasons = []

        # 1. EMA Trend & Crossover
        if prev['EMA_short'] <= prev['EMA_long'] and last['EMA_short'] > last['EMA_long']:
            bullish_votes += 2
            reasons.append("EMA Golden Cross (20 EMA > 50 EMA)")
        elif prev['EMA_short'] >= prev['EMA_long'] and last['EMA_short'] < last['EMA_long']:
            bearish_votes += 2
            reasons.append("EMA Death Cross (20 EMA < 50 EMA)")
        elif last['EMA_short'] > last['EMA_long']:
            bullish_votes += 1
        else:
            bearish_votes += 1

        # 2. RSI (Momentum)
        rsi_val = last['RSI']
        if rsi_val < 30:
            bullish_votes += 2
            reasons.append(f"RSI Oversold ({rsi_val:.1f} < 30)")
        elif rsi_val > 70:
            bearish_votes += 2
            reasons.append(f"RSI Overbought ({rsi_val:.1f} > 70)")
        elif rsi_val > 50:
            bullish_votes += 0.5
        else:
            bearish_votes += 0.5

        # 3. VWAP (Volume-Weighted Price)
        vwap_val = last['VWAP']
        vwap_diff_pct = ((price / vwap_val) - 1) * 100 if vwap_val > 0 else 0
        if price > vwap_val:
            bullish_votes += 1.5
            reasons.append(f"Trading Above VWAP (+{vwap_diff_pct:.2f}% Bullish Bias)")
        else:
            bearish_votes += 1.5
            reasons.append(f"Trading Below VWAP ({vwap_diff_pct:.2f}% Bearish Bias)")

        # 4. OBV (Volume Flow Trend)
        obv_val = last['OBV']
        obv_ema = last['OBV_EMA']
        if obv_val > obv_ema:
            bullish_votes += 1
            obv_status = "Accumulation (Money Inflow)"
        else:
            bearish_votes += 1
            obv_status = "Distribution (Money Outflow)"

        # 5. Bollinger Bands (Price Extremes & Mean Reversion)
        bb_upper = last['BB_Upper']
        bb_lower = last['BB_Lower']
        bb_mid = last['BB_Middle']
        if price <= bb_lower:
            bullish_votes += 1.5
            reasons.append("Piercing Lower Bollinger Band (Potential Oversold Reversal)")
            bb_state = "Oversold (At Lower Band)"
        elif price >= bb_upper:
            bearish_votes += 1.5
            reasons.append("Piercing Upper Bollinger Band (Overextended/Overbought)")
            bb_state = "Overbought (At Upper Band)"
        elif price > bb_mid:
            bb_state = "Upper Channel (Bullish Territory)"
        else:
            bb_state = "Lower Channel (Bearish Territory)"

        # 6. ATR (Volatility & Risk Limits)
        atr_val = last['ATR']
        stop_loss_long = price - (1.5 * atr_val)
        stop_loss_short = price + (1.5 * atr_val)

        # Decide consensus signal
        if bullish_votes - bearish_votes >= 2.5:
            overall_signal = "BUY"
        elif bearish_votes - bullish_votes >= 2.5:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"

        return {
            "signal": overall_signal,
            "reasons": reasons,
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "indicators": {
                "price": price,
                "rsi": rsi_val,
                "ema_short": last['EMA_short'],
                "ema_long": last['EMA_long'],
                "vwap": vwap_val,
                "vwap_diff_pct": vwap_diff_pct,
                "obv": obv_val,
                "obv_ema": obv_ema,
                "obv_status": obv_status,
                "atr": atr_val,
                "stop_loss_long": stop_loss_long,
                "stop_loss_short": stop_loss_short,
                "bb_upper": bb_upper,
                "bb_middle": bb_mid,
                "bb_lower": bb_lower,
                "bb_pct_b": last['BB_PctB'],
                "bb_bandwidth": last['BB_Bandwidth'],
                "bb_state": bb_state
            }
        }

    # --- MULTI-TIMEFRAME ANALYSIS ---
    @staticmethod
    def analyze_multi_timeframe(ticker, short_period="5d", short_interval="15m", long_period="6mo", long_interval="1d"):
        """
        Multi-Timeframe Analysis:
        Evaluates indicator signals across different timeframes (e.g. 15-minute intraday chart
        and daily chart) to see the macro trend while identifying high-probability short-term entry points.
        """
        try:
            yf_ticker = ticker if (".NS" in ticker or "-" in ticker or "=" in ticker) else ticker + ".NS"
            stock = yf.Ticker(yf_ticker)
            
            # 1. Higher Timeframe (Daily Trend)
            df_long = stock.history(period=long_period, interval=long_interval)
            # 2. Lower Timeframe (Intraday Momentum)
            df_short = stock.history(period=short_period, interval=short_interval)

            if df_long.empty or df_short.empty:
                return {
                    "status": "ERROR",
                    "message": "Insufficient timeframe data available from provider"
                }

            analyzer_long = StockAnalyzer(df_long)
            long_analysis = analyzer_long.get_comprehensive_analysis()

            analyzer_short = StockAnalyzer(df_short)
            short_analysis = analyzer_short.get_comprehensive_analysis()

            long_sig = long_analysis['signal']
            short_sig = short_analysis['signal']

            # Confluence logic
            if long_sig == "BUY" and short_sig == "BUY":
                confluence = "STRONG BUY (Trend + Entry Aligned 🚀)"
                recommendation = "High conviction long. Macro daily uptrend confirmed with short-term intraday momentum."
                tone = "success"
            elif long_sig == "SELL" and short_sig == "SELL":
                confluence = "STRONG SELL (Bearish Confluence 📉)"
                recommendation = "High conviction short/exit. Macro daily downtrend confirmed with short-term intraday weakness."
                tone = "error"
            elif long_sig == "BUY" and short_sig == "SELL":
                confluence = "DIP BUY OPPORTUNITY 🎯"
                recommendation = "Macro daily trend is Bullish, but short-term intraday is pulling back. Look for oversold reversal near VWAP/Lower BB for entry."
                tone = "warning"
            elif long_sig == "SELL" and short_sig == "BUY":
                confluence = "COUNTER-TREND RALLY ⚠️"
                recommendation = "Macro trend is Bearish, but short-term bounce is occurring. High risk for long trades; consider taking profits or shorting resistance."
                tone = "warning"
            else:
                confluence = "NEUTRAL / MIXED ⚖️"
                recommendation = "No strong multi-timeframe directional agreement. Await clearer breakout or consolidation test."
                tone = "info"

            return {
                "status": "SUCCESS",
                "confluence": confluence,
                "recommendation": recommendation,
                "tone": tone,
                "higher_tf": {
                    "interval": long_interval,
                    "signal": long_sig,
                    "price": long_analysis['indicators']['price'],
                    "rsi": long_analysis['indicators']['rsi'],
                    "ema_status": "Bullish (20 > 50)" if long_analysis['indicators']['ema_short'] > long_analysis['indicators']['ema_long'] else "Bearish (20 < 50)",
                    "obv_status": long_analysis['indicators']['obv_status']
                },
                "lower_tf": {
                    "interval": short_interval,
                    "signal": short_sig,
                    "price": short_analysis['indicators']['price'],
                    "rsi": short_analysis['indicators']['rsi'],
                    "vwap_status": f"{'+' if short_analysis['indicators']['vwap_diff_pct'] >= 0 else ''}{short_analysis['indicators']['vwap_diff_pct']:.2f}% vs VWAP",
                    "bb_state": short_analysis['indicators']['bb_state'],
                    "atr": short_analysis['indicators']['atr'],
                    "stop_loss": short_analysis['indicators']['stop_loss_long']
                }
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": str(e)
            }

if __name__ == "__main__":
    ticker = "RELIANCE.NS"
    print(f"Testing comprehensive analysis for {ticker}...")
    data = yf.Ticker(ticker).history(period="1mo", interval="1d")
    analyzer = StockAnalyzer(data)
    result = analyzer.get_comprehensive_analysis()
    print("Signal:", result['signal'])
    print("Reasons:", result['reasons'])
    print("Indicators:", result['indicators'])
    
    print("\nTesting Multi-Timeframe Analysis...")
    mtf = StockAnalyzer.analyze_multi_timeframe("RELIANCE.NS")
    print("MTF Confluence:", mtf['confluence'])
    print("Recommendation:", mtf['recommendation'])
