import pandas as pd
import numpy as np

class StockAnalyzer:
    def __init__(self, data):
        self.data = data

    def calculate_rsi(self, window=14):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.data['RSI'] = rsi
        return rsi

    def calculate_ema(self, span_short=20, span_long=50):
        self.data['EMA_short'] = self.data['Close'].ewm(span=span_short, adjust=False).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=span_long, adjust=False).mean()

    def get_signals(self):
        self.calculate_rsi()
        self.calculate_ema()
        
        last_row = self.data.iloc[-1]
        prev_row = self.data.iloc[-2]
        
        signal = "NEUTRAL"
        reason = ""
        
        # EMA Crossover
        if prev_row['EMA_short'] <= prev_row['EMA_long'] and last_row['EMA_short'] > last_row['EMA_long']:
            signal = "BUY"
            reason = "EMA Crossover (Golden Cross)"
        elif prev_row['EMA_short'] >= prev_row['EMA_long'] and last_row['EMA_short'] < last_row['EMA_long']:
            signal = "SELL"
            reason = "EMA Crossover (Death Cross)"
            
        # RSI Check
        if last_row['RSI'] < 30:
            if signal == "NEUTRAL" or signal == "BUY":
                signal = "BUY"
                reason += " | Oversold (RSI < 30)"
        elif last_row['RSI'] > 70:
            if signal == "NEUTRAL" or signal == "SELL":
                signal = "SELL"
                reason += " | Overbought (RSI > 70)"
                
        return signal, reason.strip(" | ")

if __name__ == "__main__":
    # Example usage with dummy data or test fetch
    import yfinance as yf
    ticker = "AAPL"
    data = yf.Ticker(ticker).history(period="6mo")
    analyzer = StockAnalyzer(data)
    signal, reason = analyzer.get_signals()
    print(f"Ticker: {ticker} | Signal: {signal} | Reason: {reason}")
