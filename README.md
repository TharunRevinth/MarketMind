# MarketMind

An automated stock analysis tool that uses Yahoo Finance data to provide BUY/SELL signals and daily stock suggestions.

## Features
- Fetches real-time and historical stock data using `yfinance`.
- Calculates technical indicators:
  - **RSI (Relative Strength Index)**: Identifies overbought (>70) and oversold (<30) conditions.
  - **EMA (Exponential Moving Average)**: Detects momentum shifts through Golden Cross and Death Cross.
- Scans a list of major stocks and provides daily "BUY" suggestions.
- Supports custom ticker symbols.

## Installation
Ensure you have Python 3 installed.
```bash
pip install yfinance pandas
```

## Usage
### Scan Default Stocks
```bash
python3 main.py
```

### Scan Specific Stocks
```bash
python3 main.py AAPL MSFT TSLA GOOGL
```

## Strategy Logic
- **BUY Signal**: Triggered if the RSI is below 30 (Oversold) or a Golden Cross occurs (20 EMA crosses above 50 EMA).
- **SELL Signal**: Triggered if the RSI is above 70 (Overbought) or a Death Cross occurs (20 EMA crosses below 50 EMA).
- **NEUTRAL**: No clear signals detected.

## Disclaimer
This tool is for educational purposes only. Always perform your own research before making financial decisions.
