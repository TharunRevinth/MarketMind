import yfinance as yf
from analyzer import StockAnalyzer
import concurrent.futures

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "V", "JPM",
    "AMD", "NFLX", "DIS", "PYPL", "INTC", "CSCO", "PEP", "KO", "ORCL", "CRM"
]

class ScanResult:
    def __init__(self, ticker, signal, reason, metrics=None):
        self.ticker = ticker
        self.signal = signal
        self.reason = reason
        self.metrics = metrics or {}

    def __iter__(self):
        return iter((self.ticker, self.signal, self.reason))

    def __getitem__(self, item):
        if item == 0: return self.ticker
        if item == 1: return self.signal
        if item == 2: return self.reason
        raise IndexError("ScanResult index out of range")

    def __lt__(self, other):
        return self.ticker < other.ticker

class StockScanner:
    def __init__(self, tickers=None):
        self.tickers = tickers or DEFAULT_TICKERS

    def scan_ticker(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="6mo")
            if data.empty:
                return ScanResult(ticker, "ERROR", "No data found")
            
            analyzer = StockAnalyzer(data)
            analysis = analyzer.get_comprehensive_analysis()
            signal = analysis['signal']
            reasons = " | ".join(analysis['reasons']) if analysis['reasons'] else "Consolidation / Rangebound"
            return ScanResult(ticker, signal, reasons, analysis.get('indicators', {}))
        except Exception as e:
            return ScanResult(ticker, "ERROR", str(e))

    def scan_all(self):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(self.scan_ticker, ticker): ticker for ticker in self.tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                results.append(future.result())
        return results

if __name__ == "__main__":
    scanner = StockScanner(["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "MSFT"])
    print("Scanning stocks with Volume & Volatility indicators...")
    results = scanner.scan_all()
    
    print("\n--- Scan Results ---")
    for res in sorted(results):
        m = res.metrics
        vwap_str = f"VWAP: {m.get('vwap', 0):.2f}" if 'vwap' in m else ""
        atr_str = f"ATR: {m.get('atr', 0):.2f}" if 'atr' in m else ""
        bb_str = f"BB: {m.get('bb_state', 'N/A')}" if 'bb_state' in m else ""
        print(f"{res.ticker:<12} | {res.signal:<8} | {vwap_str:<15} | {atr_str:<12} | {bb_str:<25} | {res.reason}")

