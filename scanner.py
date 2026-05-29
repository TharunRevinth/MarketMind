import yfinance as yf
from analyzer import StockAnalyzer
import concurrent.futures

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "V", "JPM",
    "AMD", "NFLX", "DIS", "PYPL", "INTC", "CSCO", "PEP", "KO", "ORCL", "CRM"
]

class StockScanner:
    def __init__(self, tickers=None):
        self.tickers = tickers or DEFAULT_TICKERS

    def scan_ticker(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="6mo")
            if data.empty:
                return ticker, "ERROR", "No data found"
            
            analyzer = StockAnalyzer(data)
            signal, reason = analyzer.get_signals()
            return ticker, signal, reason
        except Exception as e:
            return ticker, "ERROR", str(e)

    def scan_all(self):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(self.scan_ticker, ticker): ticker for ticker in self.tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                results.append(future.result())
        return results

if __name__ == "__main__":
    scanner = StockScanner()
    print("Scanning stocks...")
    results = scanner.scan_all()
    
    print("\n--- Scan Results ---")
    for ticker, signal, reason in sorted(results):
        print(f"{ticker:8} | {signal:8} | {reason}")
