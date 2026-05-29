import sys
from scanner import StockScanner

def print_help():
    print("Usage: python3 main.py [TICKER1 TICKER2 ...]")
    print("If no tickers are provided, a default list of major stocks will be scanned.")

def main():
    tickers = sys.argv[1:]
    
    if "-h" in tickers or "--help" in tickers:
        print_help()
        return

    scanner = StockScanner(tickers if tickers else None)
    print(f"Scanning {'default stocks' if not tickers else ' '.join(tickers)}...")
    results = scanner.scan_all()
    
    print("\n" + "="*50)
    print(f"{'TICKER':<10} | {'SIGNAL':<10} | {'REASON'}")
    print("-" * 50)
    
    suggestions = []
    
    for ticker, signal, reason in sorted(results):
        print(f"{ticker:<10} | {signal:<10} | {reason}")
        if signal == "BUY":
            suggestions.append(ticker)

    if suggestions:
        print("\n" + "="*50)
        print("DAILY SUGGESTIONS (BUY SIGNALS):")
        print(", ".join(suggestions))
        print("="*50)
    else:
        print("\nNo clear BUY signals found today.")

if __name__ == "__main__":
    main()
