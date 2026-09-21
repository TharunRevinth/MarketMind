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
    
    print("\n" + "="*100)
    print(f"{'TICKER':<10} | {'SIGNAL':<8} | {'VWAP':<10} | {'ATR':<8} | {'BOLLINGER STATUS':<24} | {'TECHNICAL REASON'}")
    print("-" * 100)
    
    suggestions = []
    
    for res in sorted(results):
        ticker, signal, reason = res
        m = getattr(res, 'metrics', {})
        vwap_val = f"₹{m.get('vwap', 0):.2f}" if 'vwap' in m else "N/A"
        atr_val = f"₹{m.get('atr', 0):.2f}" if 'atr' in m else "N/A"
        bb_state = m.get('bb_state', 'N/A')
        
        print(f"{ticker:<10} | {signal:<8} | {vwap_val:<10} | {atr_val:<8} | {bb_state:<24} | {reason}")
        if signal == "BUY":
            suggestions.append(ticker)

    if suggestions:
        print("\n" + "="*100)
        print("DAILY SUGGESTIONS (BUY SIGNALS):")
        print(", ".join(suggestions))
        print("="*100)
    else:
        print("\nNo clear BUY signals found today.")

if __name__ == "__main__":
    main()

