import os
import sys
import argparse
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import basket_universe as bu
from scripts.train_model import train_and_save

MODELS_DIR = os.path.join(REPO_ROOT, "models")


def main():
    parser = argparse.ArgumentParser(description="Batch train intraday Logistic Regression models for Nifty 500.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to train")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load full Nifty 500 universe
    symbols_df = bu.get_nifty500_symbols()
    all_symbols = symbols_df["Yahoo_Symbol"].tolist()

    if args.limit and args.limit < len(all_symbols):
        target_symbols = all_symbols[:args.limit]
    else:
        target_symbols = all_symbols

    print(f"🚀 Starting Batch Training for {len(target_symbols)} Nifty 500 tickers...")
    print(f"Resumable mode: {'OFF (Force retrain)' if args.force else 'ON (Skipping existing models)'}")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for i, ticker in enumerate(target_symbols, 1):
        # Skip known dead symbols or typos
        if "TATARELI" in ticker or "DUMMY" in ticker:
            continue

        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")

        if not args.force and os.path.exists(model_path) and os.path.exists(scaler_path):
            skipped_count += 1
            continue

        print(f"[{i}/{len(target_symbols)}] Training intraday model for {ticker}...")
        try:
            train_and_save(ticker)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to train {ticker}: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print("BATCH TRAINING COMPLETED")
    print(f"Newly trained: {success_count} | Skipped existing: {skipped_count} | Failed/No data: {failed_count}")
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
