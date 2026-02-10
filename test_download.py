"""
Simple test script to verify data download works
Run this to test if the download issue is fixed
"""

import yfinance as yf
import pandas as pd

print("="*60)
print("TESTING DATA DOWNLOAD")
print("="*60)

ticker = 'NVDA'
start_date = '2019-01-01'
end_date = '2026-02-10'

print(f"\nDownloading {ticker} from {start_date} to {end_date}...")

try:
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    print(f"✓ Download successful!")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        print("  (Flattening MultiIndex columns...)")
        df.columns = df.columns.get_level_values(0)
    
    # Reset index
    df.reset_index(inplace=True)
    
    # Show data info
    print(f"\n✓ Data structure:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show close prices
    if 'Close' in df.columns:
        close_min = float(df['Close'].min())
        close_max = float(df['Close'].max())
        print(f"  Close price range: ${close_min:.2f} - ${close_max:.2f}")
    
    # Show first few rows
    print(f"\n✓ First 3 rows:")
    print(df.head(3))
    
    # Save to test file
    df.to_csv('test_data.csv', index=False)
    print(f"\n✓ Saved to test_data.csv")
    
    print("\n" + "="*60)
    print("SUCCESS! Download is working.")
    print("="*60)
    print("\nYou can now run: python starter_script.py")
    
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try: pip install --upgrade yfinance")
    print("3. Wait a few minutes (Yahoo Finance rate limit)")
    
    import traceback
    print("\nFull error details:")
    traceback.print_exc()