"""
Data Collection Script for Event-Aware Stock Prediction Ensemble
Downloads historical data for AI winners and SaaS victims
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Stock tickers - AI Winners vs SaaS Victims
AI_WINNERS = ['NVDA', 'MSFT', 'GOOGL']  # Benefited from AI boom
SAAS_VICTIMS = ['CRM', 'ADBE', 'NOW']   # Hurt by SaaSapocalypse

ALL_STOCKS = AI_WINNERS + SAAS_VICTIMS

# Date range
START_DATE = '2019-01-01'
END_DATE = '2026-02-10'  # Up to current date

# SaaSapocalypse event date for reference
SAASAPOCALYPSE_DATE = '2026-02-03'

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        Historical stock data with OHLCV
    """
    print(f"Downloading data for {ticker}...")
    
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns (sometimes yfinance returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Add ticker column
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Basic info
        print(f"  ✓ {ticker}: {len(df)} trading days from {df['Date'].min()} to {df['Date'].max()}")
        
        # Get price range - handle potential Series/MultiIndex issues
        try:
            close_min = float(df['Close'].min())
            close_max = float(df['Close'].max())
            print(f"  Price range: ${close_min:.2f} - ${close_max:.2f}")
        except:
            print(f"  Price range: Available in data")
        
        return df
    
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {str(e)}")
        return None

def add_basic_features(df):
    """
    Add basic derived features to the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data
    
    Returns:
    --------
    pd.DataFrame
        Data with additional features
    """
    # Daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Log returns (better for modeling)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price change
    df['Price_Change'] = df['Close'] - df['Open']
    
    # High-Low range
    df['HL_Range'] = df['High'] - df['Low']
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df

def save_data(df, ticker, output_dir='data'):
    """
    Save dataframe to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data to save
    ticker : str
        Stock ticker symbol
    output_dir : str
        Directory to save data
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join(output_dir, f'{ticker}_data.csv')
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved to {filepath}")

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("STOCK DATA COLLECTION FOR EVENT-AWARE ENSEMBLE")
    print("=" * 60)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"SaaSapocalypse Event: {SAASAPOCALYPSE_DATE}")
    print(f"\nAI Winners: {', '.join(AI_WINNERS)}")
    print(f"SaaS Victims: {', '.join(SAAS_VICTIMS)}")
    print("\n" + "=" * 60 + "\n")
    
    all_data = {}
    
    # Download data for each stock
    for ticker in ALL_STOCKS:
        df = download_stock_data(ticker, START_DATE, END_DATE)
        
        if df is not None:
            # Add basic features
            df = add_basic_features(df)
            
            # Save individual stock data
            save_data(df, ticker)
            
            # Store in dictionary
            all_data[ticker] = df
        
        print()  # Empty line for readability
    
    # Create combined dataset
    if all_data:
        print("Creating combined dataset...")
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        save_data(combined_df, 'ALL_STOCKS', output_dir='data')
        
        print("\n" + "=" * 60)
        print("DATA COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"\nTotal stocks: {len(all_data)}")
        print(f"Total records: {len(combined_df)}")
        print(f"\nFiles saved in './data/' directory")
        
        # Show summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        for ticker in ALL_STOCKS:
            if ticker in all_data:
                df = all_data[ticker]
                pre_event = df[df['Date'] < SAASAPOCALYPSE_DATE]
                post_event = df[df['Date'] >= SAASAPOCALYPSE_DATE]
                
                print(f"\n{ticker}:")
                print(f"  Total days: {len(df)}")
                print(f"  Pre-SaaSapocalypse: {len(pre_event)} days")
                print(f"  Post-SaaSapocalypse: {len(post_event)} days")
                
                if len(post_event) > 0:
                    event_return = ((post_event.iloc[0]['Close'] - pre_event.iloc[-1]['Close']) / 
                                   pre_event.iloc[-1]['Close'] * 100)
                    print(f"  Event day return: {event_return:+.2f}%")
    
    else:
        print("\n✗ No data downloaded successfully!")

if __name__ == "__main__":
    import numpy as np
    main()