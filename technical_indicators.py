"""
Technical Indicators Module
Calculates various technical indicators for stock prediction
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

def add_trend_indicators(df):
    """
    Add trend-following indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data with OHLCV columns
    
    Returns:
    --------
    pd.DataFrame
        Data with trend indicators added
    """
    # Simple Moving Averages
    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    # Exponential Moving Averages
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    
    # Moving Average Crossovers (bullish/bearish signals)
    df['SMA_Cross'] = np.where(df['SMA_10'] > df['SMA_50'], 1, -1)
    
    return df

def add_momentum_indicators(df):
    """
    Add momentum indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data with OHLCV columns
    
    Returns:
    --------
    pd.DataFrame
        Data with momentum indicators added
    """
    # RSI (Relative Strength Index)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    
    # Rate of Change
    df['ROC'] = ta.roc(df['Close'], length=10)
    
    # Commodity Channel Index
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    
    return df

def add_volatility_indicators(df):
    """
    Add volatility indicators
    """
    # Bollinger Bands - handle different column name formats
    bbands = ta.bbands(df['Close'], length=20)
    
    if bbands is not None:
        # Get actual column names (they vary by pandas_ta version)
        bb_cols = bbands.columns.tolist()
        
        # Find the columns (they might be named differently)
        for col in bb_cols:
            if 'BBL' in col or 'lower' in col.lower():
                df['BB_Lower'] = bbands[col]
            elif 'BBM' in col or 'middle' in col.lower() or 'mid' in col.lower():
                df['BB_Middle'] = bbands[col]
            elif 'BBU' in col or 'upper' in col.lower():
                df['BB_Upper'] = bbands[col]
        
        # Calculate width and percent if we have the bands
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Average True Range
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    if atr is not None:
        df['ATR_14'] = atr
    
    # Historical Volatility (20-day)
    df['HV_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return df

def add_volume_indicators(df):
    """
    Add volume-based indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data with OHLCV columns
    
    Returns:
    --------
    pd.DataFrame
        Data with volume indicators added
    """
    # On-Balance Volume
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    # Volume Moving Average
    df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
    
    # Volume Ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Accumulation/Distribution Line
    df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Chaikin Money Flow
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """
    Add lagged features (past values)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data
    lags : list
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame
        Data with lagged features added
    """
    for lag in lags:
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)
        df[f'Volume_Lag{lag}'] = df['Volume'].shift(lag)
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
    
    return df

def add_statistical_features(df, window=20):
    """
    Add statistical features over rolling windows
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.DataFrame
        Data with statistical features added
    """
    # Rolling statistics
    df[f'Return_Mean_{window}'] = df['Return'].rolling(window=window).mean()
    df[f'Return_Std_{window}'] = df['Return'].rolling(window=window).std()
    df[f'Return_Skew_{window}'] = df['Return'].rolling(window=window).skew()
    df[f'Return_Kurt_{window}'] = df['Return'].rolling(window=window).kurt()
    
    # Price momentum (rate of change over different periods)
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    return df

def calculate_all_indicators(df):
    """
    Calculate all technical indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data with OHLCV
    
    Returns:
    --------
    pd.DataFrame
        Data with all technical indicators
    """
    print(f"Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate basic returns if not already present
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()
    if 'Log_Return' not in df.columns:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Add all indicator categories
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_lagged_features(df)
    df = add_statistical_features(df)
    
    # Drop NaN values created by indicators
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"  ✓ Added {len(df.columns)} total columns")
    print(f"  ✓ Dropped {dropped_rows} rows with NaN values")
    print(f"  ✓ Final dataset: {len(df)} rows")
    
    return df

def get_feature_groups():
    """
    Returns dictionary of feature groups for easy selection
    
    Returns:
    --------
    dict
        Dictionary with feature group names as keys and column lists as values
    """
    feature_groups = {
        'price': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'returns': ['Return', 'Log_Return'],
        'trend': ['SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'SMA_Cross'],
        'momentum': ['RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D', 'ROC', 'CCI'],
        'volatility': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent', 'ATR_14', 'HV_20'],
        'volume': ['OBV', 'Volume_SMA_20', 'Volume_Ratio', 'AD', 'CMF'],
        'lagged': [f'Return_Lag{i}' for i in [1,2,3,5,10]] + 
                  [f'Volume_Lag{i}' for i in [1,2,3,5,10]] +
                  [f'Close_Lag{i}' for i in [1,2,3,5,10]],
        'statistical': ['Return_Mean_20', 'Return_Std_20', 'Return_Skew_20', 'Return_Kurt_20',
                       'Momentum_5', 'Momentum_10', 'Momentum_20']
    }
    
    return feature_groups

def select_features_for_models():
    """
    Returns recommended feature sets for different models
    
    Returns:
    --------
    dict
        Dictionary with model names and their recommended features
    """
    feature_groups = get_feature_groups()
    
    model_features = {
        # For LSTM/GRU - sequential features
        'lstm': feature_groups['price'] + feature_groups['returns'] + 
                ['SMA_10', 'SMA_20', 'RSI_14', 'BB_Middle', 'Volume_SMA_20'],
        
        # For Transformer - sequential features with more context
        'transformer': feature_groups['price'] + feature_groups['returns'] + 
                      feature_groups['trend'][:4] + feature_groups['momentum'][:4],
        
        # For LightGBM - all engineered features (tabular)
        'lightgbm': (feature_groups['returns'] + feature_groups['trend'] + 
                    feature_groups['momentum'] + feature_groups['volatility'] + 
                    feature_groups['volume'] + feature_groups['lagged'] + 
                    feature_groups['statistical'])
    }
    
    return model_features

if __name__ == "__main__":
    # Example usage
    print("Technical Indicators Module")
    print("=" * 60)
    print("\nAvailable feature groups:")
    for group_name, features in get_feature_groups().items():
        print(f"  {group_name}: {len(features)} features")
    
    print("\nRecommended features per model:")
    for model_name, features in select_features_for_models().items():
        print(f"  {model_name}: {len(features)} features")
