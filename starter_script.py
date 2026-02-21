"""
STARTER SCRIPT - Phase 4: Multi-Stock Portfolio Loop
Runs the Tri-Model Ensemble across AI Winners and SaaS Victims.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our modules
from data_collection import download_stock_data, add_basic_features
from technical_indicators import calculate_all_indicators, select_features_for_models
from config import LSTM_CONFIG, TRANSFORMER_CONFIG, SEQUENCE_LENGTH, START_DATE, END_DATE

# Import Models
from lstm_model import LSTMStockPredictor
from xgb_model import XGBStockPredictor
from transformer_model import TransformerStockPredictor
from backtester import SimpleBacktester

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'results', 'results/figures', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def step1_download_data(ticker):
    """Step 1: Download stock data dynamically for a given ticker"""
    df = download_stock_data(ticker, START_DATE, END_DATE)
    if df is not None:
        df = add_basic_features(df)
        filepath = f'data/{ticker}_raw.csv'
        df.to_csv(filepath, index=False)
        return df
    else:
        raise ValueError(f"Failed to download data for {ticker}!")

def step2_add_indicators(df, ticker):
    """Step 2: Calculate technical indicators and save dynamically"""
    df_with_indicators = calculate_all_indicators(df)
    filepath = f'data/{ticker}_processed.csv'
    df_with_indicators.to_csv(filepath, index=False)
    return df_with_indicators

def step3_prepare_sequences(df):
    """Step 3: Prepare data for LSTM & Transformer"""
    model_features = select_features_for_models()
    feature_cols = model_features['lstm']
    target_col = 'Return'
    
    lstm_predictor = LSTMStockPredictor(LSTM_CONFIG)
    X, y = lstm_predictor.prepare_data(df, SEQUENCE_LENGTH, feature_cols, target_col)
    return X, y, lstm_predictor

def step4_split_data(X, y, df):
    """Step 4: Split data (time-based)"""
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    dates = df['Date'].iloc[SEQUENCE_LENGTH:].values
    dates_test = dates[train_size+val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), dates_test

def step5_train_model(model, train_data, val_data):
    """Step 5: Train deep learning models"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    history = model.train(X_train, y_train, X_val, y_val)
    return history


def main():
    print("="*70)
    print("PHASE 4: MULTI-STOCK PORTFOLIO ENSEMBLE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_directories()
    
    # Define the Portfolio
    portfolio = {
        'AI_Winners': ['NVDA', 'MSFT', 'GOOGL'],
        'SaaS_Victims': ['CRM', 'ADBE']
    }
    
    # Dictionary to store final metrics for all stocks
    portfolio_results = []
    
    # THE MASTER LOOP
    for category, tickers in portfolio.items():
        for ticker in tickers:
            print(f"\n{'='*50}")
            print(f"🚀 PROCESSING {ticker} ({category})")
            print(f"{'='*50}")
            
            try:
                # 1. Setup & Data
                df_raw = step1_download_data(ticker)
                df_processed = step2_add_indicators(df_raw, ticker)

                # 2. LSTM
                print(f"\n--- Training LSTM for {ticker} ---")
                X_lstm, y_lstm, lstm_model = step3_prepare_sequences(df_processed)
                train_lstm, val_lstm, test_lstm, dates_test = step4_split_data(X_lstm, y_lstm, df_processed)
                step5_train_model(lstm_model, train_lstm, val_lstm)
                
                # 3. XGBoost
                print(f"\n--- Training XGBoost for {ticker} ---")
                xgb_model = XGBStockPredictor()
                X_xgb, y_xgb, features = xgb_model.prepare_data(df_processed)
                
                train_size = len(train_lstm[0])
                val_size = len(val_lstm[0])
                
                X_xgb_train = X_xgb[:train_size]
                y_xgb_train = y_xgb[:train_size]
                X_xgb_val = X_xgb[train_size:train_size+val_size]
                y_xgb_val = y_xgb[train_size:train_size+val_size]
                X_xgb_test = X_xgb[train_size+val_size:]
                
                xgb_model.train(X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val)

                # 4. Transformer
                print(f"\n--- Training Transformer for {ticker} ---")
                transformer_model = TransformerStockPredictor(TRANSFORMER_CONFIG)
                transformer_model.train(train_lstm[0], train_lstm[1], val_lstm[0], val_lstm[1])

                # 5. Ensemble Evaluation
                pred_lstm = lstm_model.predict(test_lstm[0]).flatten()
                pred_xgb = xgb_model.predict(X_xgb_test)
                pred_trans = transformer_model.predict(test_lstm[0]).flatten()

                # Align arrays
                min_len = min(len(pred_lstm), len(pred_xgb), len(pred_trans))
                pred_lstm = pred_lstm[-min_len:]
                pred_xgb = pred_xgb[-min_len:]
                pred_trans = pred_trans[-min_len:]
                y_actual = test_lstm[1][-min_len:]
                dates_aligned = dates_test[-min_len:]
                
                # Majority Vote Logic
                sig_lstm = np.sign(pred_lstm)
                sig_xgb = np.sign(pred_xgb)
                sig_trans = np.sign(pred_trans)
                vote_sum = sig_lstm + sig_xgb + sig_trans
                
                avg_magnitude = (np.abs(pred_lstm) + np.abs(pred_xgb)) / 2.0
                pred_ensemble = np.sign(vote_sum) * avg_magnitude
                
                # Calculate Accuracy for this specific ticker
                correct_direction = np.sum(np.sign(y_actual) == np.sign(pred_ensemble))
                accuracy = (correct_direction / len(y_actual)) * 100
                
                # 6. Backtest
                print(f"\n--- Backtesting {ticker} ---")
                backtester = SimpleBacktester(initial_capital=10000.0)
                strat_wealth, bench_wealth = backtester.run_backtest(y_actual, pred_ensemble, dates_aligned)
                
                strat_return_pct = (strat_wealth[-1] - 10000) / 10000 * 100
                bench_return_pct = (bench_wealth[-1] - 10000) / 10000 * 100
                
                # Save the results
                portfolio_results.append({
                    'Category': category,
                    'Ticker': ticker,
                    'Accuracy': round(accuracy, 2),
                    'Strat_Return_%': round(strat_return_pct, 2),
                    'Bench_Return_%': round(bench_return_pct, 2)
                })
                
                # Clear Keras memory so the next stock doesn't crash your computer
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Error processing {ticker}: {str(e)}")
                continue

    # --- FINAL PORTFOLIO SUMMARY ---
    print("\n" + "="*70)
    print("📊 PHASE 4 FINAL PORTFOLIO SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame(portfolio_results)
    print(summary_df.to_string(index=False))
    
    # Save the summary to a CSV for your report
    summary_df.to_csv('results/phase4_portfolio_summary.csv', index=False)
    print("\n✓ Saved complete summary to results/phase4_portfolio_summary.csv")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()