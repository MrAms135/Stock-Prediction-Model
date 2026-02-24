"""
STARTER SCRIPT - Phase 5: The Master Dashboard
Runs the Tri-Model Ensemble across 10 stocks with interactive timeframe selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from datetime import datetime

# Import our modules
from data_collection import download_stock_data, add_basic_features
from technical_indicators import calculate_all_indicators, select_features_for_models
from config import LSTM_CONFIG, TRANSFORMER_CONFIG, SEQUENCE_LENGTH

# Import Models
from lstm_model import LSTMStockPredictor
from xgb_model import XGBStockPredictor
from transformer_model import TransformerStockPredictor
from backtester import SimpleBacktester

def create_directories():
    dirs = ['data', 'models', 'results', 'results/figures', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def step1_download_data(ticker, start_date, end_date):
    """Modified to accept dynamic start dates from the CLI"""
    df = download_stock_data(ticker, start_date, end_date)
    if df is not None:
        df = add_basic_features(df)
        filepath = f'data/{ticker}_raw.csv'
        df.to_csv(filepath, index=False)
        return df
    else:
        raise ValueError(f"Failed to download data for {ticker}!")

def step2_add_indicators(df, ticker):
    df_with_indicators = calculate_all_indicators(df)
    filepath = f'data/{ticker}_processed.csv'
    df_with_indicators.to_csv(filepath, index=False)
    return df_with_indicators

def step3_prepare_sequences(df):
    model_features = select_features_for_models()
    feature_cols = model_features['lstm']
    target_col = 'Return'
    
    lstm_predictor = LSTMStockPredictor(LSTM_CONFIG)
    X, y = lstm_predictor.prepare_data(df, SEQUENCE_LENGTH, feature_cols, target_col)
    return X, y, lstm_predictor

def step4_split_data(X, y, df):
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

def generate_master_dashboard(results_df, era_name):
    """Generates a grouped bar chart for returns and drawdowns across all 10 stocks"""
    tickers = results_df['Ticker'].tolist()
    strat_returns = results_df['Strat_Return_%'].tolist()
    bench_returns = results_df['Bench_Return_%'].tolist()
    strat_mdd = results_df['Strat_MDD_%'].tolist()
    bench_mdd = results_df['Bench_MDD_%'].tolist()

    x = np.arange(len(tickers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Returns
    ax1.bar(x - width/2, bench_returns, width, label='Buy & Hold Benchmark', color='gray', alpha=0.7)
    ax1.bar(x + width/2, strat_returns, width, label='AI Strategy', color='green')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title(f'Portfolio Returns by Asset ({era_name})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Plot 2: Maximum Drawdown
    ax2.bar(x - width/2, bench_mdd, width, label='Benchmark Max Drawdown', color='darkred', alpha=0.7)
    ax2.bar(x + width/2, strat_mdd, width, label='AI Strategy Max Drawdown', color='orange')
    ax2.set_ylabel('Maximum Drawdown (%) - Lower is Better')
    ax2.set_title('Risk Mitigation: Worst Peak-to-Trough Drop')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    filepath = f'results/figures/master_dashboard_{era_name.replace(" ", "_")}.png'
    plt.savefig(filepath)
    print(f"\n📈 MASTER DASHBOARD SAVED: {filepath}")

def main():
    print("="*70)
    print("PHASE 5: THE MASTER PORTFOLIO DASHBOARD")
    print("="*70)
    
    # 1. Interactive CLI for Timeframe Selection
    print("\nSelect Training Era (Lookback Window):")
    print("[1] Era 1: The Long Lookback (Since 2018-01-01)")
    print("[2] Era 2: Post-COVID Market (Since 2021-01-01)")
    print("[3] Era 3: The AI Boom (Since 2023-01-01)")
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    if choice == '1':
        start_date = '2018-01-01'
        era_name = "Era 1 - Long Lookback"
    elif choice == '2':
        start_date = '2021-01-01'
        era_name = "Era 2 - Post COVID"
    elif choice == '3':
        start_date = '2023-01-01'
        era_name = "Era 3 - AI Boom"
    else:
        print("Invalid choice. Defaulting to Era 3 (2023-01-01).")
        start_date = '2023-01-01'
        era_name = "Era 3 - AI Boom"

    end_date = '2026-02-20' # Captures the SaaSapocalypse
    print(f"\nInitializing test from {start_date} to {end_date}...\n")
    
    create_directories()
    
    # 2. Expanded 10-Stock Portfolio
    portfolio = {
        'AI_Winners': ['NVDA', 'MSFT', 'GOOGL', 'AMD', 'PLTR'],
        'SaaS_Victims': ['CRM', 'ADBE', 'WDAY', 'NOW', 'SNOW']
    }
    
    portfolio_results = []
    
    for category, tickers in portfolio.items():
        for ticker in tickers:
            print(f"\n{'='*50}")
            print(f"🚀 PROCESSING {ticker} ({category})")
            print(f"{'='*50}")
            
            try:
                # Setup
                df_raw = step1_download_data(ticker, start_date, end_date)
                df_processed = step2_add_indicators(df_raw, ticker)

                # Models
                X_lstm, y_lstm, lstm_model = step3_prepare_sequences(df_processed)
                train_lstm, val_lstm, test_lstm, dates_test = step4_split_data(X_lstm, y_lstm, df_processed)
                lstm_model.train(train_lstm[0], train_lstm[1], val_lstm[0], val_lstm[1])
                
                xgb_model = XGBStockPredictor()
                X_xgb, y_xgb, _ = xgb_model.prepare_data(df_processed)
                ts, vs = len(train_lstm[0]), len(val_lstm[0])
                xgb_model.train(X_xgb[:ts], y_xgb[:ts], X_xgb[ts:ts+vs], y_xgb[ts:ts+vs])

                transformer_model = TransformerStockPredictor(TRANSFORMER_CONFIG)
                transformer_model.train(train_lstm[0], train_lstm[1], val_lstm[0], val_lstm[1])

                # Ensemble
                pred_lstm = lstm_model.predict(test_lstm[0]).flatten()
                pred_xgb = xgb_model.predict(X_xgb[ts+vs:])
                pred_trans = transformer_model.predict(test_lstm[0]).flatten()

                min_len = min(len(pred_lstm), len(pred_xgb), len(pred_trans))
                y_actual = test_lstm[1][-min_len:]
                dates_aligned = dates_test[-min_len:]
                
                # --- V2: POSITION SIZING (SCALED VOTING) ---
                vote_sum = np.sign(pred_lstm[-min_len:]) + np.sign(pred_xgb[-min_len:]) + np.sign(pred_trans[-min_len:])
                
                # Divides the vote by 3 to create a weight: 1.0, 0.33, -0.33, or -1.0
                ensemble_weight = vote_sum / 3.0
                
                # Pass the exact weight to the backtester (no need for magnitude math anymore)
                pred_ensemble = ensemble_weight
                
                accuracy = (np.sum(np.sign(y_actual) == np.sign(pred_ensemble)) / len(y_actual)) * 100
                
                # Phase 5 Backtest with Drawdown
                backtester = SimpleBacktester(initial_capital=10000.0)
                _, _, strat_ret, bench_ret, strat_mdd, bench_mdd = backtester.run_backtest(y_actual, pred_ensemble, dates_aligned)
                
                portfolio_results.append({
                    'Category': category,
                    'Ticker': ticker,
                    'Accuracy': round(accuracy, 2),
                    'Strat_Return_%': round(strat_ret, 2),
                    'Bench_Return_%': round(bench_ret, 2),
                    'Strat_MDD_%': round(strat_mdd, 2),
                    'Bench_MDD_%': round(bench_mdd, 2)
                })
                
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Error processing {ticker}: {str(e)}")
                continue

    # --- FINAL OUTPUT ---
    summary_df = pd.DataFrame(portfolio_results)
    summary_df.to_csv(f'results/phase5_summary_{era_name.replace(" ", "_")}.csv', index=False)
    
    print("\n" + "="*80)
    print(f"📊 PHASE 5 FINAL PORTFOLIO SUMMARY ({era_name})")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Generate the beautiful Master Dashboard
    generate_master_dashboard(summary_df, era_name)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()