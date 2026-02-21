"""
STARTER SCRIPT - Phase 3: Tri-Model Ensemble
Runs LSTM, XGBoost, and Transformer models, then combines them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'results', 'results/figures', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def step1_download_data():
    """Step 1: Download stock data"""
    ticker = 'NVDA'
    df = download_stock_data(ticker, START_DATE, END_DATE)
    
    if df is not None:
        df = add_basic_features(df)
        filepath = f'data/{ticker}_raw.csv'
        df.to_csv(filepath, index=False)
        return df
    else:
        raise ValueError("Failed to download data!")

def step2_add_indicators(df):
    """Step 2: Calculate technical indicators"""
    df_with_indicators = calculate_all_indicators(df)
    filepath = 'data/NVDA_processed.csv'
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
    history = model.train(
        X_train, y_train,
        X_val, y_val
    )
    return history


def main():
    print("\n" + "="*70)
    print("PHASE 3: TRI-MODEL ENSEMBLE (LSTM + XGB + Transformer) - NVDA")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Setup & Data
    create_directories()
    df_raw = step1_download_data()
    df_processed = step2_add_indicators(df_raw)

    # --- MODEL 1: LSTM (The Historian) ---
    print("\n" + "-"*30)
    print("TRAINING MODEL 1: LSTM")
    print("-"*30)
    X_lstm, y_lstm, lstm_model = step3_prepare_sequences(df_processed)
    train_lstm, val_lstm, test_lstm, dates_test = step4_split_data(X_lstm, y_lstm, df_processed)
    step5_train_model(lstm_model, train_lstm, val_lstm)
    
    # --- MODEL 2: XGBoost (The Logic Expert) ---
    print("\n" + "-"*30)
    print("TRAINING MODEL 2: XGBOOST")
    print("-"*30)
    xgb_model = XGBStockPredictor()
    X_xgb, y_xgb, features = xgb_model.prepare_data(df_processed)
    
    # Align XGB split with LSTM split
    train_size = len(train_lstm[0])
    val_size = len(val_lstm[0])
    
    X_xgb_train = X_xgb[:train_size]
    y_xgb_train = y_xgb[:train_size]
    X_xgb_val = X_xgb[train_size:train_size+val_size]
    y_xgb_val = y_xgb[train_size:train_size+val_size]
    X_xgb_test = X_xgb[train_size+val_size:]
    
    xgb_model.train(X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val)

    # --- MODEL 3: Transformer (The Visionary) ---
    print("\n" + "-"*30)
    print("TRAINING MODEL 3: TRANSFORMER")
    print("-"*30)
    transformer_model = TransformerStockPredictor(TRANSFORMER_CONFIG)
    transformer_model.train(train_lstm[0], train_lstm[1], val_lstm[0], val_lstm[1])

    # --- ENSEMBLE: Combine 3 Predictions ---
    print("\n" + "="*70)
    print("STEP 7: TRI-MODEL ENSEMBLE EVALUATION (MAJORITY VOTE)")
    print("="*70)
    
    # 1. Get raw predictions
    pred_lstm = lstm_model.predict(test_lstm[0]).flatten()
    pred_xgb = xgb_model.predict(X_xgb_test)
    pred_trans = transformer_model.predict(test_lstm[0]).flatten()

    # 2. ALIGNMENT
    min_len = min(len(pred_lstm), len(pred_xgb), len(pred_trans))
    pred_lstm = pred_lstm[-min_len:]
    pred_xgb = pred_xgb[-min_len:]
    pred_trans = pred_trans[-min_len:]
    y_actual = test_lstm[1][-min_len:]
    dates_aligned = dates_test[-min_len:]
    
    # 3. MAJORITY RULE VOTING (The Fix)
    # Convert predictions to pure signals (+1 for Up, -1 for Down, 0 for Flat)
    sig_lstm = np.sign(pred_lstm)
    sig_xgb = np.sign(pred_xgb)
    sig_trans = np.sign(pred_trans)
    
    # Add the votes together
    # Example: (+1) + (-1) + (+1) = +1 (Up wins)
    vote_sum = sig_lstm + sig_xgb + sig_trans
    
    # The final ensemble prediction is just the winning direction
    # We multiply by the average absolute prediction to give it a realistic "magnitude" for RMSE calculations
    avg_magnitude = (np.abs(pred_lstm) + np.abs(pred_xgb)) / 2.0  # Exclude Transformer from magnitude to be safe
    pred_ensemble = np.sign(vote_sum) * avg_magnitude
    
    # 4. Metrics Helper
    def get_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
        acc = correct_direction / len(y_true) * 100
        return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Accuracy': acc}

    # 5. Calculate & Print Comparison
    metrics_list = [
        get_metrics(y_actual, pred_lstm, "LSTM Only"),
        get_metrics(y_actual, pred_xgb, "XGBoost Only"),
        get_metrics(y_actual, pred_trans, "Transformer Only"),
        get_metrics(y_actual, pred_ensemble, "Majority Vote Ensemble")
    ]
    
    results_df = pd.DataFrame(metrics_list)
    print("\n🏆 FINAL RESULTS COMPARISON:")
    print(results_df.round(4).to_string(index=False))
    
    # 6. Plot
    plt.figure(figsize=(14, 6))
    plt.plot(dates_aligned, y_actual, label='Actual Returns', color='black', alpha=0.3)
    plt.plot(dates_aligned, pred_ensemble, label='Tri-Model Ensemble', color='green', linewidth=2)
    plt.plot(dates_aligned, pred_trans, label='Transformer', color='purple', alpha=0.3, linestyle='--')
    
    plt.title(f'Phase 3 Results: Ensemble vs Actual ({min_len} days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/phase3_results.png')
    print("\n✓ Saved plot to results/figures/phase3_results.png")
    
    return pred_ensemble, y_actual

if __name__ == "__main__":
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    main()