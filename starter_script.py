"""
STARTER SCRIPT - Phase 1 Complete Workflow
Run this script to execute the basic pipeline for one stock (NVDA)

This demonstrates the complete flow from data download to LSTM training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from xgb_model import XGBStockPredictor

# Import our modules
from data_collection import download_stock_data, add_basic_features
from technical_indicators import calculate_all_indicators, select_features_for_models
from config import LSTM_CONFIG, SEQUENCE_LENGTH, START_DATE, END_DATE
from lstm_model import LSTMStockPredictor

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'results', 'results/figures', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory: {d}")

def step1_download_data():
    """Step 1: Download stock data"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATA")
    print("="*70)
    
    ticker = 'NVDA'
    df = download_stock_data(ticker, START_DATE, END_DATE)
    
    if df is not None:
        # Add basic features
        df = add_basic_features(df)
        
        # Save to CSV
        filepath = f'data/{ticker}_raw.csv'
        df.to_csv(filepath, index=False)
        print(f"\n✓ Data saved to {filepath}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        
        return df
    else:
        raise ValueError("Failed to download data!")

def step2_add_indicators(df):
    """Step 2: Calculate technical indicators"""
    print("\n" + "="*70)
    print("STEP 2: CALCULATING TECHNICAL INDICATORS")
    print("="*70)
    
    df_with_indicators = calculate_all_indicators(df)
    
    # Save processed data
    filepath = 'data/NVDA_processed.csv'
    df_with_indicators.to_csv(filepath, index=False)
    print(f"\n✓ Processed data saved to {filepath}")
    print(f"  Total features: {len(df_with_indicators.columns)}")
    
    # Show sample
    print("\nSample of data (last 5 rows, selected columns):")
    sample_cols = ['Date', 'Close', 'Return', 'SMA_10', 'RSI_14', 'BB_Middle']
    print(df_with_indicators[sample_cols].tail())
    
    return df_with_indicators

def step3_prepare_sequences(df):
    """Step 3: Prepare data for LSTM"""
    print("\n" + "="*70)
    print("STEP 3: PREPARING SEQUENCES FOR LSTM")
    print("="*70)
    
    # Get features for LSTM
    model_features = select_features_for_models()
    feature_cols = model_features['lstm']
    target_col = 'Return'
    
    print(f"\nFeatures being used ({len(feature_cols)}):")
    print(f"  {feature_cols}")
    print(f"\nTarget: {target_col}")
    print(f"Sequence Length: {SEQUENCE_LENGTH} days")
    
    # Initialize LSTM predictor
    lstm_predictor = LSTMStockPredictor(LSTM_CONFIG)
    
    # Prepare data
    X, y = lstm_predictor.prepare_data(df, SEQUENCE_LENGTH, feature_cols, target_col)
    
    print(f"\n✓ Sequences created:")
    print(f"  X shape: {X.shape} (samples, timesteps, features)")
    print(f"  y shape: {y.shape} (samples,)")
    
    return X, y, lstm_predictor

def step4_split_data(X, y, df):
    """Step 4: Split data (time-based)"""
    print("\n" + "="*70)
    print("STEP 4: SPLITTING DATA")
    print("="*70)
    
    # Time-based split (important for time series!)
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\nData splits:")
    print(f"  Training:   {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):4d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Get dates for later visualization
    # Need to account for sequence length offset
    dates = df['Date'].iloc[SEQUENCE_LENGTH:].values
    dates_test = dates[train_size+val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), dates_test

def step5_train_model(lstm_predictor, train_data, val_data):
    """Step 5: Train LSTM model"""
    print("\n" + "="*70)
    print("STEP 5: TRAINING LSTM MODEL")
    print("="*70)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Train
    print("\nStarting training (this may take 5-15 minutes)...")
    print("You'll see progress bars for each epoch.\n")
    
    history = lstm_predictor.train(
        X_train, y_train,
        X_val, y_val,
        save_path='./models/lstm_nvda'
    )
    
    print("\n✓ Training completed!")
    print(f"  Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history

def step6_evaluate_model(lstm_predictor, test_data, dates_test):
    """Step 6: Evaluate and visualize"""
    print("\n" + "="*70)
    print("STEP 6: EVALUATION AND VISUALIZATION")
    print("="*70)
    
    X_test, y_test = test_data
    
    # Evaluate
    metrics = lstm_predictor.evaluate(X_test, y_test)
    
    # Make predictions for plotting
    y_pred = lstm_predictor.predict(X_test).flatten()
    
    # Plot training history
    print("\nPlotting training history...")
    lstm_predictor.plot_training_history(save_path='results/figures/lstm_training_history.png')
    
    # Plot predictions
    print("\nPlotting predictions vs actual...")
    lstm_predictor.plot_predictions(
        y_test, y_pred,
        dates=dates_test,
        save_path='results/figures/lstm_predictions.png'
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/lstm_metrics.csv', index=False)
    print(f"\n✓ Metrics saved to results/lstm_metrics.csv")
    
    return metrics, y_pred

def main():
    print("="*70)
    print("PHASE 2: HYBRID ENSEMBLE (LSTM + XGBoost) - NVDA")
    print("="*70)
    
    # 1. Setup & Data (Same as before)
    create_directories()
    df_raw = step1_download_data()
    df_processed = step2_add_indicators(df_raw)
    
    # --- MODEL 1: LSTM (The Time Expert) ---
    print("\nTraining Model 1: LSTM...")
    X_lstm, y_lstm, lstm_model = step3_prepare_sequences(df_processed)
    train_lstm, val_lstm, test_lstm, dates_test = step4_split_data(X_lstm, y_lstm, df_processed)
    step5_train_model(lstm_model, train_lstm, val_lstm)
    
    # --- MODEL 2: XGBoost (The Technical Expert) ---
    print("\nTraining Model 2: XGBoost...")
    xgb_model = XGBStockPredictor()
    
    # Prepare tabular data (Different shape than LSTM!)
    # We use the same 'Return' target
    X_xgb, y_xgb, features = xgb_model.prepare_data(df_processed)
    
    # We must split XGB data exactly the same way as LSTM to align them
    # We use the same indices/dates
    train_size = len(train_lstm[0])
    val_size = len(val_lstm[0])
    
    X_xgb_train = X_xgb[:train_size]
    y_xgb_train = y_xgb[:train_size]
    
    X_xgb_val = X_xgb[train_size:train_size+val_size]
    y_xgb_val = y_xgb[train_size:train_size+val_size]
    
    X_xgb_test = X_xgb[train_size+val_size:]
    y_xgb_test = y_xgb[train_size+val_size:]
    
    # Train XGB
    xgb_model.train(X_xgb_train, y_xgb_train, X_xgb_val, y_xgb_val)
    
    # --- ENSEMBLE: Combine Predictions ---
    print("\n" + "="*70)
    print("STEP 7: ENSEMBLE EVALUATION & COMPARISON")
    print("="*70)
    
    # 1. Get raw predictions
    pred_lstm = lstm_model.predict(test_lstm[0]).flatten()
    pred_xgb = xgb_model.predict(X_xgb_test)
    
    # 2. ALIGNMENT: Trim all arrays to the shortest length
    min_len = min(len(pred_lstm), len(pred_xgb))
    
    pred_lstm = pred_lstm[-min_len:]
    pred_xgb = pred_xgb[-min_len:]
    y_actual = test_lstm[1][-min_len:]
    dates_aligned = dates_test[-min_len:]
    
    # 3. HYBRID: Average them (50/50 split)
    pred_ensemble = (pred_lstm * 0.5) + (pred_xgb * 0.5)
    
    # 4. Define Helper for Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    def get_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction Accuracy: % of times the sign (up/down) matches
        # We add 1e-8 to avoid division by zero or sign issues with exact 0
        correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
        acc = correct_direction / len(y_true) * 100
        
        return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Accuracy': acc}

    # 5. Calculate Metrics for All 3
    m_lstm = get_metrics(y_actual, pred_lstm, "LSTM Only")
    m_xgb = get_metrics(y_actual, pred_xgb, "XGBoost Only")
    m_ens = get_metrics(y_actual, pred_ensemble, "Ensemble (Hybrid)")
    
    # 6. Print Comparison Table
    results_df = pd.DataFrame([m_lstm, m_xgb, m_ens])
    print("\n🏆 FINAL RESULTS COMPARISON:")
    print(results_df.round(4).to_string(index=False))
    
    # 7. Plot Comparison
    plt.figure(figsize=(14, 6))
    plt.plot(dates_aligned, y_actual, label='Actual Returns', color='black', alpha=0.5)
    plt.plot(dates_aligned, pred_ensemble, label='Ensemble', color='green', linewidth=2)
    # Optional: Plot individual models faintly to see the difference
    plt.plot(dates_aligned, pred_lstm, label='LSTM', color='blue', alpha=0.3, linestyle='--')
    plt.plot(dates_aligned, pred_xgb, label='XGB', color='red', alpha=0.3, linestyle='--')
    
    plt.title('Phase 2: Ensemble vs. Individual Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/ensemble_results.png')
    print("\n✓ Saved plot to results/figures/ensemble_results.png")

if __name__ == "__main__":
    main()