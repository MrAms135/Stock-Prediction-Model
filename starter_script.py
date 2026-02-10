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
    """
    Main execution function
    Runs the complete workflow from data download to model evaluation
    """
    print("\n" + "="*70)
    print(" "*15 + "LSTM STOCK PREDICTION PIPELINE")
    print(" "*20 + "NVIDIA (NVDA) EXAMPLE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup
        create_directories()
        
        # Step 1: Download data
        df_raw = step1_download_data()
        
        # Step 2: Calculate indicators
        df_processed = step2_add_indicators(df_raw)
        
        # Step 3: Prepare sequences
        X, y, lstm_predictor = step3_prepare_sequences(df_processed)
        
        # Step 4: Split data
        train_data, val_data, test_data, dates_test = step4_split_data(X, y, df_processed)
        
        # Step 5: Train model
        history = step5_train_model(lstm_predictor, train_data, val_data)
        
        # Step 6: Evaluate
        metrics, predictions = step6_evaluate_model(lstm_predictor, test_data, dates_test)
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📊 RESULTS SUMMARY:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  R²:   {metrics['R2']:.6f}")
        print(f"  Direction Accuracy: {metrics['Direction_Accuracy']*100:.2f}%")
        
        print("\n📁 FILES CREATED:")
        print("  ✓ data/NVDA_raw.csv - Raw stock data")
        print("  ✓ data/NVDA_processed.csv - Data with indicators")
        print("  ✓ models/lstm_nvda/best_model.h5 - Trained LSTM model")
        print("  ✓ results/lstm_metrics.csv - Evaluation metrics")
        print("  ✓ results/figures/lstm_training_history.png")
        print("  ✓ results/figures/lstm_predictions.png")
        
        print("\n🎯 NEXT STEPS:")
        print("  1. Review the plots in results/figures/")
        print("  2. Try adjusting hyperparameters in config.py")
        print("  3. Add more features from technical_indicators.py")
        print("  4. Build Transformer and LightGBM models")
        print("  5. Create the ensemble!")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        return lstm_predictor, metrics, predictions
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    # Run the pipeline
    model, metrics, predictions = main()
