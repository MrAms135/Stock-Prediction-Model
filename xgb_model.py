import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class XGBStockPredictor:
    """
    XGBoost model for stock prediction (The "Logic Board")
    """
    
    def __init__(self, random_state=42):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
    def prepare_data(self, df, target_col='Return', shift=1):
        """
        Prepare tabular data for XGBoost
        """
        data = df.copy()
        
        # Create target: Shift returns back by 'shift' days
        data['Target'] = data[target_col].shift(-shift)
        
        # Drop the last 'shift' rows which now have NaN targets
        data = data.dropna()
        
        # ------------------- THE FIX IS HERE -------------------
        # Define columns to exclude (Non-numeric stuff)
        exclude_cols = ['Date', 'Target', target_col, 'Ticker']
        
        # Select features: All columns that represent numeric data
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        # -------------------------------------------------------
        
        # Double check: Ensure we only keep numeric columns
        # This protects you if you add 'News_Headline' text later!
        X = data[feature_cols].select_dtypes(include=[np.number]).values
        y = data['Target'].values
        
        print(f"XGB Data Prepared: {X.shape} samples")
        return X, y, feature_cols
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with early stopping"""
        print("Training XGBoost...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100  # Print every 100 rounds
        )
        print("✓ XGBoost training complete!")
        
    def predict(self, X):
        return self.model.predict(X)