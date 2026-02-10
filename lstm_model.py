"""
LSTM Model Implementation for Stock Prediction
Part of Event-Aware Ensemble Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

class LSTMStockPredictor:
    """
    LSTM model for stock price/return prediction
    """
    
    def __init__(self, config):
        """
        Initialize LSTM model
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with model hyperparameters
        """
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def create_sequences(self, data, sequence_length, target_col='Return'):
        """
        Create sequences for LSTM training
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with features
        sequence_length : int
            Length of input sequences (e.g., 60 days)
        target_col : str
            Column name for target variable
        
        Returns:
        --------
        X : np.array
            Input sequences (samples, timesteps, features)
        y : np.array
            Target values (samples,)
        """
        X, y = [], []
        
        # Get feature columns (all except target)
        feature_cols = [col for col in data.columns if col != target_col]
        
        for i in range(sequence_length, len(data)):
            # Input sequence
            X.append(data[feature_cols].iloc[i-sequence_length:i].values)
            # Target value
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (timesteps, features)
        
        Returns:
        --------
        model : keras.Model
            Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(units=self.config['units'][0],
                 return_sequences=True,
                 input_shape=input_shape),
            Dropout(self.config['dropout']),
            
            # Second LSTM layer
            LSTM(units=self.config['units'][1],
                 return_sequences=False),
            Dropout(self.config['dropout']),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            
            # Output layer
            Dense(1)
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer,
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    
    def prepare_data(self, df, sequence_length, feature_cols, target_col='Return'):
        """
        Prepare data for LSTM training
        """
        # 1. Clean feature columns
        feature_cols = list(dict.fromkeys(feature_cols))
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # 2. Define all unique columns needed (Features + Target)
        # Using set() ensures we don't select 'Return' twice if it's both a feature and target
        required_cols = list(set(feature_cols + [target_col]))
        
        # 3. Create clean dataframe with only unique columns
        data = df[required_cols].copy().dropna()
        
        # 4. Scale features
        # Note: We fit scaler only on the feature columns
        scaled_features = self.scaler.fit_transform(data[feature_cols])
        
        # 5. Create sequences
        X, y = [], []
        
        # Extract target values safely as a 1D array first
        target_values = data[target_col].values
        
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(target_values[i])
            
        return np.array(X), np.array(y)
    def train(self, X_train, y_train, X_val, y_val, save_path='./models/lstm_model'):
        """
        Train LSTM model
        
        Parameters:
        -----------
        X_train : np.array
            Training sequences
        y_train : np.array
            Training targets
        X_val : np.array
            Validation sequences
        y_val : np.array
            Validation targets
        save_path : str
            Path to save best model
        
        Returns:
        --------
        history : keras.History
            Training history
        """
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
        
        # Print model summary
        print("\n" + "="*60)
        print("LSTM Model Architecture")
        print("="*60)
        self.model.summary()
        print("="*60 + "\n")
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(save_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.array
            Input sequences
        
        Returns:
        --------
        predictions : np.array
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
      """
    Evaluate model performance
    """
    # Make predictions
      y_pred = self.predict(X_test)
    
    # Flatten both to 1D arrays
      y_pred = y_pred.flatten()
      y_test = np.array(y_test).flatten()
    
    # Calculate metrics
      metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'Direction_Accuracy': self._direction_accuracy(y_test, y_pred)
      }
    
    # Print metrics
      print("\n" + "="*60)
      print("MODEL EVALUATION METRICS")
      print("="*60)
      for metric_name, value in metrics.items():
          print(f"{metric_name:20s}: {value:.4f}")
      print("="*60 + "\n")
    
      return metrics
    
    def _direction_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (up/down prediction)
        
        Parameters:
        -----------
        y_true : np.array
            True values
        y_pred : np.array
            Predicted values
        
        Returns:
        --------
        float
            Directional accuracy (0-1)
        """
        correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
        return correct_direction / len(y_true)
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save figure
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, dates=None, save_path=None):
        """
        Plot predictions vs actual values
        
        Parameters:
        -----------
        y_true : np.array
            True values
        y_pred : np.array
            Predicted values
        dates : array-like, optional
            Date labels for x-axis
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(14, 6))
        
        x = dates if dates is not None else range(len(y_true))
        
        plt.plot(x, y_true, label='Actual Returns', alpha=0.7, linewidth=2)
        plt.plot(x, y_pred, label='Predicted Returns', alpha=0.7, linewidth=2)
        
        plt.xlabel('Time' if dates is None else 'Date')
        plt.ylabel('Returns')
        plt.title('LSTM Predictions vs Actual Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if dates is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage function
def example_usage():
    """
    Example of how to use the LSTMStockPredictor class
    """
    from config import LSTM_CONFIG, SEQUENCE_LENGTH
    
    print("="*70)
    print("LSTM STOCK PREDICTOR - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize model
    lstm_predictor = LSTMStockPredictor(LSTM_CONFIG)
    
    print("\n1. Load your data:")
    print("   df = pd.read_csv('data/NVDA_data.csv')")
    
    print("\n2. Prepare features:")
    print("   feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI_14']")
    
    print("\n3. Prepare data:")
    print("   X, y = lstm_predictor.prepare_data(df, SEQUENCE_LENGTH, feature_cols)")
    
    print("\n4. Split data:")
    print("   # Use time-based split (not random!)")
    print("   train_size = int(0.7 * len(X))")
    print("   val_size = int(0.15 * len(X))")
    
    print("\n5. Train model:")
    print("   history = lstm_predictor.train(X_train, y_train, X_val, y_val)")
    
    print("\n6. Evaluate:")
    print("   metrics = lstm_predictor.evaluate(X_test, y_test)")
    
    print("\n7. Make predictions:")
    print("   predictions = lstm_predictor.predict(X_test)")
    
    print("\n8. Visualize:")
    print("   lstm_predictor.plot_training_history()")
    print("   lstm_predictor.plot_predictions(y_test, predictions)")
    
    print("="*70)


if __name__ == "__main__":
    example_usage()
