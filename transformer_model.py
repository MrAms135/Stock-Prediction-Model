import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os

class TransformerStockPredictor:
    """
    Transformer model with Multi-Head Attention for Time Series
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        A single Transformer block (Attention + Feed Forward)
        """
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs  # Residual Connection (The "Skip" Logic)

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # 1. Transformer Block
        x = self.transformer_encoder(
            inputs,
            head_size=self.config['head_size'],
            num_heads=self.config['num_heads'],
            ff_dim=self.config['ff_dim'],
            dropout=self.config['dropout']
        )
        
        # 2. Global Average Pooling (Summarize the 60 days into one vector)
        x = layers.GlobalAveragePooling1D()(x)
        
        # 3. Output Layers
        x = layers.Dropout(self.config['dropout'])(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.config['dropout'])(x)
        outputs = layers.Dense(1, activation="linear")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, save_path='./models/transformer_nvda'):
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
            
        print("\n" + "="*60)
        print("Transformer Architecture")
        print("="*60)
        self.model.summary()
        
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        print("Starting Transformer training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0)