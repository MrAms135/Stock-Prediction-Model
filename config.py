"""
Configuration File for Event-Aware Stock Prediction Ensemble
Contains all hyperparameters, settings, and constants
"""

# ============================================================================
# PROJECT SETTINGS
# ============================================================================

PROJECT_NAME = "Event-Aware Stock Ensemble"
VERSION = "1.0.0"
RANDOM_SEED = 42

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Stock tickers
AI_WINNERS = ['NVDA', 'MSFT', 'GOOGL']
SAAS_VICTIMS = ['CRM', 'ADBE', 'NOW']
ALL_STOCKS = AI_WINNERS + SAAS_VICTIMS

# Date ranges
START_DATE = '2019-01-01'
END_DATE = '2026-02-10'
SAASAPOCALYPSE_DATE = '2026-02-03'

# Train/Validation/Test split (time-based)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Sequence length for LSTM/Transformer
SEQUENCE_LENGTH = 60  # 60 days of historical data

# Target variable
TARGET_TYPE = 'return'  # 'return' or 'price'
PREDICTION_HORIZON = 1  # Predict 1 day ahead

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# LSTM Model
LSTM_CONFIG = {
    'units': [128, 64],  # Number of units in each layer
    'dropout': 0.2,
    'recurrent_dropout': 0.2,
    'activation': 'tanh',
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10  # Early stopping patience
}

# Transformer Model
TRANSFORMER_CONFIG = {
    'num_layers': 4,
    'd_model': 64,  # Embedding dimension
    'num_heads': 4,
    'dff': 128,  # Feed-forward dimension
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0001,
    'patience': 10
}

# LightGBM Model
LIGHTGBM_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

# ============================================================================
# ANOMALY DETECTION SETTINGS
# ============================================================================

ANOMALY_CONFIG = {
    'method': 'isolation_forest',  # 'isolation_forest' or 'autoencoder'
    'contamination': 0.05,  # Expected proportion of outliers
    'features_to_use': ['Return', 'Volume_Change', 'HV_20', 'ATR_14'],
    
    # For Isolation Forest
    'n_estimators': 100,
    'max_samples': 256,
    
    # For rolling window analysis
    'rolling_window': 20,  # Days to look back
    'threshold_multiplier': 2.0  # Std deviations for anomaly
}

# ============================================================================
# ENSEMBLE SETTINGS
# ============================================================================

ENSEMBLE_CONFIG = {
    'fusion_method': 'adaptive_stacking',  # 'simple_avg', 'weighted_avg', 'stacking', 'adaptive_stacking'
    
    # For weighted averaging
    'initial_weights': {
        'lstm': 0.33,
        'transformer': 0.33,
        'lightgbm': 0.34
    },
    
    # For stacking meta-learner
    'meta_learner': {
        'hidden_units': [32, 16],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 30
    },
    
    # For adaptive weighting (regime-based)
    'adaptive': {
        'normal_weights': {'lstm': 0.35, 'transformer': 0.30, 'lightgbm': 0.35},
        'crisis_weights': {'lstm': 0.25, 'transformer': 0.25, 'lightgbm': 0.50},
        # LightGBM gets more weight in crisis due to its robustness
        'regime_switch_threshold': 0.7  # Anomaly score threshold
    }
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVALUATION_METRICS = [
    'rmse',
    'mae', 
    'mape',
    'r2',
    'direction_accuracy',
    'sharpe_ratio',
    'max_drawdown'
]

# Baseline strategies for comparison
BASELINE_STRATEGIES = [
    'buy_and_hold',
    'moving_average_crossover',
    'persistence'  # Previous day's return
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VIZ_CONFIG = {
    'figure_size': (12, 6),
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'save_format': 'png',
    'dpi': 300
}

# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    'data_dir': './data/',
    'raw_data': './data/raw/',
    'processed_data': './data/processed/',
    'models_dir': './models/',
    'results_dir': './results/',
    'figures_dir': './results/figures/',
    'logs_dir': './logs/'
}

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Features for each model type
FEATURES = {
    'lstm': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'Log_Return',
        'SMA_10', 'SMA_20', 'RSI_14', 'BB_Middle', 'Volume_SMA_20'
    ],
    
    'transformer': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'Log_Return',
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12',
        'RSI_14', 'MACD', 'Stoch_K', 'BB_Middle'
    ],
    
    'lightgbm': 'all_technical'  # Will use all technical indicators
}

# ============================================================================
# SAASAPOCALYPSE EVENT ANALYSIS SETTINGS
# ============================================================================

EVENT_ANALYSIS = {
    'pre_event_window': 30,  # Days before Feb 3, 2026
    'post_event_window': 7,  # Days after Feb 3, 2026
    'comparison_groups': {
        'ai_winners': AI_WINNERS,
        'saas_victims': SAAS_VICTIMS
    }
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './logs/training.log'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(section=None):
    """
    Get configuration for a specific section
    
    Parameters:
    -----------
    section : str, optional
        Configuration section name (e.g., 'LSTM_CONFIG', 'ENSEMBLE_CONFIG')
        If None, returns all configurations
    
    Returns:
    --------
    dict or object
        Configuration dictionary or specific config object
    """
    if section is None:
        return {
            'PROJECT': {
                'NAME': PROJECT_NAME,
                'VERSION': VERSION,
                'SEED': RANDOM_SEED
            },
            'DATA': {
                'STOCKS': ALL_STOCKS,
                'DATE_RANGE': (START_DATE, END_DATE),
                'SPLITS': (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
            },
            'MODELS': {
                'LSTM': LSTM_CONFIG,
                'TRANSFORMER': TRANSFORMER_CONFIG,
                'LIGHTGBM': LIGHTGBM_CONFIG
            },
            'ANOMALY': ANOMALY_CONFIG,
            'ENSEMBLE': ENSEMBLE_CONFIG,
            'PATHS': PATHS
        }
    
    # Return specific section
    config_map = {
        'lstm': LSTM_CONFIG,
        'transformer': TRANSFORMER_CONFIG,
        'lightgbm': LIGHTGBM_CONFIG,
        'anomaly': ANOMALY_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'paths': PATHS,
        'features': FEATURES,
        'event': EVENT_ANALYSIS
    }
    
    return config_map.get(section.lower(), None)

def print_config():
    """
    Print all configuration settings
    """
    print("=" * 70)
    print(f"  {PROJECT_NAME} v{VERSION}")
    print("=" * 70)
    print(f"\n📊 STOCKS: {', '.join(ALL_STOCKS)}")
    print(f"📅 DATE RANGE: {START_DATE} to {END_DATE}")
    print(f"⚠️  EVENT DATE: {SAASAPOCALYPSE_DATE}")
    print(f"\n🧠 MODELS:")
    print(f"   - LSTM: {LSTM_CONFIG['units']} units")
    print(f"   - Transformer: {TRANSFORMER_CONFIG['num_layers']} layers")
    print(f"   - LightGBM: {LIGHTGBM_CONFIG['n_estimators']} estimators")
    print(f"\n🎯 ENSEMBLE: {ENSEMBLE_CONFIG['fusion_method']}")
    print(f"\n📏 SEQUENCE LENGTH: {SEQUENCE_LENGTH} days")
    print(f"🎲 RANDOM SEED: {RANDOM_SEED}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()

# Transformer Hyperparameters
TRANSFORMER_CONFIG = {
    'head_size': 256,     # Size of attention head
    'num_heads': 4,       # Number of attention heads (Parallel focus points)
    'ff_dim': 4,          # Feed-forward layer size
    'dropout': 0.25,      # Dropout rate
    'learning_rate': 0.0001,
    'epochs': 80,
    'batch_size': 64
}