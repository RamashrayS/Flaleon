import os

# Project Path Configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

# Create necessary directories
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR, MODELS_DIR, OUTPUTS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Data Processing Configs
ALIGN_TIME_OFFSET = 0.322 # shift HEL1OS time by subtracting this offset (seconds)
ALIGN_TOLERANCE = 1.0     # tolerance for merge_asof (seconds)

# Forecasting Configurations
FORECAST_HORIZONS = {
    '5min': 300,   # seconds
    '10min': 600,  # seconds
    '30min': 1800  # seconds
}

# Machine Learning Configs
RANDOM_SEED = 42

# Label mapping for classification (Task C)
LABEL_CLASS_MAP = {
    'Quiet': 0,
    'C': 1,
    'M': 2,
    'X': 3
}

# Debug Mode Flag
DEBUG_MODE = True

# Labeling Strategy: 'overlap' (row-by-row matching) or 'daily_strongest' (Option A: entire day labeled with strongest flare in window)
LABELING_STRATEGY = 'overlap'

# Model Configuration
USE_CATBOOST = True

# Feature Selection Flag & Methods
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_METHODS = ['rf', 'xgb', 'lgb', 'correlation'] # Default methods for selection
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 1e-4

# Hyperparameter Optimization Config
USE_OPTUNA = False
OPTUNA_N_TRIALS = 15

# Training Balancing Flags
USE_CLASS_WEIGHTS = True
USE_BALANCED_SAMPLING = True

# Stage Checkpointing Flags
FORCE_RERUN = False
RETRAIN_MODELS = False



