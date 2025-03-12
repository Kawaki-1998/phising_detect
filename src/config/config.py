import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
MODEL_DIR = BASE_DIR / "src" / "models"
LOG_DIR = BASE_DIR / "src" / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "n_trials": 100,  # Number of Optuna trials
    "early_stopping_rounds": 50,
}

# Feature Configuration
FEATURE_CONFIG = {
    "selected_features": [
        "url_length",
        "special_chars",
        "num_dots",
        # Add more features based on analysis
    ]
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "experiment_name": "phishing_detection",
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "debug": bool(os.getenv("DEBUG", False)),
} 