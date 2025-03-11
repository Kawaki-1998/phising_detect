import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn
import joblib
from typing import Dict, Any, Tuple
import logging
from src.config.config import MODEL_CONFIG, MLFLOW_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_score = 0
        self.best_params = {}
        self.best_model_type = None
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray,
                 y_train: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Select model type
        model_type = trial.suggest_categorical("model_type", 
            ["rf", "xgb", "lgb"])
        
        if model_type == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
            }
            model = RandomForestClassifier(**params, random_state=MODEL_CONFIG["random_state"])
        
        elif model_type == "xgb":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=MODEL_CONFIG["random_state"])
        
        else:  # lgb
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = lgb.LGBMClassifier(**params, random_state=MODEL_CONFIG["random_state"])

        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)

        # Track with MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("accuracy", score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_params = params
                self.best_model_type = model_type

        return score

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train models using Optuna for hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=MODEL_CONFIG["n_trials"]
        )
        
        # Log best results
        logger.info(f"Best trial accuracy: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        # Save best model
        if self.best_model is not None:
            model_path = "models/best_phishing_model.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
            
            # Log final model with MLflow
            with mlflow.start_run():
                mlflow.log_params(study.best_trial.params)
                mlflow.log_metric("best_accuracy", study.best_trial.value)
                mlflow.sklearn.log_model(self.best_model, "model")
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        return self.best_model.predict_proba(X)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train("data/dataset_full.csv") 