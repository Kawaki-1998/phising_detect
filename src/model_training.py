import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import joblib
from typing import Tuple, Dict
import logging
import lightgbm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define required features
REQUIRED_FEATURES = {
    'domain_length',
    'qty_dot_domain',  # num_dots
    'qty_hyphen_domain',  # num_hyphens
    'qty_vowels_domain',  # num_digits (proxy)
    'domain_in_ip',
    'server_client_domain',
    'time_response',
    'domain_spf',
    'asn_ip',
    'time_domain_activation',
    'time_domain_expiration',
    'qty_ip_resolved',
    'qty_nameservers',
    'qty_mx_servers',
    'ttl_hostname',
    'tls_ssl_certificate',
    'qty_redirects',
    'url_google_index',
    'domain_google_index',
    'url_shortened'
}

class ModelTrainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the dataset."""
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} rows from {self.data_path}")
            
            # Add placeholder features for suspicious keywords and brand names
            df['has_suspicious_keywords'] = 0
            df['has_brand_name'] = 0
            
            # Select required features
            features = list(REQUIRED_FEATURES) + ['has_suspicious_keywords', 'has_brand_name']
            X = df[features]
            y = df['phishing']
            
            # Log feature info
            logger.info(f"Selected {len(features)} features: {', '.join(features)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_model(self, params: Dict = None) -> Tuple[LGBMClassifier, Dict]:
        """Train the model with the given parameters."""
        try:
            # Load and preprocess data
            X, y = self.load_and_preprocess_data()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features while preserving feature names
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Save the scaler
            joblib.dump(self.scaler, 'models/feature_scaler.pkl')
            logger.info("Saved feature scaler to models/feature_scaler.pkl")
            
            # Set default parameters if none provided
            if params is None:
                params = {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 10,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            
            # Train model
            model = LGBMClassifier(**params)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                eval_metric='auc',
                callbacks=[
                    lightgbm.early_stopping(50),
                    lightgbm.log_evaluation(100)
                ]
            )
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Log metrics
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score
            }
            
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            
            # Save feature names
            feature_names = list(X_train.columns)
            joblib.dump(feature_names, 'models/feature_names.pkl')
            logger.info("Saved feature names to models/feature_names.pkl")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model: LGBMClassifier, metrics: Dict):
        """Save the trained model and log metrics to MLflow."""
        try:
            # Save model
            joblib.dump(model, 'models/best_phishing_model.pkl')
            logger.info("Saved model to models/best_phishing_model.pkl")
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def train_and_save_model():
    """Main function to train and save the model."""
    try:
        trainer = ModelTrainer('data/processed/phishing_data.csv')
        model, metrics = trainer.train_model()
        trainer.save_model(model, metrics)
        logger.info("Model training and saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 