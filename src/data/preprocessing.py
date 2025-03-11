import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging
from src.features.feature_engineering import FeatureExtractor
from src.config.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
    def load_and_preprocess(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Basic cleaning
            df = df.dropna()
            
            logger.info("Extracting features")
            # Extract selected features
            features_df = self.feature_extractor.extract_features_batch(df)
            
            # Add target variable (renamed from 'phishing' to 'is_phishing')
            features_df['is_phishing'] = df['phishing'].values
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = MODEL_CONFIG["test_size"],
                   random_state: int = MODEL_CONFIG["random_state"]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        X = df.drop('is_phishing', axis=1)
        y = df['is_phishing']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic statistics for each feature."""
        stats = df.describe()
        logger.info("Feature statistics calculated")
        return stats
    
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """Detect outliers using z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        outliers = (z_scores > threshold).any(axis=1)
        
        logger.info(f"Found {outliers.sum()} samples with outliers")
        return df[outliers]
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_count = df.isnull().sum()
        if missing_count.any():
            logger.warning(f"Found missing values:\n{missing_count[missing_count > 0]}")
            df = df.fillna(df.mean())
            logger.info("Missing values handled")
        return df
    
    def scale_features(self, train_data: pd.DataFrame, 
                      test_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features using min-max scaling."""
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        
        # Calculate min and max from training data
        min_vals = train_data[numeric_cols].min()
        max_vals = train_data[numeric_cols].max()
        
        # Scale training data
        scaled_train = train_data.copy()
        scaled_train[numeric_cols] = (train_data[numeric_cols] - min_vals) / (max_vals - min_vals)
        
        # Scale test data if provided
        scaled_test = None
        if test_data is not None:
            scaled_test = test_data.copy()
            scaled_test[numeric_cols] = (test_data[numeric_cols] - min_vals) / (max_vals - min_vals)
        
        return scaled_train, scaled_test 