import os
import logging
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.visualization.visualize import (
    plot_feature_importance,
    plot_model_comparison,
    plot_confusion_matrix,
    analyze_feature_distributions,
    plot_roc_curve
)
from src.config.config import MODEL_CONFIG, MLFLOW_CONFIG, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_pipeline(data_path: str = None):
    """Run the complete training pipeline."""
    try:
        # Create necessary directories
        viz_dir = os.path.join(DATA_DIR, "visualizations")
        model_dir = os.path.join(DATA_DIR, "..", "models")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
        
        # Initialize components
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        
        # Use small dataset for quick testing if no path provided
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "dataset_small.csv")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = preprocessor.load_and_preprocess(data_path)
        
        # Analyze feature distributions
        logger.info("Analyzing feature distributions...")
        analyze_feature_distributions(
            df.drop('is_phishing', axis=1),
            save_dir=viz_dir
        )
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Train models and get best model
        logger.info("Starting model training...")
        trainer.train(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Get predictions from best model
        y_pred = trainer.best_model.predict(X_test_scaled)
        y_pred_proba = trainer.best_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        clf_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Get feature importance
        if hasattr(trainer.best_model, 'feature_importances_'):
            importance_scores = trainer.best_model.feature_importances_
        else:
            importance_scores = np.zeros(X_train.shape[1])
        
        # Plot and save visualizations
        plot_feature_importance(
            X_train.columns.tolist(),
            importance_scores,
            save_path=os.path.join(viz_dir, "feature_importance.png")
        )
        
        plot_confusion_matrix(
            conf_matrix,
            save_path=os.path.join(viz_dir, "confusion_matrix.png")
        )
        
        plot_roc_curve(
            fpr, tpr, roc_auc,
            save_path=os.path.join(viz_dir, "roc_curve.png")
        )
        
        # Log results
        logger.info("\nClassification Report:\n" + clf_report)
        logger.info(f"\nBest model type: {trainer.best_model_type}")
        logger.info(f"Best model accuracy: {trainer.best_score:.4f}")
        
        # Log final results to MLflow
        with mlflow.start_run():
            mlflow.log_params(trainer.best_params)
            mlflow.log_metric("accuracy", trainer.best_score)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log artifacts
            mlflow.log_artifacts(viz_dir, "visualizations")
            
            # Log model
            mlflow.sklearn.log_model(trainer.best_model, "model")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_training_pipeline() 