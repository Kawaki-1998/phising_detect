import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from src.config.config import DATA_DIR

def plot_feature_importance(feature_names: List[str], importance_scores: List[float], 
                          title: str = "Feature Importance", save_path: str = None):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame({
        'features': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True)
    
    sns.barplot(data=importance_df, y='features', x='importance')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(model_scores: Dict[str, float], metric: str = "Accuracy",
                         save_path: str = None):
    """Plot model performance comparison."""
    plt.figure(figsize=(10, 6))
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    sns.barplot(x=models, y=scores)
    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def analyze_feature_distributions(df: pd.DataFrame, save_dir: str = None):
    """Analyze and plot feature distributions."""
    features = df.columns
    n_features = len(features)
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 4*n_rows))
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, 2, i)
        sns.histplot(data=df, x=feature, bins=30)
        plt.title(f"{feature} Distribution")
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, 
                   save_path: str = None):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show() 