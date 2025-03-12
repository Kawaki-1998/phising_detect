import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_dataset():
    """Analyze the phishing detection dataset."""
    try:
        # Load the data
        logger.info("Loading dataset...")
        df = pd.read_csv('data/processed/phishing_data.csv')
        
        # Basic dataset information
        logger.info("\nDataset Overview:")
        print("-" * 50)
        print(f"Total samples: {len(df)}")
        print(f"Number of features: {len(df.columns) - 1}")  # Excluding target variable
        print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        # Class distribution
        logger.info("\nClass Distribution:")
        print("-" * 50)
        class_dist = df['phishing'].value_counts()
        print(f"Legitimate domains: {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)")
        print(f"Phishing domains: {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)")
        
        # Feature statistics
        logger.info("\nFeature Statistics:")
        print("-" * 50)
        print("\nNumerical Features Summary:")
        print(df.describe())
        
        # Missing values
        logger.info("\nMissing Values Analysis:")
        print("-" * 50)
        missing = df.isnull().sum()
        if missing.any():
            print("\nFeatures with missing values:")
            print(missing[missing > 0])
        else:
            print("No missing values found in the dataset")
        
        # Feature correlations with target
        logger.info("\nTop Feature Correlations with Target:")
        print("-" * 50)
        correlations = df.corr()['phishing'].sort_values(ascending=False)
        print("\nMost positively correlated features:")
        print(correlations[1:6])  # Excluding self-correlation
        print("\nMost negatively correlated features:")
        print(correlations[-5:])
        
        # Save analysis results
        logger.info("\nSaving analysis results...")
        with open('data/data_analysis_report.txt', 'w') as f:
            f.write("Phishing Domain Detection Dataset Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Number of features: {len(df.columns) - 1}\n")
            f.write(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB\n\n")
            f.write("Class Distribution:\n")
            f.write(f"Legitimate domains: {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)\n")
            f.write(f"Phishing domains: {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)\n")
        
        logger.info("Analysis complete. Results saved to data/data_analysis_report.txt")
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_dataset() 