import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the features we want to keep with their mappings
REQUIRED_FEATURES = {
    'domain_length': 'domain_length',
    'num_dots': 'qty_dot_domain',
    'num_hyphens': 'qty_hyphen_domain',
    'num_digits': 'qty_vowels_domain',  # Using vowels count as proxy for digits
    'has_suspicious_keywords': None,  # Will be computed
    'has_brand_name': None,  # Will be computed
    'domain_in_ip': 'domain_in_ip',
    'server_client_domain': 'server_client_domain',
    'time_response': 'time_response',
    'domain_spf': 'domain_spf',
    'asn_ip': 'asn_ip',
    'time_domain_activation': 'time_domain_activation',
    'time_domain_expiration': 'time_domain_expiration',
    'qty_ip_resolved': 'qty_ip_resolved',
    'qty_nameservers': 'qty_nameservers',
    'qty_mx_servers': 'qty_mx_servers',
    'ttl_hostname': 'ttl_hostname',
    'tls_ssl_certificate': 'tls_ssl_certificate',
    'qty_redirects': 'qty_redirects',
    'url_google_index': 'url_google_index',
    'domain_google_index': 'domain_google_index',
    'url_shortened': 'url_shortened'
}

def preprocess_data(input_path: str, output_path: str):
    """Preprocess the data to include only required features."""
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        # Create new features
        df['has_suspicious_keywords'] = 0  # Placeholder
        df['has_brand_name'] = 0  # Placeholder
        
        # Create new dataframe with required features
        new_df = pd.DataFrame()
        
        # Map features from old to new names
        for new_feature, old_feature in REQUIRED_FEATURES.items():
            if old_feature is None:
                # Feature already created above
                new_df[new_feature] = df[new_feature]
            else:
                # Map from old feature name
                new_df[new_feature] = df[old_feature]
        
        # Add target variable
        new_df['phishing'] = df['phishing']
        
        # Save preprocessed data
        new_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(new_df)} rows to {output_path}")
        
        # Print feature statistics
        logger.info("\nFeature statistics:")
        for col in new_df.columns:
            if col != 'phishing':
                logger.info(f"{col}:")
                logger.info(f"  Missing values: {new_df[col].isnull().sum()}")
                logger.info(f"  Unique values: {new_df[col].nunique()}")
                logger.info(f"  Mean: {new_df[col].mean():.2f}")
                logger.info(f"  Std: {new_df[col].std():.2f}")
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = "data/processed/phishing_data.csv"
    output_path = "data/processed/phishing_data_new.csv"
    preprocess_data(input_path, output_path) 