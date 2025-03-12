import joblib
import logging
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureExtractor
from src.features.brand_detection import BrandDetector
from sklearn.preprocessing import MinMaxScaler
import re
from urllib.parse import urlparse
from typing import List, Dict
import requests
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Original feature list
ORIGINAL_FEATURES = [
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
    'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
    'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
    'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
    'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
    'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
    'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
    'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
    'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
    'qty_percent_domain', 'qty_vowels_domain', 'domain_length',
    'domain_in_ip', 'server_client_domain'
]

def extract_features_for_domain(domain: str) -> Dict:
    """Extract features for a single domain."""
    domain_lower = domain.lower()
    main_domain = domain.split('/')[0]
    
    # Extract basic features
    features = {
        'qty_dot_url': domain.count('.'),
        'qty_hyphen_url': domain.count('-'),
        'qty_underline_url': domain.count('_'),
        'qty_slash_url': domain.count('/'),
        'qty_questionmark_url': domain.count('?'),
        'qty_equal_url': domain.count('='),
        'qty_at_url': domain.count('@'),
        'qty_and_url': domain.count('&'),
        'qty_exclamation_url': domain.count('!'),
        'qty_space_url': domain.count(' '),
        'qty_tilde_url': domain.count('~'),
        'qty_comma_url': domain.count(','),
        'qty_plus_url': domain.count('+'),
        'qty_asterisk_url': domain.count('*'),
        'qty_hashtag_url': domain.count('#'),
        'qty_dollar_url': domain.count('$'),
        'qty_percent_url': domain.count('%'),
        'qty_tld_url': len(domain.split('.')[-1]),
        'length_url': len(domain),
        'qty_dot_domain': main_domain.count('.'),
        'qty_hyphen_domain': main_domain.count('-'),
        'qty_underline_domain': main_domain.count('_'),
        'qty_slash_domain': main_domain.count('/'),
        'qty_questionmark_domain': main_domain.count('?'),
        'qty_equal_domain': main_domain.count('='),
        'qty_at_domain': main_domain.count('@'),
        'qty_and_domain': main_domain.count('&'),
        'qty_exclamation_domain': main_domain.count('!'),
        'qty_space_domain': main_domain.count(' '),
        'qty_tilde_domain': main_domain.count('~'),
        'qty_comma_domain': main_domain.count(','),
        'qty_plus_domain': main_domain.count('+'),
        'qty_asterisk_domain': main_domain.count('*'),
        'qty_hashtag_domain': main_domain.count('#'),
        'qty_dollar_domain': main_domain.count('$'),
        'qty_percent_domain': main_domain.count('%'),
        'qty_vowels_domain': sum(1 for c in main_domain.lower() if c in 'aeiou'),
        'domain_length': len(main_domain),
        'domain_in_ip': int(bool(sum(c.isdigit() for c in main_domain))),
        'server_client_domain': int(bool(any(x in domain_lower for x in ['server', 'client']))),
        
        # Directory features (empty for simple domains)
        'qty_dot_directory': 0,
        'qty_hyphen_directory': 0,
        'qty_underline_directory': 0,
        'qty_slash_directory': 0,
        'qty_questionmark_directory': 0,
        'qty_equal_directory': 0,
        'qty_at_directory': 0,
        'qty_and_directory': 0,
        'qty_exclamation_directory': 0,
        'qty_space_directory': 0,
        'qty_tilde_directory': 0,
        'qty_comma_directory': 0,
        'qty_plus_directory': 0,
        'qty_asterisk_directory': 0,
        'qty_hashtag_directory': 0,
        'qty_dollar_directory': 0,
        'qty_percent_directory': 0,
        'directory_length': 0,
        
        # File features (empty for simple domains)
        'qty_dot_file': 0,
        'qty_hyphen_file': 0,
        'qty_underline_file': 0,
        'qty_slash_file': 0,
        'qty_questionmark_file': 0,
        'qty_equal_file': 0,
        'qty_at_file': 0,
        'qty_and_file': 0,
        'qty_exclamation_file': 0,
        'qty_space_file': 0,
        'qty_tilde_file': 0,
        'qty_comma_file': 0,
        'qty_plus_file': 0,
        'qty_asterisk_file': 0,
        'qty_hashtag_file': 0,
        'qty_dollar_file': 0,
        'qty_percent_file': 0,
        'file_length': 0,
        
        # Parameter features (empty for simple domains)
        'qty_dot_params': 0,
        'qty_hyphen_params': 0,
        'qty_underline_params': 0,
        'qty_slash_params': 0,
        'qty_questionmark_params': 0,
        'qty_equal_params': 0,
        'qty_at_params': 0,
        'qty_and_params': 0,
        'qty_exclamation_params': 0,
        'qty_space_params': 0,
        'qty_tilde_params': 0,
        'qty_comma_params': 0,
        'qty_plus_params': 0,
        'qty_asterisk_params': 0,
        'qty_hashtag_params': 0,
        'qty_dollar_params': 0,
        'qty_percent_params': 0,
        'params_length': 0,
        'tld_present_params': 0,
        'qty_params': 0,
        
        # Additional features (default values)
        'email_in_url': 0,
        'time_response': -1,
        'domain_spf': -1,
        'asn_ip': -1,
        'time_domain_activation': -1,
        'time_domain_expiration': -1,
        'qty_ip_resolved': -1,
        'qty_nameservers': -1,
        'qty_mx_servers': -1,
        'ttl_hostname': -1,
        'tls_ssl_certificate': -1,
        'qty_redirects': -1,
        'url_google_index': -1,
        'domain_google_index': -1,
        'url_shortened': -1
    }
    
    return features

def test_model_directly(threshold: float = 0.5):
    """Test the model directly with predefined domains."""
    # Load the model components
    try:
        model = joblib.load('models/best_phishing_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        feature_extractor = FeatureExtractor()
        logger.info("Model components loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return

    # Test domains (mix of legitimate and potentially phishing)
    test_domains = [
        "google.com",
        "facebook.com",
        "g00gle.com",
        "paypal-secure.com",
        "microsoft.com",
        "micros0ft-security.com",
        "amazon.com",
        "amaz0n-account-verify.com",
        "netflix.com",
        "netflix-account-verify.com",
        "apple.com",
        "apple-id-verify.com"
    ]

    logger.info("\nRunning direct model predictions on test domains:")
    print("\nDomain Prediction Results:")
    print("-" * 75)
    print(f"{'Domain':<35} {'Prediction':<12} {'Confidence':<10} {'Risk Score':<10}")
    print("-" * 75)

    for domain in test_domains:
        try:
            # Extract features
            features = feature_extractor.extract_features(domain)
            feature_values = [features[feature] for feature in feature_names]
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction_proba = model.predict_proba(X_scaled)[0]
            risk_score = prediction_proba[1]
            prediction = "Phishing" if risk_score > threshold else "Legitimate"
            confidence = float(max(prediction_proba))

            print(f"{domain:<35} {prediction:<12} {confidence:.4f}    {risk_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {str(e)}")

def test_api_with_dataset(sample_size: int = 100, threshold: float = 0.5):
    """Test the API using domains from the dataset."""
    # Load the test data
    logger.info("Loading test data...")
    try:
        df = pd.read_csv('data/processed/phishing_data.csv')
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return

    # Take a sample for testing
    test_data = df.sample(n=sample_size, random_state=42)
    
    # API endpoint
    url = 'http://localhost:8000/check_domain'
    
    # Test each domain
    results = []
    logger.info(f"\nTesting {sample_size} domains through API...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            # Create a test domain based on features
            domain_length = max(5, int(row['domain_length']))  # Ensure minimum length of 5
            domain = f"{''.join(['a' for _ in range(domain_length)])}.com"
            
            # Make prediction request
            response = requests.post(
                url,
                json={"domain": domain, "threshold": threshold}
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'domain': domain,
                    'actual_label': row['phishing'],
                    'predicted_phishing': result['is_phishing'],
                    'risk_score': result['risk_score'],
                    'confidence': result['confidence'],
                    'suspicious_features': result['suspicious_features']
                })
            else:
                logger.error(f"Error for domain {domain}: {response.text}")
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {str(e)}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    logger.info("Results saved to test_results.csv")
    
    # Print summary
    print("\nTest Summary:")
    print("-" * 50)
    print(f"Total domains tested: {len(results)}")
    print(f"Phishing domains detected: {results_df['predicted_phishing'].sum()}")
    print(f"Average confidence: {results_df['confidence'].mean():.2f}")
    print(f"Average risk score: {results_df['risk_score'].mean():.2f}")
    
    # Calculate metrics
    accuracy = (results_df['actual_label'] == results_df['predicted_phishing']).mean()
    true_positives = ((results_df['actual_label'] == 1) & (results_df['predicted_phishing'] == True)).sum()
    false_positives = ((results_df['actual_label'] == 0) & (results_df['predicted_phishing'] == True)).sum()
    true_negatives = ((results_df['actual_label'] == 0) & (results_df['predicted_phishing'] == False)).sum()
    false_negatives = ((results_df['actual_label'] == 1) & (results_df['predicted_phishing'] == False)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nModel Metrics:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    return results_df

if __name__ == "__main__":
    # Test the model directly
    logger.info("Starting direct model testing...")
    test_model_directly(threshold=0.5)
    
    # Test the API with dataset
    logger.info("\nStarting API testing with dataset...")
    test_api_with_dataset(sample_size=100, threshold=0.5) 