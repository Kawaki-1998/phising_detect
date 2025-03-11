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

# Configure logging
logging.basicConfig(level=logging.INFO)
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

def test_predictions(threshold: float = 0.5):
    # Load the model, scaler, and feature names
    try:
        model = joblib.load('models/best_phishing_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        logger.info("Model, scaler, and feature names loaded successfully")
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

    # Extract features for each domain
    features_list = []
    for domain in test_domains:
        features = extract_features_for_domain(domain)
        # Ensure features are in the same order as during training
        features_list.append([features[feature] for feature in feature_names])

    # Convert to DataFrame with feature names
    X = pd.DataFrame(features_list, columns=feature_names)
    
    # Scale features using the saved scaler
    X_scaled = scaler.transform(X)
    
    logger.info("\nRunning predictions on test domains:")
    print("\nDomain Prediction Results:")
    print("-" * 75)
    print(f"{'Domain':<35} {'Prediction':<12} {'Confidence':<10} {'Risk Score':<10}")
    print("-" * 75)

    for domain, features in zip(test_domains, X_scaled):
        # Make prediction
        prediction_proba = model.predict_proba([features])[0]
        risk_score = prediction_proba[1]  # Probability of being phishing
        prediction = "Phishing" if risk_score > threshold else "Legitimate"
        confidence = float(max(prediction_proba))

        print(f"{domain:<35} {prediction:<12} {confidence:.4f}    {risk_score:.4f}")

if __name__ == "__main__":
    # Test with different thresholds
    print("\nTesting with default threshold (0.5):")
    test_predictions(threshold=0.5)
    
    print("\nTesting with stricter threshold (0.7):")
    test_predictions(threshold=0.7) 