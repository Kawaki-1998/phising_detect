import pytest
from src.feature_extraction.feature_extractor import FeatureExtractor

def test_feature_extraction():
    extractor = FeatureExtractor()
    test_domain = "test-domain.com"
    features = extractor.extract_features(test_domain)
    
    assert isinstance(features, dict)
    assert len(features) > 0
    assert features['num_hyphens'] == 1
    assert features['num_dots'] == 1
    assert features['length'] == len(test_domain)

def test_suspicious_domain_features():
    extractor = FeatureExtractor()
    suspicious_domain = "paypal-secure-login.com"
    features = extractor.extract_features(suspicious_domain)
    suspicious_features = extractor.get_suspicious_features(suspicious_domain)
    
    assert features['has_suspicious_keywords'] == 1
    assert features['has_brand_name'] == 1
    assert 'login' in suspicious_features.get('keywords', [])
    assert 'secure' in suspicious_features.get('keywords', [])
    assert features['num_hyphens'] == 2

def test_empty_domain():
    extractor = FeatureExtractor()
    features = extractor.extract_features("")
    suspicious_features = extractor.get_suspicious_features("")
    brand_detection = extractor.detect_brand("")
    
    assert isinstance(features, dict)
    assert features['length'] == 0
    assert features['num_digits'] == 0
    assert features['num_hyphens'] == 0
    assert features['num_dots'] == 0
    assert features['has_suspicious_keywords'] == 0
    assert features['has_brand_name'] == 0
    assert features['has_suspicious_tld'] == 0
    assert suspicious_features == {}
    assert brand_detection == {} 