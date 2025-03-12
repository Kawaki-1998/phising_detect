import pytest
from src.feature_extraction.feature_extractor import FeatureExtractor

@pytest.fixture
def feature_extractor():
    return FeatureExtractor()

def test_extract_features_legitimate_domain(feature_extractor):
    domain = "google.com"
    features = feature_extractor.extract_features(domain)
    
    assert isinstance(features, dict)
    assert len(features) > 0
    assert features["domain_length"] == len(domain)
    assert features["num_dots"] == domain.count(".")
    assert features["num_hyphens"] == domain.count("-")
    assert features["num_digits"] == sum(c.isdigit() for c in domain)

def test_extract_features_suspicious_domain(feature_extractor):
    domain = "g00gle-secure.com"
    features = feature_extractor.extract_features(domain)
    
    assert isinstance(features, dict)
    assert len(features) > 0
    assert features["domain_length"] == len(domain)
    assert features["num_dots"] == domain.count(".")
    assert features["num_hyphens"] == domain.count("-")
    assert features["num_digits"] == sum(c.isdigit() for c in domain)
    assert features["has_suspicious_keywords"] == True

def test_extract_features_empty_domain(feature_extractor):
    with pytest.raises(ValueError):
        feature_extractor.extract_features("")

def test_extract_features_invalid_domain(feature_extractor):
    with pytest.raises(ValueError):
        feature_extractor.extract_features("not_a_valid_domain")

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