from typing import Dict, Set
import re
from urllib.parse import urlparse

class BrandDetector:
    def __init__(self):
        self.common_brands = {
            'google', 'facebook', 'amazon', 'microsoft', 'apple', 'paypal', 'netflix', 
            'linkedin', 'twitter', 'instagram', 'yahoo', 'walmart', 'ebay', 'spotify',
            'github', 'dropbox', 'adobe', 'zoom', 'chase', 'wellsfargo', 'bankofamerica',
            'citibank', 'hsbc', 'barclays', 'americanexpress', 'mastercard', 'visa'
        }
        
        self.suspicious_keywords = {
            'secure', 'security', 'login', 'signin', 'verify', 'verification', 'account',
            'update', 'confirm', 'authenticate', 'wallet', 'password', 'credential',
            'billing', 'payment', 'manage', 'access', 'reset', 'recover', 'unlock'
        }
        
        self.number_letter_map = {
            '0': 'o',
            '1': 'i',
            '3': 'e',
            '4': 'a',
            '5': 's',
            '7': 't',
            '@': 'a'
        }

    def normalize_domain(self, domain: str) -> str:
        """Normalize domain by replacing common number-letter substitutions."""
        domain_lower = domain.lower()
        for num, letter in self.number_letter_map.items():
            domain_lower = domain_lower.replace(num, letter)
        return domain_lower

    def extract_brand_features(self, domain: str) -> Dict[str, float]:
        """Extract brand-related features from a domain."""
        domain_lower = domain.lower()
        main_domain = domain.split('/')[0]
        normalized_domain = self.normalize_domain(main_domain)
        
        features = {
            'contains_brand': 0,
            'brand_with_addons': 0,
            'suspicious_keywords_count': 0,
            'number_substitutions': 0,
            'brand_distance': 1.0,  # 1.0 means no brand found
            'domain_length_ratio': 0.0,  # Length ratio between domain and closest brand
            'suspicious_tld': 0,
            'has_suspicious_prefix': 0,
            'has_suspicious_suffix': 0
        }
        
        # Check for number-letter substitutions
        for num in self.number_letter_map:
            if num in main_domain:
                features['number_substitutions'] += 1
        
        # Check for suspicious TLDs
        suspicious_tlds = {'xyz', 'top', 'work', 'live', 'click', 'loan', 'online'}
        domain_tld = main_domain.split('.')[-1].lower()
        features['suspicious_tld'] = int(domain_tld in suspicious_tlds)
        
        # Count suspicious keywords
        features['suspicious_keywords_count'] = sum(
            1 for keyword in self.suspicious_keywords 
            if keyword in domain_lower
        )
        
        # Check for brand presence
        for brand in self.common_brands:
            if brand in normalized_domain:
                features['contains_brand'] = 1
                
                # Check domain length ratio
                features['domain_length_ratio'] = len(main_domain) / len(brand)
                
                # Check for suspicious additions
                if features['suspicious_keywords_count'] > 0:
                    features['brand_with_addons'] = 1
                
                # Check for suspicious prefixes/suffixes
                brand_idx = normalized_domain.find(brand)
                if brand_idx > 0:  # Has prefix
                    features['has_suspicious_prefix'] = 1
                if brand_idx + len(brand) < len(normalized_domain):  # Has suffix
                    features['has_suspicious_suffix'] = 1
                
                break
        
        return features

    def get_feature_names(self) -> list:
        """Return list of feature names in consistent order."""
        return [
            'contains_brand',
            'brand_with_addons',
            'suspicious_keywords_count',
            'number_substitutions',
            'brand_distance',
            'domain_length_ratio',
            'suspicious_tld',
            'has_suspicious_prefix',
            'has_suspicious_suffix'
        ] 