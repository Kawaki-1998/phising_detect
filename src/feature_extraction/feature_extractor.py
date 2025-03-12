import re
from urllib.parse import urlparse
import tld

class FeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'login', 'secure', 'verify', 'account', 'update', 'confirm',
            'banking', 'payment', 'password', 'credential'
        ]
        self.common_brands = [
            'google', 'facebook', 'apple', 'microsoft', 'amazon',
            'paypal', 'netflix', 'linkedin', 'twitter', 'instagram'
        ]

    def extract_features(self, domain: str) -> dict:
        """Extract features from a domain name."""
        if not domain:
            return {
                'length': 0,
                'num_digits': 0,
                'num_hyphens': 0,
                'num_dots': 0,
                'has_suspicious_keywords': 0,
                'has_brand_name': 0,
                'has_suspicious_tld': 0
            }

        features = {}
        
        # Basic features
        features['length'] = len(domain)
        features['num_digits'] = sum(c.isdigit() for c in domain)
        features['num_hyphens'] = domain.count('-')
        features['num_dots'] = domain.count('.')
        
        # Suspicious keywords
        features['has_suspicious_keywords'] = int(
            any(keyword in domain.lower() for keyword in self.suspicious_keywords)
        )
        
        # Brand impersonation
        features['has_brand_name'] = int(
            any(brand in domain.lower() for brand in self.common_brands)
        )
        
        # TLD analysis
        try:
            tld_obj = tld.get_tld(f"http://{domain}", as_object=True)
            features['has_suspicious_tld'] = int(
                tld_obj.tld in ['.xyz', '.top', '.work', '.date', '.loan']
            )
        except:
            features['has_suspicious_tld'] = 0
        
        return features

    def get_suspicious_features(self, domain: str) -> dict:
        """Get suspicious features found in the domain."""
        if not domain:
            return {}

        suspicious = {}
        domain_lower = domain.lower()

        # Check for suspicious keywords
        found_keywords = [
            keyword for keyword in self.suspicious_keywords
            if keyword in domain_lower
        ]
        if found_keywords:
            suspicious['keywords'] = found_keywords

        # Check for number-letter substitutions
        substitutions = re.findall(r'\d+', domain)
        if substitutions:
            suspicious['number_substitutions'] = substitutions

        # Check for excessive punctuation
        if domain.count('-') > 2:
            suspicious['excessive_hyphens'] = domain.count('-')
        if domain.count('.') > 2:
            suspicious['excessive_dots'] = domain.count('.')

        return suspicious

    def detect_brand(self, domain: str) -> dict:
        """Detect potential brand impersonation."""
        if not domain:
            return {}

        domain_lower = domain.lower()
        detected_brands = {}

        for brand in self.common_brands:
            if brand in domain_lower:
                # Check for exact match vs potential impersonation
                if brand == domain_lower.replace('.com', ''):
                    detected_brands[brand] = 'exact_match'
                else:
                    detected_brands[brand] = 'potential_impersonation'

        return detected_brands 