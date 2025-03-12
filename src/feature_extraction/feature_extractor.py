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
            raise ValueError("Domain cannot be empty")
            
        if not self._is_valid_domain(domain):
            raise ValueError("Invalid domain format")

        features = {}
        
        # Basic features
        features['domain_length'] = len(domain)  # Changed from 'length' to 'domain_length'
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

    def get_suspicious_features(self, domain: str) -> list:
        """Get suspicious features found in the domain."""
        if not domain:
            return []

        suspicious = []
        domain_lower = domain.lower()

        # Check for suspicious keywords
        found_keywords = [
            keyword for keyword in self.suspicious_keywords
            if keyword in domain_lower
        ]
        if found_keywords:
            suspicious.append(f"Contains suspicious keywords: {', '.join(found_keywords)}")

        # Check for number-letter substitutions
        substitutions = re.findall(r'\d+', domain)
        if substitutions:
            suspicious.append(f"Contains number substitutions: {', '.join(substitutions)}")

        # Check for excessive punctuation
        if domain.count('-') > 2:
            suspicious.append(f"Excessive hyphens: {domain.count('-')}")
        if domain.count('.') > 2:
            suspicious.append(f"Excessive dots: {domain.count('.')}")

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

    def _is_valid_domain(self, domain: str) -> bool:
        """Check if the domain format is valid."""
        if not domain or len(domain) < 4:  # Minimum valid domain length
            return False
        
        # Basic domain format check
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$'
        return bool(re.match(domain_pattern, domain)) 