import re
from urllib.parse import urlparse
import tld
import dns.resolver
import requests
import socket
from datetime import datetime
import whois
import ssl
import OpenSSL
from typing import Dict, Any, List
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.url_shorteners = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
            'is.gd', 'cli.gs', 'pic.gd', 'DwarfURL.com', 'yfrog.com'
        ]

    def extract_features(self, domain: str) -> Dict[str, Any]:
        """Extract all required features from a domain name."""
        try:
            if not domain:
                raise ValueError("Domain cannot be empty")
                
            if not self._is_valid_domain(domain):
                raise ValueError("Invalid domain format")

            # Parse domain
            parsed_domain = urlparse(f"http://{domain}")
            domain_parts = parsed_domain.netloc.split('.')

            # Initialize features with default values
            features = self._initialize_features()

            # Update URL and domain features
            features.update(self._extract_url_features(domain, domain_parts))
            features.update(self._extract_domain_features(domain))

            # Try to get additional information
            try:
                features.update(self._get_dns_info(domain))
            except Exception as e:
                logger.warning(f"Error getting DNS info for {domain}: {str(e)}")

            try:
                features.update(self._get_whois_info(domain))
            except Exception as e:
                logger.warning(f"Error getting WHOIS info for {domain}: {str(e)}")

            try:
                features.update(self._get_ssl_info(domain))
            except Exception as e:
                logger.warning(f"Error getting SSL info for {domain}: {str(e)}")

            try:
                features.update(self._get_spf_info(domain))
            except Exception as e:
                logger.warning(f"Error getting SPF info for {domain}: {str(e)}")

            # Check if domain is shortened
            features['url_shortened'] = int(any(shortener in domain.lower() for shortener in self.url_shorteners))

            # Validate all features are present
            missing_features = self._validate_features(features)
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                raise ValueError(f"Missing required features: {', '.join(missing_features)}")

            return features

        except Exception as e:
            logger.error(f"Error extracting features for {domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_features(self) -> Dict[str, Any]:
        """Initialize all features with default values."""
        return {
            # URL features
            'qty_dot_url': 0, 'qty_hyphen_url': 0, 'qty_underline_url': 0,
            'qty_slash_url': 0, 'qty_questionmark_url': 0, 'qty_equal_url': 0,
            'qty_at_url': 0, 'qty_and_url': 0, 'qty_exclamation_url': 0,
            'qty_space_url': 0, 'qty_tilde_url': 0, 'qty_comma_url': 0,
            'qty_plus_url': 0, 'qty_asterisk_url': 0, 'qty_hashtag_url': 0,
            'qty_dollar_url': 0, 'qty_percent_url': 0, 'qty_tld_url': 0,
            'length_url': 0,
            
            # Domain features
            'domain_length': 0,
            'num_dots': 0,
            'num_hyphens': 0,
            'num_digits': 0,
            'has_suspicious_keywords': 0,
            'has_brand_name': 0,
            'domain_in_ip': 0,
            'server_client_domain': 0,
            
            # Additional features
            'time_response': 0,
            'domain_spf': 0,
            'asn_ip': 0,
            'time_domain_activation': 0,
            'time_domain_expiration': 0,
            'qty_ip_resolved': 0,
            'qty_nameservers': 0,
            'qty_mx_servers': 0,
            'ttl_hostname': 0,
            'tls_ssl_certificate': 0,
            'qty_redirects': 0,
            'url_google_index': 0,
            'domain_google_index': 0,
            'url_shortened': 0,
        }

    def _extract_url_features(self, domain: str, domain_parts: list) -> Dict[str, Any]:
        """Extract URL-related features."""
        return {
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
            'qty_tld_url': len(domain_parts[-1]) if len(domain_parts) > 1 else 0,
            'length_url': len(domain)
        }

    def _extract_domain_features(self, domain: str) -> Dict[str, Any]:
        """Extract features from the domain name."""
        features = {}
        
        # Basic domain features
        features['domain_length'] = len(domain)
        features['num_dots'] = domain.count('.')
        features['num_hyphens'] = domain.count('-')
        features['num_digits'] = sum(c.isdigit() for c in domain)
        
        # Check for suspicious keywords and brand names
        features['has_suspicious_keywords'] = int(any(keyword in domain.lower() for keyword in self.suspicious_keywords))
        features['has_brand_name'] = int(any(brand in domain.lower() for brand in self.common_brands))
        
        # Check if domain is an IP address
        features['domain_in_ip'] = int(bool(re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', domain)))
        
        # Check for server/client keywords
        server_client_keywords = ['server', 'client', 'host', 'proxy']
        features['server_client_domain'] = int(any(keyword in domain.lower() for keyword in server_client_keywords))
        
        return features

    def _get_dns_info(self, domain: str) -> Dict[str, Any]:
        """Get DNS-related information."""
        info = {
            'qty_ip_resolved': 0,
            'ttl_hostname': 0,
            'qty_nameservers': 0,
            'qty_mx_servers': 0
        }

        # Get A records
        try:
            dns_info = dns.resolver.resolve(domain, 'A')
            info['qty_ip_resolved'] = len(dns_info)
            info['ttl_hostname'] = dns_info.rrset.ttl
        except Exception as e:
            logger.warning(f"Error getting A records for {domain}: {str(e)}")

        # Get NS records
        try:
            ns_info = dns.resolver.resolve(domain, 'NS')
            info['qty_nameservers'] = len(ns_info)
        except Exception as e:
            logger.warning(f"Error getting NS records for {domain}: {str(e)}")

        # Get MX records
        try:
            mx_info = dns.resolver.resolve(domain, 'MX')
            info['qty_mx_servers'] = len(mx_info)
        except Exception as e:
            logger.warning(f"Error getting MX records for {domain}: {str(e)}")

        return info

    def _get_whois_info(self, domain: str) -> Dict[str, Any]:
        """Get WHOIS information."""
        info = {
            'time_domain_activation': 0,
            'time_domain_expiration': 0
        }

        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                if isinstance(whois_info.creation_date, list):
                    creation_date = whois_info.creation_date[0]
                else:
                    creation_date = whois_info.creation_date
                info['time_domain_activation'] = int((datetime.now() - creation_date).days)
            if whois_info.expiration_date:
                if isinstance(whois_info.expiration_date, list):
                    expiration_date = whois_info.expiration_date[0]
                else:
                    expiration_date = whois_info.expiration_date
                info['time_domain_expiration'] = int((expiration_date - datetime.now()).days)
        except Exception as e:
            logger.warning(f"Error getting WHOIS info for {domain}: {str(e)}")

        return info

    def _get_ssl_info(self, domain: str) -> Dict[str, Any]:
        """Get SSL certificate information."""
        info = {'tls_ssl_certificate': 0}

        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    info['tls_ssl_certificate'] = 1
        except Exception as e:
            logger.warning(f"Error getting SSL info for {domain}: {str(e)}")

        return info

    def _get_spf_info(self, domain: str) -> Dict[str, Any]:
        """Get SPF record information."""
        info = {'domain_spf': 0}

        try:
            spf_info = dns.resolver.resolve(domain, 'TXT')
            for record in spf_info:
                if 'spf' in str(record).lower():
                    info['domain_spf'] = 1
                    break
        except Exception as e:
            logger.warning(f"Error getting SPF info for {domain}: {str(e)}")

        return info

    def _validate_features(self, features: Dict[str, Any]) -> List[str]:
        """Validate that all required features are present."""
        required_features = {
            'domain_length', 'num_dots', 'num_hyphens', 'num_digits',
            'has_suspicious_keywords', 'has_brand_name', 'domain_in_ip',
            'server_client_domain', 'time_response', 'domain_spf', 'asn_ip',
            'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
            'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
            'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened'
        }
        
        missing_features = [feature for feature in required_features if feature not in features]
        return missing_features

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

        # Check for brand impersonation
        brand_results = self.detect_brand(domain)
        if brand_results:
            for brand, status in brand_results.items():
                if status == 'potential_impersonation':
                    suspicious.append(f"Potential {brand} impersonation")

        # Check for URL shorteners
        if any(shortener in domain_lower for shortener in self.url_shorteners):
            suspicious.append("Uses URL shortener service")

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
        """Check if the domain has a valid format."""
        if not domain:
            return False
            
        # Basic domain format validation
        domain_pattern = r'^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z]{2,})+$'
        return bool(re.match(domain_pattern, domain)) 