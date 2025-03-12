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
            'qty_dot_domain': 0, 'qty_hyphen_domain': 0, 'qty_underline_domain': 0,
            'qty_slash_domain': 0, 'qty_questionmark_domain': 0, 'qty_equal_domain': 0,
            'qty_at_domain': 0, 'qty_and_domain': 0, 'qty_exclamation_domain': 0,
            'qty_space_domain': 0, 'qty_tilde_domain': 0, 'qty_comma_domain': 0,
            'qty_plus_domain': 0, 'qty_asterisk_domain': 0, 'qty_hashtag_domain': 0,
            'qty_dollar_domain': 0, 'qty_percent_domain': 0, 'qty_vowels_domain': 0,
            'domain_length': 0, 'domain_in_ip': 0, 'server_client_domain': 0,

            # Directory features
            'qty_dot_directory': 0, 'qty_hyphen_directory': 0, 'qty_underline_directory': 0,
            'qty_slash_directory': 0, 'qty_questionmark_directory': 0, 'qty_equal_directory': 0,
            'qty_at_directory': 0, 'qty_and_directory': 0, 'qty_exclamation_directory': 0,
            'qty_space_directory': 0, 'qty_tilde_directory': 0, 'qty_comma_directory': 0,
            'qty_plus_directory': 0, 'qty_asterisk_directory': 0, 'qty_hashtag_directory': 0,
            'qty_dollar_directory': 0, 'qty_percent_directory': 0, 'directory_length': 0,

            # File features
            'qty_dot_file': 0, 'qty_hyphen_file': 0, 'qty_underline_file': 0,
            'qty_slash_file': 0, 'qty_questionmark_file': 0, 'qty_equal_file': 0,
            'qty_at_file': 0, 'qty_and_file': 0, 'qty_exclamation_file': 0,
            'qty_space_file': 0, 'qty_tilde_file': 0, 'qty_comma_file': 0,
            'qty_plus_file': 0, 'qty_asterisk_file': 0, 'qty_hashtag_file': 0,
            'qty_dollar_file': 0, 'qty_percent_file': 0, 'file_length': 0,

            # Parameters features
            'qty_dot_params': 0, 'qty_hyphen_params': 0, 'qty_underline_params': 0,
            'qty_slash_params': 0, 'qty_questionmark_params': 0, 'qty_equal_params': 0,
            'qty_at_params': 0, 'qty_and_params': 0, 'qty_exclamation_params': 0,
            'qty_space_params': 0, 'qty_tilde_params': 0, 'qty_comma_params': 0,
            'qty_plus_params': 0, 'qty_asterisk_params': 0, 'qty_hashtag_params': 0,
            'qty_dollar_params': 0, 'qty_percent_params': 0, 'params_length': 0,
            'tld_present_params': 0, 'qty_params': 0,

            # Additional features
            'email_in_url': 0, 'time_response': 0, 'domain_spf': 0, 'asn_ip': 0,
            'time_domain_activation': 0, 'time_domain_expiration': 0,
            'qty_ip_resolved': 0, 'qty_nameservers': 0, 'qty_mx_servers': 0,
            'ttl_hostname': 0, 'tls_ssl_certificate': 0, 'qty_redirects': 0,
            'url_google_index': 0, 'domain_google_index': 0, 'url_shortened': 0
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
        """Extract domain-related features."""
        return {
            'qty_dot_domain': domain.count('.'),
            'qty_hyphen_domain': domain.count('-'),
            'qty_underline_domain': domain.count('_'),
            'qty_slash_domain': domain.count('/'),
            'qty_questionmark_domain': domain.count('?'),
            'qty_equal_domain': domain.count('='),
            'qty_at_domain': domain.count('@'),
            'qty_and_domain': domain.count('&'),
            'qty_exclamation_domain': domain.count('!'),
            'qty_space_domain': domain.count(' '),
            'qty_tilde_domain': domain.count('~'),
            'qty_comma_domain': domain.count(','),
            'qty_plus_domain': domain.count('+'),
            'qty_asterisk_domain': domain.count('*'),
            'qty_hashtag_domain': domain.count('#'),
            'qty_dollar_domain': domain.count('$'),
            'qty_percent_domain': domain.count('%'),
            'qty_vowels_domain': sum(c in 'aeiou' for c in domain.lower()),
            'domain_length': len(domain),
            'domain_in_ip': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain))),
            'server_client_domain': 0  # Default value
        }

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
        required_features = [
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
            'domain_in_ip', 'server_client_domain', 'qty_dot_directory',
            'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
            'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory',
            'qty_and_directory', 'qty_exclamation_directory', 'qty_space_directory',
            'qty_tilde_directory', 'qty_comma_directory', 'qty_plus_directory',
            'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
            'qty_percent_directory', 'directory_length', 'qty_dot_file',
            'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
            'qty_questionmark_file', 'qty_equal_file', 'qty_at_file',
            'qty_and_file', 'qty_exclamation_file', 'qty_space_file',
            'qty_tilde_file', 'qty_comma_file', 'qty_plus_file',
            'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
            'qty_percent_file', 'file_length', 'qty_dot_params',
            'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
            'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
            'qty_and_params', 'qty_exclamation_params', 'qty_space_params',
            'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
            'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
            'qty_percent_params', 'params_length', 'tld_present_params',
            'qty_params', 'email_in_url', 'time_response', 'domain_spf',
            'asn_ip', 'time_domain_activation', 'time_domain_expiration',
            'qty_ip_resolved', 'qty_nameservers', 'qty_mx_servers',
            'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
            'url_google_index', 'domain_google_index', 'url_shortened'
        ]
        return [f for f in required_features if f not in features]

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