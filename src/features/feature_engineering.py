import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
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

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract all features from a URL."""
        try:
            features = {
                "url_length": self._get_url_length(url),
                "num_special_chars": self._count_special_chars(url),
                "num_dots": url.count('.'),
                "num_digits": sum(c.isdigit() for c in url),
                "num_subdomains": self._count_subdomains(url),
                "has_https": int(url.startswith("https")),
                "domain_length": self._get_domain_length(url),
                "path_length": self._get_path_length(url),
                "query_length": self._get_query_length(url)
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting features from URL {url}: {str(e)}")
            return {feature: 0.0 for feature in self.feature_names}

    def _get_url_length(self, url: str) -> int:
        """Get the length of the URL."""
        return len(url)

    def _count_special_chars(self, url: str) -> int:
        """Count special characters in URL."""
        special_chars = re.findall(r'[^a-zA-Z0-9.]', url)
        return len(special_chars)

    def _count_subdomains(self, url: str) -> int:
        """Count number of subdomains in URL."""
        try:
            domain = urlparse(url).netloc
            return len(domain.split('.')) - 1
        except:
            return 0

    def _get_domain_length(self, url: str) -> int:
        """Get length of domain name."""
        try:
            return len(urlparse(url).netloc)
        except:
            return 0

    def _get_path_length(self, url: str) -> int:
        """Get length of URL path."""
        try:
            return len(urlparse(url).path)
        except:
            return 0

    def _get_query_length(self, url: str) -> int:
        """Get length of URL query parameters."""
        try:
            return len(urlparse(url).query)
        except:
            return 0

    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from the dataset."""
        try:
            # Return only the features we want to use
            return df[self.feature_names]
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return pd.DataFrame(columns=self.feature_names)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names 