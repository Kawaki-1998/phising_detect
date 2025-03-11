from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
from typing import Dict, List
import logging
from src.features.brand_detection import BrandDetector
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Domain Detection API",
    description="API for detecting phishing domains using machine learning",
    version="1.0.0"
)

# Load model components
try:
    model = joblib.load('models/best_phishing_model.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    brand_detector = BrandDetector()
    logger.info("Model components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model components: {str(e)}")
    raise

class DomainRequest(BaseModel):
    domain: str
    threshold: float = 0.5

class DomainResponse(BaseModel):
    domain: str
    is_phishing: bool
    risk_score: float
    confidence: float
    brand_detection: Dict
    suspicious_features: List[str]

def extract_features_for_domain(domain: str) -> Dict:
    """Extract all features for a domain."""
    # Basic features
    basic_features = {
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
        # Add all other features as in the original code
    }
    
    # Brand detection features
    brand_features = brand_detector.extract_brand_features(domain)
    
    # Combine all features
    features = {**basic_features, **brand_features}
    return features

def get_suspicious_features(features: Dict, domain: str) -> List[str]:
    """Identify suspicious features in the domain."""
    suspicious = []
    
    # Check for number-letter substitutions
    if features.get('number_substitutions', 0) > 0:
        suspicious.append("Contains number-letter substitutions")
    
    # Check for brand impersonation
    if features.get('contains_brand', 0) == 1:
        if features.get('brand_with_addons', 0) == 1:
            suspicious.append("Brand name with suspicious additions")
        if features.get('has_suspicious_prefix', 0) == 1:
            suspicious.append("Suspicious prefix before brand name")
        if features.get('has_suspicious_suffix', 0) == 1:
            suspicious.append("Suspicious suffix after brand name")
    
    # Check for suspicious TLD
    if features.get('suspicious_tld', 0) == 1:
        suspicious.append("Suspicious top-level domain")
    
    # Check for suspicious keywords
    if features.get('suspicious_keywords_count', 0) > 0:
        suspicious.append("Contains suspicious keywords")
    
    return suspicious

@app.post("/check_domain", response_model=DomainResponse)
async def check_domain(request: DomainRequest):
    try:
        # Extract features
        features = extract_features_for_domain(request.domain)
        feature_vector = [features[feature] for feature in feature_names]
        
        # Scale features
        X_scaled = scaler.transform([feature_vector])
        
        # Make prediction
        prediction_proba = model.predict_proba(X_scaled)[0]
        risk_score = float(prediction_proba[1])
        is_phishing = risk_score > request.threshold
        confidence = float(max(prediction_proba))
        
        # Get brand detection results and suspicious features
        brand_features = brand_detector.extract_brand_features(request.domain)
        suspicious_features = get_suspicious_features(features, request.domain)
        
        return DomainResponse(
            domain=request.domain,
            is_phishing=is_phishing,
            risk_score=risk_score,
            confidence=confidence,
            brand_detection=brand_features,
            suspicious_features=suspicious_features
        )
        
    except Exception as e:
        logger.error(f"Error processing domain {request.domain}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 