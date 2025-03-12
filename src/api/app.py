from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from src.features.brand_detection import BrandDetector
import uvicorn
from src.feature_extraction.feature_extractor import FeatureExtractor
import mlflow
from src.config.config import MLFLOW_CONFIG
import json
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

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

# Initialize feature extractor
feature_extractor = FeatureExtractor()

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

class TimelineEntry(BaseModel):
    timestamp: str
    is_phishing: bool
    confidence: float

class DashboardStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_predictions: int
    phishing_ratio: float
    avg_confidence: float
    top_suspicious_features: List[Dict[str, int]]
    detection_timeline: List[TimelineEntry]
    brand_impersonation_stats: Dict[str, int]
    confidence_distribution: Dict[str, int]

def log_prediction_to_mlflow(domain: str, features: Dict, prediction_result: Dict, model_version: str = "1.0.0"):
    """Enhanced MLflow logging with comprehensive metrics and artifacts."""
    timestamp = datetime.now().isoformat()
    
    # Create a run name with timestamp for easier identification
    run_name = f"prediction_{domain}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Log basic parameters
        mlflow.log_params({
            "domain": domain,
            "threshold": prediction_result["threshold"],
            "model_version": model_version,
            "timestamp": timestamp
        })
        
        # Log detailed metrics
        mlflow.log_metrics({
            "prediction_confidence": prediction_result["confidence"],
            "phishing_probability": prediction_result["risk_score"],
            "domain_length": len(domain),
            "num_suspicious_features": len(prediction_result["suspicious_features"]),
            "num_dots": domain.count('.'),
            "num_hyphens": domain.count('-'),
            "num_digits": sum(c.isdigit() for c in domain)
        })
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')
        
        # Log prediction details as artifacts
        prediction_details = {
            "input_features": features,
            "prediction_result": prediction_result,
            "brand_detection": prediction_result["brand_detection"],
            "suspicious_features": prediction_result["suspicious_features"]
        }
        with open('prediction_details.json', 'w') as f:
            json.dump(prediction_details, f, indent=2)
        mlflow.log_artifact('prediction_details.json')
        
        # Add tags for easier filtering
        mlflow.set_tags({
            "prediction_type": "phishing" if prediction_result["is_phishing"] else "legitimate",
            "confidence_level": "high" if prediction_result["confidence"] > 0.8 else "medium" if prediction_result["confidence"] > 0.5 else "low",
            "has_brand_match": "true" if prediction_result["brand_detection"] else "false",
            "has_suspicious_features": "true" if prediction_result["suspicious_features"] else "false"
        })

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
        features = feature_extractor.extract_features(request.domain)
        
        # Map feature names to match model expectations
        feature_dict = {
            'qty_dot_url': features['num_dots'],
            'qty_hyphen_url': features['num_hyphens'],
            'qty_underline_url': 0,
            'qty_slash_url': 0,
            'qty_questionmark_url': 0,
            'qty_equal_url': 0,
            'qty_at_url': 0,
            'qty_and_url': 0,
            'qty_exclamation_url': 0,
            'qty_space_url': 0,
            'qty_tilde_url': 0,
            'qty_comma_url': 0,
            'qty_plus_url': 0,
            'qty_asterisk_url': 0,
            'qty_hashtag_url': 0,
            'qty_dollar_url': 0,
            'qty_percent_url': 0,
            'qty_tld_url': 1,
            'length_url': features['length'],
            'qty_dot_domain': features['num_dots'],
            'qty_hyphen_domain': features['num_hyphens'],
            'qty_underline_domain': 0,
            'qty_slash_domain': 0,
            'qty_questionmark_domain': 0,
            'qty_equal_domain': 0,
            'qty_at_domain': 0,
            'qty_and_domain': 0,
            'qty_exclamation_domain': 0,
            'qty_space_domain': 0,
            'qty_tilde_domain': 0,
            'qty_comma_domain': 0,
            'qty_plus_domain': 0,
            'qty_asterisk_domain': 0,
            'qty_hashtag_domain': 0,
            'qty_dollar_domain': 0,
            'qty_percent_domain': 0,
            'qty_vowels_domain': sum(c in 'aeiou' for c in request.domain.lower()),
            'domain_length': features['length'],
            'domain_in_ip': 0,
            'server_client_domain': 0,
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
            'email_in_url': 0,
            'time_response': 0,
            'domain_spf': 0,
            'asn_ip': 0,
            'time_domain_activation': 0,
            'time_domain_expiration': 0,
            'qty_ip_resolved': 1,
            'qty_nameservers': 1,
            'qty_mx_servers': 1,
            'ttl_hostname': 0,
            'tls_ssl_certificate': 0,
            'qty_redirects': 0,
            'url_google_index': 0,
            'domain_google_index': 0,
            'url_shortened': 0
        }
        
        # Scale features
        feature_vector = np.array([feature_dict[feature] for feature in feature_names]).reshape(1, -1)
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction_proba = model.predict_proba(scaled_features)[0]
        is_phishing = prediction_proba[1] >= request.threshold
        confidence = prediction_proba[1] if is_phishing else prediction_proba[0]
        
        # Get suspicious features and brand detection
        suspicious_features_dict = feature_extractor.get_suspicious_features(request.domain)
        suspicious_features = []
        
        # Convert dictionary to list of strings
        if 'keywords' in suspicious_features_dict:
            suspicious_features.extend([f"Contains suspicious keyword: {kw}" for kw in suspicious_features_dict['keywords']])
        if 'number_substitutions' in suspicious_features_dict:
            suspicious_features.extend([f"Contains number substitution: {sub}" for sub in suspicious_features_dict['number_substitutions']])
        if 'excessive_hyphens' in suspicious_features_dict:
            suspicious_features.append(f"Excessive hyphens: {suspicious_features_dict['excessive_hyphens']}")
        if 'excessive_dots' in suspicious_features_dict:
            suspicious_features.append(f"Excessive dots: {suspicious_features_dict['excessive_dots']}")
        
        # Get brand detection results
        brand_detection = feature_extractor.detect_brand(request.domain)
        
        # Prepare prediction result
        prediction_result = {
            "domain": request.domain,
            "is_phishing": is_phishing,
            "confidence": confidence,
            "risk_score": prediction_proba[1],
            "suspicious_features": suspicious_features,
            "brand_detection": brand_detection,
            "threshold": request.threshold
        }
        
        # Log prediction with enhanced tracking
        log_prediction_to_mlflow(request.domain, feature_dict, prediction_result)
        
        return DomainResponse(**prediction_result)
    except Exception as e:
        logger.error(f"Error processing domain {request.domain}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/dashboard/stats")
async def get_dashboard_stats(days: Optional[int] = 7):
    """Get aggregated statistics for the monitoring dashboard."""
    try:
        # Calculate the start date for filtering
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get all runs from the experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_CONFIG["experiment_name"])
        
        if not experiment:
            raise HTTPException(status_code=404, detail="No experiment found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.start_time > '{start_date}'"
        )
        
        # Initialize counters and collectors
        total_predictions = len(runs)
        phishing_count = 0
        confidence_sum = 0
        suspicious_features_count = {}
        timeline_data = []
        brand_impersonation_count = {}
        confidence_levels = {"high": 0, "medium": 0, "low": 0}
        
        # Process each run
        for run in runs:
            # Get metrics and tags
            metrics = run.data.metrics
            tags = run.data.tags
            
            # Update counters
            if tags.get("prediction_type") == "phishing":
                phishing_count += 1
            
            confidence = metrics.get("prediction_confidence", 0)
            confidence_sum += confidence
            
            # Update confidence distribution
            confidence_level = tags.get("confidence_level", "low")
            confidence_levels[confidence_level] += 1
            
            # Get prediction details
            try:
                prediction_path = client.download_artifacts(run.info.run_id, "prediction_details.json")
                with open(prediction_path) as f:
                    prediction_details = json.load(f)
                
                # Update suspicious features count
                for feature in prediction_details["suspicious_features"]:
                    suspicious_features_count[feature] = suspicious_features_count.get(feature, 0) + 1
                
                # Update brand impersonation stats
                if prediction_details["brand_detection"]:
                    for brand in prediction_details["brand_detection"]:
                        brand_impersonation_count[brand] = brand_impersonation_count.get(brand, 0) + 1
                
                # Add to timeline
                timeline_data.append({
                    "timestamp": run.info.start_time,
                    "is_phishing": prediction_details["prediction_result"]["is_phishing"],
                    "confidence": confidence
                })
            except:
                continue
        
        # Prepare the response
        stats = DashboardStats(
            total_predictions=total_predictions,
            phishing_ratio=phishing_count / total_predictions if total_predictions > 0 else 0,
            avg_confidence=confidence_sum / total_predictions if total_predictions > 0 else 0,
            top_suspicious_features=sorted(
                [{"feature": k, "count": v} for k, v in suspicious_features_count.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:10],
            detection_timeline=sorted(timeline_data, key=lambda x: x["timestamp"]),
            brand_impersonation_stats=brand_impersonation_count,
            confidence_distribution=confidence_levels
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/predictions/{domain}")
async def get_domain_predictions(domain: str):
    """Get prediction history for a specific domain."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_CONFIG["experiment_name"])
        
        if not experiment:
            raise HTTPException(status_code=404, detail="No experiment found")
        
        # Search for runs with the specific domain
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.domain = '{domain}'"
        )
        
        predictions = []
        for run in runs:
            predictions.append({
                "timestamp": run.info.start_time,
                "is_phishing": run.data.tags.get("prediction_type") == "phishing",
                "confidence": run.data.metrics.get("prediction_confidence"),
                "risk_score": run.data.metrics.get("phishing_probability"),
                "run_id": run.info.run_id
            })
        
        return sorted(predictions, key=lambda x: x["timestamp"], reverse=True)
        
    except Exception as e:
        logger.error(f"Error retrieving domain predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 