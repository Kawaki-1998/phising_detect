from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, validator
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from logging.handlers import RotatingFileHandler
import traceback
from src.features.brand_detection import BrandDetector
import uvicorn
from src.feature_extraction.feature_extractor import FeatureExtractor
import mlflow
from src.config.config import MLFLOW_CONFIG, LOG_DIR
import json
from datetime import datetime, timedelta
import pandas as pd
import os

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, 'app.log')

# Remove any existing handlers
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Log file: {log_file}")

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
    model_path = os.path.join('models', 'best_phishing_model.pkl')
    scaler_path = os.path.join('models', 'feature_scaler.pkl')
    feature_names_path = os.path.join('models', 'feature_names.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    brand_detector = BrandDetector()
    logger.info("Model components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model components: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Initialize feature extractor
feature_extractor = FeatureExtractor()

class DomainRequest(BaseModel):
    domain: str
    threshold: float = 0.5

    @validator('domain')
    def validate_domain(cls, v):
        if not v:
            raise HTTPException(status_code=400, detail='Domain cannot be empty')
        if len(v) > 253:  # Maximum length of a domain name
            raise HTTPException(status_code=400, detail='Domain name too long')
        return v.lower()

    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v

class DomainResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    domain: str
    is_phishing: bool
    risk_score: float
    confidence: float
    brand_detection: Dict[str, Any]
    suspicious_features: List[str]

class TimelineEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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
    confidence_distribution: Dict[str, float]

def log_prediction_to_mlflow(domain: str, features: dict, prediction_result: dict):
    """Log prediction details to MLflow."""
    try:
        with mlflow.start_run():
            # Convert all values to JSON serializable format
            features_json = {}
            for k, v in features.items():
                if isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                    features_json[k] = float(v)
                elif isinstance(v, bool):
                    features_json[k] = bool(v)
                else:
                    features_json[k] = str(v)
            
            prediction_json = {}
            for k, v in prediction_result.items():
                if isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                    prediction_json[k] = float(v)
                elif isinstance(v, bool):
                    prediction_json[k] = bool(v)
                else:
                    prediction_json[k] = str(v)
            
            prediction_details = {
                "domain": domain,
                "features": features_json,
                "prediction": prediction_json
            }
            
            # Log parameters
            mlflow.log_params({"domain": domain})
            
            # Log metrics
            mlflow.log_metrics({
                "risk_score": float(prediction_result["risk_score"]),
                "confidence": float(prediction_result["confidence"])
            })
            
            # Save prediction details as JSON artifact
            with open("prediction_details.json", "w") as f:
                json.dump(prediction_details, f, indent=2)
            mlflow.log_artifact("prediction_details.json")
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Don't raise the exception since MLflow logging is not critical

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/check_domain", response_model=DomainResponse)
async def check_domain(request: DomainRequest):
    """Check if a domain is potentially phishing."""
    try:
        logger.info(f"Processing domain: {request.domain}")

        # Extract features
        try:
            features = feature_extractor.extract_features(request.domain)
        except ValueError as e:
            logger.error(f"Error extracting features for domain {request.domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Error extracting features: {str(e)}")
        
        # Validate features
        required_features = {
            'domain_length', 'num_dots', 'num_hyphens', 'num_digits',
            'has_suspicious_keywords', 'has_brand_name', 'domain_in_ip',
            'server_client_domain', 'time_response', 'domain_spf', 'asn_ip',
            'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
            'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
            'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened'
        }
        missing_features = required_features - set(features.keys())
        if missing_features:
            logger.error(f"Missing features for domain {request.domain}: {missing_features}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {', '.join(missing_features)}"
            )
        
        # Scale features
        try:
            feature_values = []
            feature_names = ['domain_google_index', 'domain_in_ip', 'num_hyphens', 'qty_redirects', 'ttl_hostname',
                           'qty_mx_servers', 'server_client_domain', 'has_suspicious_keywords', 'time_domain_activation',
                           'asn_ip', 'domain_length', 'num_dots', 'has_brand_name', 'time_response',
                           'tls_ssl_certificate', 'qty_nameservers', 'num_digits', 'url_shortened',
                           'time_domain_expiration', 'qty_ip_resolved', 'url_google_index', 'domain_spf']
            
            for feature_name in feature_names:
                feature_values.append(float(features[feature_name]))
            
            X = pd.DataFrame([feature_values], columns=feature_names)
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.debug(f"Scaled features shape: {X_scaled.shape}")
        except Exception as e:
            logger.error(f"Error scaling features for domain {request.domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error scaling features: {str(e)}"
            )
        
        # Get prediction and probability
        try:
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            logger.debug(f"Model prediction: {prediction}, probabilities: {probabilities}")
        except Exception as e:
            logger.error(f"Error making prediction for domain {request.domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error making prediction: {str(e)}"
            )
        
        # Get brand detection results
        try:
            brand_results = brand_detector.check_brand_impersonation(request.domain)
            logger.debug(f"Brand detection results: {brand_results}")
        except Exception as e:
            logger.error(f"Error checking brand impersonation for domain {request.domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            brand_results = {}  # Continue with empty results
        
        # Calculate risk score and confidence
        risk_score = float(probabilities[1])
        confidence = float(max(probabilities))
        is_phishing = bool(risk_score >= request.threshold)
        
        # Get suspicious features
        try:
            suspicious_features = feature_extractor.get_suspicious_features(request.domain)
            logger.debug(f"Suspicious features: {suspicious_features}")
        except Exception as e:
            logger.error(f"Error getting suspicious features for domain {request.domain}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            suspicious_features = []  # Continue with empty list
        
        response = {
            "domain": request.domain,
            "is_phishing": is_phishing,
            "risk_score": risk_score,
            "confidence": confidence,
            "brand_detection": brand_results,
            "suspicious_features": suspicious_features[:5]  # Top 5 suspicious features
        }
        
        # Log prediction
        try:
            log_prediction_to_mlflow(request.domain, features, response)
        except Exception as e:
            logger.error(f"Error logging prediction to MLflow: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without MLflow logging
        
        logger.info(f"Successfully processed domain {request.domain}: phishing={is_phishing}, risk_score={risk_score:.2f}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error for domain {request.domain}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing domain {request.domain}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(days: int = 7):
    """Get dashboard statistics for the specified number of days."""
    try:
        logger.info(f"Getting dashboard stats for last {days} days")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Query MLflow for runs in the specified time range
        try:
            runs = mlflow.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(MLFLOW_CONFIG["experiment_name"]).experiment_id],
                filter_string=f"attributes.start_time >= {int(start_time.timestamp() * 1000)} and attributes.start_time <= {int(end_time.timestamp() * 1000)}"
            )
            logger.debug(f"Found {len(runs)} MLflow runs")
        except Exception as e:
            logger.error(f"Error querying MLflow: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            runs = pd.DataFrame()  # Empty DataFrame
        
        if len(runs) == 0:
            logger.info("No MLflow runs found, returning empty stats")
            return {
                "total_predictions": 0,
                "phishing_ratio": 0.0,
                "avg_confidence": 0.0,
                "top_suspicious_features": [],
                "detection_timeline": [],
                "brand_impersonation_stats": {},
                "confidence_distribution": {"low": 0, "medium": 0, "high": 0}
            }
        
        # Calculate statistics
        total_predictions = len(runs)
        phishing_predictions = runs[runs["metrics.risk_score"] >= 0.5]
        phishing_ratio = len(phishing_predictions) / total_predictions if total_predictions > 0 else 0
        avg_confidence = runs["metrics.confidence"].mean()
        
        # Create timeline entries
        timeline = []
        for _, run in runs.iterrows():
            timeline.append({
                "timestamp": datetime.fromtimestamp(run["start_time"] / 1000).isoformat(),
                "is_phishing": run["metrics.risk_score"] >= 0.5,
                "confidence": float(run["metrics.confidence"])
            })
        
        # Calculate confidence distribution
        confidence_dist = {
            "low": len(runs[runs["metrics.confidence"] < 0.6]),
            "medium": len(runs[(runs["metrics.confidence"] >= 0.6) & (runs["metrics.confidence"] < 0.8)]),
            "high": len(runs[runs["metrics.confidence"] >= 0.8])
        }
        
        stats = {
            "total_predictions": total_predictions,
            "phishing_ratio": float(phishing_ratio),
            "avg_confidence": float(avg_confidence),
            "top_suspicious_features": [],  # This would need feature importance analysis
            "detection_timeline": timeline,
            "brand_impersonation_stats": {},  # This would need aggregation of brand detection results
            "confidence_distribution": confidence_dist
        }
        
        logger.info(f"Successfully generated dashboard stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 