from flask import Flask, request, jsonify
import joblib
import os
from datetime import datetime
from src.database.cassandra_client import CassandraClient
from src.features.feature_engineering import FeatureExtractor
from src.config.config import MODEL_DIR
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
feature_extractor = FeatureExtractor()
cassandra_client = CassandraClient()

# Load the model
model_path = os.path.join(MODEL_DIR, "best_phishing_model.pkl")
try:
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.before_first_request
def initialize():
    """Initialize database connection before first request."""
    try:
        cassandra_client.connect()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a domain is phishing or not."""
    try:
        data = request.get_json()
        if not data or 'domain' not in data:
            return jsonify({"error": "No domain provided"}), 400

        domain = data['domain']
        
        # Check if prediction exists in database
        existing_prediction = cassandra_client.get_prediction(domain)
        if existing_prediction:
            return jsonify({
                "domain": domain,
                "prediction": existing_prediction.prediction,
                "confidence": existing_prediction.confidence,
                "features": existing_prediction.features,
                "timestamp": existing_prediction.timestamp.isoformat(),
                "model_version": existing_prediction.model_version,
                "source": "cache"
            })

        # Extract features
        features = feature_extractor.extract_features(domain)
        
        # Make prediction
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        prediction_proba = model.predict_proba([list(features.values())])[0]
        prediction = bool(prediction_proba[1] > 0.5)
        confidence = float(max(prediction_proba))

        # Store prediction in database
        cassandra_client.insert_prediction(
            domain=domain,
            prediction=prediction,
            confidence=confidence,
            features=features,
            model_version="1.0.0"
        )

        return jsonify({
            "domain": domain,
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "source": "model"
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up database connection."""
    cassandra_client.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 