import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import mlflow
from src.api.app import app

client = TestClient(app)

def test_check_domain_legitimate():
    with patch('mlflow.start_run') as mock_mlflow:
        response = client.post(
            "/check_domain",
            json={"domain": "google.com", "threshold": 0.5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "google.com"
        assert isinstance(data["is_phishing"], bool)
        assert isinstance(data["risk_score"], float)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["brand_detection"], dict)
        assert isinstance(data["suspicious_features"], list)

def test_check_domain_phishing():
    with patch('mlflow.start_run') as mock_mlflow:
        response = client.post(
            "/check_domain",
            json={"domain": "g00gle-secure.com", "threshold": 0.5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "g00gle-secure.com"
        assert isinstance(data["is_phishing"], bool)
        assert isinstance(data["risk_score"], float)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["brand_detection"], dict)
        assert isinstance(data["suspicious_features"], list)

def test_invalid_domain():
    with patch('mlflow.start_run') as mock_mlflow:
        response = client.post(
            "/check_domain",
            json={"domain": "", "threshold": 0.5}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Domain cannot be empty"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"} 