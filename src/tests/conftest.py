import sys
from unittest.mock import MagicMock
import pytest
import numpy as np
import joblib

# Create a mock Cassandra package
class MockCassandra:
    def __init__(self):
        # Set up cluster mock
        self.cluster = MagicMock()
        mock_session = MagicMock()
        mock_session.execute = MagicMock()
        
        # Configure cluster instance
        mock_cluster_instance = MagicMock()
        mock_cluster_instance.connect = MagicMock(return_value=mock_session)
        self.cluster.Cluster = MagicMock(return_value=mock_cluster_instance)
        
        # Set up auth mock
        self.auth = MagicMock()
        self.auth.PlainTextAuthProvider = MagicMock

# Create and install the mock
mock_cassandra = MockCassandra()
sys.modules['cassandra'] = mock_cassandra
sys.modules['cassandra.cluster'] = mock_cassandra.cluster
sys.modules['cassandra.auth'] = mock_cassandra.auth

# Mock MLflow
mock_mlflow = MagicMock()
sys.modules['mlflow'] = mock_mlflow 

@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr('mlflow.start_run', mock)
    monkeypatch.setattr('mlflow.log_params', mock)
    monkeypatch.setattr('mlflow.log_metrics', mock)
    monkeypatch.setattr('mlflow.log_artifact', mock)
    return mock

@pytest.fixture(autouse=True)
def mock_model_components(monkeypatch):
    # Mock model prediction
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    
    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])
    
    # Mock feature names
    mock_feature_names = ['feature1', 'feature2', 'feature3']
    
    monkeypatch.setattr('joblib.load', lambda x: {
        'models/best_phishing_model.pkl': mock_model,
        'models/feature_scaler.pkl': mock_scaler,
        'models/feature_names.pkl': mock_feature_names
    }[x])
    
    return mock_model, mock_scaler, mock_feature_names 