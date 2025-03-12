import sys
from unittest.mock import MagicMock

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