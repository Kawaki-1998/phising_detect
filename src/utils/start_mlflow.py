import os
import subprocess
import logging
from src.config.config import MLFLOW_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_mlflow_server():
    """Start MLflow tracking server."""
    try:
        # Create mlruns directory if it doesn't exist
        os.makedirs("mlruns", exist_ok=True)
        
        # Start MLflow server
        cmd = [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "mlruns",
            "--default-artifact-root", "mlruns"
        ]
        
        logger.info("Starting MLflow server...")
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Error starting MLflow server: {str(e)}")
        raise

if __name__ == "__main__":
    start_mlflow_server() 