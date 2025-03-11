@echo off
echo Starting Phishing Domain Detection Project...

REM Create Python virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Start MLflow server in background
start "MLflow Server" python src/utils/start_mlflow.server.py

REM Wait for MLflow server to start
timeout /t 5

REM Start Cassandra (assuming it's installed and in PATH)
echo Starting Cassandra...
net start "Apache Cassandra"

REM Wait for Cassandra to start
timeout /t 10

REM Run training pipeline
echo Running training pipeline...
python src/models/run_training.py

REM Start API server
echo Starting API server...
python src/api/app.py 