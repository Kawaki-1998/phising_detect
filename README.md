# Phishing Domain Detection

A machine learning-based system for detecting phishing domains using advanced feature engineering and ensemble models.

## Features

- Real-time phishing domain detection via REST API
- Advanced URL feature extraction
- Multiple ML models (RandomForest, XGBoost, LightGBM, CatBoost)
- Automated hyperparameter optimization using Optuna
- MLflow experiment tracking
- Cassandra database for caching predictions
- Deployment configuration for Render

## Project Structure

```
phishing_detection/
├── src/
│   ├── api/              # Flask API implementation
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── utils/            # Utility functions
│   ├── visualization/    # Visualization scripts
│   ├── config/          # Configuration files
│   ├── database/        # Database operations
│   ├── logs/            # Application logs
│   ├── notebooks/       # Jupyter notebooks
│   └── tests/           # Unit tests
├── deployment/          # Deployment configurations
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Cassandra:
```bash
# For Windows:
# Download and install Apache Cassandra
# Start the service
```

3. Set up environment variables:
```bash
CASSANDRA_HOST=localhost
CASSANDRA_PORT=9042
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Usage

1. Train the model:
```bash
python src/models/train.py
```

2. Start the API:
```bash
python src/api/app.py
```

3. Make predictions:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"domain": "example.com"}'
```

## API Endpoints

- `POST /predict`: Predict if a domain is phishing
  - Request body: `{"domain": "example.com"}`
  - Response: Prediction results with confidence score

- `GET /health`: Health check endpoint

## Model Training

The system uses multiple models and selects the best performing one:
- RandomForest
- XGBoost
- LightGBM
- CatBoost

Hyperparameter optimization is performed using Optuna, and all experiments are tracked using MLflow.

## Feature Engineering

URL features extracted include:
- URL length
- Number of special characters
- Number of dots
- Number of subdomains
- HTTPS usage
- Domain length
- Path length
- Query parameters

## Deployment

The project includes configuration for deployment on Render:

1. Connect your repository to Render
2. Configure environment variables
3. Deploy using the provided `render.yaml`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 