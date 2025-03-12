# Phishing Domain Detection

[![CI/CD Pipeline](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Kawaki-1998/phising_detect/branch/main/graph/badge.svg)](https://codecov.io/gh/Kawaki-1998/phising_detect)

A machine learning-based system for detecting phishing domains using FastAPI and MLflow.

## Features

- Real-time phishing domain detection
- Brand impersonation detection
- Suspicious feature analysis
- MLflow integration for experiment tracking
- Comprehensive monitoring dashboard
- REST API with FastAPI
- Automated testing and CI/CD pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kawaki-1998/phising_detect.git
cd phising_detect
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. Start the FastAPI application:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

3. Access the API documentation:
- OpenAPI documentation: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc

4. Access MLflow UI:
- MLflow dashboard: http://localhost:5000

## API Endpoints

### Check Domain
```http
POST /check_domain
```
Check if a domain is potentially phishing.

Request body:
```json
{
    "domain": "example.com",
    "threshold": 0.5
}
```

### Dashboard Statistics
```http
GET /dashboard/stats?days=7
```
Get aggregated statistics for the monitoring dashboard.

### Domain Predictions History
```http
GET /dashboard/predictions/{domain}
```
Get prediction history for a specific domain.

## Development

1. Run tests:
```bash
pytest src/tests/ -v --cov=src
```

2. Format code:
```bash
black src/
```

3. Check code style:
```bash
flake8 src/
```

## Project Structure

```
Phishing_Domain_Detection/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── src/
│   ├── api/
│   │   └── app.py
│   ├── features/
│   │   ├── brand_detection.py
│   │   └── feature_extractor.py
│   ├── models/
│   │   └── best_phishing_model.pkl
│   └── tests/
│       ├── test_api.py
│       └── test_features.py
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FastAPI for the web framework
- MLflow for experiment tracking
- scikit-learn and LightGBM for machine learning
- GitHub Actions for CI/CD 