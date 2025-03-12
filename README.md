# 🛡️ Phishing Domain Detection

[![CI/CD Pipeline](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Kawaki-1998/phising_detect/branch/main/graph/badge.svg)](https://codecov.io/gh/Kawaki-1998/phising_detect)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.0-0194E2.svg?logo=mlflow)](https://mlflow.org)

## 📋 Overview

A production-ready machine learning system for real-time phishing domain detection. Built with FastAPI and MLflow, this solution leverages advanced feature engineering and the LightGBM algorithm to identify potentially malicious domains with high accuracy.

### 🎯 Model Performance

- **Accuracy**: 86.45% on test set
- **AUC-ROC**: 0.942
- **Binary Log Loss**: 0.312

### 🌟 Key Features

- **Real-time Detection**: Instant analysis of domain legitimacy through RESTful API endpoints
- **Comprehensive Feature Analysis**: 22 carefully engineered features covering:
  - Domain characteristics (length, special characters)
  - DNS information (nameservers, MX records, TTL)
  - SSL/TLS certificate validation
  - WHOIS data (domain age, expiration)
  - Search engine presence
  - Network characteristics (ASN, IP resolution)
- **MLflow Integration**: 
  - Experiment tracking
  - Model versioning
  - Performance metrics logging
  - Artifact management
- **Automated CI/CD**: 
  - Continuous testing across Python 3.9, 3.10, and 3.11
  - Automated deployments to Render
  - Code quality checks

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kawaki-1998/phising_detect.git
cd phising_detect
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

2. Launch the API:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

3. Access documentation:
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

## 🔍 API Endpoints

### Check Domain
```http
POST /check_domain
Content-Type: application/json

{
    "domain": "example.com",
    "threshold": 0.5
}
```

Response:
```json
{
    "domain": "example.com",
    "is_phishing": false,
    "probability": 0.12,
    "features": {
        "domain_length": 11,
        "num_dots": 1,
        "num_hyphens": 0,
        // ... other features
    },
    "prediction_time": "2024-03-12T02:05:52"
}
```

### Health Check
```http
GET /health
```

## 📊 Features Extracted

Our model analyzes the following features for each domain:

1. **Domain Structure**
   - `domain_length`: Length of the domain name
   - `num_dots`: Number of dots in the domain
   - `num_hyphens`: Number of hyphens in the domain
   - `num_digits`: Number of numerical digits

2. **Security Indicators**
   - `domain_in_ip`: Whether the domain contains an IP address
   - `tls_ssl_certificate`: SSL/TLS certificate validation
   - `domain_spf`: SPF record presence and validity

3. **DNS Information**
   - `qty_nameservers`: Number of nameservers
   - `qty_mx_servers`: Number of MX records
   - `qty_ip_resolved`: Number of resolved IP addresses
   - `ttl_hostname`: TTL value of the hostname

4. **Temporal Features**
   - `time_domain_activation`: Domain age
   - `time_domain_expiration`: Time until domain expiration
   - `time_response`: Response time for DNS queries

5. **Network Features**
   - `asn_ip`: Autonomous System Number
   - `server_client_domain`: Server-client domain relationship

6. **Search Engine Presence**
   - `domain_google_index`: Domain indexed by Google
   - `url_google_index`: URL indexed by Google
   - `url_shortened`: Whether URL is shortened

7. **Brand Protection**
   - `has_suspicious_keywords`: Presence of suspicious terms
   - `has_brand_name`: Detection of brand impersonation

## 🧪 Development

### Running Tests

```bash
# Run tests with coverage
pytest src/tests/ -v --cov=src/ --cov-report=term-missing

# Run specific test file
pytest src/tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black src/

# Run linter
flake8 src/
```

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/        # CI/CD configuration
├── data/
│   └── processed/       # Processed datasets
├── models/              # Trained models and artifacts
├── src/
│   ├── api/            # FastAPI application
│   │   └── app.py      # Main API endpoints
│   ├── config/         # Configuration files
│   ├── feature_extraction/
│   │   └── feature_extractor.py
│   ├── features/       # Feature engineering
│   │   └── brand_detection.py
│   ├── tests/         # Test suite
│   ├── model_training.py
│   └── preprocess_data.py
├── mlartifacts/        # MLflow artifacts
├── README.md
└── requirements.txt
```

## 🔄 Recent Updates

- Added feature preservation during model scaling
- Fixed feature order consistency in API predictions
- Improved error handling and validation
- Updated CI/CD workflow with proper test models
- Enhanced logging and monitoring

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools 