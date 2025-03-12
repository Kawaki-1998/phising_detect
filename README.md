# 🛡️ Phishing Domain Detection

[![CI/CD Pipeline](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Kawaki-1998/phising_detect/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Kawaki-1998/phising_detect/branch/main/graph/badge.svg)](https://codecov.io/gh/Kawaki-1998/phising_detect)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

## 📋 Overview

A robust machine learning system that detects phishing domains in real-time. Built with FastAPI and MLflow, this solution provides enterprise-grade security through advanced feature analysis and brand impersonation detection.

### 🌟 Key Features

- **Real-time Detection**: Instant analysis of domain legitimacy
- **Brand Protection**: Advanced impersonation detection
- **ML-Powered**: Utilizing LightGBM for accurate predictions
- **Comprehensive API**: RESTful endpoints with FastAPI
- **MLflow Integration**: Experiment tracking and model versioning
- **Automated Pipeline**: CI/CD with GitHub Actions

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
mlflow server --host 127.0.0.1 --port 5000
```

2. Launch the API:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

3. Access documentation:
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

## 🔍 API Reference

### Check Domain
```http
POST /check_domain
Content-Type: application/json

{
    "domain": "example.com",
    "threshold": 0.5
}
```

### Dashboard Statistics
```http
GET /dashboard/stats?days=7
```

For complete API documentation, visit the [/docs](http://localhost:8000/docs) endpoint.

## 🧪 Development

Run tests and quality checks:
```bash
# Run tests with coverage
pytest src/tests/ -v --cov=src

# Code formatting
black src/

# Linting
flake8 src/
```

## 📁 Project Structure

```
.
├── .github/workflows/    # CI/CD configuration
├── src/
│   ├── api/             # FastAPI application
│   ├── features/        # Feature extraction
│   ├── models/          # Trained models
│   └── tests/           # Test suite
├── README.md
└── requirements.txt
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Phishing Domain Detection

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools 