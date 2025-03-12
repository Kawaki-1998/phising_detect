# Low-Level Design Document: Phishing Domain Detection System

## 1. Component Details

### 1.1 API Layer (src/api/app.py)
```python
class DomainRequest(BaseModel):
    domain: str
    threshold: float = 0.5

@app.post("/check_domain")
async def check_domain(request: DomainRequest):
    features = extract_features(request.domain)
    prediction = model.predict_proba(features)
    return {
        "domain": request.domain,
        "is_phishing": prediction > request.threshold,
        "probability": prediction,
        "features": features
    }
```

### 1.2 Feature Extraction (src/feature_extraction/feature_extractor.py)
```python
class FeatureExtractor:
    def extract_features(self, domain: str) -> Dict[str, Any]:
        return {
            "domain_length": self._get_domain_length(domain),
            "num_dots": self._count_dots(domain),
            # ... other features
        }
```

## 2. Database Schema

### 2.1 MLflow Metadata (SQLite)
```sql
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    artifact_location VARCHAR(255),
    lifecycle_stage VARCHAR(20)
);

CREATE TABLE runs (
    run_id VARCHAR(32) PRIMARY KEY,
    experiment_id INTEGER,
    status VARCHAR(20),
    start_time BIGINT,
    end_time BIGINT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
```

## 3. Class Diagrams

### 3.1 Feature Engineering Classes
```
FeatureExtractor
├── DomainAnalyzer
├── DNSAnalyzer
├── WHOISAnalyzer
├── SSLAnalyzer
└── SearchEngineAnalyzer
```

### 3.2 Model Training Classes
```
ModelTrainer
├── DataPreprocessor
├── FeatureScaler
├── ModelValidator
└── ModelSaver
```

## 4. Sequence Diagrams

### 4.1 Domain Check Flow
```
Client -> API: POST /check_domain
API -> FeatureExtractor: extract_features()
FeatureExtractor -> DNSAnalyzer: get_dns_info()
FeatureExtractor -> WHOISAnalyzer: get_whois_info()
FeatureExtractor -> SSLAnalyzer: verify_ssl()
API -> Model: predict()
API -> Client: Response
```

## 5. Data Structures

### 5.1 Feature Vector
```python
features = {
    "domain_length": int,
    "num_dots": int,
    "num_hyphens": int,
    "num_digits": int,
    "domain_in_ip": bool,
    "tls_ssl_certificate": bool,
    "domain_spf": int,
    "qty_nameservers": int,
    "qty_mx_servers": int,
    "qty_ip_resolved": int,
    "ttl_hostname": int,
    "time_domain_activation": int,
    "time_domain_expiration": int,
    "time_response": float,
    "asn_ip": int,
    "server_client_domain": bool,
    "domain_google_index": bool,
    "url_google_index": bool,
    "url_shortened": bool,
    "has_suspicious_keywords": bool,
    "has_brand_name": bool
}
```

## 6. API Endpoints

### 6.1 Check Domain
- **Endpoint**: `/check_domain`
- **Method**: POST
- **Request Body**:
```json
{
    "domain": "example.com",
    "threshold": 0.5
}
```
- **Response**:
```json
{
    "domain": "example.com",
    "is_phishing": false,
    "probability": 0.12,
    "features": {...}
}
```

### 6.2 Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Response**:
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

## 7. Error Handling

### 7.1 Exception Classes
```python
class DomainValidationError(Exception):
    pass

class FeatureExtractionError(Exception):
    pass

class ModelPredictionError(Exception):
    pass
```

### 7.2 Error Responses
```python
@app.exception_handler(DomainValidationError)
async def domain_validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )
```

## 8. Configuration

### 8.1 Environment Variables
```env
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MODEL_PATH=models/best_phishing_model.pkl
FEATURE_SCALER_PATH=models/feature_scaler.pkl
LOG_LEVEL=INFO
```

## 9. Logging

### 9.1 Log Format
```python
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

### 9.2 Log Categories
- API requests/responses
- Feature extraction steps
- Model predictions
- Error events
- Performance metrics

## 10. Testing Strategy

### 10.1 Unit Tests
```python
def test_feature_extraction():
    extractor = FeatureExtractor()
    features = extractor.extract_features("example.com")
    assert "domain_length" in features
    assert features["num_dots"] == 1
```

### 10.2 Integration Tests
```python
def test_domain_check_endpoint():
    response = client.post(
        "/check_domain",
        json={"domain": "example.com"}
    )
    assert response.status_code == 200
    assert "is_phishing" in response.json()
```

## 11. Performance Optimization

### 11.1 Caching Strategy
```python
@cache(ttl=3600)
async def get_domain_features(domain: str):
    return await feature_extractor.extract_features(domain)
```

### 11.2 Batch Processing
```python
@app.post("/check_domains_batch")
async def check_domains_batch(domains: List[str]):
    return await process_domains_in_parallel(domains)
``` 