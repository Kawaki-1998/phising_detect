# Wireframe Documentation: Phishing Domain Detection System

## 1. API Documentation Interface

### 1.1 Swagger UI Homepage
```
┌────────────────────────────────────────────┐
│ Phishing Domain Detection API              │
│                                            │
│ ▼ POST /check_domain                       │
│   Check if a domain is potentially phishing│
│                                            │
│ ▼ GET /health                             │
│   Check API health status                  │
│                                            │
│ ▼ GET /metrics                            │
│   Get model performance metrics            │
└────────────────────────────────────────────┘
```

### 1.2 Domain Check Request
```
┌────────────────────────────────────────────┐
│ POST /check_domain                         │
│                                            │
│ Request Body:                              │
│ ┌────────────────────────────────────┐     │
│ │ {                                  │     │
│ │   "domain": "example.com",         │     │
│ │   "threshold": 0.5                 │     │
│ │ }                                  │     │
│ └────────────────────────────────────┘     │
│                                            │
│ Execute                                    │
└────────────────────────────────────────────┘
```

### 1.3 Domain Check Response
```
┌────────────────────────────────────────────┐
│ Response (200 OK)                          │
│                                            │
│ ┌────────────────────────────────────┐     │
│ │ {                                  │     │
│ │   "domain": "example.com",         │     │
│ │   "is_phishing": false,            │     │
│ │   "probability": 0.12,             │     │
│ │   "features": {                    │     │
│ │     "domain_length": 11,           │     │
│ │     "num_dots": 1,                │     │
│ │     ...                           │     │
│ │   }                               │     │
│ │ }                                  │     │
│ └────────────────────────────────────┘     │
└────────────────────────────────────────────┘
```

## 2. MLflow Interface

### 2.1 Experiment Tracking
```
┌────────────────────────────────────────────┐
│ MLflow Experiments                         │
│                                            │
│ ┌────────────────┐ ┌────────────────┐     │
│ │ Experiment 1   │ │ Experiment 2   │     │
│ │ Runs: 5        │ │ Runs: 3        │     │
│ └────────────────┘ └────────────────┘     │
│                                            │
│ Metrics | Parameters | Artifacts           │
└────────────────────────────────────────────┘
```

### 2.2 Model Performance
```
┌────────────────────────────────────────────┐
│ Model Metrics                              │
│                                            │
│ Accuracy     [==========] 86.45%           │
│ AUC-ROC      [==========] 0.942            │
│ Log Loss     [========--] 0.312            │
│                                            │
│ Feature Importance                         │
│ domain_length [========--] 0.82            │
│ num_dots     [======----] 0.64            │
└────────────────────────────────────────────┘
```

## 3. Error Pages

### 3.1 Validation Error
```
┌────────────────────────────────────────────┐
│ 400 Bad Request                            │
│                                            │
│ Error: Invalid domain format               │
│                                            │
│ Details:                                   │
│ Domain must be a valid hostname            │
│                                            │
│ [Try Again]                                │
└────────────────────────────────────────────┘
```

### 3.2 Server Error
```
┌────────────────────────────────────────────┐
│ 500 Internal Server Error                  │
│                                            │
│ Error: Feature extraction failed           │
│                                            │
│ Request ID: abc-123                        │
│                                            │
│ [Contact Support]                          │
└────────────────────────────────────────────┘
```

## 4. Monitoring Dashboard

### 4.1 System Health
```
┌────────────────────────────────────────────┐
│ System Status                              │
│                                            │
│ API Service    [✓] Healthy                 │
│ MLflow Server  [✓] Healthy                 │
│ Database      [✓] Connected                │
│                                            │
│ Last Updated: 2024-03-12 02:05:52         │
└────────────────────────────────────────────┘
```

### 4.2 Request Metrics
```
┌────────────────────────────────────────────┐
│ Request Statistics                         │
│                                            │
│ Requests/min  [/////|     ] 50/100        │
│ Avg Response  [///|       ] 150ms         │
│ Error Rate    [|          ] 0.1%          │
│                                            │
│ Last 24 Hours ▁▂▅▂▁▂▃▂▁▁▂▃▄▅▂▁▂▃▄▅▆▇█▆   │
└────────────────────────────────────────────┘
```

## 5. Batch Processing Interface

### 5.1 Batch Upload
```
┌────────────────────────────────────────────┐
│ Batch Domain Check                         │
│                                            │
│ [Select File] domains.txt                  │
│                                            │
│ Threshold: [0.5]                           │
│                                            │
│ [Start Processing]                         │
└────────────────────────────────────────────┘
```

### 5.2 Batch Results
```
┌────────────────────────────────────────────┐
│ Processing Results                         │
│                                            │
│ Total Domains: 100                         │
│ Processed: 95                              │
│ Failed: 5                                  │
│                                            │
│ [Download Report]                          │
└────────────────────────────────────────────┘
```

## 6. Mobile Interface

### 6.1 Mobile Check Domain
```
┌────────────────────┐
│ Check Domain       │
│                    │
│ [example.com    ]  │
│                    │
│ Threshold: [0.5]   │
│                    │
│ [Check]            │
└────────────────────┘
```

### 6.2 Mobile Results
```
┌────────────────────┐
│ Results            │
│                    │
│ example.com        │
│                    │
│ Status: Safe       │
│ Risk: 12%         │
│                    │
│ [View Details]     │
└────────────────────┘
``` 