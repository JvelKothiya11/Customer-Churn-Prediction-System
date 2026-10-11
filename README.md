# Customer Churn Prediction System

A production-ready machine learning system that predicts telecommunications customer churn. The project features an end-to-end predictive pipeline using **Random Forest**, decision threshold optimization (**0.35 threshold raising recall to 71%**), feature importance extraction (**tenure, service type, contract risk factors**), model serialization via **Joblib**, real-time REST API integration with **FastAPI**, and full containerization via **Docker**.

---

## Resume Highlights & Project Alignment

| Feature / Objective | Implementation | Metric / Outcome |
| :--- | :--- | :--- |
| **Pipeline & Algorithm** | Random Forest pipeline with preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) | **ROC-AUC: 0.844** |
| **Threshold Optimization** | Shifted decision threshold from default 0.50 to **0.35** | **Churner Recall raised to 71%** (catches ~71% of churners early) |
| **Feature Importance Analysis** | Extracted and ranked feature importances using Pandas and NumPy | Identified **tenure (0.209)**, **TotalCharges (0.131)**, **Fiber optic (0.102)** as primary risk factors |
| **Serialization & Containerization** | Serialized complete pipeline with **Joblib**; packaged in lightweight **Docker** container | Real-time REST API powered by **FastAPI** for downstream system integration |

---

## Dataset

- **Dataset:** Telco Customer Churn
- **Records:** 7,043 customers (21 features)
- **Target:** `Churn` (Yes / No)
- **Class Distribution:** Non-churn: 5,174 (73.5%), Churn: 1,869 (26.5%)

### Key Features
- `tenure` — Number of months the customer has stayed with the company
- `MonthlyCharges` — Monthly amount billed
- `TotalCharges` — Total amount billed to date
- `Contract` — Contract duration (Month-to-month, One year, Two year)
- `InternetService` — DSL, Fiber optic, No
- `PaymentMethod` — Electronic check, Mailed check, Bank transfer, Credit card

---

## Model Evaluation & Threshold Tuning

| Metric | Baseline (Logistic Regression) | Random Forest (Tuned @ 0.35) |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.840 | **0.844** |
| **Recall (Churn)** | 0.54 | **0.71** |
| **Precision (Churn)** | 0.65 | 0.57 |
| **Decision Threshold** | 0.50 | **0.35** |

### Business Rationale for Threshold = 0.35:
In customer retention, the business cost of losing a customer (false negative) far outweighs the marginal cost of offering a retention discount to a non-churner (false positive). Lowering the decision threshold to **0.35** boosts churner identification from ~54% to **71%**, giving customer retention teams maximum proactive intervention leverage.

---

## Feature Importance (Top Risk Drivers)

| Rank | Feature | Importance | Business Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `tenure` | 0.209 | Newer customers are at highest risk; focus on early onboarding. |
| 2 | `TotalCharges` | 0.131 | Low cumulative spend indicates nascent engagement. |
| 3 | `InternetService_Fiber optic` | 0.102 | Fiber optic customers churn at higher rates; service quality focus needed. |
| 4 | `PaymentMethod_Electronic check` | 0.091 | Manual payment methods correlate with lower commitment; encourage auto-pay. |
| 5 | `Contract_Two year` | 0.087 | Long-term contracts dramatically protect retention. |

---

## Repository Structure

```bash
CustomerChurn/
├── .dockerignore
├── Dockerfile                  # Containerization specification (Python 3.11 slim)
├── docker-compose.yml          # Single-command container deployment
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── data/
│   ├── raw/
│   │   ├── TelcoChurnData.csv  # Full raw dataset
│   │   └── sample_customer.csv # Sample records for testing inference
│   ├── processed/
│   │   └── processedTelcoChurnData.csv
│   └── sample_request.json     # Sample API request payload
├── models/
│   └── churn_model.pkl         # Serialized Joblib pipeline
├── notebooks/
│   ├── DataUnderstanding.ipynb
│   ├── DataCleaning.ipynb
│   ├── EDA.ipynb
│   ├── BaselineLogisticRegression.ipynb
│   └── TreeModels.ipynb
└── src/
    ├── __init__.py
    ├── api.py                  # FastAPI RESTful service for real-time inference
    ├── preprocessing.py        # Feature cleaning, encoding & scaling
    ├── modeling.py             # Random Forest pipeline builder & trainer
    ├── evaluation.py           # Threshold analysis & metrics
    ├── feature_importance.py   # Feature importance analysis (Pandas & NumPy)
    ├── inference.py            # Real-time inference function (Joblib model loader)
    └── utils.py                # Business logic, risk levels & retention recommendations
```

---

## Installation & Setup

### 1. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Or on Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running Real-Time Inference

### Option A: Python Module
```python
import pandas as pd
from src.inference import predict_churn

sample = pd.read_csv("data/raw/sample_customer.csv")
predictions = predict_churn(sample)
print(predictions)
```

### Option B: FastAPI REST Server
Start the real-time inference server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

#### API Endpoints:
- `GET /health` — Service health and threshold parameters
- `POST /predict` — Real-time single customer churn prediction
- `POST /predict_batch` — High-throughput batch inference for downstream systems
- `GET /features/importance` — Ranked feature importances

---

## Containerization (Docker)

The project is containerized for deployment across downstream systems and cloud services.

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t customer-churn-api:latest .

# Run the container
docker run -d -p 8000:8000 --name churn_container customer-churn-api:latest
```

### Or using Docker Compose:
```bash
docker compose up -d --build
```

Test the containerized endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @data/sample_request.json
```