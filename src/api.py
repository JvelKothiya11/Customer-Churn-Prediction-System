from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from src.inference import predict_churn, THRESHOLD
from src.feature_importance import get_feature_importances

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time inference API for Customer Churn Prediction using Random Forest and Joblib serialization.",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    customerID: Optional[str] = "CUST-001"
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="No")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=2)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=85.5)
    TotalCharges: float = Field(..., example=171.0)

class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    threshold_applied: float
    risk_level: str
    proactive_retention_recommendations: List[str]

class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput]

class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churners: int
    predictions: List[PredictionResponse]

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Customer Churn Prediction Service is online.",
        "threshold": THRESHOLD,
        "model": "Random Forest Classifier (Joblib serialized)",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "decision_threshold": THRESHOLD,
        "target_recall": 0.71,
        "target_roc_auc": 0.844
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(customer: CustomerInput):
    try:
        cust_dict = customer.dict()
        cid = cust_dict.pop("customerID", None)
        df_result = predict_churn(cust_dict, threshold=THRESHOLD)
        res = df_result.iloc[0].to_dict()
        return PredictionResponse(
            customerID=cid,
            churn_probability=res["churn_probability"],
            churn_prediction=res["churn_prediction"],
            threshold_applied=THRESHOLD,
            risk_level=res["risk_level"],
            proactive_retention_recommendations=res["proactive_retention_recommendations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchCustomerInput):
    try:
        raw_list = [c.dict() for c in batch.customers]
        customer_ids = [c.get("customerID") for c in raw_list]
        for c in raw_list:
            c.pop("customerID", None)

        df_result = predict_churn(raw_list, threshold=THRESHOLD)
        predictions = []
        churners_count = 0

        for i, row in df_result.iterrows():
            if row["churn_prediction"] == 1:
                churners_count += 1
            predictions.append(PredictionResponse(
                customerID=customer_ids[i],
                churn_probability=row["churn_probability"],
                churn_prediction=row["churn_prediction"],
                threshold_applied=THRESHOLD,
                risk_level=row["risk_level"],
                proactive_retention_recommendations=row["proactive_retention_recommendations"]
            ))

        return BatchPredictionResponse(
            total_customers=len(predictions),
            predicted_churners=churners_count,
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/importance", tags=["Analysis"])
def feature_importance(top_n: int = 10):
    try:
        df_imp = get_feature_importances(top_n=top_n)
        return df_imp.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
