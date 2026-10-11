import os
import joblib
import pandas as pd
from typing import Dict, Any, List

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl"
)

_cached_model = None

def load_churn_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads and caches the Joblib serialized pipeline model."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _cached_model = joblib.load(model_path)
    return _cached_model

def get_risk_level(probability: float, threshold: float = 0.35) -> str:
    """Classifies risk level based on churn probability."""
    if probability < 0.20:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    elif probability < 0.60:
        return "High"
    else:
        return "Critical"

def generate_retention_recommendations(customer: Dict[str, Any], probability: float) -> List[str]:
    """Generates tailored proactive retention recommendations based on customer attributes."""
    recs = []

    tenure = customer.get("tenure", 0)
    contract = customer.get("Contract", "")
    internet = customer.get("InternetService", "")
    payment = customer.get("PaymentMethod", "")
    monthly_charges = customer.get("MonthlyCharges", 0)

    if tenure <= 12:
        recs.append("Early Onboarding: Enroll customer in high-touch onboarding program.")
    if "Month-to-month" in contract:
        recs.append("Contract Retention: Offer discount or loyalty incentive to transition to 1-year or 2-year contract.")
    if "Fiber optic" in internet:
        recs.append("Technical Support: Proactively check service reliability and speed satisfaction for Fiber Optic lines.")
    if "Electronic check" in payment:
        recs.append("Auto-Pay Incentive: Provide one-time billing discount to switch to automatic credit card or bank transfer.")
    if monthly_charges > 75:
        recs.append("Value Optimization: Review service bundle to ensure competitive pricing and perceived value.")

    if not recs:
        recs.append("Standard Engagement: Maintain regular service check-ins and loyalty rewards.")

    return recs
