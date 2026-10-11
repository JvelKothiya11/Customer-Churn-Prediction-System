import os
import pandas as pd
from typing import Union, Dict, Any, List
from src.preprocessing import clean_total_charges
from src.utils import load_churn_model, get_risk_level, generate_retention_recommendations

THRESHOLD = 0.35
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl"
)

def predict_churn(
    customer_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    threshold: float = THRESHOLD,
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """
    Real-time inference function for customer churn prediction.

    Parameters:
        customer_data: DataFrame or dictionary/list of dictionaries containing customer features.
        threshold: Decision threshold for churn classification (default: 0.35).
        model_path: Path to serialized Joblib model.

    Returns:
        pd.DataFrame with churn_probability, churn_prediction, risk_level, and recommendations.
    """
    model = load_churn_model(model_path)

    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    else:
        raise TypeError("customer_data must be a pandas DataFrame, dict, or list of dicts")

    # Clean TotalCharges numeric conversion
    df_clean = clean_total_charges(df)

    # Drop non-feature columns if present
    features_df = df_clean.drop(columns=[col for col in ["customerID", "Churn"] if col in df_clean.columns])

    # Compute probabilities using serialized Joblib pipeline
    probas = model.predict_proba(features_df)[:, 1]
    predictions = (probas >= threshold).astype(int)

    results = []
    for idx, (prob, pred) in enumerate(zip(probas, predictions)):
        cust_row = df.iloc[idx].to_dict()
        results.append({
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(pred),
            "risk_level": get_risk_level(prob, threshold),
            "proactive_retention_recommendations": generate_retention_recommendations(cust_row, prob)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    sample_cust = {
        "customerID": "9237-HQITU",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0
    }
    output = predict_churn(sample_cust)
    print("Inference Result:")
    print(output.to_dict(orient="records"))
