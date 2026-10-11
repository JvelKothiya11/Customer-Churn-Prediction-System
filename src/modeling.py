import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from src.preprocessing import get_preprocessor, clean_total_charges

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl"
)
RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "raw", "TelcoChurnData.csv"
)

def build_random_forest_pipeline(
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_leaf: int = 50,
    random_state: int = 42,
    n_jobs: int = 1
) -> Pipeline:
    """Builds the complete end-to-end churn prediction pipeline."""
    preprocessor = get_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

def train_and_save_model(
    data_path: str = RAW_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    **rf_params
) -> Pipeline:
    """Loads raw data, trains the Random Forest pipeline, and saves using Joblib."""
    df = pd.read_csv(data_path)
    df = clean_total_charges(df)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        y = df["Churn"].astype(int)

    pipeline = build_random_forest_pipeline(**rf_params)
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

    # Evaluate on data
    evaluate_model(pipeline, X, y)
    return pipeline

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.35):
    """
    Evaluates the model and computes ROC-AUC and classification metrics
    at the specified decision threshold.
    """
    X_clean = clean_total_charges(X)
    if "customerID" in X_clean.columns:
        X_clean = X_clean.drop(columns=["customerID"])
    if "Churn" in X_clean.columns:
        X_clean = X_clean.drop(columns=["Churn"])

    y_proba = model.predict_proba(X_clean)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y, y_proba)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Classification Report (Threshold = {threshold}):")
    print(classification_report(y, y_pred))
    return {
        "roc_auc": auc,
        "threshold": threshold,
        "classification_report": classification_report(y, y_pred, output_dict=True)
    }

if __name__ == "__main__":
    train_and_save_model()
