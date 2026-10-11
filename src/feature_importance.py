import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")

def get_feature_importances(
    model=None,
    model_path: str = DEFAULT_MODEL_PATH,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extracts and ranks feature importances from the trained Random Forest pipeline
    using NumPy and Pandas.

    Identifies key operational risk factors (e.g., tenure, service type, contract type).
    """
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Retrieve feature names after one-hot encoding and scaling
    feature_names = preprocessor.get_feature_names_out()

    # Clean feature name prefixes ('num__', 'cat__') for clarity
    cleaned_feature_names = [
        col.replace("num__", "").replace("cat__", "") for col in feature_names
    ]

    importances = classifier.feature_importances_

    # Create sorted DataFrame using Pandas and NumPy
    df_importance = pd.DataFrame({
        "feature": cleaned_feature_names,
        "importance": importances,
        "importance_pct": np.round(importances * 100, 2)
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    if top_n is not None:
        return df_importance.head(top_n)

    return df_importance

if __name__ == "__main__":
    df_imp = get_feature_importances(top_n=10)
    print("Top Feature Importances:")
    print(df_imp.to_string(index=False))
