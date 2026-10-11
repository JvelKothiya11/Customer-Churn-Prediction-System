import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Fix TotalCharges missing values and convert to float."""
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def get_preprocessor(numerical_cols=None, categorical_cols=None) -> ColumnTransformer:
    """
    Returns a ColumnTransformer with StandardScaler for numeric features
    and OneHotEncoder (drop='first', handle_unknown='ignore') for categorical features.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )
    return preprocessor
