import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning. Contains only fixed rules (no statistics learned from data)."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges has blank strings (customers with tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    df["HasSecuritySupport"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scaling + one-hot encoding. Fit on TRAIN data only (avoids leakage)."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])


def get_train_test(test_size=0.2, random_state=42):
    df = add_features(clean_data(load_data()))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate (train):", round(y_train.mean(), 3))
    