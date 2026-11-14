from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_churn(record: dict, threshold: float = 0.5) -> dict:
    df = add_features(pd.DataFrame([record]))
    proba = float(get_model().predict_proba(df)[0, 1])
    label = "Likely to Churn" if proba >= threshold else "Unlikely to Churn"
    return {"prediction": label, "probability": round(proba, 4)}