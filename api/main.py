import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import get_model, predict_churn

logger = logging.getLogger("churn-api")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 1, "PhoneService": "No",
                "MultipleLines": "No phone service", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85, "TotalCharges": 29.85,
            }
        }
    }


class Prediction(BaseModel):
    prediction: str
    probability: float


@app.on_event("startup")
def load_model_on_startup():
    try:
        get_model()
    except Exception:
        logger.exception("Could not load model. Run `python -m src.train` first.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        return predict_churn(customer.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found. Train the model first.")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")