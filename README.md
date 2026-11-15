# Customer Churn Prediction System

An end-to-end machine learning system that predicts whether a telecom customer is likely to leave, based on historical customer data, and serves predictions through a REST API. Built for the Cognirex Solutions AI/ML Engineer Technical Assessment (Round 2).

**Author:** Amshala Sravanthi

## Live Demo

- **API:** https://churn-api-dhrd.onrender.com
- **API docs (Swagger):** https://churn-api-dhrd.onrender.com/docs
- **Demo UI:** https://ai-ml-assessment-frontend.vercel.app
- **Demo video:** <https://drive.google.com/file/d/1RdrIi29hQqiHXhol_TUnzl8PnOc7w-2j/view?usp=sharing>
- **Evaluation report:** [`reports/evaluation_report.md`](reports/evaluation_report.md)

> The API runs on Render's free tier, so it sleeps when idle. The first request
> after a break can take 30-60 seconds to wake it up.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Architecture and Workflow](#3-architecture-and-workflow)
4. [Setup and Installation](#4-setup-and-installation)
5. [Dataset Understanding](#5-dataset-understanding)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Preprocessing and Feature Engineering](#7-preprocessing-and-feature-engineering)
8. [Algorithms Implemented](#8-algorithms-implemented)
9. [Model Comparison and Final Model Selection](#9-model-comparison-and-final-model-selection)
10. [Model Evaluation](#10-model-evaluation)
11. [Prediction API](#11-prediction-api)
12. [Deployment](#12-deployment)
13. [Challenges and Solutions](#13-challenges-and-solutions)
14. [Future Improvements and Real-World Applications](#14-future-improvements-and-real-world-applications)
15. [AI Tool Usage Disclosure](#15-ai-tool-usage-disclosure)

---

## 1. Project Overview

**Problem.** Acquiring a new customer costs far more than keeping an existing one. If a company can identify which customers are likely to churn, it can target them with retention offers before they leave.

**Goal.** Analyze customer data, find the patterns behind churn, train and compare classification models, select the best one, and expose it as an API that returns a churn prediction with a probability for any new customer record.

**Dataset.** IBM Telco Customer Churn (Kaggle), 7,043 customers and 21 columns. The CSV is stored in `dataset/`.

**Tech stack.** Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, FastAPI, Uvicorn, joblib, Docker, Render, Vercel.

**Headline result.** The final XGBoost model reaches a test ROC-AUC of **0.847** and a recall of **0.80** on the churn class, meaning it identifies about 8 out of 10 customers who actually leave.

## 2. Project Structure

```
AI-ML-Assessment/
│
├── dataset/                  # Raw Telco churn CSV
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Cleaning, feature engineering, train/test split, preprocessor
│   ├── train.py              # Trains, tunes and compares 3 models; saves best pipeline + metrics
│   └── predict.py            # Loads the saved pipeline and predicts for new customers
├── models/                   # Saved trained pipeline (churn_model.joblib)
├── api/
│   └── main.py               # FastAPI application
├── frontend/
│   └── index.html            # Demo web form (hosted on Vercel)
├── reports/
│   ├── evaluation_report.md  # Model evaluation report
│   ├── metrics.json          # Metrics for all three models
│   └── confusion_matrix.png  # Confusion matrix of the final model
├── Dockerfile                # Container build used for deployment
├── requirements.txt          # Full environment (training, notebooks)
├── requirements-deploy.txt   # Minimal dependencies for the API container
├── .gitignore
└── README.md
```

## 3. Architecture and Workflow

```
Raw CSV
   │
   ▼
Cleaning (types, blanks, ID column)
   │
   ▼
Feature engineering (NumServices, AvgMonthlySpend, HasSecuritySupport, TenureGroup)
   │
   ▼
Stratified 80/20 train/test split
   │
   ▼
Pipeline = ColumnTransformer (StandardScaler + OneHotEncoder) → Classifier
   │
   ▼
RandomizedSearchCV with stratified k-fold CV (scored on ROC-AUC)
   │
   ▼
Compare Logistic Regression / Random Forest / XGBoost on the held-out test set
   │
   ▼
Select best model → save full pipeline to models/ → metrics to reports/
   │
   ▼
FastAPI  POST /predict  →  { "prediction": ..., "probability": ... }
```

**Key design decision:** the preprocessing and the classifier are saved together as one scikit-learn Pipeline. The API therefore applies exactly the same scaling and encoding that was used during training, which removes the risk of training/serving mismatch.

## 4. Setup and Installation

**Requirements:** Python 3.10+ (developed on Python 3.14 with scikit-learn 1.9.1). Exact package versions are pinned in `requirements.txt`; install them as-is so the saved model loads correctly.

### 1. Clone the repository

```bash
git clone https://github.com/sravanthi-28/AI-ML-Assessment.git
cd AI-ML-Assessment
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` was exported on Windows and includes Windows-only Jupyter
> packages (such as `pywinpty`). On macOS or Linux, install the minimal set
> instead: `pip install -r requirements-deploy.txt`. It is enough to train the
> models and run the API.

### Train the models

Run from the project root:

```bash
python -m src.train
```

This loads and preprocesses the data, tunes and trains all three models, prints the comparison, and saves the best pipeline to `models/churn_model.joblib` and the metrics to `reports/metrics.json`.

### Run the API

```bash
uvicorn api.main:app --reload
```

- API base URL: http://127.0.0.1:8000
- Interactive documentation (Swagger UI): http://127.0.0.1:8000/docs

Run these commands from the project root (not from inside `api/` or `src/`), otherwise the `src` imports will fail.

## 5. Dataset Understanding

- **Source:** IBM Telco Customer Churn dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns (target: `Churn`)
- **Location in repo:** `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Preprocessing:** implemented in `src/preprocessing.py` (cleaning, feature engineering, encoding, scaling, train/test split)

Each row is one customer. The columns fall into four groups:

| Group | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Target / ID | Churn (Yes/No), customerID |

**Target balance.** Roughly 26.5% of customers churned and 73.5% stayed, so the dataset is imbalanced. This shaped both the training strategy and the choice of evaluation metrics (see Section 10).

## 6. Exploratory Data Analysis

The full analysis is in `notebooks/eda.ipynb`. It covers:

- Dataset shape, data types and summary statistics
- Missing values, duplicates and outliers
- Distribution of the target variable
- Churn rate across categorical features
- Distributions of tenure, MonthlyCharges and TotalCharges by churn status
- A correlation heatmap of the numeric features and the churn flag

### Key insights

**Tenure is the strongest churn signal.** Churn falls steadily as customers stay longer:

| Tenure group | Churn rate |
|---|---|
| 0-12 months | 47.4% |
| 13-24 months | 28.7% |
| 25-48 months | 20.4% |
| 49-72 months | 9.5% |

Customers in their first year are about five times as likely to leave as customers with more than four years of tenure. This motivated the `TenureGroup` engineered feature.

Other patterns found in the data:

- **Contract type.** Customers on month-to-month contracts churn far more than those on one-year or two-year contracts, which lock customers in.
- **Internet service.** Fiber optic customers churn more than DSL customers, which may reflect higher prices or service quality issues.
- **Payment method.** Customers paying by electronic check churn more than those on automatic payments.
- **Support services.** Customers without online security or tech support churn more than those who have them, so these services appear to increase stickiness.
- **Charges.** Higher monthly charges are associated with higher churn, while TotalCharges is largely driven by tenure.

## 7. Preprocessing and Feature Engineering

### Data cleaning

- `TotalCharges` is stored as text and contains blank entries (these belong to customers with tenure = 0). It is converted to a numeric type and the blanks are handled in `src/preprocessing.py` so no missing values reach the model.
- `customerID` is dropped, since it is an identifier with no predictive value.
- The target `Churn` is converted to a binary flag (Yes = 1, No = 0).
- Duplicate records and outliers were checked during EDA.

### Feature engineering

| Feature | Definition | Rationale |
|---|---|---|
| NumServices | Number of add-on services the customer subscribes to | Customers using more services are more embedded |
| AvgMonthlySpend | Total charges divided by months of tenure | Captures actual spending behavior over time |
| HasSecuritySupport | Has online security or tech support | Support services were linked to lower churn |
| TenureGroup | Tenure bucket: 0-12, 13-24, 25-48, 49-72 months | Churn is non-linear in tenure (see table above) |

### Encoding and scaling

Handled by a scikit-learn `ColumnTransformer` inside the pipeline:

- **Numeric features** (SeniorCitizen, tenure, MonthlyCharges, TotalCharges, NumServices, AvgMonthlySpend, HasSecuritySupport) are standardized with `StandardScaler`.
- **Categorical features** (16 columns, including TenureGroup) are one-hot encoded with `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`. Ignoring unknown categories means an unexpected value in an API request does not crash the model.

After encoding, the classifier receives 47 input features.

### Train/test split

A stratified 80/20 split with `random_state=42` gives 5,634 training rows and 1,409 test rows, with the same churn ratio in both sets.

## 8. Algorithms Implemented

| Model | Why it was included |
|---|---|
| Logistic Regression | Simple, fast and interpretable; a strong baseline for tabular data |
| Random Forest | Captures non-linear relationships and interactions; robust to noise |
| XGBoost | Gradient boosting; typically the strongest family of models on structured data |

### How performance was improved

- **Hyperparameter tuning** with `RandomizedSearchCV` using `StratifiedKFold` cross-validation, optimizing ROC-AUC.
- **Feature engineering** (four engineered features, described above).
- **Class imbalance handling** through class weighting: `class_weight="balanced"` for Logistic Regression and Random Forest, and `scale_pos_weight` (about 2.77) for XGBoost.
- **Regularization** through simple models (shallow trees, subsampling) to limit overfitting.

### Best hyperparameters found

| Model | Best parameters |
|---|---|
| Logistic Regression | C = 0.1 |
| Random Forest | n_estimators = 200, max_depth = 8, min_samples_leaf = 10, max_features = "sqrt" |
| XGBoost | n_estimators = 100, max_depth = 2, learning_rate = 0.1, subsample = 0.7, colsample_bytree = 0.85 |

## 9. Model Comparison and Final Model Selection

All three models were evaluated on the same held-out test set.

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC | Train ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8460 | 0.7374 | 0.5034 | 0.7861 | 0.6138 | 0.8421 | 0.8501 |
| Random Forest | 0.8482 | 0.7537 | 0.5237 | 0.7968 | 0.6320 | 0.8436 | 0.8868 |
| **XGBoost (final)** | **0.8504** | 0.7466 | 0.5146 | **0.7995** | 0.6262 | **0.8467** | 0.8630 |

### Final model: XGBoost

Why XGBoost was selected:

1. **Best ranking quality.** It has the highest cross-validated ROC-AUC (0.8504) and the highest test ROC-AUC (0.8467).
2. **Better generalization than Random Forest.** Its train-to-test ROC-AUC gap is small (0.863 to 0.847), while Random Forest drops from 0.887 to 0.844, a clear sign of overfitting.
3. **Highest recall (0.80).** For churn, missing a customer who leaves is more costly than sending an unnecessary retention offer, so catching more churners matters most.
4. **Efficient and lightweight.** Shallow trees (`max_depth = 2`) and 100 estimators give fast predictions suitable for an API.

**An honest note:** the three models perform very similarly, with ROC-AUC differing by less than 0.005 on the test set. XGBoost is chosen because it is best on the most reliable measure (cross-validated ROC-AUC), generalizes well, and has the highest recall, not because it wins by a large margin. This is expected for a small, relatively clean tabular dataset.

## 10. Model Evaluation

The full written evaluation is in [`reports/evaluation_report.md`](reports/evaluation_report.md).

### Metrics for the final model (XGBoost, test set)

| Metric | Value |
|---|---|
| Accuracy | 0.7466 |
| Precision | 0.5146 |
| Recall | 0.7995 |
| F1-Score | 0.6262 |
| ROC-AUC | 0.8467 |

### Confusion matrix (test set, 1,409 customers)

| | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 753 (true negatives) | 282 (false positives) |
| **Actual: Churn** | 75 (false negatives) | 299 (true positives) |

The model correctly catches 299 of the 374 customers who actually churned (80%). The trade-off is 282 false alarms, which is why precision is about 0.51. The confusion matrix plot and the metrics JSON for all three models are in the `reports/` folder.

### How the key evaluation concerns were handled

**Class imbalance.** About 26.5% of customers churn. The imbalance was handled by:

- using a stratified train/test split and stratified cross-validation, so every fold keeps the same churn ratio;
- applying class weighting so the model does not ignore the minority class;
- judging models on ROC-AUC, recall and F1 instead of accuracy.

**Overfitting.** It was identified by comparing training and test performance and by using cross-validation. It was prevented by tuning for cross-validated ROC-AUC, using shallow trees (`max_depth = 2`), row and column subsampling (0.7 and 0.85), a low learning rate and a limited number of trees. The final model's train and test ROC-AUC are close (0.863 vs 0.847), unlike the Random Forest, which overfits more.

**Data leakage.** It was avoided by:

- splitting the data into train and test sets before fitting any transformer;
- keeping scaling and encoding inside the scikit-learn Pipeline, so they are fitted only on training data (and only on the training folds during cross-validation);
- never using the target variable to build features;
- dropping the identifier column.

**Why accuracy alone is not sufficient.** About 73.5% of customers do not churn, so a model that always predicts "will stay" would score about 73.5% accuracy while catching zero churners and being useless for retention. Our final model's accuracy of 0.747 looks only slightly better than that useless baseline, but its recall of 0.80 shows it is actually finding churners. Precision shows how many alerts are correct, recall shows how many churners are caught, F1 balances the two, and ROC-AUC measures how well the model ranks customers regardless of the decision threshold.

## 11. Prediction API

The API is built with FastAPI. It loads the saved pipeline once at startup, validates the incoming JSON, computes the engineered features, and returns the prediction with the churn probability.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` when the model is loaded (503 otherwise) |
| POST | `/predict` | Returns a churn prediction and probability |

### Input fields for `POST /predict`

| Field | Type | Allowed values |
|---|---|---|
| gender | string | Female, Male |
| SeniorCitizen | integer | 0, 1 |
| Partner | string | Yes, No |
| Dependents | string | Yes, No |
| tenure | integer | months with the company (0-100) |
| PhoneService | string | Yes, No |
| MultipleLines | string | Yes, No, No phone service |
| InternetService | string | DSL, Fiber optic, No |
| OnlineSecurity | string | Yes, No, No internet service |
| OnlineBackup | string | Yes, No, No internet service |
| DeviceProtection | string | Yes, No, No internet service |
| TechSupport | string | Yes, No, No internet service |
| StreamingTV | string | Yes, No, No internet service |
| StreamingMovies | string | Yes, No, No internet service |
| Contract | string | Month-to-month, One year, Two year |
| PaperlessBilling | string | Yes, No |
| PaymentMethod | string | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges | float | monthly bill amount (>= 0) |
| TotalCharges | float | total billed so far (>= 0) |

### Sample request

```json
{"Contract": "Month-to-month",
  "Dependents": "No",
  "DeviceProtection": "No",
  "InternetService": "DSL",
  "MonthlyCharges": 29.85,
  "MultipleLines": "No phone service",
  "OnlineBackup": "Yes",
  "OnlineSecurity": "No",
  "PaperlessBilling": "Yes",
  "Partner": "Yes",
  "PaymentMethod": "Electronic check",
  "PhoneService": "No",
  "SeniorCitizen": 0,
  "StreamingMovies": "No",
  "StreamingTV": "No",
  "TechSupport": "No",
  "TotalCharges": 29.85,
  "gender": "Female",
  "tenure": 1
}
```

### Sample response

```json
{
  "prediction": "Likely to Churn",
  "probability": 0.8676
}
```

`prediction` is `"Likely to Churn"` or `"Unlikely to Churn"` (probability at or above 0.5 gives "Likely to Churn"), and `probability` is the model's estimated probability that the customer churns.

### How to test the API

**Option 1: Swagger UI (easiest).** Open `/docs` (locally http://127.0.0.1:8000/docs, or the live link at the top), expand `POST /predict`, click **Try it out**, paste the sample request, and click **Execute**.

**Option 2: PowerShell**

```powershell
$body = @{
  gender="Female"; SeniorCitizen=0; Partner="Yes"; Dependents="No"; tenure=2
  PhoneService="Yes"; MultipleLines="No"; InternetService="Fiber optic"
  OnlineSecurity="No"; OnlineBackup="No"; DeviceProtection="No"; TechSupport="No"
  StreamingTV="Yes"; StreamingMovies="Yes"; Contract="Month-to-month"
  PaperlessBilling="Yes"; PaymentMethod="Electronic check"
  MonthlyCharges=95.5; TotalCharges=191.0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
```

**Option 3: curl**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":191.0}'
```

**Sanity check.** A low-risk customer (for example `tenure = 60`, `Contract = "Two year"`, `TechSupport = "Yes"`) should return a clearly lower probability than the example above.

### Error handling

- Missing fields or wrong data types (for example `"tenure": "abc"`) are rejected by FastAPI's request validation with an HTTP 422 response that explains which field is invalid, instead of crashing the server.
- Unexpected category values do not crash the model, because the encoder is configured with `handle_unknown="ignore"`. The API schema also restricts each categorical field to its valid values.
- If the model file is missing, `/predict` returns HTTP 503, and unexpected server-side errors return HTTP 500 with a clear message instead of a raw stack trace.

## 12. Deployment

- **API:** packaged with the `Dockerfile` and deployed as a Docker web service on Render's free tier. The container installs `requirements-deploy.txt` (no Jupyter or plotting packages), and Render checks the `/health` endpoint. CORS is enabled so the browser demo can call the API.
- **Demo UI:** the static page in `frontend/` is hosted on Vercel and calls the Render API. It is a convenience for trying the model, and the API itself is the main deliverable.

To run the container locally:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Then open http://localhost:8000/docs.

## 13. Challenges and Solutions

| Challenge | Solution |
|---|---|
| `TotalCharges` is stored as text and has blank values for new customers | Converted to numeric and handled the blanks during preprocessing so no missing values reach the model |
| Imbalanced target (about 26.5% churn) | Stratified splits, class weighting (`class_weight`, `scale_pos_weight`), and evaluation on ROC-AUC, recall and F1 instead of accuracy |
| Preventing leakage between training and test data | Split first, and kept scaling and encoding inside a single scikit-learn Pipeline |
| Random Forest overfit the training data (train ROC-AUC 0.887 vs test 0.844) | Constrained tree depth and leaf size during tuning; the final model was chosen partly for its smaller train-test gap |
| The API receives raw customer fields, but the model needs engineered features | Feature engineering is applied in the prediction code before the pipeline is called, so clients only send the original fields |
| Saved models can fail to load if library versions differ | Pinned exact package versions in `requirements.txt` and `requirements-deploy.txt` |
| The full `requirements.txt` includes Windows-only packages that break a Linux container build | Created a minimal `requirements-deploy.txt` used only by the Docker image |
| Free hosting sleeps when idle, so the first request is slow | Added a `/health` endpoint, documented the cold start, and warm up the service before demos |

## 14. Future Improvements and Real-World Applications

### Future improvements

- Tune the decision threshold using the actual cost of losing a customer versus the cost of a retention offer.
- Explain predictions with SHAP values so retention teams see why a customer was flagged.
- Try more models (LightGBM, CatBoost), probability calibration, and resampling methods such as SMOTE.
- Add automated tests and CI.
- Monitor in production for data drift and performance decay, and retrain on a schedule.
- Add authentication, logging and batch prediction endpoints to the API.

### Real-world applications

- Targeting retention campaigns (discounts, contract upgrades, proactive support calls) at high-risk customers.
- Prioritizing customer-success outreach by churn probability.
- Identifying the drivers of churn to inform pricing, contract and service decisions.
- The same approach applies to other subscription businesses such as banking, SaaS, insurance and streaming.

## 15. AI Tool Usage Disclosure

In line with the assessment guidelines, AI tools were used as follows:

- **Claude (Anthropic):** used to review progress against the assessment requirements, to help write the Dockerfile and deployment configuration (Render and Vercel), to help build the demo web page in `frontend/`, to troubleshoot deployment errors, and to help draft and structure this README and the evaluation report.

All AI-assisted content was reviewed, edited and verified by the author. The models were trained, the evaluation metrics were produced, and the API was run and tested by the author. The author understands the complete implementation, including the preprocessing pipeline, model selection, evaluation, API design and deployment, and can explain it in detail.