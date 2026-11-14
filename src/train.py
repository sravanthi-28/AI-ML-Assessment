import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from src.preprocessing import get_train_test, build_preprocessor

RANDOM_STATE = 42


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    X_train, X_test, y_train, y_test = get_train_test()

    # Class imbalance: class weights (LR, RF) and scale_pos_weight (XGBoost)
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
            5,
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [200, 300, 500],
             "clf__max_depth": [5, 8, 12, None],
             "clf__min_samples_leaf": [1, 3, 5, 10],
             "clf__max_features": ["sqrt", 0.5]},
            15,
        ),
        "XGBoost": (
            XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [100, 200, 300],
             "clf__max_depth": [2, 3, 4, 6],
             "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0]},
            20,
        ),
    }

    # Cross-validation inside the training set only (test set stays untouched)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, (clf, grid, n_iter) in candidates.items():
        print(f"\nTuning {name} ...")
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc",
                                    cv=cv, n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best
        results[name] = {
            "cv_roc_auc": search.best_score_,
            "best_params": search.best_params_,
            "train": evaluate(best, X_train, y_train),
            "test": evaluate(best, X_test, y_test),
        }
        print("  CV ROC-AUC:", round(search.best_score_, 4))
        print("  Test:", {k: round(v, 3) for k, v in results[name]["test"].items()})

    # Select the final model by CV score (not by test score, to avoid leakage)
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    print("\nFinal model:", best_name)
    final = fitted[best_name]

    joblib.dump(final, "models/churn_model.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump({"final_model": best_name, "results": results},
                  f, indent=2, default=str)

    cm = confusion_matrix(y_test, final.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved model to models/churn_model.joblib and reports to reports/")


if __name__ == "__main__":
    main()