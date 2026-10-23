
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

TARGET = "isFraud"
RANDOM_STATE = 42


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature dataset...")
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype("int8")

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Test fraud cases: {int(y_test.sum()):,}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        started = time.time()
        pipeline.fit(X_train, y_train)
        training_seconds = round(time.time() - started, 2)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "average_precision_pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
            "training_seconds": training_seconds,
            "classification_report": classification_report(
                y_test,
                predictions,
                zero_division=0,
                output_dict=True,
            ),
        }

        results[model_name] = metrics

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{model_name}.joblib",
        )

        print(json.dumps({
            key: value for key, value in metrics.items()
            if key != "classification_report"
        }, indent=4))

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved metrics to {REPORT_DIR / 'metrics.json'}")
    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
