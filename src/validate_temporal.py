
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "transactions_features.csv"
REPORT_DIR = ROOT / "reports" / "model_evaluation"
MODEL_DIR = ROOT / "models"

TARGET = "isFraud"
RANDOM_STATE = 42

# Features intended to be available when authorization is requested.
NUMERIC_FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "oldbalanceDest",
]
CATEGORICAL_FEATURES = ["type"]

# Thresholds to compare using the same held-out temporal test set.
THRESHOLDS = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
MINIMUM_PRECISION = 0.95


def evaluate_thresholds(y_true, probabilities):
    """Evaluate classification metrics at multiple probability thresholds."""
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": float(
                precision_score(y_true, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, predictions, zero_division=0)
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    return results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    required_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    )
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Hold out the latest 20% of transaction steps.
    cutoff = df["step"].quantile(0.80)

    train = df[df["step"] <= cutoff]
    test = df[df["step"] > cutoff]

    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty dataset.")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train[features]
    y_train = train[TARGET].astype("int8")
    X_test = test[features]
    y_test = test[TARGET].astype("int8")

    print(f"Step cutoff: {cutoff}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Test fraud cases: {int(y_test.sum()):,}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\nTraining temporal-validation model...")
    started = time.time()
    model.fit(X_train, y_train)
    training_seconds = round(time.time() - started, 2)

    print("\nScoring held-out transactions...")
    probabilities = model.predict_proba(X_test)[:, 1]

    # Overall ranking quality does not depend on the chosen threshold.
    pr_auc = float(average_precision_score(y_test, probabilities))

    # Evaluate multiple thresholds on the same test set.
    threshold_results = evaluate_thresholds(y_test, probabilities)

    threshold_df = pd.DataFrame(threshold_results)

    print("\nThreshold comparison:")
    print(
        threshold_df.round(4).to_string(index=False)
    )

    # Exploratory selection: maximize recall subject to minimum precision.
    # Final threshold selection should use a separate validation period.
    eligible = [
        result for result in threshold_results
        if result["precision"] >= MINIMUM_PRECISION
    ]

    best_threshold_result = None

    if eligible:
        best_threshold_result = max(
            eligible,
            key=lambda result: (
                result["recall"],
                result["precision"],
            ),
        )

        print(
            "\nBest tested threshold at "
            f">= {MINIMUM_PRECISION:.0%} precision:"
        )
        print(json.dumps(best_threshold_result, indent=4))
    else:
        print(
            "\nNo tested threshold achieved the minimum precision "
            f"of {MINIMUM_PRECISION:.0%}."
        )

    # Keep the original 0.5 threshold metrics for comparison.
    baseline = next(
        result for result in threshold_results
        if result["threshold"] == 0.50
    )

    metrics = {
        "features": features,
        "step_cutoff": float(cutoff),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "training_fraud_cases": int(y_train.sum()),
        "test_fraud_cases": int(y_test.sum()),
        "average_precision_pr_auc": pr_auc,
        "baseline_threshold": baseline["threshold"],
        "baseline_precision": baseline["precision"],
        "baseline_recall": baseline["recall"],
        "baseline_f1_score": baseline["f1_score"],
        "baseline_confusion_matrix": [
            [
                baseline["true_negatives"],
                baseline["false_positives"],
            ],
            [
                baseline["false_negatives"],
                baseline["true_positives"],
            ],
        ],
        "minimum_precision_target": MINIMUM_PRECISION,
        "best_threshold_result": best_threshold_result,
        "training_seconds": training_seconds,
        "note": (
            "Threshold selection on this test set is exploratory. "
            "Use a separate validation period to select the final "
            "threshold and preserve an untouched test set for "
            "final performance reporting."
        ),
    }

    metrics_file = REPORT_DIR / "temporal_validation_metrics.json"
    metrics_file.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    thresholds_file = REPORT_DIR / "threshold_comparison.json"
    thresholds_file.write_text(
        json.dumps(threshold_results, indent=4),
        encoding="utf-8",
    )

    model_file = MODEL_DIR / "temporal_random_forest.joblib"
    joblib.dump(model, model_file)

    print("\nBaseline results at threshold 0.50:")
    print(json.dumps(baseline, indent=4))

    print(f"\nSaved metrics to: {metrics_file}")
    print(f"Saved threshold results to: {thresholds_file}")
    print(f"Saved model to: {model_file}")


if __name__ == "__main__":
    main()
