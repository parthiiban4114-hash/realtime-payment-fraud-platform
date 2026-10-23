
from pathlib import Path
import json

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def load_data(file_path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the raw transaction dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns and basic data types."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "isFraud must contain only 0 and 1."
        )


def generate_quality_report(df: pd.DataFrame) -> dict:
    """Summarize dataset quality before cleaning."""
    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },
        "fraud_distribution": {
            str(key): int(value)
            for key, value in df[TARGET].value_counts().items()
        },
        "fraud_rate_percent": round(
            float(df[TARGET].mean() * 100), 4
        ),
        "negative_amount_rows": int(
            (df["amount"] < 0).sum()
        ),
        "invalid_step_rows": int(
            (df["step"] < 1).sum()
        ),
    }

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative data-cleaning rules."""
    cleaned = df.copy()

    # Remove exact duplicate records.
    cleaned = cleaned.drop_duplicates()

    # Remove rows with missing required values.
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    # Remove records with invalid transaction amounts or time steps.
    cleaned = cleaned[
        (cleaned["amount"] >= 0)
        & (cleaned["step"] >= 1)
    ].copy()

    # Normalize transaction type.
    cleaned["type"] = (
        cleaned["type"].astype(str).str.strip().str.upper()
    )

    # Convert numeric fields safely.
    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="raise"
        )

    # Ensure the target remains binary.
    if not cleaned[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Invalid target values detected after cleaning."
        )

    if cleaned.empty:
        raise ValueError("No records remain after cleaning.")

    return cleaned


def save_outputs(
    cleaned: pd.DataFrame,
    report: dict
) -> None:
    """Save processed data and its quality report."""
    output_file = PROCESSED_DIR / "transactions_cleaned.csv"
    report_file = PROCESSED_DIR / "data_quality_report.json"

    cleaned.to_csv(output_file, index=False)

    report["rows_after_cleaning"] = int(len(cleaned))
    report["rows_removed"] = int(
        report["row_count"] - len(cleaned)
    )

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Cleaned data saved to: {output_file}")
    print(f"Quality report saved to: {report_file}")


def main() -> None:
    print("Loading raw transaction data...")
    df = load_data()

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    print("Validating schema...")
    validate_schema(df)

    print("Generating data quality report...")
    report = generate_quality_report(df)

    print(json.dumps(report, indent=2))

    print("Cleaning transaction data...")
    cleaned = clean_data(df)

    save_outputs(cleaned, report)

    print("\nData processing completed successfully.")


if __name__ == "__main__":
    main()
