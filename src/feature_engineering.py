
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_cleaned.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "isFraud"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction features without using the target as a feature."""

    df = df.copy()

    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["errorBalanceDest"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    # Exclude identifiers, the target, and the existing fraud flag.
    excluded = [
        "nameOrig",
        "nameDest",
        "isFlaggedFraud",
        TARGET,
    ]

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    features = df[feature_columns].copy()
    features[TARGET] = df[TARGET].astype("int8")

    return features


def main():
    print("Loading cleaned transaction data...")
    df = pd.read_csv(DATA_FILE)

    print(f"Input rows: {len(df):,}")

    model_df = create_features(df)

    output_file = OUTPUT_DIR / "transactions_features.csv"
    model_df.to_csv(output_file, index=False)

    print(f"Output rows: {len(model_df):,}")
    print(f"Feature columns: {model_df.shape[1] - 1}")
    print(f"Target distribution:\n{model_df[TARGET].value_counts()}")
    print(f"Saved feature dataset to: {output_file}")


if __name__ == "__main__":
    main()
