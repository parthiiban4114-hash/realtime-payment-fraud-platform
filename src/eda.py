
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading transaction data...")
    df = pd.read_csv(DATA_FILE)

    print(f"Total transactions: {len(df):,}")

    # Fraud vs legitimate transactions
    counts = df["isFraud"].value_counts().sort_index()
    fraud_summary = {
        "legitimate_transactions": int(counts.get(0, 0)),
        "fraud_transactions": int(counts.get(1, 0)),
        "fraud_rate_percent": round(float(df["isFraud"].mean() * 100), 4),
    }

    print("\nFraud summary:")
    print(json.dumps(fraud_summary, indent=4))

    # Transaction types and fraud rates
    type_summary = (
        df.groupby("type", observed=True)
        .agg(
            transaction_count=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    type_summary["fraud_rate_percent"] = (
        type_summary["fraud_count"]
        / type_summary["transaction_count"]
        * 100
    ).round(4)

    print("\nTransaction type summary:")
    print(type_summary.to_string(index=False))

    # Save results
    type_summary.to_csv(
        OUTPUT_DIR / "transaction_type_summary.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "fraud_summary.json", "w") as f:
        json.dump(fraud_summary, f, indent=4)

    # Plot class imbalance
    plt.figure(figsize=(7, 5))
    counts.rename(index={0: "Legitimate", 1: "Fraud"}).plot(
        kind="bar"
    )
    plt.title("Legitimate vs Fraud Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_distribution.png", dpi=150)
    plt.close()

    # Plot fraud rates by transaction type
    plt.figure(figsize=(9, 5))
    type_summary.plot(
        x="type",
        y="fraud_rate_percent",
        kind="bar",
        legend=False,
    )
    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_rate_by_type.png", dpi=150)
    plt.close()

    print(f"\nEDA completed. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
