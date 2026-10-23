
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "random_forest.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"


def main():
    model = joblib.load(MODEL_FILE)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(output_file, index=False)

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    df_columns = pd.read_csv(DATA_FILE, nrows=5).columns.tolist()

    print("\nTarget column present:", "isFraud" in df_columns)
    print("Identifier columns excluded:",
          "nameOrig" not in df_columns and "nameDest" not in df_columns)
    print("isFlaggedFraud excluded:", "isFlaggedFraud" not in df_columns)

    print(f"\nSaved feature report to: {output_file}")


if __name__ == "__main__":
    main()
