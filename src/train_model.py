"""
train_model.py
----------------
End-to-end training script:
  1. Load CSV(s) from data/ (real CIC-IDS2017 or the generated demo set)
  2. Clean/prepare with data_preprocessing
  3. Build scikit-learn Pipelines (imputer + scaler + classifier) for a
     binary model (Normal vs Attack) and, if enough classes are present,
     a multiclass model (attack category)
  4. Train, evaluate, and save everything needed for inference with joblib

Run directly:  python -m src.train_model --data data/demo_traffic.csv
"""

import os
import sys
import time
import argparse
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_csv, load_multiple_csvs, prepare_dataset
from src.evaluation import evaluate_model, get_feature_importance

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


def build_pipeline(classifier):
    """Standard preprocessing + model pipeline.

    Bundling imputation/scaling with the classifier in a single sklearn
    Pipeline guarantees inference-time preprocessing exactly matches
    training-time preprocessing (requirement: no train/serve skew).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test, classifier, name):
    pipeline = build_pipeline(classifier)

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    metrics, cm, report, y_pred = evaluate_model(pipeline, X_test, y_test)
    metrics["training_time_seconds"] = train_time
    metrics["model_name"] = name

    print(f"\n=== {name} ===")
    print(f"  Train time: {train_time:.2f}s")
    for k in ("accuracy", "precision", "recall", "f1_score"):
        print(f"  {k.capitalize()}: {metrics[k]:.4f}")

    return pipeline, metrics, cm, report


def main():
    parser = argparse.ArgumentParser(description="Train the Network IDS models")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to a single CSV file OR a directory of CIC-IDS2017 CSVs. "
             "Defaults to data/demo_traffic.csv (generating it if missing).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also train Logistic Regression and Decision Tree for comparison.",
    )
    args = parser.parse_args()

    data_arg = args.data
    if data_arg is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_path = os.path.join(base_dir, "data", "demo_traffic.csv")
        if not os.path.exists(demo_path):
            print("No --data provided and no demo dataset found. Generating demo data...")
            from src.generate_demo_data import generate_demo_dataset
            generate_demo_dataset(n_rows=6000, out_path=demo_path)
        data_arg = demo_path
        print(f"[INFO] Using demo dataset at {demo_path}. This is SYNTHETIC data for "
              f"pipeline testing only -- see data/README.md to use real CIC-IDS2017 data.")

    print(f"Loading data from: {data_arg}")
    if os.path.isdir(data_arg):
        df = load_multiple_csvs(data_arg)
    else:
        df = load_csv(data_arg)
    print(f"Loaded {len(df)} raw rows.")

    df_clean, feature_cols, label_col = prepare_dataset(df)
    print(f"After cleaning: {len(df_clean)} rows, {len(feature_cols)} numeric features.")

    if label_col is None or "binary_label" not in df_clean.columns:
        raise ValueError(
            "Could not locate a label column in the dataset. Expected a column "
            "named 'Label' (or similar) as used in CIC-IDS2017 CSVs."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    X = df_clean[feature_cols]

    # ---------------------------------------------------------------
    # BINARY MODEL: Normal vs Attack
    # ---------------------------------------------------------------
    y_binary = df_clean["binary_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    binary_pipeline, binary_metrics, binary_cm, binary_report = train_and_evaluate(
        X_train, X_test, y_train, y_test, rf, "RandomForest_Binary"
    )

    comparison_results = {}
    if args.compare:
        for clf, name in [
            (LogisticRegression(max_iter=500, class_weight="balanced"), "LogisticRegression_Binary"),
            (DecisionTreeClassifier(random_state=42, class_weight="balanced"), "DecisionTree_Binary"),
        ]:
            _, m, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, clf, name)
            comparison_results[name] = m

    feature_importance = get_feature_importance(
        binary_pipeline.named_steps["classifier"], feature_cols
    )

    joblib.dump(binary_pipeline, os.path.join(MODELS_DIR, "binary_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    # ---------------------------------------------------------------
    # MULTICLASS MODEL: attack category (only if >=2 classes present
    # with enough samples each -- do not force it if data doesn't support it)
    # ---------------------------------------------------------------
    multiclass_metrics = None
    multiclass_cm = None
    multiclass_report = None
    class_counts = df_clean["attack_category"].value_counts()
    viable_classes = class_counts[class_counts >= 10].index.tolist()

    if len(viable_classes) >= 2:
        df_multi = df_clean[df_clean["attack_category"].isin(viable_classes)]
        X_multi = df_multi[feature_cols]
        y_multi = df_multi["attack_category"]

        Xm_train, Xm_test, ym_train, ym_test = train_test_split(
            X_multi, y_multi, test_size=0.25, random_state=42, stratify=y_multi
        )

        rf_multi = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1,
        )
        multi_pipeline, multiclass_metrics, multiclass_cm, multiclass_report = train_and_evaluate(
            Xm_train, Xm_test, ym_train, ym_test, rf_multi, "RandomForest_Multiclass"
        )
        joblib.dump(multi_pipeline, os.path.join(MODELS_DIR, "multiclass_model.pkl"))
        joblib.dump(sorted(y_multi.unique().tolist()), os.path.join(MODELS_DIR, "multiclass_labels.pkl"))
    else:
        print("\n[INFO] Not enough distinct attack categories with sufficient samples "
              "to train a reliable multiclass model. Skipping (binary model only).")

    # ---------------------------------------------------------------
    # Save summary artifacts for the dashboard
    # ---------------------------------------------------------------
    summary = {
        "binary_metrics": {k: v for k, v in binary_metrics.items() if k != "model_name"},
        "multiclass_metrics": (
            {k: v for k, v in multiclass_metrics.items() if k != "model_name"}
            if multiclass_metrics else None
        ),
        "comparison_results": {
            name: {k: v for k, v in m.items() if k != "model_name"}
            for name, m in comparison_results.items()
        },
        "n_rows_used": len(df_clean),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "class_distribution": class_counts.to_dict(),
        "has_multiclass_model": multiclass_metrics is not None,
    }

    with open(os.path.join(OUTPUTS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    feature_importance.to_csv(
        os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False
    )
    pd.DataFrame(binary_cm).to_csv(
        os.path.join(OUTPUTS_DIR, "binary_confusion_matrix.csv"), index=False
    )
    if multiclass_cm is not None:
        pd.DataFrame(multiclass_cm).to_csv(
            os.path.join(OUTPUTS_DIR, "multiclass_confusion_matrix.csv"), index=False
        )

    with open(os.path.join(OUTPUTS_DIR, "classification_report_binary.json"), "w") as f:
        json.dump(binary_report, f, indent=2)
    if multiclass_report is not None:
        with open(os.path.join(OUTPUTS_DIR, "classification_report_multiclass.json"), "w") as f:
            json.dump(multiclass_report, f, indent=2)

    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Evaluation outputs saved to: {OUTPUTS_DIR}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
