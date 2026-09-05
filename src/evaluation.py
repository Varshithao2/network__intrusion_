"""
evaluation.py
--------------
Model evaluation utilities: metrics, confusion matrix, classification report,
and feature importance extraction. Kept separate from train_model.py so the
Streamlit app can re-run evaluation-only logic on saved artifacts without
retraining.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, average="weighted"):
    """Compute standard classification metrics plus prediction timing.

    average='weighted' is used because intrusion-detection datasets are
    class-imbalanced (mostly Normal traffic); weighted averaging accounts
    for support per class rather than treating rare attack classes and the
    dominant Normal class equally.
    """
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
        "predict_time_seconds": predict_time,
        "n_test_samples": len(y_test),
    }

    cm = confusion_matrix(y_test, y_pred, labels=sorted(pd.unique(y_test)))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    return metrics, cm, report, y_pred


def get_feature_importance(model, feature_names, top_n=20):
    """Extract and sort feature importances from a tree-based model.

    Returns a DataFrame sorted descending by importance, truncated to
    top_n rows for readable dashboard display.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def why_precision_recall_matter() -> str:
    """Returns a short explanation used in the dashboard/README.

    Kept as a function (not a hardcoded string in multiple places) so the
    explanation stays consistent across the app and documentation.
    """
    return (
        "In intrusion detection, the dataset is heavily imbalanced -- the vast "
        "majority of traffic is Normal. A model that always predicts 'Normal' "
        "could score 95%+ accuracy while catching zero attacks, which is "
        "useless for security. Recall matters because it measures how many "
        "actual attacks were caught (missed attacks = false negatives = "
        "breaches that go undetected). Precision matters because it measures "
        "how many alerts are real (too many false positives = alert fatigue "
        "for the SOC team). F1-score balances both, which is why it is "
        "reported alongside accuracy rather than accuracy alone."
    )
