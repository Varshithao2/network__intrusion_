"""
predict.py
-----------
Inference utilities used by the Streamlit dashboard. Loads the saved
joblib pipelines (which bundle imputation + scaling + classifier, so
inference-time preprocessing exactly matches training) and exposes simple
functions for single-flow and batch (CSV) prediction.
"""

import os
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import load_csv, prepare_dataset, get_feature_columns
from src.risk_assessment import assess_risk, build_alert

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def models_available():
    """Check whether trained models exist on disk."""
    return os.path.exists(os.path.join(MODELS_DIR, "binary_model.pkl"))


def load_artifacts():
    """Load all saved model artifacts. Raises FileNotFoundError with a
    clear message if training hasn't been run yet."""
    binary_path = os.path.join(MODELS_DIR, "binary_model.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

    if not os.path.exists(binary_path):
        raise FileNotFoundError(
            "No trained model found. Run training first: "
            "python -m src.train_model"
        )

    binary_model = joblib.load(binary_path)
    feature_columns = joblib.load(features_path)

    multiclass_model = None
    multiclass_labels = None
    multi_path = os.path.join(MODELS_DIR, "multiclass_model.pkl")
    if os.path.exists(multi_path):
        multiclass_model = joblib.load(multi_path)
        multiclass_labels = joblib.load(os.path.join(MODELS_DIR, "multiclass_labels.pkl"))

    return {
        "binary_model": binary_model,
        "multiclass_model": multiclass_model,
        "multiclass_labels": multiclass_labels,
        "feature_columns": feature_columns,
    }


def align_features(df, feature_columns):
    """Ensure the dataframe has exactly the columns the model expects,
    in the correct order -- filling any missing ones with 0 and dropping
    any extras. This makes the dashboard tolerant of CSVs that don't
    contain every original CIC-IDS2017 column.
    """
    df_aligned = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            df_aligned[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df_aligned[col] = 0.0
    df_aligned = df_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_aligned


def predict_single(feature_values: dict, artifacts: dict):
    """Predict a single network flow given a dict of {feature_name: value}.

    Returns a result dict with binary prediction, attack category (if a
    multiclass model is available), risk level, and confidence.
    """
    feature_columns = artifacts["feature_columns"]
    row = pd.DataFrame([feature_values])
    X = align_features(row, feature_columns)

    binary_model = artifacts["binary_model"]
    pred = binary_model.predict(X)[0]
    proba = binary_model.predict_proba(X)[0]
    classes = list(binary_model.classes_)
    confidence = float(proba[classes.index(pred)])

    result = {
        "prediction": pred,
        "confidence": confidence,
        "attack_category": None,
    }

    if pred == "Attack" and artifacts["multiclass_model"] is not None:
        multi_model = artifacts["multiclass_model"]
        cat_pred = multi_model.predict(X)[0]
        cat_proba = multi_model.predict_proba(X)[0]
        cat_classes = list(multi_model.classes_)
        cat_confidence = float(cat_proba[cat_classes.index(cat_pred)])
        result["attack_category"] = cat_pred
        result["confidence"] = cat_confidence
    elif pred == "Normal":
        result["attack_category"] = "Normal"

    result["risk_level"] = assess_risk(result["attack_category"] or ("Attack" if pred == "Attack" else "Normal"))
    return result


def predict_batch(df: pd.DataFrame, artifacts: dict):
    """Run predictions on an uploaded CSV's worth of flows.

    Returns the original dataframe with added columns: Prediction,
    Attack_Type, Risk_Level, Confidence.
    """
    feature_columns = artifacts["feature_columns"]
    X = align_features(df, feature_columns)

    binary_model = artifacts["binary_model"]
    preds = binary_model.predict(X)
    probas = binary_model.predict_proba(X)
    classes = list(binary_model.classes_)

    confidences = np.array([probas[i, classes.index(p)] for i, p in enumerate(preds)])

    attack_types = np.array(["Normal"] * len(df), dtype=object)
    if artifacts["multiclass_model"] is not None:
        attack_mask = preds == "Attack"
        if attack_mask.any():
            multi_model = artifacts["multiclass_model"]
            X_attack = X[attack_mask]
            cat_preds = multi_model.predict(X_attack)
            cat_probas = multi_model.predict_proba(X_attack)
            cat_classes = list(multi_model.classes_)
            cat_conf = np.array([
                cat_probas[i, cat_classes.index(p)] for i, p in enumerate(cat_preds)
            ])
            attack_types[attack_mask] = cat_preds
            confidences[attack_mask] = cat_conf
    else:
        attack_types[preds == "Attack"] = "Attack (category unknown)"

    risk_levels = [assess_risk(cat) for cat in attack_types]

    result_df = df.copy()
    result_df["Prediction"] = preds
    result_df["Attack_Type"] = attack_types
    result_df["Risk_Level"] = risk_levels
    result_df["Confidence"] = np.round(confidences * 100, 2)

    return result_df
