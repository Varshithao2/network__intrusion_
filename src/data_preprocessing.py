"""
data_preprocessing.py
----------------------
Handles loading and cleaning of CIC-IDS2017-style network traffic CSV files.

Designed to be defensive: CIC-IDS2017 CSVs are notoriously messy (extra
whitespace in column names, mixed-case labels, inf/NaN values, duplicate
rows, non-numeric junk). This module normalizes all of that so training
and inference always see the same shape of data.
"""

import os
import re
import numpy as np
import pandas as pd

# Columns that are identifiers / metadata rather than predictive features.
# We keep them around for the dashboard (to show Source IP, Port, etc.)
# but drop them before feeding the model.
NON_FEATURE_COLUMNS = [
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "src ip addr", "dst ip addr", "flow id ",
]

# The label column has appeared under several names across CIC-IDS2017
# CSV releases.
LABEL_COLUMN_CANDIDATES = ["label", "class", "attack", "attack_type"]


def clean_column_names(columns):
    """Strip whitespace, lower-case, and normalize separators in column names.

    CIC-IDS2017 CSVs frequently have leading/trailing spaces in headers
    (e.g. ' Destination Port'), which silently break naive column lookups.
    """
    cleaned = []
    for col in columns:
        c = str(col).strip()
        c = re.sub(r"\s+", " ", c)
        cleaned.append(c)
    return cleaned


def find_label_column(df):
    """Locate the label column regardless of exact naming/casing."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    # Fall back: any column literally named "Label"
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    return None


def normalize_labels(series):
    """Normalize inconsistent attack label strings.

    CIC-IDS2017 label text varies release to release, e.g.
    'BENIGN' / 'Benign' / 'benign', 'DoS Hulk' / 'DoS_Hulk', etc.
    This function trims whitespace and standardizes casing/underscores
    while leaving the semantic attack name intact.
    """
    def _norm(v):
        if pd.isna(v):
            return v
        s = str(v).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("�", "-")  # some CICIDS releases mangle en-dashes
        return s
    return series.apply(_norm)


def map_to_attack_category(label):
    """Group fine-grained CIC-IDS2017 labels into broader attack categories.

    Returns one of: Normal, DoS, Port Scan, Brute Force, Botnet, Web Attack,
    Infiltration, Other. This mapping only fires for labels it recognizes;
    anything unrecognized (but not benign) falls into 'Other' rather than
    being forced into a wrong bucket.
    """
    if pd.isna(label):
        return "Unknown"
    l = str(label).strip().lower()

    if l in ("benign", "normal", "0"):
        return "Normal"
    if "ddos" in l or "dos" in l:
        return "DoS"
    if "portscan" in l or "port scan" in l:
        return "Port Scan"
    if "ftp-patator" in l or "ssh-patator" in l or "brute" in l or "bruteforce" in l:
        return "Brute Force"
    if "bot" in l:
        return "Botnet"
    if "web attack" in l or "sql injection" in l or "xss" in l:
        return "Web Attack"
    if "infiltration" in l:
        return "Infiltration"
    if "heartbleed" in l:
        return "Other"
    return "Other"


def load_csv(path_or_buffer):
    """Load a CIC-IDS2017 CSV (or list of CSVs) with robust encoding handling."""
    try:
        df = pd.read_csv(path_or_buffer, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path_or_buffer, low_memory=False, encoding="latin1")
    df.columns = clean_column_names(df.columns)
    return df


def load_multiple_csvs(directory):
    """Load and concatenate every CSV file found in a directory."""
    frames = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".csv"):
            fpath = os.path.join(directory, fname)
            try:
                frames.append(load_csv(fpath))
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}")
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return pd.concat(frames, axis=0, ignore_index=True)


def basic_clean(df, label_col=None):
    """Core cleaning routine shared by training and inference.

    Steps:
      1. Drop exact duplicate rows.
      2. Replace inf/-inf with NaN, then drop rows that are all-NaN in
         numeric feature columns (a common CIC-IDS2017 artifact from
         division-by-zero flow-rate calculations).
      3. Impute remaining NaNs with the column median (numeric only).
      4. Drop columns that are entirely empty or constant (zero variance),
         since they carry no predictive signal and can break scalers.
    """
    df = df.copy()

    # 1. Duplicates
    df.drop_duplicates(inplace=True)

    # Identify numeric feature columns (exclude label + identifier columns)
    exclude = set(NON_FEATURE_COLUMNS)
    if label_col:
        exclude.add(label_col.lower())

    numeric_cols = []
    for c in df.columns:
        if c.lower() in exclude:
            continue
        # Try to coerce to numeric; non-numeric feature columns are rare
        # in CIC-IDS2017 aside from the label/identifier fields.
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() > 0:
            df[c] = coerced
            numeric_cols.append(c)

    # 2. Replace inf/-inf with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows where every numeric feature is NaN (unrecoverable rows)
    if numeric_cols:
        df = df[df[numeric_cols].notna().any(axis=1)]

    # 3. Impute remaining NaNs with median
    for c in numeric_cols:
        if df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val if not pd.isna(median_val) else 0)

    # 4. Drop zero-variance numeric columns
    zero_var_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        df.drop(columns=zero_var_cols, inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in zero_var_cols]

    return df, numeric_cols


def get_feature_columns(df, label_col=None, id_cols=None):
    """Return the list of columns usable as model features.

    Dynamically determined from whatever is actually present in the
    dataframe -- nothing is hard-coded, so the pipeline degrades
    gracefully on CIC-IDS2017 CSVs that only contain a subset of the
    ~78 standard columns.
    """
    id_cols = set([c.lower() for c in (id_cols or [])]) | set(NON_FEATURE_COLUMNS)
    if label_col:
        id_cols.add(label_col.lower())

    features = []
    for c in df.columns:
        if c.lower() in id_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
    return features


def prepare_dataset(df):
    """Full preparation: locate label, clean, normalize labels, derive category.

    Returns:
        df_clean: cleaned dataframe (still contains label + id columns)
        feature_cols: list of numeric feature column names
        label_col: name of the raw label column (or None if not found)
    """
    label_col = find_label_column(df)

    if label_col is not None:
        df[label_col] = normalize_labels(df[label_col])

    df_clean, numeric_cols = basic_clean(df, label_col=label_col)

    if label_col is not None and label_col in df_clean.columns:
        df_clean["attack_category"] = df_clean[label_col].apply(map_to_attack_category)
        df_clean["binary_label"] = df_clean["attack_category"].apply(
            lambda x: "Normal" if x == "Normal" else "Attack"
        )

    feature_cols = get_feature_columns(df_clean, label_col=label_col)
    return df_clean, feature_cols, label_col
