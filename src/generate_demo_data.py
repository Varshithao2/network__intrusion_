"""
generate_demo_data.py
-----------------------
Generates a small, clearly-labeled SYNTHETIC/DEMO dataset that mimics the
structure of CIC-IDS2017 (same column names, same label style) so the
project can be installed, trained, and demoed end-to-end before the real
dataset is downloaded.

THIS IS NOT REAL NETWORK TRAFFIC. It is randomly generated data shaped to
resemble CIC-IDS2017 feature ranges, purely so the pipeline has something
to run against. Accuracy numbers produced from this demo data are NOT
meaningful and must not be reported as real results -- swap in the actual
CIC-IDS2017 CSVs (see data/README.md) before drawing any conclusions.
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Total", "Bwd IAT Total",
    "SYN Flag Count", "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
]

CATEGORIES = ["Normal", "DoS", "Port Scan", "Brute Force", "Botnet", "Web Attack"]
# Rough class weights to mimic CIC-IDS2017's heavy imbalance (mostly Normal).
CATEGORY_WEIGHTS = [0.78, 0.09, 0.06, 0.03, 0.02, 0.02]


def _simulate_row(category):
    """Generate one synthetic flow with feature ranges loosely shaped by
    the attack category, so the demo classifier has *some* learnable signal
    (this is a stand-in for real traffic, not a realistic attack simulator).
    """
    row = {}
    if category == "Normal":
        row["Destination Port"] = RNG.choice([80, 443, 22, 53, 8080])
        row["Flow Duration"] = RNG.normal(500000, 150000)
        row["Total Fwd Packets"] = RNG.poisson(15)
        row["Total Backward Packets"] = RNG.poisson(14)
        row["SYN Flag Count"] = RNG.integers(0, 2)
        row["RST Flag Count"] = 0
    elif category == "DoS":
        row["Destination Port"] = RNG.choice([80, 443])
        row["Flow Duration"] = RNG.normal(2000, 800)
        row["Total Fwd Packets"] = RNG.poisson(500)
        row["Total Backward Packets"] = RNG.poisson(2)
        row["SYN Flag Count"] = RNG.integers(50, 200)
        row["RST Flag Count"] = RNG.integers(0, 5)
    elif category == "Port Scan":
        row["Destination Port"] = RNG.integers(1, 65535)
        row["Flow Duration"] = RNG.normal(100, 50)
        row["Total Fwd Packets"] = RNG.poisson(2)
        row["Total Backward Packets"] = RNG.poisson(0.2)
        row["SYN Flag Count"] = 1
        row["RST Flag Count"] = RNG.integers(0, 2)
    elif category == "Brute Force":
        row["Destination Port"] = RNG.choice([22, 21, 3389])
        row["Flow Duration"] = RNG.normal(3000, 1000)
        row["Total Fwd Packets"] = RNG.poisson(40)
        row["Total Backward Packets"] = RNG.poisson(38)
        row["SYN Flag Count"] = RNG.integers(1, 5)
        row["RST Flag Count"] = RNG.integers(0, 3)
    elif category == "Botnet":
        row["Destination Port"] = RNG.choice([443, 6667, 8080])
        row["Flow Duration"] = RNG.normal(800000, 300000)
        row["Total Fwd Packets"] = RNG.poisson(8)
        row["Total Backward Packets"] = RNG.poisson(8)
        row["SYN Flag Count"] = RNG.integers(0, 3)
        row["RST Flag Count"] = RNG.integers(0, 2)
    else:  # Web Attack
        row["Destination Port"] = 80
        row["Flow Duration"] = RNG.normal(50000, 20000)
        row["Total Fwd Packets"] = RNG.poisson(25)
        row["Total Backward Packets"] = RNG.poisson(20)
        row["SYN Flag Count"] = RNG.integers(0, 2)
        row["RST Flag Count"] = 0

    row["Flow Duration"] = max(1, row["Flow Duration"])
    row["Total Fwd Packets"] = max(0, row["Total Fwd Packets"])
    row["Total Backward Packets"] = max(0, row["Total Backward Packets"])
    row["Total Length of Fwd Packets"] = row["Total Fwd Packets"] * RNG.uniform(40, 1500)
    row["Total Length of Bwd Packets"] = row["Total Backward Packets"] * RNG.uniform(40, 1500)
    row["Fwd Packet Length Max"] = RNG.uniform(40, 1500)
    row["Fwd Packet Length Min"] = RNG.uniform(0, 40)
    row["Fwd Packet Length Mean"] = (row["Fwd Packet Length Max"] + row["Fwd Packet Length Min"]) / 2
    row["Bwd Packet Length Max"] = RNG.uniform(40, 1500)
    row["Bwd Packet Length Min"] = RNG.uniform(0, 40)
    row["Bwd Packet Length Mean"] = (row["Bwd Packet Length Max"] + row["Bwd Packet Length Min"]) / 2
    duration_s = max(row["Flow Duration"] / 1e6, 1e-6)
    total_bytes = row["Total Length of Fwd Packets"] + row["Total Length of Bwd Packets"]
    total_packets = row["Total Fwd Packets"] + row["Total Backward Packets"]
    row["Flow Bytes/s"] = total_bytes / duration_s
    row["Flow Packets/s"] = total_packets / duration_s
    row["Flow IAT Mean"] = RNG.uniform(10, 100000)
    row["Flow IAT Std"] = RNG.uniform(0, 50000)
    row["Fwd IAT Total"] = RNG.uniform(0, row["Flow Duration"])
    row["Bwd IAT Total"] = RNG.uniform(0, row["Flow Duration"])
    row["ACK Flag Count"] = RNG.integers(0, 10)
    row["PSH Flag Count"] = RNG.integers(0, 5)

    row["Label"] = category
    return row


def generate_demo_dataset(n_rows=6000, out_path=None):
    """Generate a synthetic demo dataset shaped like CIC-IDS2017.

    Args:
        n_rows: total number of synthetic flows to generate.
        out_path: if provided, writes the CSV there.

    Returns:
        pandas.DataFrame
    """
    categories = RNG.choice(CATEGORIES, size=n_rows, p=CATEGORY_WEIGHTS)
    rows = [_simulate_row(cat) for cat in categories]
    df = pd.DataFrame(rows)
    # Reorder columns for readability
    df = df[FEATURE_COLUMNS + ["Label"]]

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Demo dataset written to {out_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(base_dir, "data", "demo_traffic.csv")
    generate_demo_dataset(n_rows=6000, out_path=default_path)
