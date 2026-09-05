# ML-Based Network Intrusion Detection and Security Monitoring System

An educational, ML-powered Network Intrusion Detection System (IDS) that
analyzes network traffic flow data (CIC-IDS2017 format), classifies each
flow as **Normal** or an **attack category**, assigns a simple **risk
level**, and presents everything through a professional Streamlit security
dashboard.

Built as a portfolio project connecting hands-on AI/ML skills with concepts
from an **AICTE–EduSkills Fortinet Security Associate internship**
(network security, firewalls, IDS/IPS, traffic monitoring, threat detection).

> ⚠️ **This is an educational/prototype system, not a production security
> product.** It does not perform real-time packet capture, does not
> integrate with live network hardware, and never blocks or modifies any
> traffic. It only analyzes traffic data provided to it (CSV files) and
> displays predictions with recommended actions.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Installation & Environment Setup](#installation--environment-setup)
- [Dataset Setup](#dataset-setup)
- [Training Instructions](#training-instructions)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Project Structure](#project-structure)
- [ML Methodology](#ml-methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Screenshots](#screenshots)
- [Connection to Fortinet Security Associate Internship](#connection-to-fortinet-security-associate-internship)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [How to Explain This Project in an Interview](#how-to-explain-this-project-in-an-interview)
- [Resume Description](#resume-description)

---

## Problem Statement

Traditional network security relies heavily on signature-based detection
(firewalls, IPS rules) that catch known attack patterns but struggle with
new or subtly-varied attacks. Security teams also generate huge volumes of
traffic logs that are impractical to review manually. This project explores
whether a supervised machine learning model, trained on labeled network
flow statistics, can learn to distinguish normal traffic from malicious
traffic (and categorize the attack type) directly from flow-level features
— complementing traditional rule-based detection.

## Objectives

1. Build a working ML pipeline that classifies network flows as Normal or Attack.
2. Where the data supports it, further classify attacks into categories
   (DoS, Port Scan, Brute Force, Botnet, Web Attack).
3. Attach a simple, explainable risk level to each prediction.
4. Present results through an interactive, professional dashboard suitable
   for demoing in an interview.
5. Keep the system honest about what it is: an educational prototype
   inspired by, not equivalent to, enterprise tools like FortiGate/FortiAnalyzer.

## Architecture

```
CIC-IDS2017 CSV
      ↓
Data Loading            (src/data_preprocessing.py)
      ↓
Data Cleaning           (whitespace, NaN/inf, duplicates, dynamic feature detection)
      ↓
Feature Engineering/Selection
      ↓
Train/Test Split        (stratified, split BEFORE any fitting -> no leakage)
      ↓
ML Model                (Random Forest inside an sklearn Pipeline: impute -> scale -> classify)
      ↓
Prediction              (src/predict.py)
      ↓
Attack Classification   (binary model -> multiclass model for attack subtype)
      ↓
Risk Assessment         (src/risk_assessment.py — rule-based severity mapping)
      ↓
Streamlit Security Dashboard (app.py)
```

## Features

- **Binary classification**: Normal vs Attack
- **Multiclass classification**: attack category (DoS, Port Scan, Brute
  Force, Botnet, Web Attack, ...) — trained automatically only if the
  dataset has enough samples per category
- **Risk scoring layer**: LOW / MEDIUM / HIGH based on predicted category
- **3-section Streamlit dashboard**: CSV Analysis (upload & batch
  predictions), Traffic Analysis (charts/statistics), and Threat
  Detection (flagged-flow table with risk levels)
- **CSV upload & download**: analyze any compatible CSV and export results
- **Dynamic feature detection**: works even if a CSV only has a subset of
  the standard CIC-IDS2017 columns
- **Demo/synthetic data generator** so the whole thing runs before you
  download the real dataset

## Technology Stack

| Component            | Technology              |
|-----------------------|--------------------------|
| Language              | Python 3.10+             |
| Data handling         | Pandas, NumPy            |
| Machine Learning      | Scikit-learn (Random Forest, Logistic Regression, Decision Tree) |
| Visualization         | Plotly, Matplotlib       |
| Dashboard             | Streamlit                |
| Model persistence     | Joblib                   |
| Dataset               | CIC-IDS2017 (or generated demo data) |

## Dataset

**Primary dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)**
by the Canadian Institute for Cybersecurity. It contains ~78-85 flow-level
statistical features (duration, packet/byte counts, inter-arrival times,
TCP flag counts, etc.) per network flow, labeled as `BENIGN` or one of
several attack types (DoS variants, PortScan, FTP/SSH-Patator, Web Attacks,
Botnet, Infiltration, Heartbleed).

Because the dataset is several gigabytes and requires registration, it is
**not included in this repository**. See [Dataset Setup](#dataset-setup)
below, and `data/README.md`, for exact download and setup instructions.
A synthetic demo dataset generator is included so the project can be run
immediately without downloading anything.

## Installation & Environment Setup

```bash
# 1. Clone or copy the project, then move into it
cd network_intrusion_detection

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

**Fastest path (no download) — synthetic demo data:**

```bash
python -m src.generate_demo_data
```

This creates `data/demo_traffic.csv`, a small dataset shaped like
CIC-IDS2017 so you can run the entire pipeline immediately. It is clearly
synthetic — see the warning printed by the script and in `data/README.md`.

**Real dataset:**

1. Download from `https://www.unb.ca/cic/datasets/ids-2017.html`
   (`MachineLearningCSV.zip`).
2. Extract the CSVs into `data/` (or a subfolder like `data/cicids2017/`).
3. Train against them (see below).

Full details, expected CSV format, and directory layout are in
[`data/README.md`](data/README.md).

## Training Instructions

```bash
# Train on the demo dataset (auto-generates it if missing)
python -m src.train_model

# Train on a specific real CIC-IDS2017 CSV
python -m src.train_model --data data/Monday-WorkingHours.pcap_ISCX.csv

# Train on a whole folder of CIC-IDS2017 CSVs
python -m src.train_model --data data/cicids2017/

# Also compare against Logistic Regression and Decision Tree
python -m src.train_model --compare
```

Training will:
- Load and clean the CSV(s)
- Dynamically detect usable numeric feature columns
- Split into train/test **before** fitting anything (no leakage)
- Train a Random Forest binary model (`Normal` vs `Attack`) inside a
  Pipeline (imputer → scaler → classifier)
- Train a Random Forest multiclass model for attack category, **only if**
  at least 2 categories have ≥10 samples each
- Save models to `models/*.pkl` (Joblib) and metrics/plots data to `outputs/`

## Running the Streamlit Application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
The app will tell you if no trained model is found yet and give you the
exact command to run.

## Project Structure

```
network_intrusion_detection/
│
├── app.py                     # Streamlit dashboard (CSV Analysis, Traffic Analysis, Threat Detection)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── README.md              # dataset download & setup instructions
│   └── demo_traffic.csv       # generated locally, gitignored
│
├── models/                    # trained .pkl artifacts (gitignored, generated)
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # loading, cleaning, label normalization
│   ├── generate_demo_data.py  # synthetic CIC-IDS2017-shaped dataset
│   ├── train_model.py         # training + evaluation + saving pipeline
│   ├── predict.py             # inference used by the dashboard
│   ├── evaluation.py          # metrics, confusion matrix, feature importance
│   └── risk_assessment.py     # rule-based LOW/MEDIUM/HIGH risk mapping
│
├── notebooks/
│   └── optional_exploration.ipynb   # optional EDA, not required to run the app
│
└── outputs/                   # metrics/plots data generated by training
    └── .gitkeep
```

## ML Methodology

- **Model**: Random Forest is the primary classifier — chosen because it
  handles tabular, mixed-scale numeric data well, is robust to outliers
  compared to linear models, requires little tuning, and (unlike a neural
  net) produces feature importances that are easy to explain to a
  non-ML audience or in an interview.
- **Pipeline**: `SimpleImputer(median) → StandardScaler → RandomForestClassifier`,
  wrapped in a single `sklearn.pipeline.Pipeline` and saved as one Joblib
  file. This guarantees inference-time preprocessing is *identical* to
  training-time preprocessing — a common source of bugs when preprocessing
  and modeling are handled separately.
- **No data leakage**: the train/test split happens first (`train_test_split`
  with stratification), and the imputer/scaler are fit only on the
  training fold (via the Pipeline's `.fit()`), never on the full dataset.
- **Class imbalance**: handled with `class_weight="balanced"` on all
  models rather than SMOTE, since CIC-IDS2017's imbalance (mostly benign
  traffic) is well-handled by class weighting without the added complexity
  and synthetic-sample risk of oversampling.
- **Dynamic feature detection**: `get_feature_columns()` inspects whatever
  numeric columns are actually present in a given CSV rather than
  hard-coding a fixed feature list — so the pipeline still works on a
  partial or differently-formatted CIC-IDS2017 export.
- **Attack category grouping**: `map_to_attack_category()` groups raw
  CIC-IDS2017 labels (e.g. `DoS Hulk`, `DoS GoldenEye`) into broader,
  interview-friendly categories (`DoS`, `Port Scan`, `Brute Force`,
  `Botnet`, `Web Attack`, `Infiltration`, `Other`) without forcing a
  category grouping the label data doesn't support — anything
  unrecognized falls into `Other`.
- **Model comparison**: Logistic Regression and a single Decision Tree are
  available via `--compare` as lightweight baselines, not extensively tuned.

## Evaluation Metrics

The dashboard's **Model Performance** page reports Accuracy, Precision,
Recall, F1-score, a confusion matrix, the full classification report,
training/prediction time, and Random Forest feature importances.

**Why not just accuracy?** Network traffic is heavily imbalanced — the
vast majority of flows are normal. A model that predicts "Normal" for
everything can score 95%+ accuracy while catching zero attacks, which is
useless for security. **Recall** measures how many real attacks were
caught (a missed attack = a false negative = an undetected breach).
**Precision** measures how many raised alerts were real attacks (too many
false positives = alert fatigue for a SOC analyst). **F1-score** balances
both, which is why all three are reported together rather than relying on
accuracy alone.

### Actual results from this repository's own training run

These numbers come from an actual `python -m src.train_model --compare`
run included with this project, using the **synthetic demo dataset**
(`data/demo_traffic.csv`, 6,000 rows) — they demonstrate the pipeline
works end-to-end and are reproducible by running the same command, but
**they are not real-world intrusion-detection performance** since the
demo data is randomly generated, not real traffic. Re-run training on the
real CIC-IDS2017 CSVs (see Dataset Setup) to get results that reflect
genuine network traffic patterns.

| Model                        | Accuracy | Precision | Recall | F1-score | Train time |
|-------------------------------|----------|-----------|--------|----------|------------|
| Random Forest (binary)        | 99.60%   | 99.60%    | 99.60% | 99.60%   | 1.67s      |
| Decision Tree (binary)        | 99.00%   | 99.00%    | 99.00% | 98.99%   | 0.08s      |
| Logistic Regression (binary)  | 97.93%   | 97.92%    | 97.93% | 97.93%   | 0.03s      |
| Random Forest (multiclass)    | 99.47%   | 99.47%    | 99.47% | 99.43%   | 1.55s      |

Test set size: 1,500 flows (25% holdout, stratified). Prediction time for
the binary Random Forest on the full test set: ~0.025s.

Top features by Random Forest importance (on the demo data): `Flow
Duration`, `Flow Packets/s`, `Flow Bytes/s`, `Fwd IAT Total`, `Total Fwd
Packets` — consistent with the kind of flow-rate and timing features that
matter in real CIC-IDS2017 analyses too, since DoS/scan traffic tends to
show abnormal packet timing and volume.

## Screenshots

_(Add screenshots here after running the app locally — e.g. `streamlit run app.py`,
then capture the Home, Traffic Analysis, Threat Detection, and Model
Performance pages.)_

- `docs/screenshot_home.png`
- `docs/screenshot_traffic_analysis.png`
- `docs/screenshot_threat_detection.png`
- `docs/screenshot_model_performance.png`

## Connection to Fortinet Security Associate Internship

This project was built to directly apply concepts from the **AICTE–EduSkills
Fortinet Security Associate internship**:

- **Network security fundamentals**: the project operates on TCP/IP flow
  data — source/destination ports, protocols, packet/byte counts — the
  same kind of metadata a firewall or IPS inspects.
- **Firewall / IPS monitoring concepts**: just as a FortiGate firewall
  inspects traffic against rules/signatures, this system inspects flow
  features against a learned ML model to flag anomalous traffic.
- **Intrusion detection/prevention**: the binary + multiclass models
  mirror IDS/IPS goals — detecting DoS, port scanning, brute force, and
  web attacks — though this project **detects and reports only, it does
  not prevent/block** (no IPS-style enforcement is implemented).
- **Network traffic inspection**: the preprocessing pipeline works
  directly with flow-level statistics of the kind produced by NetFlow/IPFIX
  or Fortinet's traffic logging, rather than raw application content.
- **Threat detection & security alerts**: the Security Alerts page
  produces structured alerts (attack type, risk level, confidence,
  recommended action) analogous to how FortiAnalyzer surfaces prioritized
  security events from FortiGate logs.
- **Security logs**: the CSV Analysis feature treats uploaded traffic data
  as a stand-in for exported security/traffic logs, which is how this
  kind of batch analysis would work against real firewall log exports.
- **Risk assessment**: the LOW/MEDIUM/HIGH severity model reflects how
  Fortinet products (and SIEM tools generally) prioritize which alerts an
  analyst should look at first.

**This project is an educational, ML-based IDS inspired by enterprise
network-security and Fortinet security concepts. It is an independent
academic/portfolio project and is not a Fortinet product, is not built on
Fortinet software, and is not affiliated with or endorsed by Fortinet.**

## Limitations

- Trained/evaluated on CIC-IDS2017 (or synthetic demo data resembling
  it) — a lab-generated dataset from 2017; real-world enterprise traffic,
  encrypted traffic patterns, and newer attack techniques may differ
  significantly, so performance would need re-validation on live traffic.
- No real-time packet capture — the system analyzes traffic that has
  already been converted to flow-level CSV data (e.g. via CICFlowMeter),
  it does not sniff live packets.
- Detection only, no prevention — it never blocks, drops, or modifies
  traffic; "recommended actions" are informational only.
- Multiclass attack categorization is only as reliable as the label
  quality and sample size per category in the training data; rare attack
  types with very few samples may not train a usable multiclass model at all.
- Random Forest, while accurate on this type of tabular data, can be
  fooled by adversarial or previously-unseen attack patterns not
  represented in training data (a general limitation of supervised ML-based
  detection vs. purely rule/signature-based systems).
- No authentication, encryption, or hardening has been added to the
  Streamlit app itself — it is meant for local/demo use, not exposure on
  a public network.

## Future Enhancements

- Real-time packet capture (e.g. via `scapy` or `pyshark`) feeding flows
  into the existing model pipeline.
- Integration with Wireshark/Zeek exports for richer, live flow generation.
- SIEM integration (e.g. exporting alerts in a Splunk/Elastic-friendly format).
- Real-time alerting via email/SMS/Slack webhook when a HIGH risk flow is detected.
- Scheduled/continuous model retraining as new labeled traffic becomes available.
- More advanced anomaly detection (e.g. Isolation Forest or autoencoders)
  to catch attack types not present in the training labels at all.
- Tighter integration with firewall/IPS infrastructure so ML-flagged
  events could feed into (not replace) rule-based enforcement.
- Exploring secure SD-WAN / network telemetry sources as additional
  real-world data feeds.

## How to Explain This Project in an Interview

### 1-minute explanation

"I built an ML-based network intrusion detection system trained on the
CIC-IDS2017 dataset, which contains labeled network flow statistics for
both normal and malicious traffic. I used a Random Forest classifier
inside a scikit-learn pipeline to first flag traffic as Normal or Attack,
and then, where the data supports it, classify the specific attack type —
like DoS, port scanning, or brute force. Each prediction gets a simple
risk level, and I built a Streamlit dashboard so you can upload traffic
data, see attack breakdowns, review model performance metrics like
precision and recall, and see security-style alerts with recommended
actions. I built this to apply what I learned in a Fortinet Security
Associate internship — network security, IDS/IPS concepts, and traffic
monitoring — combined with my machine learning skills."

### Q&A Prep

**1. Why did you choose this project?**
It let me combine my Fortinet Security Associate internship knowledge
(network security, firewalls, IDS/IPS) with practical machine learning —
a project that's both technically substantial and directly relevant to
security-focused roles.

**2. What is an IDS?**
An Intrusion Detection System monitors network or system traffic for
signs of malicious activity or policy violations and alerts on them,
without necessarily taking action to stop the traffic.

**3. Difference between IDS and IPS?**
An IDS detects and alerts; an IPS (Intrusion Prevention System) sits
inline and can actively block or drop malicious traffic. My project is
IDS-style — detection and alerting only, no traffic blocking.

**4. Why did you choose Random Forest?**
It handles tabular, mixed-scale numeric features well, is robust to
outliers, needs relatively little hyperparameter tuning, resists
overfitting better than a single decision tree via ensembling, and
produces feature importances — making it both accurate and easy to
explain, which matters for a security tool where analysts need to trust
and understand its decisions.

**5. Why CIC-IDS2017?**
It's a widely used, realistically-labeled benchmark dataset for network
intrusion detection research, generated from real attack simulations
against a testbed network, with a broad variety of attack categories
(DoS, port scan, brute force, botnet, web attacks) and normal background traffic.

**6. What features are used?**
Flow-level statistics: flow duration, forward/backward packet counts,
packet length statistics, flow byte/packet rates, inter-arrival times,
destination port, and TCP flag counts — the pipeline dynamically detects
which of these are present rather than hard-coding a fixed feature list.

**7. How did you handle class imbalance?**
Used `class_weight="balanced"` in the classifiers rather than SMOTE,
since it's simpler, avoids generating synthetic feature vectors that
might not represent realistic traffic, and was sufficient for this
dataset's imbalance level.

**8. How did you prevent data leakage?**
I split into train/test sets before any preprocessing was fit, and
wrapped imputation + scaling + the classifier in a single scikit-learn
Pipeline so the scaler/imputer statistics are learned only from the
training fold, never from the test fold or the full dataset.

**9. Why is recall important in intrusion detection?**
Because a missed attack (false negative) means a real intrusion goes
undetected — in security, that's far more costly than a false positive,
which just means an analyst double-checks something benign. Accuracy
alone can look great while recall on the (rare) attack class is poor.

**10. How does this relate to Fortinet?**
The project mirrors concepts from my Fortinet Security Associate
internship — traffic inspection, IDS/IPS logic, security logging, alerts,
and risk-based prioritization — implemented with an ML approach rather
than Fortinet's proprietary signature/rule engine. It's inspired by, not
built on, Fortinet technology.

**11. What are the limitations?**
It's trained on a 2017 lab dataset (or synthetic demo data), doesn't
capture live packets, only detects rather than prevents, and like any
supervised model, may not generalize to attack patterns not present in
training data.

**12. How would you make it real-time?**
Add a live packet/flow capture layer (e.g. `scapy`, `pyshark`, or a tool
like CICFlowMeter running continuously), feed generated flow records into
the existing preprocessing + prediction pipeline in near real time instead
of via CSV upload, and stream results to the dashboard.

**13. How would you deploy it in an enterprise?**
Run it alongside existing IDS/IPS/firewall logging (not as a replacement),
feed it exported flow logs continuously, route HIGH risk alerts into a
SIEM or ticketing system for analyst review, and keep it in an
advise-only role until it's been validated against real production
traffic over time.

**14. What is the difference between a firewall and IDS?**
A firewall enforces access-control rules (allow/deny traffic based on
ports, IPs, protocols) proactively at the network boundary. An IDS
passively monitors traffic that's already allowed through and flags
suspicious patterns/anomalies — they're complementary layers, not substitutes.

**15. How can ML improve traditional rule-based detection?**
Rule/signature-based detection only catches known attack patterns; ML
models can generalize from training examples to catch variations of known
attack types and potentially flag anomalous behavior that doesn't match
any specific signature — though it can also produce false positives/negatives
differently than rules do, so the two approaches work best combined.

## Resume Description

- Built an ML-based Network Intrusion Detection System using the
  CIC-IDS2017 dataset, applying a Random Forest classifier inside a
  scikit-learn pipeline to detect malicious network traffic and classify
  attack types including DoS, Port Scanning, Brute Force, and Botnet activity.
- Designed a data-cleaning and feature pipeline that dynamically handles
  inconsistent CIC-IDS2017 CSV formats (missing values, infinite values,
  duplicate records, and label inconsistencies) while preventing train/test
  data leakage.
- Developed an interactive Streamlit security dashboard presenting model
  performance metrics (Accuracy, Precision, Recall, F1-score), traffic
  analysis, and risk-prioritized security alerts, connecting hands-on ML
  engineering with concepts from a Fortinet Security Associate internship.
