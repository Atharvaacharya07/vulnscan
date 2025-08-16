# vulnscan

A **two-stage hybrid framework** for cyber risk analysis.

- **Stage-1**: ML-based intrusion detection (IDS) on CIC/UNSW datasets.
- **Stage-2**: Vulnerability prioritisation and enrichment on Nessus-like findings.

---

## 📜 Overview

### 🎯 Goals
- Detect likely-attacked assets from network flow features (**Stage-1**).
- Prioritise and contextualise vulnerabilities on those assets (**Stage-2**).

### 🔑 Key Features
- **LightGBM** models with GPU acceleration (fallback to CPU).
- **SMOTE** for handling class imbalance.
- **Asset correlation**: Joins IDS alerts with vulnerability data via IPs.
- **Enrichment**: Maps services/protocols → MITRE ATT&CK TTP + remediation tips.
- **Risk scoring**: Asset criticality, exposure, KEV-like heuristics.
- **Outputs**: Alerts CSV, prioritised vulnerabilities, feature importances, risk bands, optional SHAP explainability.

---

## 📦 Dataset Links

> **Note:** Due to large file sizes, datasets must be stored externally (Google Drive, Dropbox, etc.) and linked here.

- CIC-IDS-2017 → `https://drive.google.com/drive/folders/1L0Fth5OQrjNqw11TTAoFZQsXaUbIuARX?usp=sharing`
- CSE-CIC-IDS-2018 → `https://drive.google.com/drive/folders/11Xr267ncPzJ7gjHXdGLtURYOaAzTZj3Y?usp=sharing`
- UNSW-NB15 → `https://drive.google.com/drive/folders/150zpa5fepFIs0FlPcFUXLH1sS-XXRtQh?usp=sharing`
- Synthetic Nessus dataset (`findings_realistic_v3.csv`) → `https://drive.google.com/drive/folders/1Y_CpC3W05azQjeuN3pdMxD6zkxiPIXbe?usp=sharing`

---

## ⚡ Colab Quickstart

1. Upload scripts to **Google Colab** or clone the repository.
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## 📚 Python Requirements

Install required packages:
```bash
pip install numpy pandas scikit-learn lightgbm imbalanced-learn matplotlib seaborn shap
```

---

## 🚀 Usage

### **Stage-0**: Generate Synthetic Dataset
**Script:** `generate_realistic_dataset_auto_v3.py`

**Example:**
```bash
python generate_realistic_dataset_auto_v3.py   --rows 6250000   --target_auc 0.90   --tolerance 0.01   --max_iters 20   --init_overlap 0.40   --init_noise 0.04   --feat_noise 0.06   --random_noise 0.05   --proportions 0.05 0.15 0.30 0.30 0.20   --probe_rows 250000   --probe_sample_per_class 8000   --output Dissertation_Project/simulated_nessus_data/findings_realistic_v3.csv   --plots
```

**Outputs:**
- `findings_realistic_v3.csv`
- `sanity_plots/*` (if `--plots` used)

---

### **Stage-1 & Stage-2**: Hybrid Run
**Script:** `trial_run_for_stage_1_and_2.py`

**Purpose:**
- **Stage-1:** Train IDS model on CIC/CSE/UNSW data, export high-probability alerts.
- **Stage-2:** Join Nessus-like data with Stage-1 alerts, train prioritisation model, enrich with TTP + remediation.

**Run:**
```bash
python trial_run_for_stage_1_and_2.py
```

**Outputs:**
- `alerts_stage1.csv`
- `vuln_prioritized_enriched.csv`
- `feature_importances.csv`
- `risk_bands_summary.csv`
- `shap_summary.png` (optional)

---

## 🔍 Script Details

### **`generate_realistic_dataset_auto_v3.py`**
- Generates large Nessus-like vulnerability dataset.
- Tunable parameters for class balance, noise, and AUC target.
- Outputs numeric & categorical features plus `remediation_priority`.

### **`trial_run_for_stage_1_and_2.py`**
- **Stage-1:** LightGBM binary classifier with SMOTE + ColumnTransformer preprocessing.
- **Stage-2:** Risk-based prioritisation with contextual features (`asset_criticality`, `internet_exposed`, etc.) and TTP mapping.

---

## 📊 Results Interpretation

- **Stage-1:** `alerts_stage1.csv` → probable attacks & asset IPs.
- **Stage-2:** `vuln_prioritized_enriched.csv` → vulnerabilities ranked by `risk_score` & `predicted_priority`.
- `priority_band` (`P1`..`P5`) + `sla_hint` → operational patch guidance.

---

## 🙏 Acknowledgements

- **Datasets:**
  - CIC-IDS-2017 / CSE-CIC-IDS-2018 — *Canadian Institute for Cybersecurity*
  - UNSW-NB15 — *University of New South Wales / IXIA*
- **Note:** Synthetic Nessus-like dataset is for research purposes only.

---
