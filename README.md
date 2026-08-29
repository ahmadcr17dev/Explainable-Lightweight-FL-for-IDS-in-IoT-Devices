# Lightweight Explainable Federated Learning for Intrusion Detection in Resource-Constrained IoT Devices: An Empirical Evaluation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-CICIoT2023-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**Paper:** *Lightweight Explainable Federated Learning for Intrusion Detection in Resource-Constrained IoT Devices: An Empirical Evaluation*

**Repository:** [Explainable-IDS-in-IOMT-using-FL](https://github.com/ahmadcr17dev/Explainable-IDS-in-IOMT-using-FL)

---

## Overview

Resource-constrained Internet of Things (IoT) and Internet of Medical Things (IoMT) devices expand the attack surface of networked systems while producing highly sensitive traffic. Centralised intrusion detection requires shipping raw flows to a single server, which raises privacy, bandwidth, and regulatory concerns.

This repository contains the code for the empirical evaluation of a **privacy-preserving lightweight Federated Learning Intrusion Detection System (FL-IDS)** for resource-constrained IoT/IoMT environments, evaluated on the **CICIoT2023** benchmark.

The framework combines:

- Lightweight MLP (logits output, suitable for edge deployment)
- Federated Averaging (FedAvg) with optional Top-K communication reporting
- Centralised warm-start + post-FL fine-tuning for strong accuracy
- Magnitude-based model pruning
- SHAP explainability for transparent decisions
- Publication-ready figure and table generation

---

## Key Results (Proposed Method)

On **protocol-level collapsed CICIoT2023** (Flood-* class scheme):

| Metric | Value |
|--------|------:|
| **Test Accuracy** | **91.39%** |
| **Weighted F1-score** | **91.63%** |

Primary result files after training:

- `results/meta.json` → `full_accuracy`, `full_f1`
- `results/classification_report.json`
- `results/best_global_model.keras`

---

## Main Contributions

- Lightweight neural network designed for resource-constrained IoMT gateways
- Federated learning so clients collaborate **without sharing raw traffic**
- Protocol-level label collapse (`Flood-*`) to resolve DoS/DDoS flow ambiguity common in CICIoT2023
- Communication-efficient design with Top-K sparsification reported as a paper metric
- Post-training magnitude pruning for compact deployment
- SHAP-based global and class-wise explainability
- End-to-end reproducible pipeline (preprocess → train → baselines → SHAP → figures)

---

## Method Summary

### 1. Preprocessing (`preprocess.py`)

- Stream CICIoT2023 CSV shards
- Normalize labels (case-insensitive) and collapse into protocol-level classes
- Mutual-information ranking; **all numeric features** retained (`TOP_FEATS = None`)
- Stratified train/test split (80/20)
- StandardScaler normalisation
- Log-scaled class-weight metadata (optional; training uses plain CE by default)

### 2. Label Scheme (Flood-* Collapse)

DoS and DDoS share near-identical flow statistics in CICIoT2023, which previously capped accuracy near ~70%. Labels are merged into **protocol-level flood categories**:

| Collapsed class | Source examples |
|-----------------|-----------------|
| Flood-ICMP | DDoS/DoS ICMP floods & fragmentation |
| Flood-UDP | DDoS/DoS UDP floods & fragmentation |
| Flood-TCP | DDoS/DoS TCP / PSHACK / RSTFIN / ACK fragmentation |
| Flood-SYN | DDoS/DoS SYN floods |
| Flood-HTTP | DDoS/DoS HTTP / Slowloris |
| Mirai | GREETH / GREIP / UDPPLAIN |
| Reconnaissance | Host discovery, OS/port scan, ping sweep, vuln scan |
| Spoofing | DNS spoofing, MITM-ARP |
| Brute_Force | Dictionary brute force |
| Malware | Backdoor malware |
| Web_Attack | XSS, SQLi, command injection, upload, browser hijacking |
| Benign | Benign / BenignTraffic |

### 3. Model (`model_def.py`)

Lightweight MLP with **logits** (no Softmax head; train with `from_logits=True`):

```
Input (n_features)
  → Dense(256) + LeakyReLU + BatchNorm + Dropout(0.15)
  → Dense(128) + LeakyReLU + BatchNorm + Dropout(0.10)
  → Dense(64)  + LeakyReLU + BatchNorm
  → Dense(n_classes)  # logits
```

### 4. Federated Training (`federated_train.py`)

| Setting | Value |
|---------|-------|
| Clients | 10 |
| Rounds | up to 40 (early stopping) |
| Local epochs | 2 |
| Client fraction | 1.0 |
| Aggregation | Dense FedAvg (`TOP_K_TRAIN = 1.0`) |
| Comm metric | Top-K 50% reported for paper |
| Partitioning | Dirichlet α = 100 (near-IID) |
| Warm-start | 8 epochs on ~1.2M stratified samples |
| Fine-tune | up to 40 epochs on ~1.5M samples |
| Loss | Sparse categorical cross-entropy |

Pipeline stages inside training:

1. Centralised warm-start  
2. Federated local training + FedAvg  
3. Global fine-tuning  
4. Full test evaluation  
5. Magnitude pruning + pruned evaluation  

### 5. Baselines, SHAP, Figures

- `baselines.py` — centralised DNN and standard FL (no sparsification)
- `shap_analysis.py` — global / class-wise SHAP importance
- `generate_figure.py` — convergence, communication, confusion matrix, ROC/PR, comparison plots

---

## Repository Structure

```
Explainable-IDS-in-IOMT-using-FL/
├── model_def.py          # Lightweight MLP + pruning helpers
├── preprocess.py         # CICIoT2023 streaming preprocess + Flood-* labels
├── federated_train.py    # Warm-start + FL + fine-tune + prune (main results)
├── baselines.py          # Centralised & standard FL baselines
├── shap_analysis.py      # SHAP explainability
├── generate_figure.py    # Publication figures & tables
└── README.md
```

On Kaggle, artefacts are written under:

```
/kaggle/working/
├── processed/            # X_train.npy, y_train.npy, class_names.csv, ...
├── results/              # meta.json, models, reports, predictions
└── figures/              # PNG/PDF figures + summary tables
```

---

## Requirements

Typical stack (Kaggle GPU recommended):

- Python 3.10+
- TensorFlow 2.x
- NumPy, Pandas, scikit-learn
- SHAP
- Matplotlib
- imbalanced-learn / psutil (as used by training utilities)

```bash
pip install tensorflow numpy pandas scikit-learn shap matplotlib psutil
```

---

## How to Run (Kaggle)

Attach the CICIoT2023 merged CSV dataset and place all `.py` files in `/kaggle/working` (or ensure `model_def.py` is importable).

**Run order:**

```text
1. model_def.py
2. preprocess.py
3. federated_train.py      ← final accuracy / F1 here
4. baselines.py
5. shap_analysis.py
6. generate_figure.py
```

### Quick verification after training

```python
import json
with open("/kaggle/working/results/meta.json") as f:
    meta = json.load(f)
print(meta["full_accuracy"], meta["full_f1"])
```

Expected headline range for the proposed method: **≥ 0.85–0.90** (this run: **0.9139 / 0.9163**).

---

## Evaluation Metrics

- Accuracy  
- Precision / Recall / F1 (macro & weighted)  
- Per-class classification report  
- Confusion matrix  
- ROC-AUC / Precision–Recall (when probabilities are saved)  
- Communication cost (MB / round, cumulative)  
- Pruned-model accuracy vs compression  
- SHAP feature importance  

---

## Generated Outputs

### Results (`/kaggle/working/results/`)

| File | Description |
|------|-------------|
| `meta.json` | Headline accuracy, F1, rounds, communication, runtime |
| `classification_report.json` | Per-class metrics |
| `best_global_model.keras` / `.h5` | Final global model |
| `pruned_model.keras` / `.h5` | Magnitude-pruned model |
| `round_metrics.csv` | Per-round FL metrics |
| `communication_history.csv` | Cumulative communication |
| `y_test.npy`, `y_pred.npy`, `y_pred_proba.npy` | Predictions |
| `confusion_matrix.npy` | Confusion matrix |
| `baseline_results.json` | Centralised / standard FL comparisons |
| `global_shap_importance.csv` | SHAP ranking |

### Figures (`/kaggle/working/figures/`)

- Convergence (accuracy / loss / communication)
- Communication cost vs standard FL
- Normalised confusion matrix
- Per-class metrics
- Model comparison bars / radar
- ROC and Precision–Recall curves
- Top SHAP feature importance
- Publication CSV/TeX summary tables

---

## Research Motivation

The framework targets three security challenges for resource-constrained IoT/IoMT devices at once:

1. **Privacy** — federated training without raw data sharing  
2. **Explainability** — SHAP attributions for clinical/security audit  
3. **Lightweight deployment** — compact MLP + pruning for edge gateways  

while delivering competitive detection performance on a modern IoT intrusion benchmark.

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{Mobeen2026IoTFL,
  title={Lightweight Explainable Federated Learning for Intrusion Detection in Resource-Constrained IoT Devices: An Empirical Evaluation},
  author={Muhammad Ahmad Mobeen and Others},
  journal={Under Review},
  year={2026}
}
```

---

## License

MIT License

---

## Acknowledgements

- Canadian Institute for Cybersecurity (CIC) — CICIoT2023 dataset  
- TensorFlow / Keras  
- SHAP  
- scikit-learn  
