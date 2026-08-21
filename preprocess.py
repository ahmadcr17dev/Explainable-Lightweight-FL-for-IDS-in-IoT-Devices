import os
import json
import warnings
import logging
import gc
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Preprocess")

RAW_DIR = "/kaggle/input/datasets/ahmadcr17/ciciot2023/MERGED_CSV"
PROC_DIR = "/kaggle/working/processed"
os.makedirs(PROC_DIR, exist_ok=True)

LABEL_COL = "Label"
TOP_FEATS = 35
TEST_SIZE = 0.2
DTYPE = np.float32
SAMPLE_FOR_MI = 200_000  # enough for MI estimation
CHUNK_SIZE = 30_000       # small chunks = low peak RAM

# ============================================================
# LABEL COLLAPSE MAPPING
# Keys MUST be uppercase — apply normalize_label() before .map()
# ============================================================
COLLAPSE = OrderedDict({
    "DDOS-ICMP_FLOOD":          "DDoS-ICMP",
    "DDOS-UDP_FLOOD":           "DDoS-UDP",
    "DDOS-TCP_FLOOD":           "DDoS-TCP",
    "DDOS-PSHACK_FLOOD":        "DDoS-TCP",
    "DDOS-SYN_FLOOD":           "DDoS-SYN",
    "DDOS-RSTFINFLOOD":         "DDoS-TCP",
    "DDOS-SYNONYMOUSIP_FLOOD":  "DDoS-UDP",
    "DDOS-HTTP_FLOOD":          "DDoS-HTTP",
    "DDOS-SLOWLORIS":           "DDoS-HTTP",
    "DDOS-ICMP_FRAGMENTATION":  "DDoS-ICMP",
    "DDOS-UDP_FRAGMENTATION":   "DDoS-UDP",
    "DDOS-ACK_FRAGMENTATION":   "DDoS-TCP",
    "DOS-UDP_FLOOD":            "DoS-UDP",
    "DOS-TCP_FLOOD":            "DoS-TCP",
    "DOS-SYN_FLOOD":            "DoS-SYN",
    "DOS-HTTP_FLOOD":           "DoS-HTTP",
    "MIRAI-GREETH_FLOOD":       "Mirai",
    "MIRAI-GREIP_FLOOD":        "Mirai",
    "MIRAI-UDPPLAIN":           "Mirai",
    "RECON-HOSTDISCOVERY":      "Reconnaissance",
    "RECON-OSSCAN":             "Reconnaissance",
    "RECON-PORTSCAN":           "Reconnaissance",
    "RECON-PINGSWEEP":          "Reconnaissance",
    "VULNERABILITYSCAN":        "Reconnaissance",
    "DNS_SPOOFING":             "Spoofing",
    "MITM-ARPSPOOFING":         "Spoofing",
    "DICTIONARYBRUTEFORCE":     "Brute_Force",
    "BACKDOOR_MALWARE":         "Malware",
    "XSS":                      "Web_Attack",
    "BROWSERHIJACKING":         "Web_Attack",
    "SQLINJECTION":             "Web_Attack",
    "COMMANDINJECTION":         "Web_Attack",
    "UPLOADING_ATTACK":         "Web_Attack",
    "BENIGN":                   "Benign",
    "BENIGNTRAFFIC":            "Benign",  # CICIoT2023 official name
})


def normalize_label(series):
    """Uppercase + strip so mixed-case CICIoT labels match COLLAPSE keys."""
    return series.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)


def fmt_mem(nbytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def integrity_assert(X, y, stage=""):
    assert not np.any(np.isnan(X)), f"[{stage}] NaN in features"
    assert not np.any(np.isinf(X)), f"[{stage}] Inf in features"
    assert not np.any(np.isnan(y)), f"[{stage}] NaN in labels"


# ============================================================
# PHASE 1 — SCHEMA + SAMPLE FOR LABEL ENCODER + MI
# ============================================================
logger.info("=" * 60)
logger.info(" PHASE 1 — SCHEMA + STRATIFIED SAMPLING")
logger.info("=" * 60)

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
logger.info(f"CSV files: {len(files)}")

# Discover schema from first file
probe = pd.read_csv(os.path.join(RAW_DIR, files[0]), nrows=5, low_memory=False)
if LABEL_COL not in probe.columns:
    raise RuntimeError("Label column not found")
numeric_cols = probe.select_dtypes(include=[np.number]).columns.tolist()
n_total_features = len(numeric_cols)
logger.info(f"Numeric features: {n_total_features}")

# Stream ALL files ONCE: collect stratified sample for MI + fit label encoder + count classes
rng = np.random.RandomState(SEED)
sample_X_list = []
sample_y_list = []
sample_counts = {}
class_counts = {}
total_rows = 0
per_class_limit = max(1, SAMPLE_FOR_MI // 20)  # rough per-class cap for sample

for idx, fname in enumerate(files, 1):
    fpath = os.path.join(RAW_DIR, fname)
    try:
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            if LABEL_COL not in chunk.columns:
                continue
            # Collapse labels (uppercase first — CICIoT uses mixed case)
            chunk[LABEL_COL] = normalize_label(chunk[LABEL_COL]).map(COLLAPSE)
            unmapped = int(chunk[LABEL_COL].isna().sum())
            if unmapped:
                logger.warning(f"  Dropping {unmapped} rows with unmapped labels in {fname}")
            chunk = chunk.dropna(subset=[LABEL_COL])
            if len(chunk) == 0:
                continue

            y_str = chunk[LABEL_COL].values
            total_rows += len(y_str)

            # Count classes (streaming)
            for lbl in y_str:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1

            # Collect sample rows for MI (cap per class)
            for lbl in np.unique(y_str):
                already = sample_counts.get(lbl, 0)
                if already >= per_class_limit:
                    continue
                mask = (y_str == lbl)
                indices = np.where(mask)[0]
                need = min(per_class_limit - already, len(indices))
                if need > 0:
                    sel = rng.choice(indices, size=need, replace=False)
                    X_sel = chunk[numeric_cols].iloc[sel].values.astype(DTYPE)
                    sample_X_list.append(X_sel)
                    sample_y_list.extend([lbl] * len(sel))
                    sample_counts[lbl] = already + len(sel)

            del chunk
            gc.collect()

        logger.info(f"  [{idx:02d}/{len(files)}] {fname:<50s} | total: {total_rows:,}")

    except Exception as exc:
        logger.error(f"  ✗ {fname}: {exc}")

# Build sample arrays for MI
logger.info("Assembling sample for MI...")
X_sample = np.concatenate(sample_X_list, axis=0).astype(DTYPE)
y_sample_raw = np.array(sample_y_list, dtype=str)
del sample_X_list, sample_y_list
gc.collect()

# Fit label encoder on all unique classes
all_classes = sorted(class_counts.keys())
label_encoder = LabelEncoder()
label_encoder.fit(all_classes)
class_names = list(label_encoder.classes_)
n_classes = len(class_names)
y_sample = label_encoder.transform(y_sample_raw).astype(np.int32)

logger.info(f"Classes: {n_classes} | Sample size: {len(X_sample):,} | Total rows: {total_rows:,}")

# Save class metadata
pd.Series(class_names, dtype=str).to_csv(
    os.path.join(PROC_DIR, "class_names.csv"), index=False, header=False
)
pd.DataFrame({"encoded_label": range(n_classes), "class_name": class_names}).to_csv(
    os.path.join(PROC_DIR, "label_mapping.csv"), index=False
)

# Clean sample for MI
X_sample = np.where(np.isinf(X_sample), np.nan, X_sample)
col_medians = np.nanmedian(X_sample, axis=0)
nan_mask = np.isnan(X_sample)
for col_idx in range(X_sample.shape[1]):
    if nan_mask[:, col_idx].any():
        X_sample[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]

# ============================================================
# PHASE 2 — FEATURE SELECTION ON SAMPLE (25 features only)
# ============================================================
logger.info("=" * 60)
logger.info(f" PHASE 2 — FEATURE SELECTION (Top-{TOP_FEATS} MI on {len(X_sample):,} samples)")
logger.info("=" * 60)

mi_scores = mutual_info_classif(X_sample, y_sample, random_state=SEED, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)

feature_ranking_df = pd.DataFrame({
    "rank": range(1, len(mi_series) + 1),
    "feature": mi_series.index,
    "mutual_information": mi_series.values.round(6),
})
feature_ranking_df.to_csv(os.path.join(PROC_DIR, "feature_ranking.csv"), index=False)
feature_ranking_df.to_csv(os.path.join(PROC_DIR, "feature_importance.csv"), index=False)

selected_features = mi_series.head(TOP_FEATS).index.tolist()
selected_indices = [numeric_cols.index(f) for f in selected_features]

logger.info(f"Top-{TOP_FEATS} features selected:")
for rank, (feat, score) in enumerate(mi_series.head(TOP_FEATS).items(), 1):
    logger.info(f"  {rank:>2d}. {feat:<40s} MI={score:.6f}")

pd.Series(selected_features).to_csv(
    os.path.join(PROC_DIR, "selected_features.csv"), index=False, header=False
)

del X_sample, y_sample, y_sample_raw
gc.collect()

# ============================================================
# PHASE 3 — STREAMING LOAD → DIRECT WRITE (25 COLS ONLY)
# ============================================================
logger.info("=" * 60)
logger.info(f" PHASE 3 — STREAMING → TRAIN/TEST ({TOP_FEATS} features)")
logger.info("=" * 60)

# Compute per-class train/test targets
class_counts_arr = np.zeros(n_classes, dtype=np.int64)
for lbl_name, cnt in class_counts.items():
    cls_idx = label_encoder.transform([lbl_name])[0]
    class_counts_arr[cls_idx] = cnt

train_target = np.zeros(n_classes, dtype=np.int64)
test_target = np.zeros(n_classes, dtype=np.int64)
for c in range(n_classes):
    n_test = max(1, int(class_counts_arr[c] * TEST_SIZE))
    train_target[c] = class_counts_arr[c] - n_test
    test_target[c] = n_test

n_train_total = int(train_target.sum())
n_test_total = int(test_target.sum())
logger.info(f"Train: {n_train_total:,}  |  Test: {n_test_total:,}")

# Pre-allocate arrays for selected features only
X_train = np.zeros((n_train_total, TOP_FEATS), dtype=DTYPE)
y_train = np.zeros(n_train_total, dtype=np.int32)
X_test = np.zeros((n_test_total, TOP_FEATS), dtype=DTYPE)
y_test = np.zeros(n_test_total, dtype=np.int32)

logger.info(f"Pre-allocated: X_train={fmt_mem(X_train.nbytes)}, "
            f"X_test={fmt_mem(X_test.nbytes)}, "
            f"y_train={fmt_mem(y_train.nbytes)}, y_test={fmt_mem(y_test.nbytes)}")
logger.info(f"Total allocation: {fmt_mem(X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes)}")

# Cursor per class
train_cursor = np.zeros(n_classes, dtype=np.int64)
test_cursor = np.zeros(n_classes, dtype=np.int64)

# Compute base offsets
train_base = np.zeros(n_classes, dtype=np.int64)
test_base = np.zeros(n_classes, dtype=np.int64)
pos = 0
for c in range(n_classes):
    train_base[c] = pos
    pos += train_target[c]
pos = 0
for c in range(n_classes):
    test_base[c] = pos
    pos += test_target[c]

# Stream all files again → direct write
total_assigned = 0
rng = np.random.RandomState(SEED + 1)

for idx, fname in enumerate(files, 1):
    fpath = os.path.join(RAW_DIR, fname)
    try:
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            if LABEL_COL not in chunk.columns:
                continue
            chunk[LABEL_COL] = normalize_label(chunk[LABEL_COL]).map(COLLAPSE)
            chunk = chunk.dropna(subset=[LABEL_COL])
            if len(chunk) == 0:
                continue

            y_enc = label_encoder.transform(chunk[LABEL_COL].values).astype(np.int32)
            X_chunk = chunk[selected_features].values.astype(DTYPE)  # selected cols only
            del chunk
            gc.collect()

            # Clean
            X_chunk = np.where(np.isinf(X_chunk), np.nan, X_chunk)
            col_med = np.nanmedian(X_chunk, axis=0)
            nm = np.isnan(X_chunk)
            for ci in range(X_chunk.shape[1]):
                if nm[:, ci].any():
                    fill = col_med[ci] if not np.isnan(col_med[ci]) else 0.0
                    X_chunk[nm[:, ci], ci] = fill

            # Shuffle
            perm = rng.permutation(len(X_chunk))
            X_chunk = X_chunk[perm]
            y_enc = y_enc[perm]

            # Assign per class
            for c in range(n_classes):
                c_mask = (y_enc == c)
                c_idx = np.where(c_mask)[0]
                if len(c_idx) == 0:
                    continue

                train_needed = train_target[c] - train_cursor[c]
                test_needed = test_target[c] - test_cursor[c]

                n_train = min(train_needed, len(c_idx))
                n_test = min(test_needed, len(c_idx) - n_train)

                if n_train > 0:
                    ws = train_base[c] + train_cursor[c]
                    we = ws + n_train
                    X_train[ws:we] = X_chunk[c_idx[:n_train]]
                    y_train[ws:we] = y_enc[c_idx[:n_train]]
                    train_cursor[c] += n_train

                if n_test > 0:
                    ws = test_base[c] + test_cursor[c]
                    we = ws + n_test
                    X_test[ws:we] = X_chunk[c_idx[n_train:n_train + n_test]]
                    y_test[ws:we] = y_enc[c_idx[n_train:n_train + n_test]]
                    test_cursor[c] += n_test

            total_assigned += len(X_chunk)
            del X_chunk, y_enc
            gc.collect()

        logger.info(f"  [{idx:02d}/{len(files)}] {fname:<50s} | assigned: {total_assigned:,}")

    except Exception as exc:
        logger.error(f"  ✗ {fname}: {exc}")

# Verify
for c in range(n_classes):
    assert train_cursor[c] == train_target[c], f"Class {c} train: {train_cursor[c]} != {train_target[c]}"
    assert test_cursor[c] == test_target[c], f"Class {c} test: {test_cursor[c]} != {test_target[c]}"

logger.info(f"✓ All {total_assigned:,} rows assigned")
integrity_assert(X_train, y_train, stage="train-raw")
integrity_assert(X_test, y_test, stage="test-raw")

# Final shuffle
train_perm = rng.permutation(n_train_total)
test_perm = rng.permutation(n_test_total)
X_train = np.ascontiguousarray(X_train[train_perm])
y_train = y_train[train_perm]
X_test = np.ascontiguousarray(X_test[test_perm])
y_test = y_test[test_perm]
gc.collect()

# ============================================================
# PHASE 4 — NORMALISATION
# ============================================================
logger.info("=" * 60)
logger.info(" PHASE 4 — NORMALISATION")
logger.info("=" * 60)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train).astype(DTYPE)
X_test_norm = scaler.transform(X_test).astype(DTYPE)
del X_train, X_test
gc.collect()

pd.DataFrame({
    "feature": selected_features,
    "mean": scaler.mean_.round(6),
    "std": scaler.scale_.round(6),
}).to_csv(os.path.join(PROC_DIR, "scaler_params.csv"), index=False)

integrity_assert(X_train_norm, y_train, stage="train-norm")
integrity_assert(X_test_norm, y_test, stage="test-norm")

logger.info(f"X_train_norm: {X_train_norm.shape} ({fmt_mem(X_train_norm.nbytes)})")
logger.info(f"X_test_norm:  {X_test_norm.shape} ({fmt_mem(X_test_norm.nbytes)})")

# ============================================================
# PHASE 5 — CLASS WEIGHTS + DISTRIBUTION
# ============================================================
logger.info("=" * 60)
logger.info(" PHASE 5 — CLASS WEIGHTS")
logger.info("=" * 60)

train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()

pd.DataFrame({
    "class_id": range(n_classes),
    "class_name": class_names,
    "train_count": [train_dist.get(i, 0) for i in range(n_classes)],
    "test_count": [test_dist.get(i, 0) for i in range(n_classes)],
}).to_csv(os.path.join(PROC_DIR, "class_distribution.csv"), index=False)

cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
# Cap extreme weights — rare classes otherwise destabilise CE training
cw = np.clip(cw, 0.5, 8.0)
cw_dict = {int(i): round(float(cw[i]), 6) for i in range(n_classes)}
with open(os.path.join(PROC_DIR, "class_weights.json"), "w") as f:
    json.dump(cw_dict, f, indent=2)
logger.info(f"Class weights (clipped): {cw_dict}")

# ============================================================
# PHASE 6 — SAVE
# ============================================================
logger.info("=" * 60)
logger.info(" SAVING")
logger.info("=" * 60)

np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train_norm)
np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROC_DIR, "X_test.npy"), X_test_norm)
np.save(os.path.join(PROC_DIR, "y_test.npy"), y_test)

dataset_info = OrderedDict({
    "dataset": "CICIoT2023",
    "preprocessing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "random_seed": SEED,
    "total_raw_samples": int(total_rows),
    "train_samples": int(len(X_train_norm)),
    "test_samples": int(len(X_test_norm)),
    "n_features_total": n_total_features,
    "n_features_selected": TOP_FEATS,
    "selected_features": selected_features,
    "n_classes": n_classes,
    "class_names": class_names,
    "test_split_ratio": TEST_SIZE,
    "normalisation": "StandardScaler",
    "feature_selection": f"Mutual Information top-{TOP_FEATS} on {SAMPLE_FOR_MI:,} samples",
    "smote_applied": False,
})
with open(os.path.join(PROC_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2)

total_mem = X_train_norm.nbytes + y_train.nbytes + X_test_norm.nbytes + y_test.nbytes

logger.info("\n" + "█" * 60)
logger.info("█" + "  DONE".center(58) + "█")
logger.info("█" * 60)
logger.info(f"  Total raw: {total_rows:,} | Train: {len(X_train_norm):,} | Test: {len(X_test_norm):,}")
logger.info(f"  Features: {TOP_FEATS} | Classes: {n_classes}")
logger.info(f"  Final arrays: {fmt_mem(total_mem)}")
logger.info("█" * 60)

del X_train_norm, y_train, X_test_norm, y_test
gc.collect()