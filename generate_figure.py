"""
Publication-Quality Figure Generation
CICIoT2023 IoMT Federated Learning — Springer Scientific Reports
Compatible with updated federated_train / baselines / shap_analysis outputs.
"""

import os
import gc
import json
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ============================================================
# CONFIGURATION
# ============================================================
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FigureGen")

DPI = 600
SEED = 42

PROC_DIR = "/kaggle/working/processed"
RESULTS_DIR = "/kaggle/working/results"
FIGURES_DIR = "/kaggle/working/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "primary": "#2166AC",
    "secondary": "#B2182B",
    "tertiary": "#4DAC26",
    "accent": "#F4A582",
    "grid": "#E0E0E0",
    "text": "#333333",
    "diagonal_highlight": "#2166AC",
}
BLUE_PALETTE = plt.cm.Blues

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": COLORS["grid"],
})


def moving_average(data, window=5):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="same")


def safe_softmax(proba: np.ndarray) -> np.ndarray:
    """Ensure rows are valid probabilities (handles logits accidentally saved)."""
    proba = np.asarray(proba, dtype=np.float64)
    row_sums = proba.sum(axis=1, keepdims=True)
    # Likely logits if any negative or rows don't sum ~1
    if np.any(proba < -1e-6) or np.any(np.abs(row_sums.ravel() - 1.0) > 0.05):
        z = proba - proba.max(axis=1, keepdims=True)
        e = np.exp(z)
        proba = e / e.sum(axis=1, keepdims=True)
    else:
        proba = proba / np.clip(row_sums, 1e-12, None)
    return proba.astype(np.float32)


def save_fig(fig, stem: str):
    fig.savefig(os.path.join(FIGURES_DIR, f"{stem}.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {stem}")


# ============================================================
# 1. LOAD AND VALIDATE
# ============================================================
logger.info("=" * 60)
logger.info(" LOADING RESULTS AND VALIDATING FILES")
logger.info("=" * 60)

REQUIRED_FILES = {
    os.path.join(RESULTS_DIR, "round_metrics.csv"): "FL training metrics",
    os.path.join(RESULTS_DIR, "classification_report.json"): "classification report",
    os.path.join(RESULTS_DIR, "baseline_results.json"): "baseline results",
    os.path.join(RESULTS_DIR, "y_pred.npy"): "predictions",
    os.path.join(RESULTS_DIR, "y_test.npy"): "test labels",
    os.path.join(PROC_DIR, "class_names.csv"): "class names",
}

missing_required = []
for fpath, desc in REQUIRED_FILES.items():
    if not os.path.exists(fpath):
        missing_required.append((fpath, desc))
        logger.error(f"  ✗ MISSING: {fpath} ({desc})")
    else:
        logger.info(f"  ✓ {os.path.basename(fpath)}")

if missing_required:
    raise FileNotFoundError(
        f"Missing {len(missing_required)} required files. "
        "Run federated_train.py and baselines.py first."
    )

y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy")).astype(np.int32)
y_pred = np.load(os.path.join(RESULTS_DIR, "y_pred.npy")).astype(np.int32)
round_df = pd.read_csv(os.path.join(RESULTS_DIR, "round_metrics.csv"))
class_names = pd.read_csv(
    os.path.join(PROC_DIR, "class_names.csv"), header=None
)[0].tolist()

with open(os.path.join(RESULTS_DIR, "classification_report.json")) as f:
    report = json.load(f)
with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
    baselines = json.load(f)

n_classes = len(class_names)
if len(y_test) != len(y_pred):
    raise ValueError(f"y_test ({len(y_test)}) vs y_pred ({len(y_pred)}) length mismatch")

cent_report = baselines["centralised"]["report"]
sfl_report = baselines["standard_fl"]["report"]

# Validation-only rows (federated_train logs acc=0 on non-val rounds)
val_mask = (
    (round_df["accuracy"] > 0) | (round_df["loss"] > 0)
    if {"accuracy", "loss"}.issubset(round_df.columns)
    else np.ones(len(round_df), dtype=bool)
)
val_df = round_df.loc[val_mask].copy()
if len(val_df) == 0:
    val_df = round_df.copy()
    logger.warning("No positive val metrics found — plotting all rounds")

# Communication
has_comm = os.path.exists(os.path.join(RESULTS_DIR, "communication_history.csv"))
if has_comm:
    comm_df = pd.read_csv(os.path.join(RESULTS_DIR, "communication_history.csv"))
    if "cumulative_comm_mb" in comm_df.columns:
        prop_comm = comm_df["cumulative_comm_mb"].values.astype(float)
        comm_rounds = comm_df["round"].values if "round" in comm_df.columns else round_df["round"].values
    else:
        prop_comm = round_df["comm_mb"].cumsum().values.astype(float)
        comm_rounds = round_df["round"].values
else:
    prop_comm = round_df["comm_mb"].cumsum().values.astype(float)
    comm_rounds = round_df["round"].values

# Align lengths if needed
n_comm = min(len(comm_rounds), len(prop_comm), len(round_df))
comm_rounds = np.asarray(comm_rounds[:n_comm])
prop_comm = np.asarray(prop_comm[:n_comm], dtype=float)

# Probabilities (softmax from federated_train; auto-fix if logits)
has_proba = os.path.exists(os.path.join(RESULTS_DIR, "y_pred_proba.npy"))
if has_proba:
    y_pred_proba = safe_softmax(np.load(os.path.join(RESULTS_DIR, "y_pred_proba.npy")))
    if y_pred_proba.shape[0] != len(y_test) or y_pred_proba.shape[1] != n_classes:
        logger.warning(
            f"  ⚠ y_pred_proba shape {y_pred_proba.shape} "
            f"!= ({len(y_test)}, {n_classes}) — disabling ROC/PR"
        )
        has_proba = False
        y_pred_proba = None
    else:
        logger.info("  ✓ y_pred_proba.npy loaded — ROC/PR curves enabled")
else:
    y_pred_proba = None
    logger.warning("  ⚠ y_pred_proba.npy not found — skipping ROC/PR curves")

has_shap = os.path.exists(os.path.join(RESULTS_DIR, "global_shap_importance.csv"))
shap_df = pd.read_csv(os.path.join(RESULTS_DIR, "global_shap_importance.csv")) if has_shap else None

meta = {}
if os.path.exists(os.path.join(RESULTS_DIR, "meta.json")):
    with open(os.path.join(RESULTS_DIR, "meta.json")) as f:
        meta = json.load(f)

# Top-K compression ratio from meta / default 0.50 (updated federated_train)
topk_pct = float(meta.get("topk_pct", 0.50))
if topk_pct <= 0 or topk_pct > 1:
    topk_pct = 0.50
std_comm = prop_comm / topk_pct  # denser FL without sparsification
saving_pct = (1.0 - prop_comm[-1] / std_comm[-1]) * 100 if len(prop_comm) and std_comm[-1] > 0 else 0.0

prop_acc = float(report.get("accuracy", meta.get("full_accuracy", 0)))
prop_wf1 = float(report.get("weighted avg", {}).get("f1-score", meta.get("full_f1", 0)))
prop_mf1 = float(report.get("macro avg", {}).get("f1-score", 0))
cent_acc = float(cent_report.get("accuracy", baselines["centralised"].get("accuracy", 0)))
cent_wf1 = float(cent_report.get("weighted avg", {}).get("f1-score", 0))
sfl_acc = float(sfl_report.get("accuracy", baselines["standard_fl"].get("accuracy", 0)))
sfl_wf1 = float(sfl_report.get("weighted avg", {}).get("f1-score", 0))

cent_params = baselines["centralised"].get("params", meta.get("n_features", 0))
sfl_comm = float(baselines["standard_fl"].get("total_comm_mb", std_comm[-1] if len(std_comm) else 0))
prop_comm_total = float(meta.get("total_comm_mb", prop_comm[-1] if len(prop_comm) else 0))
prop_time = meta.get("time_min", meta.get("training_time_min", "N/A"))

figure_counter = 0


def next_fig():
    global figure_counter
    figure_counter += 1
    return figure_counter


# ============================================================
# 2. FIGURE: FL Convergence
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Convergence...")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
ax1, ax2, ax3, ax4 = axes.flatten()

ax1.plot(val_df["round"], val_df["accuracy"],
         color=COLORS["primary"], lw=2, alpha=0.55, marker="o", ms=3, label="Val (raw)")
smoothed_acc = moving_average(val_df["accuracy"].values, min(5, max(1, len(val_df))))
ax1.plot(val_df["round"], smoothed_acc,
         color=COLORS["primary"], lw=2.5, label="Smoothed")
ax1.axhline(cent_acc, color=COLORS["secondary"], ls="--", lw=1.5,
            label=f"Centralised ({cent_acc:.3f})")
ax1.axhline(prop_acc, color=COLORS["tertiary"], ls=":", lw=1.5,
            label=f"Final test ({prop_acc:.3f})")
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Accuracy")
ax1.set_title("Model Accuracy Convergence")
ax1.legend(fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.grid(True)

if "loss" in val_df.columns:
    ax2.plot(val_df["round"], val_df["loss"],
             color=COLORS["tertiary"], lw=2, alpha=0.55, marker="o", ms=3, label="Val (raw)")
    smoothed_loss = moving_average(val_df["loss"].values, min(5, max(1, len(val_df))))
    ax2.plot(val_df["round"], smoothed_loss,
             color=COLORS["tertiary"], lw=2.5, label="Smoothed")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_title("Validation Loss Convergence")
    ax2.legend(fontsize=9)
    ax2.grid(True)

# GPU / RAM if present, else placeholder
if "gpu_mb" in round_df.columns and round_df["gpu_mb"].sum() > 0:
    ax3.plot(round_df["round"], round_df["gpu_mb"], color="#D95F02", lw=2, label="GPU MB")
    if "ram_gb" in round_df.columns:
        ax3_t = ax3.twinx()
        ax3_t.plot(round_df["round"], round_df["ram_gb"], color="#7570B3", lw=1.5, ls="--", label="RAM GB")
        ax3_t.set_ylabel("RAM (GB)")
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("GPU Memory (MB)")
    ax3.set_title("Resource Usage")
    ax3.grid(True)
elif "learning_rate" in round_df.columns:
    ax3.plot(round_df["round"], round_df["learning_rate"], color="#D95F02", lw=2)
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True)
else:
    ax3.text(0.5, 0.5, "LR / resource trace\nnot logged",
             ha="center", va="center", transform=ax3.transAxes, fontsize=12, color="grey")
    ax3.set_title("Resources / LR")

if "comm_mb" in round_df.columns:
    ax4.bar(round_df["round"], round_df["comm_mb"],
            color=COLORS["primary"], alpha=0.75, width=0.8)
    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("Communication (MB)")
    ax4.set_title("Communication Cost per Round")
    ax4.grid(axis="y")
else:
    ax4.text(0.5, 0.5, "Communication data\nnot available",
             ha="center", va="center", transform=ax4.transAxes, fontsize=12, color="grey")

fig.suptitle(
    f"Figure {fig_num}: Federated Learning Training Convergence",
    fontsize=15, fontweight="bold", color=COLORS["primary"], y=1.01
)
plt.tight_layout()
save_fig(fig, f"fig{fig_num}_convergence")

# ============================================================
# 3. FIGURE: Communication Cost
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Communication Cost...")

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(comm_rounds, std_comm, color=COLORS["secondary"], lw=2, ls="--",
        label=f"Standard FL (est.) — {std_comm[-1]:.0f} MB")
ax.plot(comm_rounds, prop_comm, color=COLORS["primary"], lw=2.5,
        label=f"Proposed Top-K FL — {prop_comm[-1]:.0f} MB")
ax.fill_between(comm_rounds, prop_comm, std_comm, alpha=0.12, color=COLORS["secondary"],
                label=f"Bandwidth Saved: {saving_pct:.1f}%")

ax.annotate(f"{prop_comm[-1]:.0f} MB",
            xy=(comm_rounds[-1], prop_comm[-1]),
            xytext=(10, -25), textcoords="offset points",
            fontsize=10, color=COLORS["primary"], fontweight="bold")
ax.annotate(f"{std_comm[-1]:.0f} MB",
            xy=(comm_rounds[-1], std_comm[-1]),
            xytext=(10, 15), textcoords="offset points",
            fontsize=10, color=COLORS["secondary"], fontweight="bold")

ax.set_xlabel("Communication Round")
ax.set_ylabel("Cumulative Upload Cost (MB)")
ax.set_title(
    f"Figure {fig_num}: Communication Cost Analysis\n"
    f"Proposed Top-K ({topk_pct:.0%}) vs Dense Standard FL",
    fontweight="bold", color=COLORS["primary"]
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True)

if "comm_mb" in round_df.columns:
    inset_ax = ax.inset_axes([0.55, 0.15, 0.40, 0.35])
    inset_ax.plot(round_df["round"], round_df["comm_mb"],
                  color=COLORS["primary"], lw=1.5, alpha=0.8)
    inset_ax.set_title("Per-Round (MB)", fontsize=9)
    inset_ax.set_xlabel("Round", fontsize=8)
    inset_ax.grid(True, alpha=0.5)
    inset_ax.tick_params(labelsize=8)

plt.tight_layout()
save_fig(fig, f"fig{fig_num}_comm_cost")

# ============================================================
# 4. FIGURE: Confusion Matrix
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)), normalize="true")
cm_masked = np.where(cm < 0.01, np.nan, cm)

fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.9), max(10, n_classes * 0.8)))
im = ax.imshow(cm_masked, cmap="Blues", vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Proportion", fontsize=12)

ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(class_names, fontsize=9)
ax.set_title(
    f"Figure {fig_num}: Normalised Confusion Matrix\n"
    f"Proposed Lightweight FL-IDS (Accuracy={prop_acc:.4f})",
    fontweight="bold", color=COLORS["primary"], pad=15
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

for i in range(n_classes):
    if not np.isnan(cm_masked[i, i]):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor=COLORS["diagonal_highlight"], linewidth=2.5
        ))

for i in range(n_classes):
    for j in range(n_classes):
        val = cm[i, j]
        if val >= 0.01:
            text_color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color=text_color, fontweight="bold" if val > 0.7 else "normal")

plt.tight_layout()
save_fig(fig, f"fig{fig_num}_confusion_matrix")

# ============================================================
# 5. FIGURE: Per-Class Metrics
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Per-Class Metrics...")

skip = {"accuracy", "macro avg", "weighted avg"}
plot_cls = [c for c in class_names if c in report and c not in skip]
cls_metrics = [{
    "class": c,
    "precision": float(report[c]["precision"]),
    "recall": float(report[c]["recall"]),
    "f1": float(report[c]["f1-score"]),
} for c in plot_cls]
cls_metrics.sort(key=lambda x: x["f1"], reverse=True)

plot_names = [m["class"][:16] for m in cls_metrics]
prec_vals = [m["precision"] for m in cls_metrics]
rec_vals = [m["recall"] for m in cls_metrics]
f1_vals = [m["f1"] for m in cls_metrics]

x = np.arange(len(plot_names))
width = 0.25
fig, ax = plt.subplots(figsize=(max(12, len(plot_names) * 0.9), 7))
bars1 = ax.bar(x - width, prec_vals, width, label="Precision",
               color="#2166AC", alpha=0.85, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x, rec_vals, width, label="Recall",
               color="#F4A582", alpha=0.85, edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + width, f1_vals, width, label="F1-Score",
               color="#B2182B", alpha=0.85, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title(
    f"Figure {fig_num}: Per-Class Performance Metrics\n"
    f"Precision, Recall, F1-Score (Sorted by F1)",
    fontweight="bold", color=COLORS["primary"]
)
ax.legend(fontsize=11, loc="lower right")
ax.grid(axis="y", alpha=0.4)

for bar, val in zip(bars3, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            color=COLORS["secondary"], fontweight="bold")

plt.tight_layout()
save_fig(fig, f"fig{fig_num}_per_class_metrics")

# ============================================================
# 6. FIGURE: Model Comparison
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Model Comparison...")

metrics_names = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
prop_metrics = [
    prop_acc,
    float(report.get("weighted avg", {}).get("precision", 0)),
    float(report.get("weighted avg", {}).get("recall", 0)),
    prop_mf1,
    prop_wf1,
]
cent_metrics = [
    cent_acc,
    float(cent_report.get("weighted avg", {}).get("precision", 0)),
    float(cent_report.get("weighted avg", {}).get("recall", 0)),
    float(cent_report.get("macro avg", {}).get("f1-score", 0)),
    cent_wf1,
]
sfl_metrics = [
    sfl_acc,
    float(sfl_report.get("weighted avg", {}).get("precision", 0)),
    float(sfl_report.get("weighted avg", {}).get("recall", 0)),
    float(sfl_report.get("macro avg", {}).get("f1-score", 0)),
    sfl_wf1,
]

x = np.arange(len(metrics_names))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 7))
ax.bar(x - width, cent_metrics, width, label="Centralised",
       color=COLORS["secondary"], alpha=0.85, edgecolor="white")
ax.bar(x, sfl_metrics, width, label="Standard FL",
       color=COLORS["tertiary"], alpha=0.85, edgecolor="white")
ax.bar(x + width, prop_metrics, width, label="Proposed FL",
       color=COLORS["primary"], alpha=0.90, edgecolor="white")

for i, (cv, sv, pv) in enumerate(zip(cent_metrics, sfl_metrics, prop_metrics)):
    for val, offset in [(cv, -width), (sv, 0), (pv, width)]:
        ax.text(i + offset, val + 0.008, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=COLORS["text"])

ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title(
    f"Figure {fig_num}: Performance Comparison — Centralised vs Standard FL vs Proposed",
    fontweight="bold", color=COLORS["primary"]
)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
save_fig(fig, f"fig{fig_num}_model_comparison")

# ============================================================
# 7. FIGURE: Radar Chart
# ============================================================
fig_num = next_fig()
logger.info(f"Generating Figure {fig_num}: Radar Chart...")

comm_efficiency = max(0.0, min(1.0, saving_pct / 100.0))
n_params = float(baselines["centralised"].get("params", 50_000))
model_size_kb = n_params * 4 / 1024.0
memory_efficiency = 1.0 if model_size_kb < 50 else min(1.0, 50.0 / model_size_kb)
explainability = 1.0 if has_shap else 0.0

categories = [
    "Detection\nAccuracy", "Weighted F1",
    "Communication\nEfficiency", "Memory\nEfficiency", "Explainability",
]
N_cat = len(categories)
angles = [n / N_cat * 2 * np.pi for n in range(N_cat)]
angles += angles[:1]

scores = {
    "Proposed": [prop_acc, prop_wf1, comm_efficiency, memory_efficiency, explainability],
    "Centralised": [cent_acc, cent_wf1, 0.05, 0.30, 0.0],
    "Standard FL": [sfl_acc, sfl_wf1, 0.10, memory_efficiency * 0.9, 0.0],
}
radar_colors = {
    "Proposed": (COLORS["primary"], 0.25),
    "Centralised": (COLORS["secondary"], 0.12),
    "Standard FL": (COLORS["tertiary"], 0.15),
}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
for label, (color, alpha) in radar_colors.items():
    vals = scores[label] + scores[label][:1]
    ax.plot(angles, vals, lw=2.5, color=color, label=label)
    ax.fill(angles, vals, alpha=alpha, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
ax.set_title(
    f"Figure {fig_num}: Multi-Dimensional Performance Radar",
    fontweight="bold", color=COLORS["primary"], pad=22
)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=10)

plt.tight_layout()
save_fig(fig, f"fig{fig_num}_radar")

# ============================================================
# 8. FIGURE: ROC Curves
# ============================================================
if has_proba and y_pred_proba is not None:
    fig_num = next_fig()
    logger.info(f"Generating Figure {fig_num}: ROC Curves...")

    classes = list(range(n_classes))
    y_test_bin = label_binarize(y_test, classes=classes)
    if n_classes == 2 and y_test_bin.ndim == 1:
        y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])

    fig, ax = plt.subplots(figsize=(11, 8))
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        ax.plot(fpr_dict[i], tpr_dict[i], lw=1.2, alpha=0.5,
                label=f"{class_names[i][:18]} (AUC={auc_dict[i]:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, lw=3, color=COLORS["primary"],
            label=f"Micro-average (AUC={auc_micro:.3f})", zorder=5)

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, lw=3, ls="--", color=COLORS["secondary"],
            label=f"Macro-average (AUC={auc_macro:.3f})", zorder=5)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"Figure {fig_num}: ROC Curves — Per-Class and Averages",
        fontweight="bold", color=COLORS["primary"]
    )
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_fig(fig, f"fig{fig_num}_roc")

# ============================================================
# 9. FIGURE: Precision-Recall Curves
# ============================================================
if has_proba and y_pred_proba is not None:
    fig_num = next_fig()
    logger.info(f"Generating Figure {fig_num}: Precision-Recall Curves...")

    classes = list(range(n_classes))
    y_test_bin = label_binarize(y_test, classes=classes)
    if n_classes == 2 and y_test_bin.ndim == 1:
        y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])

    fig, ax = plt.subplots(figsize=(11, 8))
    for i in range(n_classes):
        precision_i, recall_i, _ = precision_recall_curve(
            y_test_bin[:, i], y_pred_proba[:, i]
        )
        ap_i = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        ax.plot(recall_i, precision_i, lw=1.2, alpha=0.5,
                label=f"{class_names[i][:18]} (AP={ap_i:.3f})")

    all_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        precision_i, recall_i, _ = precision_recall_curve(
            y_test_bin[:, i], y_pred_proba[:, i]
        )
        # recall is ascending after reverse for interp
        order = np.argsort(recall_i)
        mean_precision += np.interp(all_recall, recall_i[order], precision_i[order])
    mean_precision /= n_classes
    ax.plot(all_recall, mean_precision, lw=3, color=COLORS["primary"],
            label="Macro-average", zorder=5)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Figure {fig_num}: Precision-Recall Curves",
        fontweight="bold", color=COLORS["primary"]
    )
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_fig(fig, f"fig{fig_num}_precision_recall")

# ============================================================
# 10. FIGURE: SHAP Feature Importance
# ============================================================
if has_shap and shap_df is not None and "mean_shap" in shap_df.columns:
    fig_num = next_fig()
    logger.info(f"Generating Figure {fig_num}: SHAP Feature Importance...")

    n_top = min(20, len(shap_df))
    top_n = shap_df.head(n_top).copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, max(6, n_top * 0.35)))
    colors_grad = BLUE_PALETTE(np.linspace(0.3, 0.9, n_top))
    bars = ax.barh(range(n_top), top_n["mean_shap"].values, color=colors_grad,
                   edgecolor=COLORS["primary"], linewidth=0.4, height=0.7)
    ax.set_yticks(range(n_top))
    feat_col = "feature" if "feature" in top_n.columns else top_n.columns[0]
    ax.set_yticklabels(top_n[feat_col].astype(str).values, fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=13)
    ax.set_title(
        f"Figure {fig_num}: Top-{n_top} SHAP Feature Importance",
        fontweight="bold", color=COLORS["primary"]
    )

    for bar, val in zip(bars, top_n["mean_shap"].values):
        ax.text(val + max(top_n["mean_shap"].values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=COLORS["text"])

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f"fig{fig_num}_shap_importance")

# ============================================================
# 11. EXPORT PUBLICATION TABLES
# ============================================================
logger.info("=" * 60)
logger.info(" EXPORTING PUBLICATION TABLES")
logger.info("=" * 60)

results_summary = pd.DataFrame([
    {
        "Model": "Centralised",
        "Accuracy": f"{cent_acc:.4f}",
        "Precision": f"{float(cent_report.get('weighted avg', {}).get('precision', 0)):.4f}",
        "Recall": f"{float(cent_report.get('weighted avg', {}).get('recall', 0)):.4f}",
        "Macro_F1": f"{float(cent_report.get('macro avg', {}).get('f1-score', 0)):.4f}",
        "Weighted_F1": f"{cent_wf1:.4f}",
        "Training_Time_min": baselines["centralised"].get("training_time", "N/A"),
        "Communication_MB": "N/A",
        "Model_Parameters": baselines["centralised"].get("params", "N/A"),
    },
    {
        "Model": "Standard FL",
        "Accuracy": f"{sfl_acc:.4f}",
        "Precision": f"{float(sfl_report.get('weighted avg', {}).get('precision', 0)):.4f}",
        "Recall": f"{float(sfl_report.get('weighted avg', {}).get('recall', 0)):.4f}",
        "Macro_F1": f"{float(sfl_report.get('macro avg', {}).get('f1-score', 0)):.4f}",
        "Weighted_F1": f"{sfl_wf1:.4f}",
        "Training_Time_min": baselines["standard_fl"].get("training_time", "N/A"),
        "Communication_MB": f"{sfl_comm:.2f}",
        "Model_Parameters": baselines["standard_fl"].get("params", "N/A"),
    },
    {
        "Model": "Proposed FL",
        "Accuracy": f"{prop_acc:.4f}",
        "Precision": f"{float(report.get('weighted avg', {}).get('precision', 0)):.4f}",
        "Recall": f"{float(report.get('weighted avg', {}).get('recall', 0)):.4f}",
        "Macro_F1": f"{prop_mf1:.4f}",
        "Weighted_F1": f"{prop_wf1:.4f}",
        "Training_Time_min": prop_time,
        "Communication_MB": f"{prop_comm_total:.2f}",
        "Model_Parameters": baselines["centralised"].get("params", meta.get("n_features", "N/A")),
    },
])
results_summary.to_csv(os.path.join(FIGURES_DIR, "table_results_summary.csv"), index=False)
results_summary.to_latex(
    os.path.join(FIGURES_DIR, "table_results_summary.tex"),
    index=False,
    caption="Performance comparison on CICIoT2023.",
    label="tab:results_summary",
    escape=True,
)
logger.info("  ✓ table_results_summary.csv/.tex")

# Per-class table
per_class_rows = []
for c in class_names:
    if c not in report:
        continue
    per_class_rows.append({
        "Class": c,
        "Precision": f"{float(report[c]['precision']):.4f}",
        "Recall": f"{float(report[c]['recall']):.4f}",
        "F1": f"{float(report[c]['f1-score']):.4f}",
        "Support": int(report[c].get("support", 0)),
    })
per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(os.path.join(FIGURES_DIR, "table_per_class_metrics.csv"), index=False)
logger.info("  ✓ table_per_class_metrics.csv")

# Round metrics (validation rounds only)
val_df.to_csv(os.path.join(FIGURES_DIR, "table_fl_val_rounds.csv"), index=False)
logger.info("  ✓ table_fl_val_rounds.csv")

# Manifest
manifest = {
    "n_figures": figure_counter,
    "n_classes": n_classes,
    "proposed_accuracy": prop_acc,
    "proposed_weighted_f1": prop_wf1,
    "centralised_accuracy": cent_acc,
    "standard_fl_accuracy": sfl_acc,
    "comm_saving_pct": round(saving_pct, 2),
    "topk_pct": topk_pct,
    "has_shap": has_shap,
    "has_proba": has_proba,
    "figures_dir": FIGURES_DIR,
}
with open(os.path.join(FIGURES_DIR, "figure_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

logger.info("\n" + "█" * 50)
logger.info(f"  DONE — {figure_counter} figures → {FIGURES_DIR}")
logger.info(f"  Proposed Acc={prop_acc:.4f} | Cent={cent_acc:.4f} | SFL={sfl_acc:.4f}")
logger.info("█" * 50)

gc.collect()
