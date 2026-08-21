"""
Research-Grade SHAP Explainability Analysis
CICIoT2023 IoMT Federated Learning — Springer Scientific Reports
Kaggle: 30 GB RAM, Tesla P100 15 GB VRAM
"""

import os
import gc
import json
import time
import warnings
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIGURATION
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SHAP-Analysis")

SEED = 42
SHAP_SAMPLES = 5000
BG_SAMPLES = 500
DPI = 600

PROC_DIR = "/kaggle/working/processed"
RESULTS_DIR = "/kaggle/working/results"
FIGURES_DIR = "/kaggle/working/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# COLOR PALETTE (consistent publication style)
# ============================================================
BLUE_PALETTE = plt.cm.Blues
COLOR_PRIMARY = "#003A6B"
COLOR_POSITIVE = "#d73027"
COLOR_NEGATIVE = "#4575b4"


# ============================================================
# HELPER: Balanced Background Selection
# ============================================================
def balanced_background(X, y, n_total, seed=SEED):
    """Select balanced background with equal class representation."""
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(y)
    n_per_class = max(1, n_total // len(unique_classes))
    indices = []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        n_select = min(n_per_class, len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n_select, replace=False).tolist())
    # If still short, fill randomly
    if len(indices) < n_total:
        remaining = list(set(range(len(X))) - set(indices))
        indices.extend(rng.choice(remaining, size=n_total - len(indices), replace=False).tolist())
    return np.array(indices[:n_total], dtype=np.int64)


# ============================================================
# HELPER: Stratified Sampling
# ============================================================
def stratified_sample(X, y, n_total, seed=SEED):
    """Stratified sampling ensuring all classes represented."""
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    n_per_class = max(1, n_total // len(unique_classes))
    indices = []
    for cls, count in zip(unique_classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_select = min(n_per_class, len(cls_idx))
        if n_select > 0:
            indices.extend(rng.choice(cls_idx, size=n_select, replace=False).tolist())
    return np.array(indices[:n_total], dtype=np.int64)


# ============================================================
# HELPER: GPU Memory Check
# ============================================================
def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    try:
        mem = tf.config.experimental.get_memory_info("GPU:0")
        return mem["current"] / 1024**2
    except Exception:
        return 0.0


# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================
logger.info("=" * 60)
logger.info(" LOADING MODEL AND DATA")
logger.info("=" * 60)

logger.info("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(RESULTS_DIR, "best_global_model.keras"),
    compile=False,
)
# Model outputs logits — compile with from_logits=True
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
logger.info(f"Model input shape: {model.input_shape}")
logger.info(f"Model output shape: {model.output_shape}")

logger.info("Loading data...")
X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))
X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
y_train_full = np.load(os.path.join(PROC_DIR, "y_train.npy"))

feat_names = pd.read_csv(
    os.path.join(PROC_DIR, "selected_features.csv"), header=None
)[0].tolist()
class_names = pd.read_csv(
    os.path.join(PROC_DIR, "class_names.csv"), header=None
)[0].tolist()

n_features = len(feat_names)
n_classes = len(class_names)

# Feature name validation
assert n_features == X_test.shape[1], (
    f"Feature mismatch: {n_features} names vs {X_test.shape[1]} columns. "
    "Check selected_features.csv consistency with preprocess.py."
)
logger.info(f"Features: {n_features}  |  Classes: {n_classes}")
logger.info(f"X_test shape: {X_test.shape}")

# ============================================================
# 2. SELECT BALANCED BACKGROUND
# ============================================================
logger.info("=" * 60)
logger.info(" SELECTING BALANCED BACKGROUND")
logger.info("=" * 60)

bg_idx = balanced_background(X_train, y_train_full, BG_SAMPLES)
X_bg = X_train[bg_idx].astype(np.float32)
logger.info(f"Background samples: {len(X_bg)} (balanced across {n_classes} classes)")

# ============================================================
# 3. STRATIFIED SHAP SAMPLES
# ============================================================
logger.info("=" * 60)
logger.info(" SELECTING STRATIFIED SHAP SAMPLES")
logger.info("=" * 60)

shap_idx = stratified_sample(X_test, y_test, SHAP_SAMPLES)
X_shap = X_test[shap_idx].astype(np.float32)
y_shap_true = y_test[shap_idx]

# Verify all classes present
present_classes = np.unique(y_shap_true)
missing = set(range(n_classes)) - set(present_classes)
if missing:
    logger.warning(f"Classes missing from SHAP samples: {missing}")
logger.info(f"SHAP samples: {len(X_shap)}  |  {len(present_classes)}/{n_classes} classes represented")

# ============================================================
# 4. SELECT EXPLAINER (auto-detect)
# ============================================================
logger.info("=" * 60)
logger.info(" INITIALISING SHAP EXPLAINER")
logger.info("=" * 60)

try:
    logger.info("Attempting DeepExplainer...")
    explainer = shap.DeepExplainer(model, X_bg)
    logger.info("✓ Using DeepExplainer")
except Exception as e:
    logger.warning(f"DeepExplainer failed: {e}")
    logger.info("Falling back to GradientExplainer...")
    explainer = shap.GradientExplainer(model, X_bg)
    logger.info("✓ Using GradientExplainer")

# ============================================================
# 5. COMPUTE SHAP VALUES
# ============================================================
logger.info("=" * 60)
logger.info(" COMPUTING SHAP VALUES")
logger.info("=" * 60)

t0 = time.time()
gpu_before = get_gpu_memory()

# Compute SHAP in batches if needed
try:
    shap_values = explainer.shap_values(X_shap)
except tf.errors.ResourceExhaustedError:
    logger.warning("OOM — computing SHAP in batches of 1000")
    batch_size = 1000
    shap_values = []
    for i in range(0, len(X_shap), batch_size):
        batch = X_shap[i:i + batch_size]
        sv = explainer.shap_values(batch)
        if i == 0:
            shap_values = [sv[c] for c in range(len(sv))]
        else:
            for c in range(len(sv)):
                shap_values[c] = np.vstack([shap_values[c], sv[c]])
        gc.collect()
    logger.info(f"Batch SHAP complete — {len(X_shap)} samples processed")

sv_array = np.array(shap_values, dtype=np.float32)
del shap_values
gc.collect()

t_elapsed = time.time() - t0
gpu_after = get_gpu_memory()
logger.info(f"SHAP array shape: {sv_array.shape}")
logger.info(f"SHAP computation time: {t_elapsed:.1f}s  ({t_elapsed/60:.1f}min)")
logger.info(f"GPU memory used: {gpu_after - gpu_before:.0f}MB")

# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ============================================================
logger.info("=" * 60)
logger.info(" GLOBAL FEATURE IMPORTANCE")
logger.info("=" * 60)

# Mean |SHAP| across samples, then across classes
mean_abs_all = np.mean(np.abs(sv_array), axis=1)  # (n_classes, n_features)
mean_abs_global = np.mean(mean_abs_all, axis=0)   # (n_features,)
std_abs_global = np.std(np.abs(sv_array).reshape(-1, n_features), axis=0)
median_abs_global = np.median(np.abs(sv_array).reshape(-1, n_features), axis=0)

imp_df = pd.DataFrame({
    "feature": feat_names,
    "mean_shap": mean_abs_global,
    "median_shap": median_abs_global,
    "std_shap": std_abs_global,
}).sort_values("mean_shap", ascending=False).reset_index(drop=True)

imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv(
    os.path.join(RESULTS_DIR, "global_shap_importance.csv"), index=False
)

logger.info("\nTop-10 Global Features:")
logger.info(imp_df.head(10)[["rank", "feature", "mean_shap"]].to_string(index=False))

# Top-K contributions
total_importance = imp_df["mean_shap"].sum()
top_contributions = {}
for k in [5, 10, 15]:
    top_sum = imp_df["mean_shap"].values[:k].sum()
    pct = 100 * top_sum / total_importance if total_importance > 0 else 0
    top_contributions[f"top_{k}_pct"] = round(pct, 2)
    logger.info(f"Top-{k} features = {pct:.1f}% of total predictive variance")

# ============================================================
# 7. PER-CLASS SHAP IMPORTANCE
# ============================================================
logger.info("=" * 60)
logger.info(" PER-CLASS SHAP IMPORTANCE")
logger.info("=" * 60)

per_class_rows = []
for c in range(n_classes):
    class_mean = np.mean(np.abs(sv_array[c]), axis=0)
    class_rank = np.argsort(class_mean)[::-1]
    for rank, feat_idx in enumerate(class_rank, 1):
        per_class_rows.append({
            "attack_class": class_names[c],
            "feature": feat_names[feat_idx],
            "mean_shap": round(float(class_mean[feat_idx]), 6),
            "rank": rank,
        })

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(
    os.path.join(RESULTS_DIR, "per_class_shap.csv"), index=False
)
logger.info(f"Per-class SHAP saved: {len(per_class_rows)} rows")

# ============================================================
# 8. FIGURE: Global SHAP Bar
# ============================================================
logger.info("Generating Global SHAP Bar...")

fig, ax = plt.subplots(figsize=(10, 7))
top10 = imp_df.head(10).copy()
colors = BLUE_PALETTE(np.linspace(0.4, 0.9, 10))[::-1]

bars = ax.barh(
    range(10),
    top10["mean_shap"].values,
    color=colors,
    edgecolor=COLOR_PRIMARY,
    linewidth=0.5,
    height=0.7,
)
ax.set_yticks(range(10))
ax.set_yticklabels(top10["feature"].values, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP Value|", fontsize=13, color=COLOR_PRIMARY)
ax.set_title(
    f"Global Feature Importance (SHAP)\n"
    f"Top-10 Features — All {n_classes} Attack Classes",
    fontsize=14,
    fontweight="bold",
    color=COLOR_PRIMARY,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=10)

for bar, val in zip(bars, top10["mean_shap"].values):
    ax.text(
        val + 0.0002,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        fontsize=10,
        color=COLOR_PRIMARY,
    )

ax.text(
    0.98, 0.02,
    f"Top-5: {top_contributions['top_5_pct']:.1f}%  |  "
    f"Top-10: {top_contributions['top_10_pct']:.1f}%  |  "
    f"Top-15: {top_contributions['top_15_pct']:.1f}%",
    transform=ax.transAxes,
    ha="right",
    fontsize=9,
    color=COLOR_PRIMARY,
    style="italic",
)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_shap_global_bar.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "fig_shap_global_bar.pdf"), bbox_inches="tight")
plt.close(fig)
logger.info("  ✓ fig_shap_global_bar.png + .pdf")

# ============================================================
# 9. FIGURE: SHAP Beeswarm
# ============================================================
logger.info("Generating SHAP Beeswarm...")

# Average SHAP across classes for beeswarm
sv_avg = np.mean(sv_array, axis=0)  # (n_samples, n_features)

fig = plt.figure(figsize=(14, 8))
shap.summary_plot(
    sv_avg,
    X_shap,
    feature_names=feat_names,
    show=False,
    plot_type="dot",
    max_display=15,
    color_bar=True,
)
fig.savefig(os.path.join(FIGURES_DIR, "fig_beeswarm.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "fig_beeswarm.pdf"), bbox_inches="tight")
plt.close("all")
logger.info("  ✓ fig_beeswarm.png + .pdf")

# ============================================================
# 10. FIGURE: SHAP Summary Dot Plot
# ============================================================
logger.info("Generating SHAP Summary Dot Plot...")

fig = plt.figure(figsize=(14, 8))
shap.summary_plot(
    sv_avg,
    X_shap,
    feature_names=feat_names,
    show=False,
    plot_type="dot",
    max_display=15,
)
fig.savefig(os.path.join(FIGURES_DIR, "fig_summary_dot.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "fig_summary_dot.pdf"), bbox_inches="tight")
plt.close("all")
logger.info("  ✓ fig_summary_dot.png + .pdf")

# ============================================================
# 11. FIGURE: Per-Class Heatmap
# ============================================================
logger.info("Generating Per-Class Heatmap...")

class_feat = np.array([
    np.mean(np.abs(sv_array[c]), axis=0) for c in range(n_classes)
])

fig, ax = plt.subplots(figsize=(18, max(6, n_classes * 0.4)))
im = ax.imshow(class_feat, aspect="auto", cmap="Blues", interpolation="nearest")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean |SHAP Value|", fontsize=11, color=COLOR_PRIMARY)
ax.set_xticks(range(n_features))
ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_classes))
ax.set_yticklabels(class_names, fontsize=9)
ax.set_title(
    "SHAP Feature Importance — Per Attack Class",
    fontsize=13,
    fontweight="bold",
    color=COLOR_PRIMARY,
)
ax.set_xlabel("Features", fontsize=11)
ax.set_ylabel("Attack Classes", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_shap_heatmap.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "fig_shap_heatmap.pdf"), bbox_inches="tight")
plt.close(fig)
logger.info("  ✓ fig_shap_heatmap.png + .pdf")

# ============================================================
# 12. LOCAL EXPLANATIONS: Intelligent Sample Selection
# ============================================================
logger.info("Generating Local Explanations (Waterfall + Bar)...")

local_dir = os.path.join(FIGURES_DIR, "local_explanations")
os.makedirs(local_dir, exist_ok=True)

# Predictions for selection
y_pred_proba = model.predict(X_shap, batch_size=256, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
confidence = np.max(y_pred_proba, axis=1)
correct = (y_pred == y_shap_true)

# Select representative samples
sample_configs = OrderedDict({
    "correct_high_conf": {"condition": correct & (confidence > 0.9), "label": "Correct — High Confidence"},
    "correct_low_conf": {"condition": correct & (confidence < 0.6), "label": "Correct — Low Confidence"},
    "correct_attack": {"condition": correct & (y_shap_true != 0), "label": "Correct Attack"},
    "correct_benign": {"condition": correct & (y_shap_true == 0), "label": "Correct Benign"},
    "misclassified": {"condition": ~correct, "label": "Misclassified"},
})

selected_samples = []
for config_key, config in sample_configs.items():
    candidates = np.where(config["condition"])[0]
    if len(candidates) > 0:
        selected_samples.append((candidates[0], config["label"]))
    logger.info(f"  {config['label']}: {len(candidates)} candidates")

for idx, (sample_i, label_desc) in enumerate(selected_samples):
    sample = X_shap[[sample_i]]
    true_cls = int(y_shap_true[sample_i])
    pred_cls = int(y_pred[sample_i])
    conf = confidence[sample_i]

    true_label = class_names[true_cls]
    pred_label = class_names[pred_cls]

    sv = sv_array[pred_cls][sample_i]
    order = np.argsort(np.abs(sv))[::-1][:15]
    vals = sv[order]
    flabs = [feat_names[j] for j in order]

    # ── Waterfall-style Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in vals]

    ax.barh(
        range(len(vals)),
        vals,
        color=bar_colors,
        edgecolor="grey",
        linewidth=0.3,
        height=0.7,
    )
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(flabs, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value", fontsize=11, color=COLOR_PRIMARY)
    ax.set_title(
        f"Local Explanation — {label_desc}\n"
        f"True: {true_label}  →  Pred: {pred_label}  "
        f"(Conf: {conf:.3f})",
        fontsize=12,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_POSITIVE, label="↑ Increases probability"),
            mpatches.Patch(color=COLOR_NEGATIVE, label="↓ Decreases probability"),
        ],
        fontsize=9,
        loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(local_dir, f"local_exp_{idx+1}_{label_desc.replace(' ', '_')}.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(local_dir, f"local_exp_{idx+1}_{label_desc.replace(' ', '_')}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

logger.info(f"  ✓ {len(selected_samples)} local explanations → {local_dir}")

# ============================================================
# 13. FIGURE: Waterfall Plots (3 correct, 3 misclassified)
# ============================================================
logger.info("Generating Waterfall Plots...")

waterfall_dir = os.path.join(FIGURES_DIR, "waterfall_plots")
os.makedirs(waterfall_dir, exist_ok=True)

correct_idx = np.where(correct)[0]
misclass_idx = np.where(~correct)[0]

selected_waterfall = []
if len(correct_idx) >= 3:
    selected_waterfall.extend(correct_idx[:3].tolist())
if len(misclass_idx) >= 3:
    selected_waterfall.extend(misclass_idx[:3].tolist())

for wf_i, sample_i in enumerate(selected_waterfall):
    true_cls = int(y_shap_true[sample_i])
    pred_cls = int(y_pred[sample_i])
    conf = confidence[sample_i]
    is_correct = "Correct" if correct[sample_i] else "Misclassified"

    sv = sv_array[pred_cls][sample_i]
    order = np.argsort(np.abs(sv))[::-1][:10]
    vals = sv[order]
    flabs = [feat_names[j] for j in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in vals]

    ax.barh(
        range(len(vals)),
        vals,
        color=bar_colors,
        edgecolor="grey",
        linewidth=0.3,
        height=0.6,
    )
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(flabs, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value", fontsize=11, color=COLOR_PRIMARY)
    ax.set_title(
        f"Waterfall — {is_correct}\n"
        f"True: {class_names[true_cls]}  →  "
        f"Pred: {class_names[pred_cls]} ({conf:.3f})",
        fontsize=11,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(
        os.path.join(waterfall_dir, f"waterfall_{wf_i+1}_{is_correct}.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(waterfall_dir, f"waterfall_{wf_i+1}_{is_correct}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

logger.info(f"  ✓ {len(selected_waterfall)} waterfall plots → {waterfall_dir}")

# ============================================================
# 14. SAVE ARTIFACTS
# ============================================================
logger.info("Saving SHAP arrays...")
np.save(os.path.join(RESULTS_DIR, "shap_values.npy"), sv_array.astype(np.float32))
np.save(os.path.join(RESULTS_DIR, "shap_test_idx.npy"), shap_idx)

# ============================================================
# 15. SHAP SUMMARY JSON
# ============================================================
shap_summary = OrderedDict({
    "global_top_features": imp_df.head(15)[["rank", "feature", "mean_shap"]].to_dict(orient="records"),
    "top_k_contributions": top_contributions,
    "n_shap_samples": int(len(X_shap)),
    "n_background_samples": int(len(X_bg)),
    "n_classes": n_classes,
    "n_features": n_features,
    "explainer_type": explainer.__class__.__name__,
})
with open(os.path.join(RESULTS_DIR, "shap_summary.json"), "w") as f:
    json.dump(shap_summary, f, indent=2)

# ============================================================
# 16. SHAP META
# ============================================================
shap_meta = OrderedDict({
    "total_runtime_seconds": round(t_elapsed, 2),
    "avg_explanation_time_ms": round((t_elapsed / len(X_shap)) * 1000, 2),
    "gpu_memory_used_mb": round(gpu_after - gpu_before, 1),
    "shap_samples": int(len(X_shap)),
    "background_samples": int(len(X_bg)),
    "background_selection": "balanced",
    "sampling": "stratified",
    "explainer": explainer.__class__.__name__,
    "dpi": DPI,
    "seed": SEED,
})
with open(os.path.join(RESULTS_DIR, "shap_meta.json"), "w") as f:
    json.dump(shap_meta, f, indent=2)

# ============================================================
# CLEANUP
# ============================================================
del sv_array, X_shap, X_bg, explainer, model
gc.collect()
tf.keras.backend.clear_session()

# ============================================================
# FINAL SUMMARY
# ============================================================
logger.info("\n")
logger.info("█" * 60)
logger.info("█" + " " * 58 + "█")
logger.info("█" + "  SHAP ANALYSIS COMPLETE".center(58) + "█")
logger.info("█" + " " * 58 + "█")
logger.info("█" * 60)
logger.info(f"  Total runtime:       {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
logger.info(f"  Avg per sample:      {shap_meta['avg_explanation_time_ms']:.1f} ms")
logger.info(f"  GPU memory used:     {shap_meta['gpu_memory_used_mb']:.0f} MB")
logger.info(f"  Samples explained:   {shap_meta['shap_samples']}")
logger.info(f"  Explainer:           {shap_meta['explainer']}")
logger.info(f"  Figures directory:   {FIGURES_DIR}")
logger.info(f"  Results directory:   {RESULTS_DIR}")
logger.info("█" * 60)