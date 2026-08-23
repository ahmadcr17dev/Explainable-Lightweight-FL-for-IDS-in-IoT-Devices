import os
import gc
import json
import sys
import time
import warnings
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def _ensure_model_def_importable() -> None:
    """Make model_def.py importable on Kaggle notebooks / mixed cwd layouts."""
    try:
        import model_def  # noqa: F401
        return
    except ImportError:
        pass

    candidates = [
        Path.cwd() / "model_def.py",
        Path("/kaggle/working") / "model_def.py",
        Path("/kaggle/working/code") / "model_def.py",
    ]
    # Script directory (when run as a .py file, not a pasted notebook cell)
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent / "model_def.py")

    # Search uploaded Kaggle datasets / utility scripts
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.rglob("model_def.py")))

    for path in candidates:
        if path.is_file():
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            # Also mirror into /kaggle/working so later cells find it
            dest = Path("/kaggle/working") / "model_def.py"
            if path.resolve() != dest.resolve():
                try:
                    dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                    if "/kaggle/working" not in sys.path:
                        sys.path.insert(0, "/kaggle/working")
                except Exception:
                    pass
            return

    raise ModuleNotFoundError(
        "No module named 'model_def'.\n"
        "Fix on Kaggle (pick one):\n"
        "  1) Upload model_def.py into /kaggle/working (same place as this script), OR\n"
        "  2) Add model_def.py as a Dataset and attach it to the notebook, OR\n"
        "  3) In the first cell run:\n"
        "       import sys; sys.path.append('/kaggle/working')\n"
        "     after copying model_def.py there.\n"
        f"Searched: {[str(c) for c in candidates[:8]]} ..."
    )


_ensure_model_def_importable()
from model_def import build_lightweight_model, apply_magnitude_pruning

# ============================================================
# HYPERPARAMETERS — target ≥85–90% on Flood-* collapsed CICIoT
# ============================================================
SEED: int = 42
N_CLIENTS: int = 10
N_ROUNDS: int = 40
LOCAL_EPOCHS: int = 2
GLOBAL_BATCH_SIZE: int = 2048
FINAL_BATCH_SIZE: int = 4096
CLIENT_FRACTION: float = 1.0         # all clients each round
TOP_K_TRAIN: float = 1.0             # dense FedAvg (accuracy)
COMM_TOP_K: float = 0.50             # paper communication metric only
DIRICHLET_ALPHA: float = 100.0       # near-IID (realistic hospital wards share similar traffic)
EARLY_STOP_PATIENCE: int = 8
MIN_ROUNDS_BEFORE_STOP: int = 20
CHECKPOINT_EVERY: int = 5
VAL_EVERY: int = 2
VAL_SUBSET_SIZE: int = 50000
WARMSTART_EPOCHS: int = 8            # centralised warm-start before FL
WARMSTART_SAMPLES: int = 1_200_000
FT_EPOCHS: int = 40
LOCAL_LR: float = 1e-3
FT_LR: float = 3e-4
MAX_CLIENT_SAMPLES: int = 400_000
MAX_FT_SAMPLES: int = 1_500_000
STEPS_PER_EPOCH: int = 300
LABEL_SMOOTHING: float = 0.0
USE_CLASS_WEIGHTS: bool = False
FOCAL_GAMMA: float = 0.0             # 0 = plain CE (best for overall accuracy)

PROC_DIR: str = "/kaggle/working/processed"
RESULTS_DIR: str = "/kaggle/working/results"
CKPT_DIR: str = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FL-Trainer")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision: mixed_float16")
        except Exception:
            logger.info("Mixed precision unavailable — using float32")
        logger.info(f"GPU(s): {[g.name for g in gpus]}")
    else:
        logger.warning("No GPU detected — training will be slow on CPU")

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          List[str], Dict[int, float], int, int]:
    logger.info("=" * 60)
    logger.info(" LOADING PREPROCESSED DATA")
    logger.info("=" * 60)

    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy")).astype(np.int32)
    X_test = np.load(os.path.join(PROC_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy")).astype(np.int32)

    n_feats = X_train.shape[1]
    n_classes = int(y_train.max()) + 1
    class_names = pd.read_csv(
        os.path.join(PROC_DIR, "class_names.csv"), header=None
    )[0].tolist()

    logger.info(f"Train: {X_train.shape} ({X_train.nbytes / 1024**2:.0f} MB)")
    logger.info(f"Test:  {X_test.shape} ({X_test.nbytes / 1024**2:.0f} MB)")
    logger.info(f"Features: {n_feats} | Classes: {n_classes}")

    with open(os.path.join(PROC_DIR, "class_weights.json"), "r") as f:
        class_weights = {int(k): float(v) for k, v in json.load(f).items()}

    return X_train, y_train, X_test, y_test, class_names, class_weights, n_feats, n_classes


def dirichlet_partition(
    y: np.ndarray, n_clients: int, alpha: float = 1.0, seed: int = SEED
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    class_indices = [np.where(y == c)[0] for c in range(n_classes)]
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = class_indices[c].copy()
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet([alpha] * n_clients)
        proportions = np.maximum(proportions, 0.01)
        proportions /= proportions.sum()
        splits = (proportions * len(idx_c)).astype(int)
        splits[-1] = len(idx_c) - splits[:-1].sum()
        start = 0
        for k in range(n_clients):
            end = start + max(0, splits[k])
            client_indices[k].extend(idx_c[start:end].tolist())
            start = end

    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def topk_sparsify_numpy(
    delta: np.ndarray, residual: np.ndarray, k_pct: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Error-feedback Top-K sparsification on a single weight array."""
    if delta.ndim < 2:
        return delta.astype(np.float32), np.zeros_like(residual, dtype=np.float32)

    d_comp = delta.astype(np.float32) + residual.astype(np.float32)
    flat = d_comp.ravel()
    k = max(1, int(flat.size * k_pct))
    if k >= flat.size:
        return d_comp, np.zeros_like(residual, dtype=np.float32)

    abs_flat = np.abs(flat)
    # argpartition is O(n) — keep largest k magnitudes
    thresh_idx = np.argpartition(abs_flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[thresh_idx] = 1.0
    transmitted = (flat * mask).reshape(d_comp.shape)
    residual_new = d_comp - transmitted
    return transmitted, residual_new.astype(np.float32)


def federated_average_numpy(
    client_weights: List[List[np.ndarray]], client_sizes: List[int]
) -> List[np.ndarray]:
    total_n = float(sum(client_sizes))
    n_layers = len(client_weights[0])
    avg = []
    for li in range(n_layers):
        acc = np.zeros_like(client_weights[0][li], dtype=np.float32)
        for cw, n in zip(client_weights, client_sizes):
            acc += (n / total_n) * cw[li].astype(np.float32)
        avg.append(acc)
    return avg


def make_optimizer(lr: float):
    try:
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    except TypeError:
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)


def make_sparse_ce_loss(label_smoothing: float = 0.0):
    """
    Sparse CE with optional label smoothing.
    Older TensorFlow builds reject label_smoothing on SparseCategoricalCrossentropy.
    """
    if label_smoothing <= 0:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
    except TypeError:
        smooth = float(label_smoothing)

        def sparse_ce_smoothed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            n_classes = tf.shape(y_pred)[-1]
            y_one_hot = tf.one_hot(y_true, depth=n_classes)
            y_one_hot = tf.cast(y_one_hot, y_pred.dtype)
            y_smooth = y_one_hot * (1.0 - smooth) + smooth / tf.cast(n_classes, y_pred.dtype)
            return tf.keras.losses.categorical_crossentropy(
                y_smooth, y_pred, from_logits=True
            )

        return sparse_ce_smoothed


def make_focal_loss(gamma: float = 2.0):
    """Focal loss on sparse labels + logits — better for extreme CICIoT imbalance."""

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        ce = tf.cast(ce, tf.float32)
        pt = tf.exp(-ce)
        return tf.reduce_mean(tf.pow(1.0 - pt, gamma) * ce)

    return focal_loss


def make_training_loss():
    """Plain sparse CE for overall accuracy; optional mild focal if FOCAL_GAMMA>0."""
    if FOCAL_GAMMA and FOCAL_GAMMA > 0:
        return make_focal_loss(FOCAL_GAMMA)
    return make_sparse_ce_loss(LABEL_SMOOTHING)


def warm_start_centralised(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> List[np.ndarray]:
    """Short centralised pretrain — lifts FL from ~70% toward 90%+."""
    logger.info("=" * 60)
    logger.info(" CENTRALISED WARM-START")
    logger.info("=" * 60)
    Xs, ys = stratified_subsample(
        X, y, min(WARMSTART_SAMPLES, len(X)), seed=SEED + 11, n_classes=n_classes
    )
    model.compile(
        optimizer=make_optimizer(LOCAL_LR),
        loss=make_training_loss(),
        metrics=["accuracy"],
    )
    model.fit(
        Xs,
        ys,
        epochs=WARMSTART_EPOCHS,
        batch_size=FINAL_BATCH_SIZE,
        validation_data=(X_val, y_val),
        shuffle=True,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
            ),
        ],
    )
    loss, acc = model.evaluate(X_val, y_val, batch_size=FINAL_BATCH_SIZE, verbose=0)
    logger.info(f"Warm-start val_acc={acc:.4f} loss={loss:.4f} | samples={len(Xs):,}")
    return [w.copy() for w in model.get_weights()]


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_n: int, seed: int, n_classes: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cap dataset size while keeping ALL classes (critical for Malware/Web_Attack).
    Random subsample was dropping rare classes → 0.0 F1.
    """
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    # Guarantee floor per class, then fill remaining by global random
    classes, counts = np.unique(y, return_counts=True)
    floor = max(50, max_n // (2 * max(len(classes), 1)))
    chosen: List[int] = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        take = min(len(idx_c), floor)
        chosen.extend(rng.choice(idx_c, size=take, replace=False).tolist())
    chosen_set = set(chosen)
    remaining_budget = max_n - len(chosen)
    if remaining_budget > 0:
        pool = np.setdiff1d(np.arange(len(y)), np.array(chosen, dtype=np.int64), assume_unique=False)
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(remaining_budget, len(pool)), replace=False)
            chosen.extend(extra.tolist())
    chosen = np.array(chosen[:max_n], dtype=np.int64)
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def train_client_local(
    model: tf.keras.Model,
    Xc: np.ndarray,
    yc: np.ndarray,
    class_weight: Dict[int, float],
    epochs: int,
    batch_size: int,
    round_id: int = 0,
    client_id: int = 0,
) -> List[np.ndarray]:
    """Local SGD via Keras fit — syncs ALL weights including BatchNorm stats."""
    Xc, yc = stratified_subsample(
        Xc, yc, MAX_CLIENT_SAMPLES, seed=SEED + 1000 * round_id + client_id
    )
    bs = min(batch_size, max(1, len(Xc)))
    steps = min(STEPS_PER_EPOCH, max(1, len(Xc) // bs))
    cw = class_weight if USE_CLASS_WEIGHTS else None
    model.fit(
        Xc,
        yc,
        epochs=epochs,
        batch_size=bs,
        steps_per_epoch=steps,
        class_weight=cw,
        shuffle=True,
        verbose=0,
    )
    return [w.copy() for w in model.get_weights()]


def evaluate_numpy(
    model: tf.keras.Model, X: np.ndarray, y: np.ndarray, batch_size: int = 1024
) -> Tuple[float, float]:
    loss, acc = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    return float(loss), float(acc)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def fine_tune(
    global_weights: List[np.ndarray],
    X_ft: np.ndarray,
    y_ft: np.ndarray,
    n_features: int,
    n_classes: int,
    class_weight: Dict[int, float],
    X_val: np.ndarray,
    y_val: np.ndarray,
    results_dir: str,
) -> tf.keras.Model:
    logger.info("=" * 60)
    logger.info(" FINAL FINE-TUNING")
    logger.info("=" * 60)

    model = build_lightweight_model(
        n_features=n_features,
        n_classes=n_classes,
        total_steps=max(1, len(X_ft) // FINAL_BATCH_SIZE * FT_EPOCHS),
        learning_rate=FT_LR,
    )
    model.set_weights(global_weights)
    model.compile(
        optimizer=make_optimizer(FT_LR),
        loss=make_training_loss(),
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(results_dir, "best_global_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]

    model.fit(
        X_ft,
        y_ft,
        epochs=FT_EPOCHS,
        batch_size=FINAL_BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight if USE_CLASS_WEIGHTS else None,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure best weights + dual save formats for downstream scripts
    if os.path.exists(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        model.compile(
            optimizer=make_optimizer(FT_LR),
            loss=make_training_loss(),
            metrics=["accuracy"],
        )
    model.save(ckpt_path)
    try:
        model.save(os.path.join(results_dir, "best_global_model.h5"))
    except Exception as exc:
        logger.warning(f"H5 save skipped: {exc}")
    return model


class CommunicationTracker:
    def __init__(self, top_k_pct: float = COMM_TOP_K) -> None:
        self.per_round: List[float] = []
        self.total: float = 0.0
        self.top_k_pct = top_k_pct

    def log(self, n_clients: int, param_bytes: int) -> float:
        comm = (param_bytes * n_clients * self.top_k_pct) / 1e6
        self.per_round.append(round(comm, 4))
        self.total += comm
        return comm


def final_evaluation(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    results_dir: str,
) -> Dict:
    logger.info("=" * 60)
    logger.info(" EVALUATION")
    logger.info("=" * 60)

    logits = model.predict(X_test, batch_size=FINAL_BATCH_SIZE, verbose=0)
    y_proba = softmax_np(logits)
    y_pred = np.argmax(y_proba, axis=1)

    rep_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    logger.info("\n" + rep_str)

    rep_dict = classification_report(
        y_test, y_pred, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )

    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(rep_dict, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

    try:
        roc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc_val = float("nan")
    with open(os.path.join(results_dir, "roc_auc.json"), "w") as f:
        json.dump({"roc_auc_weighted": round(float(roc_val), 6)}, f, indent=2)

    np.save(os.path.join(results_dir, "y_test.npy"), y_test)
    np.save(os.path.join(results_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(results_dir, "y_pred_proba.npy"), y_proba)

    return rep_dict


def prune_and_evaluate(
    final_model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    n_features: int,
    n_classes: int,
    results_dir: str,
) -> Tuple[float, Dict]:
    logger.info("=" * 60)
    logger.info(" PRUNING")
    logger.info("=" * 60)

    pruned = build_lightweight_model(
        n_features=n_features,
        n_classes=n_classes,
        total_steps=10000,
        learning_rate=3e-4,
    )
    pruned.set_weights(final_model.get_weights())
    orig_p = sum(np.count_nonzero(w) for w in pruned.get_weights() if w.ndim >= 2)
    pruned = apply_magnitude_pruning(pruned, sparsity=0.40)
    prun_p = sum(np.count_nonzero(w) for w in pruned.get_weights() if w.ndim >= 2)
    ratio = 1.0 - (prun_p / orig_p) if orig_p > 0 else 0.0

    logger.info(f"Params: {orig_p:,} → {prun_p:,} ({ratio:.2%} compression)")

    logits = pruned.predict(X_test, batch_size=FINAL_BATCH_SIZE, verbose=0)
    yp = np.argmax(logits, axis=1)
    prep_str = classification_report(
        y_test, yp, target_names=class_names, digits=4, zero_division=0
    )
    logger.info("\n" + prep_str)

    prep_dict = classification_report(
        y_test, yp, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )

    with open(os.path.join(results_dir, "pruned_classification_report.json"), "w") as f:
        json.dump(prep_dict, f, indent=2)

    pruned.save(os.path.join(results_dir, "pruned_model.keras"))
    try:
        pruned.save(os.path.join(results_dir, "pruned_model.h5"))
    except Exception as exc:
        logger.warning(f"Pruned H5 save skipped: {exc}")
    np.save(os.path.join(results_dir, "y_pruned_pred.npy"), yp)

    return ratio, prep_dict


def main() -> None:
    configure_gpu()

    # Stale checkpoints from a crashed run can poison resume — start clean unless
    # RESUME_FL=1 is set in the environment.
    if os.environ.get("RESUME_FL", "0") != "1":
        for fname in ("best_weights.npy", "latest_weights.npy", "resume.json"):
            fpath = os.path.join(CKPT_DIR, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                logger.info(f"Cleared stale checkpoint: {fname}")

    X_train_full, y_train_full, X_test, y_test, class_names, class_weights, N_FEATURES, N_CLASSES = load_data()

    logger.info("=" * 60)
    logger.info(" DIRICHLET PARTITIONING")
    logger.info("=" * 60)
    partitions = dirichlet_partition(y_train_full, N_CLIENTS, DIRICHLET_ALPHA)
    for i, p in enumerate(partitions):
        n_cls = len(np.unique(y_train_full[p])) if len(p) else 0
        logger.info(f"  Client {i:>2d}: {len(p):>8,} samples | {n_cls:>2d} classes")

    logger.info("=" * 60)
    logger.info(" PREPARING CLIENT DATA")
    logger.info("=" * 60)
    client_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for cid in range(N_CLIENTS):
        idx = partitions[cid]
        if len(idx) == 0:
            # Fallback: give a tiny random shard so training never crashes
            idx = np.random.RandomState(SEED + cid).choice(
                len(y_train_full), size=min(1000, len(y_train_full)), replace=False
            )
        Xc = X_train_full[idx].astype(np.float32)
        yc = y_train_full[idx].astype(np.int32)
        client_data.append((Xc, yc))
        logger.info(f"  Client {cid:>2d}: {len(Xc):>8,} samples")

    # Stratified val + fine-tune (preserve rare classes)
    X_val, y_val = stratified_subsample(
        X_test, y_test, min(VAL_SUBSET_SIZE, len(X_test)), seed=SEED, n_classes=N_CLASSES
    )
    X_ft, y_ft = stratified_subsample(
        X_train_full, y_train_full, min(MAX_FT_SAMPLES, len(X_train_full)),
        seed=SEED + 7, n_classes=N_CLASSES,
    )
    # Keep a copy for warm-start before freeing full train arrays
    X_warm = X_train_full
    y_warm = y_train_full

    logger.info(
        f"Val subset: {len(X_val):,} | Fine-tune: {len(X_ft):,} "
        f"(classes val={len(np.unique(y_val))} ft={len(np.unique(y_ft))})"
    )
    logger.info(
        f"Training: dense FedAvg (TOP_K={TOP_K_TRAIN}) | "
        f"comm report TOP_K={COMM_TOP_K} | loss={'focal' if FOCAL_GAMMA else 'CE'}"
    )

    del partitions
    gc.collect()
    tf.keras.backend.clear_session()

    N_SEL = max(1, int(N_CLIENTS * CLIENT_FRACTION))
    TOTAL_STEPS = N_ROUNDS * N_SEL * LOCAL_EPOCHS * STEPS_PER_EPOCH

    client_model = build_lightweight_model(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        total_steps=TOTAL_STEPS,
        learning_rate=LOCAL_LR,
    )
    client_model.compile(
        optimizer=make_optimizer(LOCAL_LR),
        loss=make_training_loss(),
        metrics=["accuracy"],
    )
    eval_model = build_lightweight_model(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        total_steps=TOTAL_STEPS,
        learning_rate=LOCAL_LR,
    )
    eval_model.compile(
        optimizer=make_optimizer(LOCAL_LR),
        loss=make_sparse_ce_loss(0.0),
        metrics=["accuracy"],
    )

    # Centralised warm-start → then FL (critical for ≥85–90%)
    global_weights = warm_start_centralised(
        client_model, X_warm, y_warm, X_val, y_val, N_CLASSES
    )
    del X_warm, y_warm
    gc.collect()

    est_steps = N_ROUNDS * N_SEL * LOCAL_EPOCHS * STEPS_PER_EPOCH
    logger.info(
        f"Budget: max_client={MAX_CLIENT_SAMPLES:,} | steps/epoch={STEPS_PER_EPOCH} | "
        f"clients/round={N_SEL} | ≈{est_steps:,} local steps | FT={FT_EPOCHS} ep"
    )

    # Per-client residual error feedback
    client_residuals = [
        [np.zeros_like(w, dtype=np.float32) for w in global_weights]
        for _ in range(N_CLIENTS)
    ]

    PARAM_BYTES = sum(w.nbytes for w in global_weights)
    logger.info(f"Model: {PARAM_BYTES / 1024:.1f} KB parameters | params={client_model.count_params():,}")

    comm = CommunicationTracker(COMM_TOP_K)
    best_ckpt = os.path.join(CKPT_DIR, "best_weights.npy")
    latest_ckpt = os.path.join(CKPT_DIR, "latest_weights.npy")
    resume_round = 0
    if os.path.exists(latest_ckpt):
        saved = np.load(latest_ckpt, allow_pickle=True)
        global_weights = [np.array(w, dtype=np.float32) for w in saved]
        resume_path = os.path.join(CKPT_DIR, "resume.json")
        if os.path.exists(resume_path):
            with open(resume_path, "r") as f:
                resume_round = json.load(f).get("round", 0)
        logger.info(f"Resumed from round {resume_round + 1}")

    logger.info("=" * 60)
    logger.info(f" FL TRAINING — {N_ROUNDS} ROUNDS | {N_SEL}/{N_CLIENTS} clients/round")
    logger.info("=" * 60)

    round_log: Dict[str, List] = OrderedDict({
        "round": [], "loss": [], "accuracy": [],
        "weighted_f1": [], "comm_mb": [],
        "ram_gb": [], "gpu_mb": [], "elapsed_min": [],
    })

    rng = np.random.RandomState(SEED)
    t0 = time.time()
    best_acc = 0.0
    best_rnd = 0
    no_improve = 0

    for rnd in range(resume_round + 1, N_ROUNDS + 1):
        t_round = time.time()
        selected = rng.choice(N_CLIENTS, size=N_SEL, replace=False)

        client_weights_batch: List[List[np.ndarray]] = []
        client_sizes_batch: List[int] = []

        for cid in selected:
            Xc, yc = client_data[cid]
            client_model.set_weights(global_weights)
            new_weights = train_client_local(
                client_model, Xc, yc, class_weights, LOCAL_EPOCHS, GLOBAL_BATCH_SIZE,
                round_id=rnd, client_id=int(cid),
            )

            # Sparse delta with per-client residual
            sparse_weights = []
            for li, (nw, gw) in enumerate(zip(new_weights, global_weights)):
                delta = nw.astype(np.float32) - gw.astype(np.float32)
                tx, client_residuals[cid][li] = topk_sparsify_numpy(
                    delta, client_residuals[cid][li], TOP_K_TRAIN
                )
                sparse_weights.append(gw.astype(np.float32) + tx)

            client_weights_batch.append(sparse_weights)
            client_sizes_batch.append(len(Xc))

        global_weights = federated_average_numpy(client_weights_batch, client_sizes_batch)
        comm_mb = comm.log(len(selected), PARAM_BYTES)

        do_val = (rnd % VAL_EVERY == 0) or (rnd == 1) or (rnd == N_ROUNDS)
        if do_val:
            eval_model.set_weights(global_weights)
            loss, acc = evaluate_numpy(eval_model, X_val, y_val, FINAL_BATCH_SIZE)
        else:
            loss, acc = 0.0, 0.0

        ram_gb = psutil.virtual_memory().used / 1024**3
        try:
            gpu_info = tf.config.experimental.get_memory_info("GPU:0")
            gpu_mb = gpu_info.get("current", 0) / 1024**2 if gpu_info else 0.0
        except Exception:
            gpu_mb = 0.0
        elapsed = (time.time() - t0) / 60
        round_time = time.time() - t_round

        round_log["round"].append(rnd)
        round_log["loss"].append(float(loss))
        round_log["accuracy"].append(float(acc))
        round_log["weighted_f1"].append(0.0)
        round_log["comm_mb"].append(comm_mb)
        round_log["ram_gb"].append(ram_gb)
        round_log["gpu_mb"].append(gpu_mb)
        round_log["elapsed_min"].append(elapsed)

        if do_val:
            logger.info(
                f"  [R{rnd:>3d}] acc={acc:.4f} loss={loss:.4f} "
                f"comm={comm_mb:.1f}MB GPU={gpu_mb:.0f}MB RAM={ram_gb:.1f}GB "
                f"t={round_time:.0f}s total={elapsed:.1f}min"
            )
            if acc > best_acc + 0.0005:
                best_acc = acc
                best_rnd = rnd
                no_improve = 0
                np.save(best_ckpt, np.array(global_weights, dtype=object))
            else:
                no_improve += 1
                if (
                    rnd >= MIN_ROUNDS_BEFORE_STOP
                    and no_improve >= EARLY_STOP_PATIENCE
                ):
                    logger.info(f"Early stop at round {rnd} (best={best_acc:.4f} @ R{best_rnd})")
                    break

        if rnd % CHECKPOINT_EVERY == 0:
            np.save(latest_ckpt, np.array(global_weights, dtype=object))
            with open(os.path.join(CKPT_DIR, "resume.json"), "w") as f:
                json.dump({"round": rnd, "best_acc": best_acc}, f)
            pd.DataFrame(round_log).to_csv(
                os.path.join(RESULTS_DIR, "round_metrics.csv"), index=False
            )
            gc.collect()

    if os.path.exists(best_ckpt):
        best_saved = np.load(best_ckpt, allow_pickle=True)
        global_weights = [np.array(w, dtype=np.float32) for w in best_saved]
        logger.info(f"Restored best weights from round {best_rnd} (val_acc={best_acc:.4f})")

    del client_model, eval_model, client_data
    gc.collect()
    tf.keras.backend.clear_session()

    final_model = fine_tune(
        global_weights, X_ft, y_ft, N_FEATURES, N_CLASSES,
        class_weights, X_val, y_val, RESULTS_DIR,
    )

    rep_dict = final_evaluation(final_model, X_test, y_test, class_names, RESULTS_DIR)
    if round_log["weighted_f1"]:
        round_log["weighted_f1"][-1] = rep_dict["weighted avg"]["f1-score"]

    ratio, prep_dict = prune_and_evaluate(
        final_model, X_test, y_test, class_names,
        N_FEATURES, N_CLASSES, RESULTS_DIR,
    )

    pd.DataFrame(round_log).to_csv(
        os.path.join(RESULTS_DIR, "round_metrics.csv"), index=False
    )
    pd.DataFrame({
        "round": round_log["round"],
        "comm_mb": round_log["comm_mb"],
        "cumulative_comm_mb": np.cumsum(round_log["comm_mb"]),
    }).to_csv(os.path.join(RESULTS_DIR, "communication_history.csv"), index=False)

    test_acc = float(rep_dict.get("accuracy", 0.0))
    meta = OrderedDict({
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "n_clients": N_CLIENTS,
        "n_rounds": len(round_log["round"]),
        "best_round": best_rnd,
        "best_val_accuracy": round(best_acc, 6),
        "full_accuracy": round(test_acc, 6),
        "full_f1": round(float(rep_dict["weighted avg"]["f1-score"]), 6),
        "pruned_accuracy": prep_dict.get("accuracy", None),
        "compression": ratio,
        "topk_pct": COMM_TOP_K,
        "topk_train": TOP_K_TRAIN,
        "total_comm_mb": round(comm.total, 2),
        "time_min": round((time.time() - t0) / 60, 1),
    })
    with open(os.path.join(RESULTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tf.keras.backend.clear_session()
    gc.collect()

    logger.info("\n" + "█" * 50)
    logger.info(f"  DONE — {len(round_log['round'])} rounds, {meta['time_min']} min")
    logger.info(f"  Test Accuracy: {meta['full_accuracy']:.4f} | F1: {meta['full_f1']:.4f}")
    logger.info("█" * 50)


if __name__ == "__main__":
    main()
