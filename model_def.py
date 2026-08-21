"""
Lightweight MLP for CICIoT2023 federated IDS.

On Kaggle: running this file installs a copy to /kaggle/working/model_def.py
so federated_train.py / baselines.py can `import model_def`.

Notebook tip — if you paste into a cell, make the first line:
    %%writefile /kaggle/working/model_def.py
then run the cell once (writes the file), then import normally.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# Kaggle / local path bootstrap
# ============================================================
KAGGLE_WORKING = Path("/kaggle/working")


def install_to_kaggle_working(verbose: bool = True) -> Path:
    """
    Ensure model_def.py is importable from /kaggle/working (and cwd).
    Returns the path that should be used for imports.
    """
    # Always prefer /kaggle/working on Kaggle; else cwd
    if KAGGLE_WORKING.exists():
        target_dir = KAGGLE_WORKING
    else:
        target_dir = Path.cwd()

    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "model_def.py"

    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))

    # Copy from this file when executed/imported as a real .py
    src = None
    try:
        src = Path(__file__).resolve()
    except NameError:
        src = None

    if src is not None and src.is_file():
        try:
            same = dest.exists() and src.resolve() == dest.resolve()
        except Exception:
            same = False
        if not same:
            shutil.copy2(src, dest)
            if verbose:
                print(f"[model_def] Installed → {dest}")
        elif verbose:
            print(f"[model_def] Ready at {dest}")
        return dest

    # Pasted notebook cell (no __file__): try IPython write if dest missing
    if not dest.exists() and verbose:
        print(
            "[model_def] WARNING: running as a pasted notebook cell (no __file__).\n"
            "  Re-run this cell with the first line:\n"
            "      %%writefile /kaggle/working/model_def.py\n"
            "  so federated_train.py can import model_def."
        )
    return dest


# Run on import so a prior `import model_def` / `%run model_def.py` fixes the path
_MODEL_DEF_PATH = install_to_kaggle_working(verbose=False)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass


def _leaky_relu():
    """TF/Keras compatibility: negative_slope (Keras 3) vs alpha (TF 2.x)."""
    try:
        return layers.LeakyReLU(negative_slope=0.1)
    except TypeError:
        return layers.LeakyReLU(alpha=0.1)


def _make_optimizer(learning_rate, total_steps, weight_decay, clipnorm):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(int(total_steps), 1),
        alpha=0.1,
    )
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )
    except (TypeError, AttributeError):
        try:
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=clipnorm)
        except TypeError:
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def build_lightweight_model(
    n_features,
    n_classes,
    total_steps=10000,
    learning_rate=3e-4,
    weight_decay=1e-4,
    clipnorm=1.0,
):
    """
    Lightweight MLP that outputs LOGITS (no softmax).
    Train / compile with from_logits=True.
    """
    inp = layers.Input(shape=(n_features,), name="features")

    x = layers.Dense(
        256,
        kernel_regularizer=regularizers.l2(weight_decay),
        kernel_initializer="he_normal",
    )(inp)
    x = _leaky_relu()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(weight_decay),
        kernel_initializer="he_normal",
    )(x)
    x = _leaky_relu()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(
        64,
        kernel_regularizer=regularizers.l2(weight_decay),
        kernel_initializer="he_normal",
    )(x)
    x = _leaky_relu()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)

    # Logits in float32 — required under mixed_float16 for stable CE
    out = layers.Dense(n_classes, activation=None, dtype="float32", name="logits")(x)
    model = models.Model(inputs=inp, outputs=out, name="lightweight_mlp")

    optimizer = _make_optimizer(learning_rate, total_steps, weight_decay, clipnorm)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def apply_magnitude_pruning(model, sparsity=0.40, compile_after=True):
    weights = model.get_weights()
    pruned = []
    total = active = 0
    for w in weights:
        if w.ndim >= 2:
            cutoff = np.percentile(np.abs(w), sparsity * 100)
            mask = (np.abs(w) >= cutoff).astype(w.dtype)
            pruned.append(w * mask)
            total += w.size
            active += int(mask.sum())
        else:
            pruned.append(w)
    model.set_weights(pruned)
    retained = 100.0 * active / total if total else 0.0
    print(f"  Pruning done: {active}/{total} active ({retained:.1f}% retained)")
    return model


def build_and_test_model():
    model = build_lightweight_model(n_features=35, n_classes=16, total_steps=10000)
    model.summary(print_fn=lambda s: print(s))
    x = np.zeros((2, 35), dtype=np.float32)
    y = model.predict(x, verbose=0)
    print(f"Model test output shape: {y.shape} (logits)")
    return model


def verify_kaggle_import() -> None:
    """Confirm other scripts can import model_def from /kaggle/working."""
    dest = install_to_kaggle_working(verbose=True)
    if str(dest.parent) not in sys.path:
        sys.path.insert(0, str(dest.parent))
    # Drop cached failed imports if any
    sys.modules.pop("model_def", None)
    import model_def as md  # noqa: F401
    assert hasattr(md, "build_lightweight_model")
    print(f"[model_def] Import OK from {dest}")
    print(f"[model_def] sys.path[0]={sys.path[0]}")


if __name__ == "__main__":
    install_to_kaggle_working(verbose=True)
    verify_kaggle_import()
    build_and_test_model()
