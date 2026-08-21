# baselines.py

import os, sys, time, json, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

def _ensure_model_def_importable() -> None:
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
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent / "model_def.py")
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.rglob("model_def.py")))
    for path in candidates:
        if path.is_file():
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
    raise ModuleNotFoundError(
        "No module named 'model_def'. Upload model_def.py to /kaggle/working "
        "or attach it as a Kaggle dataset."
    )

_ensure_model_def_importable()
from model_def import build_lightweight_model

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {[g.name for g in gpus]}")
else:
    print("CPU only")

PROC_DIR    = '/kaggle/working/processed'
RESULTS_DIR = '/kaggle/working/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = np.load(os.path.join(PROC_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROC_DIR, 'y_train.npy'))
X_test  = np.load(os.path.join(PROC_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(PROC_DIR, 'y_test.npy'))

N_FEATURES  = X_train.shape[1]
N_CLASSES   = int(y_train.max()) + 1
class_names = pd.read_csv(
    os.path.join(PROC_DIR, 'class_names.csv'), header=None)[0].tolist()

print(f"Features={N_FEATURES} Classes={N_CLASSES}")
print(f"Train={len(X_train):,} Test={len(X_test):,}")

results = {}

def baseline_model():
    return build_lightweight_model(
        n_features=N_FEATURES,
        n_classes=N_CLASSES
    )

# ── BASELINE 1: Centralised ───────────────────────────────────
print("\n" + "="*55)
print("BASELINE 1: Centralised Deep Model (GPU)")
print("="*55)

# Load class weights if available
cw_path = os.path.join(PROC_DIR, "class_weights.json")
class_weight = None
if os.path.exists(cw_path):
    with open(cw_path, "r") as f:
        class_weight = {int(k): float(v) for k, v in json.load(f).items()}

cb = baseline_model()
print(f"Params: {cb.count_params():,}")

t0 = time.time()
cb.fit(
    X_train, y_train,
    epochs=40,
    batch_size=512,
    validation_split=0.1,
    class_weight=class_weight,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ],
)
cb_time = time.time() - t0

def logits_to_pred(model, X):
    logits = model.predict(X, batch_size=1024, verbose=0)
    return np.argmax(logits, axis=1)

y_pred_cb = logits_to_pred(cb, X_test)
rep_cb    = classification_report(
    y_test, y_pred_cb,
    target_names=class_names,
    digits=4,
    output_dict=True
)
print(classification_report(
    y_test, y_pred_cb, target_names=class_names, digits=4))


results['centralised'] = {
    'report': rep_cb,
    'training_time': round(cb_time / 60, 2),
    'params': cb.count_params(),
    'size_kb': round(cb.count_params() * 4 / 1024, 2),
    'accuracy': rep_cb['accuracy'],
    'macro_f1': rep_cb['macro avg']['f1-score'],
    'weighted_f1': rep_cb['weighted avg']['f1-score'],
    'precision': rep_cb['weighted avg']['precision'],
    'recall': rep_cb['weighted avg']['recall'],
}
cb.save(os.path.join(RESULTS_DIR, 'centralised_model.h5'))
del cb
gc.collect()
tf.keras.backend.clear_session()
# re-enable memory growth after clear_session
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

# ── BASELINE 2: Standard FL (manual, no compression) ─────────
print("\n" + "="*55)
print("BASELINE 2: Standard FL — no sparsification")
print("="*55)

N_CLIENTS   = 10
N_ROUNDS    = 100
SEED        = 42
rng         = np.random.RandomState(SEED)
splits      = np.array_split(rng.permutation(len(X_train)), N_CLIENTS)

# init weights from big_model
init_b      = baseline_model()
g_weights   = [w.copy() for w in init_b.get_weights()]
param_bytes = sum(w.nbytes for w in g_weights)
del init_b
gc.collect()

comm_total = 0.0

# ONE persistent model — reuse weights, no rebuild per round
with tf.device('/CPU:0'):
    client_model = baseline_model()

eval_b = baseline_model()

t0 = time.time()
def fedavg_inplace(g_weights, updates):
    total = sum(n for n, _ in updates)

    for i in range(len(g_weights)):
        agg = np.zeros_like(g_weights[i], dtype=np.float32)

        for n, w in updates:
            agg += (n / total) * w[i]

        np.copyto(g_weights[i], agg)

    return g_weights


for rnd in range(1, N_ROUNDS + 1):

    sel = rng.choice(N_CLIENTS, size=8, replace=False)
    updates = []

    for cid in sel:
        with tf.device('/CPU:0'):
            client_model.set_weights(g_weights)

            client_model.fit(
                X_train[splits[cid]],
                y_train[splits[cid]],
                epochs=3,
                batch_size=512,
                verbose=0
            )

            updates.append(
                (len(splits[cid]),
                 [w.copy() for w in client_model.get_weights()])
            )

    g_weights = fedavg_inplace(g_weights, updates)
    comm_total += (param_bytes * len(sel)) / 1e6

    del updates

    if rnd % 10 == 0:
        val_idx = rng.choice(
            len(X_test),
            size=min(2000, len(X_test)),
            replace=False
        )

        eval_b.set_weights(g_weights)

        loss, acc = eval_b.evaluate(
            X_test[val_idx],
            y_test[val_idx],
            batch_size=256,
            verbose=0
        )

        print(
            f"Round {rnd:3d} | comm={comm_total:.0f} MB | "
            f"val_loss={loss:.4f} | val_acc={acc:.4f}"
        )

    gc.collect()

sfl_time = time.time() - t0
del client_model, eval_b
gc.collect()
tf.keras.backend.clear_session()

# re-enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

# final eval on test set
sfl_m = baseline_model()
sfl_m.set_weights(g_weights)
y_pred_sfl = logits_to_pred(sfl_m, X_test)
rep_sfl = classification_report(
    y_test, y_pred_sfl,
    target_names=class_names,
    digits=4,
    output_dict=True
)
print(classification_report(
    y_test, y_pred_sfl, target_names=class_names, digits=4))

results['standard_fl'] = {
    'report':        rep_sfl,
    'training_time': round(sfl_time/60, 2),
    'total_comm_mb': round(comm_total, 2),
    'params':        sfl_m.count_params(),
    'accuracy': rep_sfl['accuracy'],
    'macro_f1': rep_sfl['macro avg']['f1-score'],
    'weighted_f1': rep_sfl['weighted avg']['f1-score'],
    'precision': rep_sfl['weighted avg']['precision'],
    'recall': rep_sfl['weighted avg']['recall'],
}
sfl_m.save(os.path.join(RESULTS_DIR, 'standard_fl_model.h5'))
del sfl_m
gc.collect()

# ── SAVE ──────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Baselines saved → {RESULTS_DIR}")
print(f"  Centralised time : {results['centralised']['training_time']} min")
print(f"  Centralised acc  : {results['centralised']['accuracy']:.4f}")
print(f"  Standard FL time : {results['standard_fl']['training_time']} min")
print(f"  Standard FL acc  : {results['standard_fl']['accuracy']:.4f}")
print(f"  Standard FL comm : {comm_total:.0f} MB")