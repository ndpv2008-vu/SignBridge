"""Train the LSTM on data/sequences and report honest held-out metrics.

Usage:  python -m signbridge.train --epochs 100
Split:  ~70% train / ~15% validation (early stopping) / ~15% test (reported only, never trained on).
"""
import argparse
import json
import math
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import config, dataset
from .model import build_lstm, save_model


class TrainingError(RuntimeError):
    """Raised for problems the user can fix (missing/too little data, missing TensorFlow)."""


def check_counts(counts: dict) -> None:
    if not counts:
        raise TrainingError("No samples found. Collect data in the app or import ISL videos first.")
    if len(counts) < 2:
        raise TrainingError("Need at least 2 different signs to train a classifier.")
    low = {k: v for k, v in counts.items() if v < config.MIN_SAMPLES_PER_CLASS}
    if low:
        details = ", ".join(f"{k}: {v}" for k, v in low.items())
        raise TrainingError(f"Each sign needs at least {config.MIN_SAMPLES_PER_CLASS} samples. Too few -> {details}")


def split_data(X, y, seed: int = 42):
    """Stratified train/val/test split. Raises TrainingError if the data is too small to split."""
    n, c = len(y), len(set(y.tolist()))
    n_hold = max(math.ceil(0.15 * n), c)  # size of each of val and test (at least one per class)
    if 2 * n_hold > 0.5 * n:
        raise TrainingError("Too few samples to hold out validation and test sets. Collect more samples.")
    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=2 * n_hold, stratify=y, random_state=seed)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    except ValueError as e:
        raise TrainingError(f"Could not split the data evenly across signs ({e}). Collect more samples per sign.") from e
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def train(epochs: int = 100, batch_size: int = 16, seed: int = 42, progress_cb=None) -> dict:
    """Train, evaluate on the held-out test set, save model + report. Returns the report dict.

    progress_cb(epoch, total_epochs, logs_dict) is called after every epoch (for the UI).
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise TrainingError("TensorFlow is not installed. Run: pip install -r requirements.txt") from e

    check_counts(dataset.class_counts())
    try:
        X, y, labels = dataset.load_dataset()
    except RuntimeError as e:
        raise TrainingError(str(e)) from e

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    X_tr, y_tr, X_val, y_val, X_te, y_te = split_data(X, y, seed)
    weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_tr)
    model = build_lstm(num_classes=len(labels), feature_dim=X.shape[2], seq_len=X.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]
    if progress_cb is not None:
        callbacks.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: progress_cb(ep + 1, epochs, dict(logs or {}))))

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        class_weight=dict(enumerate(weights)), callbacks=callbacks, verbose=0)

    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    all_idx = list(range(len(labels)))
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels": labels,
        "samples_per_label": dataset.class_counts(),
        "split_sizes": {"train": len(y_tr), "validation": len(y_val), "test": len(y_te)},
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float((pred == y_te).mean()),
        "per_label": classification_report(y_te, pred, labels=all_idx, target_names=labels,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=all_idx).tolist(),
        "note": ("Test accuracy is measured on held-out samples from YOUR dataset. If the samples were "
                 "recorded by one person in one session, it overestimates accuracy for new signers, "
                 "lighting and camera positions. With a small test set it is also a noisy estimate."),
    }
    save_model(model, labels)
    config.REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_report():
    """Return the last saved training report, or None if there is none / it is unreadable."""
    try:
        return json.loads(config.REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the SignBridge LSTM.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    try:
        rep = train(args.epochs, args.batch_size,
                    progress_cb=lambda e, t, l: print(f"epoch {e}/{t} loss={l.get('loss', 0):.4f} "
                                                      f"val_acc={l.get('val_accuracy', 0):.3f}"))
    except TrainingError as err:
        raise SystemExit(f"Training not started: {err}")
    print(f"\nHeld-out test accuracy: {rep['test_accuracy']:.3f} on {rep['split_sizes']['test']} samples")
    print(rep["note"])
