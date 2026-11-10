"""LSTM model definition and model/metadata persistence.

TensorFlow is imported lazily so that modules which only need the error classes
(or the smoothing logic) load quickly and work without TensorFlow installed.
"""
import json
import shutil
from . import config


class ModelNotFoundError(RuntimeError):
    """Raised when no usable trained model exists (not trained yet, or files unreadable)."""


def build_lstm(num_classes: int, feature_dim: int, seq_len: int = config.SEQUENCE_LENGTH,
               learning_rate: float = 1e-3):
    """Two stacked LSTMs -> dense head. Input: (seq_len, feature_dim); output: softmax over signs."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="signbridge_lstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_model(model, labels: list) -> None:
    """Save model + labels.json. A previous model is kept once as *.prev.keras."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if config.MODEL_PATH.exists():
        shutil.copy2(config.MODEL_PATH, config.MODEL_PATH.with_suffix(".prev.keras"))
    model.save(config.MODEL_PATH)
    meta = {"labels": list(labels),
            "sequence_length": int(model.input_shape[1]),
            "feature_dim": int(model.input_shape[2])}
    config.LABELS_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def model_exists() -> bool:
    return config.MODEL_PATH.exists() and config.LABELS_PATH.exists()


def load_trained():
    """Return (keras_model, meta). meta = {labels, sequence_length, feature_dim}.

    Raises ModelNotFoundError with a user-readable message on any problem.
    """
    if not model_exists():
        raise ModelNotFoundError("No trained model found. Collect/import data and train a model first.")
    try:
        meta = json.loads(config.LABELS_PATH.read_text(encoding="utf-8"))
        labels = meta["labels"]
        int(meta["sequence_length"]), int(meta["feature_dim"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ModelNotFoundError(f"labels.json is unreadable or incomplete ({e}). Retrain the model.") from e
    try:
        from tensorflow import keras
        model = keras.models.load_model(config.MODEL_PATH, compile=False)
    except ImportError as e:
        raise ModelNotFoundError("TensorFlow is not installed. Run: pip install -r requirements.txt") from e
    except Exception as e:  # noqa: BLE001 - keras raises many types for corrupt files
        raise ModelNotFoundError(f"The saved model could not be loaded ({e}). Retrain the model.") from e
    if model.output_shape[-1] != len(labels):
        raise ModelNotFoundError("Model and labels.json do not match. Retrain the model.")
    return model, meta
