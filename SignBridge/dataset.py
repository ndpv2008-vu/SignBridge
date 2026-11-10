"""Read/write labelled landmark sequences stored as data/sequences/<label>/<n>.npy."""
import re
import numpy as np
from . import config


def clean_label(label: str) -> str:
    """Make a label safe to use as a folder name (keeps letters, digits, _ and -)."""
    label = re.sub(r"[^\w\-]+", "_", label.strip(), flags=re.UNICODE).strip("_")
    if not label:
        raise ValueError("Label cannot be empty.")
    return label


def save_sequence(label: str, sequence: np.ndarray) -> str:
    """Save one (SEQUENCE_LENGTH, FEATURE_DIM) sequence. Never overwrites an existing file."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] != config.SEQUENCE_LENGTH:
        raise ValueError(f"Expected {config.SEQUENCE_LENGTH} frames, got shape {sequence.shape}.")
    folder = config.DATA_DIR / clean_label(label)
    folder.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem) for p in folder.glob("*.npy") if p.stem.isdigit()]
    path = folder / f"{(max(existing) + 1) if existing else 0}.npy"
    np.save(path, sequence)
    return str(path)


def class_counts() -> dict:
    """{label: number_of_samples}, sorted by label."""
    if not config.DATA_DIR.exists():
        return {}
    return {d.name: len(list(d.glob("*.npy")))
            for d in sorted(config.DATA_DIR.iterdir()) if d.is_dir() and any(d.glob("*.npy"))}


def delete_label(label: str) -> int:
    """Delete all samples of one label. Returns number of files removed."""
    folder = config.DATA_DIR / clean_label(label)
    n = 0
    if folder.exists():
        for p in folder.glob("*.npy"):
            p.unlink()
            n += 1
        try:
            folder.rmdir()
        except OSError:
            pass
    return n


def load_dataset():
    """Return X (N, T, F), y_idx (N,), labels (list[str]). Raises if the dataset is empty."""
    labels = list(class_counts().keys())
    if not labels:
        raise RuntimeError("No samples found. Collect data (or import ISL videos) first.")
    X, y = [], []
    for idx, label in enumerate(labels):
        for p in sorted((config.DATA_DIR / label).glob("*.npy")):
            arr = np.load(p)
            if arr.shape[0] != config.SEQUENCE_LENGTH:
                continue  # skip samples recorded with a different sequence length
            X.append(arr)
            y.append(idx)
    return np.stack(X).astype(np.float32), np.array(y), labels
