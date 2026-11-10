"""Central configuration for SignBridge. Edit values here, not in other modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Storage ---------------------------------------------------------------
DATA_DIR = ROOT / "data" / "sequences"      # data/sequences/<label>/<n>.npy
VIDEO_DIR = ROOT / "data" / "videos"        # optional: ISL videos to import (data/videos/<label>/*.mp4)
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "signbridge_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
REPORT_PATH = MODEL_DIR / "training_report.json"

# ---- Sequence / model ------------------------------------------------------
SEQUENCE_LENGTH = 30        # frames per sample (project requirement)
MIN_SAMPLES_PER_CLASS = 10  # below this, training refuses to run (too little to evaluate)

# ---- Inference -------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80  # default; adjustable in the UI
SMOOTHING_WINDOW = 5         # number of recent predictions to vote over
MIN_AGREEMENT = 3            # votes (of the window) needed to accept a word

# ---- Added in step 2 -------------------------------------------------------
PREDICT_EVERY = 3            # run the LSTM every N frames once the 30-frame window is full
RESET_AFTER_IDLE = 15        # frames without any visible hand before the window/vote state resets
IMPORT_MANIFEST = ROOT / "data" / "imported_videos.json"   # remembers which videos were imported

# ---- Added in step 3 -------------------------------------------------------
HOLISTIC_COMPLEXITY = 1      # MediaPipe model_complexity: 0 = faster, 1 = more accurate.
                             # Keep it identical for collecting, importing and recognising; change it => recollect/retrain.
