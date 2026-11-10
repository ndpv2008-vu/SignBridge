"""Real-time prediction: rolling 30-frame window -> LSTM -> confidence threshold -> vote smoothing.

Feed one feature vector per frame (from landmarks.LandmarkExtractor) to SignPredictor.push().
It does not import MediaPipe, and imports TensorFlow only when a model is loaded.
"""
from collections import Counter, deque
from dataclasses import dataclass, field, replace
import numpy as np

from . import config
from .model import ModelNotFoundError


@dataclass
class Prediction:
    label: str | None = None        # smoothed label accepted by threshold + voting (None = nothing confident)
    confidence: float = 0.0         # mean confidence of the votes that agree on `label`
    raw_label: str | None = None    # this frame's top class (before threshold / smoothing)
    raw_confidence: float = 0.0
    top: list = field(default_factory=list)   # [(label, probability), ...] top 3 of the latest raw output
    new_word: bool = False          # True exactly once when a newly accepted word should be appended/spoken
    frames: int = 0                 # frames currently in the window (of SEQUENCE_LENGTH)


class PredictionSmoother:
    """Majority vote over the last `window` raw predictions.

    A raw prediction only counts as a vote if its confidence >= threshold. A label is accepted when it has
    at least `min_agreement` votes. `new_word` fires once per acceptance until reset() (hands left the frame).
    """

    def __init__(self, threshold=config.CONFIDENCE_THRESHOLD, window=config.SMOOTHING_WINDOW,
                 min_agreement=config.MIN_AGREEMENT):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        if window < 1 or not 1 <= min_agreement <= window:
            raise ValueError("Need window >= 1 and 1 <= min_agreement <= window.")
        self.threshold = threshold
        self.min_agreement = min_agreement
        self.votes = deque(maxlen=window)
        self.last_emitted = None

    def update(self, idx: int, conf: float):
        """Returns (accepted_idx | None, mean_confidence, new_word)."""
        self.votes.append((idx, conf) if conf >= self.threshold else (None, conf))
        counts = Counter(i for i, _ in self.votes if i is not None)
        if not counts:
            return None, 0.0, False
        best, n = counts.most_common(1)[0]
        if n < self.min_agreement:
            return None, 0.0, False
        mean_conf = float(np.mean([c for i, c in self.votes if i == best]))
        new = best != self.last_emitted
        if new:
            self.last_emitted = best
        return best, mean_conf, new

    def reset(self):
        self.votes.clear()
        self.last_emitted = None


class SignPredictor:
    """One instance per live session (it holds that session's frame buffer)."""

    def __init__(self, threshold=config.CONFIDENCE_THRESHOLD, window=config.SMOOTHING_WINDOW,
                 min_agreement=config.MIN_AGREEMENT, predict_every=config.PREDICT_EVERY):
        from .model import load_trained
        self.model, meta = load_trained()          # raises ModelNotFoundError with a clear message
        self.labels = meta["labels"]
        self.feature_dim = int(meta["feature_dim"])
        self.seq_len = int(meta["sequence_length"])
        if self.seq_len != config.SEQUENCE_LENGTH:
            raise ModelNotFoundError(f"Model expects {self.seq_len}-frame sequences but config says "
                                     f"{config.SEQUENCE_LENGTH}. Retrain the model.")
        self.predict_every = max(1, int(predict_every))
        self.smoother = PredictionSmoother(threshold, window, min_agreement)
        self.buffer = deque(maxlen=self.seq_len)
        self._since_predict = 0
        self._idle = 0
        self._last = Prediction()

    def set_threshold(self, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        self.smoother.threshold = threshold

    def reset(self) -> None:
        self.buffer.clear()
        self.smoother.reset()
        self._since_predict = 0
        self._idle = 0
        self._last = Prediction()

    def push(self, features: np.ndarray, hands_visible: bool = True) -> Prediction:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.feature_dim,):
            raise ValueError(f"Feature vector has shape {features.shape}, model expects ({self.feature_dim},). "
                             "The model was trained with a different landmark format - retrain it.")
        self.buffer.append(features)           # always append, to keep the window temporally continuous

        if not hands_visible:
            self._idle += 1
            if self._idle >= config.RESET_AFTER_IDLE:   # signer is resting: start fresh for the next sign
                self.reset()
            return Prediction(frames=len(self.buffer))
        self._idle = 0

        if len(self.buffer) < self.seq_len:
            return Prediction(frames=len(self.buffer))
        self._since_predict += 1
        if self._since_predict < self.predict_every:
            return replace(self._last, new_word=False, frames=len(self.buffer))
        self._since_predict = 0

        probs = self.model(np.asarray(self.buffer, dtype=np.float32)[None], training=False).numpy()[0]
        order = np.argsort(probs)[::-1]
        idx, conf = int(order[0]), float(probs[order[0]])
        stable, stable_conf, new = self.smoother.update(idx, conf)
        self._last = Prediction(
            label=self.labels[stable] if stable is not None else None,
            confidence=stable_conf,
            raw_label=self.labels[idx], raw_confidence=conf,
            top=[(self.labels[int(i)], float(probs[i])) for i in order[:3]],
            new_word=new, frames=len(self.buffer))
        return self._last
