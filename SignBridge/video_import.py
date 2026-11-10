"""Turn labelled sign videos into training sequences.

Expected layout:   data/videos/<sign_name>/<any_name>.mp4      (one folder per sign; folder name = label)
Use videos from a genuine ISL dataset (e.g. AI4Bharat INCLUDE - check its licence). WLASL is American Sign
Language and must not be used as ISL data.

For each video: landmarks are extracted from every frame, optionally trimmed to the span where a hand is
visible, then resampled to exactly SEQUENCE_LENGTH frames (evenly spaced) and saved via dataset.save_sequence.
Caveat: a resampled video spans the *whole* sign in 30 frames, while the live webcam window is 30 consecutive
frames, so signing speed/fps can differ between imported and live data. Mixing in some webcam-collected samples
of your own helps. Videos already imported are skipped on re-runs (tracked in data/imported_videos.json).

Usage:  python -m signbridge.video_import
"""
import json
import numpy as np
from . import config, dataset

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MIN_HAND_FRAMES = 10   # videos with fewer hand-visible frames are rejected


class VideoImportError(RuntimeError):
    """Raised for problems with the video folder as a whole."""


def resample(frames: np.ndarray, n: int = config.SEQUENCE_LENGTH) -> np.ndarray:
    """Pick n evenly spaced frames (repeating frames if the clip is shorter than n)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def video_to_sequence(path, trim_to_hands: bool = True, max_width: int = 640) -> np.ndarray:
    """Return a (SEQUENCE_LENGTH, FEATURE_DIM) array for one video. Raises ValueError on unusable videos."""
    import cv2
    from .landmarks import LandmarkExtractor, hands_visible

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    extractor = LandmarkExtractor()      # fresh instance per video: tracking state must not leak between clips
    feats, hands = [], []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if w > max_width:
                bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)))
            f, res = extractor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            feats.append(f)
            hands.append(hands_visible(res))
    finally:
        cap.release()
        extractor.close()

    if not feats:
        raise ValueError("no frames could be read")
    feats, hands = np.stack(feats), np.array(hands)
    if hands.sum() < MIN_HAND_FRAMES:
        raise ValueError(f"hands detected in only {int(hands.sum())} frames (need {MIN_HAND_FRAMES})")
    if trim_to_hands:
        visible = np.flatnonzero(hands)
        feats = feats[visible[0]: visible[-1] + 1]
    return resample(feats)


def _load_manifest() -> set:
    try:
        return set(json.loads(config.IMPORT_MANIFEST.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def import_videos(video_dir=config.VIDEO_DIR, trim_to_hands: bool = True, progress_cb=None) -> dict:
    """Import every new video under video_dir. progress_cb(done, total, path) is optional.

    Returns {"imported": n, "skipped_already_imported": n, "failed": [(path, reason)], "per_label": {...}}.
    """
    if not video_dir.exists():
        raise VideoImportError(f"Folder not found: {video_dir}. Create it as data/videos/<sign_name>/*.mp4")
    jobs = []
    for folder in sorted(p for p in video_dir.iterdir() if p.is_dir()):
        try:
            label = dataset.clean_label(folder.name)
        except ValueError:
            continue
        jobs += [(label, v) for v in sorted(folder.iterdir()) if v.suffix.lower() in VIDEO_EXTS]
    if not jobs:
        raise VideoImportError(f"No videos found under {video_dir} (supported: {', '.join(sorted(VIDEO_EXTS))}).")

    done_keys = _load_manifest()
    summary = {"imported": 0, "skipped_already_imported": 0, "failed": [], "per_label": {}}
    try:
        for i, (label, path) in enumerate(jobs, 1):
            key = f"{label}/{path.name}:{path.stat().st_size}"
            if progress_cb:
                progress_cb(i, len(jobs), str(path))
            if key in done_keys:
                summary["skipped_already_imported"] += 1
                continue
            try:
                dataset.save_sequence(label, video_to_sequence(path, trim_to_hands))
            except ValueError as e:
                summary["failed"].append((str(path), str(e)))
                continue
            done_keys.add(key)
            summary["imported"] += 1
            summary["per_label"][label] = summary["per_label"].get(label, 0) + 1
    finally:   # keep progress even if interrupted
        config.IMPORT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        config.IMPORT_MANIFEST.write_text(json.dumps(sorted(done_keys), indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        s = import_videos(progress_cb=lambda d, t, p: print(f"[{d}/{t}] {p}"))
    except VideoImportError as err:
        raise SystemExit(f"Import not started: {err}")
    print(f"\nImported {s['imported']}, already imported {s['skipped_already_imported']}, failed {len(s['failed'])}")
    for p, why in s["failed"]:
        print(f"  FAILED {p}: {why}")
