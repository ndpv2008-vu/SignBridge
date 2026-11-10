"""MediaPipe Holistic wrapper: turns one RGB frame into one fixed-length feature vector.

Feature layout per frame (all 3D, normalised):
    pose  : 33 x (x, y, z, visibility)
    face  : contour landmarks only (eyes, brows, lips, face oval) x (x, y, z)
    hands : left 21 x (x, y, z) + right 21 x (x, y, z)
Missing body parts (e.g. a hand out of frame) are filled with zeros.

Requires the legacy `mp.solutions` API, which is why requirements.txt pins
mediapipe==0.10.14 (newer releases removed `mp.solutions`).
"""
import numpy as np
import mediapipe as mp
from . import config

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Contour subset of the 468 face-mesh points: keeps expression cues (non-manual
# markers matter in ISL) while shrinking the input from 1404 to a few hundred values.
FACE_IDX = np.array(sorted({i for pair in mp.solutions.face_mesh.FACEMESH_CONTOURS for i in pair}))

N_POSE, N_HAND = 33, 21
FEATURE_DIM = N_POSE * 4 + len(FACE_IDX) * 3 + 2 * N_HAND * 3

_L_SHOULDER, _R_SHOULDER = 11, 12


def _xyz(landmark_list, n):
    """NormalizedLandmarkList -> (n, 3) array, or None if the part was not detected."""
    if landmark_list is None:
        return None
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], dtype=np.float32)


def results_to_features(results) -> np.ndarray:
    """Convert a MediaPipe Holistic result into a (FEATURE_DIM,) float32 vector.

    Coordinates are made translation- and scale-invariant: origin = mid-shoulder,
    unit = shoulder width. If the pose is not detected, raw coordinates are used.
    """
    pose = None
    if results.pose_landmarks is not None:
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
                        dtype=np.float32)
    face = _xyz(results.face_landmarks, 468)
    lh = _xyz(results.left_hand_landmarks, N_HAND)
    rh = _xyz(results.right_hand_landmarks, N_HAND)

    origin, scale = np.zeros(3, np.float32), 1.0
    if pose is not None:
        ls, rs = pose[_L_SHOULDER, :3], pose[_R_SHOULDER, :3]
        width = float(np.linalg.norm(ls - rs))
        if width > 1e-3:
            origin, scale = (ls + rs) / 2.0, width

    def norm(a):
        return (a - origin) / scale

    pose_f = np.zeros((N_POSE, 4), np.float32)
    if pose is not None:
        pose_f[:, :3] = norm(pose[:, :3])
        pose_f[:, 3] = pose[:, 3]
    face_f = norm(face[FACE_IDX]) if face is not None else np.zeros((len(FACE_IDX), 3), np.float32)
    lh_f = norm(lh) if lh is not None else np.zeros((N_HAND, 3), np.float32)
    rh_f = norm(rh) if rh is not None else np.zeros((N_HAND, 3), np.float32)

    return np.concatenate([pose_f.ravel(), face_f.ravel(), lh_f.ravel(), rh_f.ravel()]).astype(np.float32)


def hands_visible(results) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


class LandmarkExtractor:
    """Holds one Holistic instance. Not thread-safe: create one per stream/session."""

    def __init__(self, static_image_mode=False, model_complexity=None,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=config.HOLISTIC_COMPLEXITY if model_complexity is None else model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray):
        """rgb_frame: HxWx3 uint8 RGB. Returns (features, raw_results)."""
        rgb_frame = np.ascontiguousarray(rgb_frame)
        rgb_frame.flags.writeable = False
        results = self._holistic.process(rgb_frame)
        return results_to_features(results), results

    def close(self):
        self._holistic.close()


def draw_landmarks(rgb_frame: np.ndarray, results) -> np.ndarray:
    """Return a copy of the frame with pose + hand + face-contour landmarks drawn."""
    out = rgb_frame.copy()
    mp_drawing.draw_landmarks(out, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                              None, mp_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(out, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(out, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return out
