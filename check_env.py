"""Run `python check_env.py` after installing. Verifies every dependency SignBridge needs
and runs MediaPipe Holistic on a blank frame. Prints PASS/FAIL per check; changes nothing."""
import sys

ok_all = True


def check(name, fn):
    global ok_all
    try:
        print(f"[PASS] {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        ok_all = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def py():
    v = sys.version_info
    if not (3, 9) <= (v.major, v.minor) <= (3, 12):
        raise RuntimeError(f"Python {v.major}.{v.minor} - use 3.10 or 3.11 for best compatibility")
    return f"{v.major}.{v.minor}.{v.micro}"


def numpy_v():
    import numpy
    if int(numpy.__version__.split(".")[0]) >= 2:
        raise RuntimeError(f"numpy {numpy.__version__}: mediapipe 0.10.14 needs numpy<2")
    return numpy.__version__


def mediapipe_v():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError(f"mediapipe {mp.__version__} has no mp.solutions; install mediapipe==0.10.14")
    return mp.__version__


def holistic():
    import numpy as np
    from signbridge.landmarks import LandmarkExtractor, FEATURE_DIM
    ex = LandmarkExtractor()
    feats, _ = ex.process(np.zeros((480, 640, 3), np.uint8))
    ex.close()
    assert feats.shape == (FEATURE_DIM,)
    return f"Holistic ran, feature size = {FEATURE_DIM}"


def tf_v():
    import tensorflow as tf
    return f"tensorflow {tf.__version__}, GPUs: {len(tf.config.list_physical_devices('GPU'))}"


def sk_v():
    import sklearn
    return sklearn.__version__


def gradio_v():
    import gradio
    return gradio.__version__


def gtts_v():
    import gtts  # TTS is network-based, so only the import is checked here
    return "gTTS importable (speech synthesis itself needs internet access)"


for n, f in [("Python", py), ("NumPy", numpy_v), ("MediaPipe", mediapipe_v), ("Holistic run", holistic),
             ("TensorFlow", tf_v), ("scikit-learn", sk_v), ("Gradio", gradio_v), ("gTTS", gtts_v)]:
    check(n, f)
print("\nAll checks passed." if ok_all else "\nSome checks failed - fix those before continuing.")
sys.exit(0 if ok_all else 1)
