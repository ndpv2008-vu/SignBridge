"""SignBridge - Gradio web app.  Run:  python app.py   (then open the printed URL)

Tabs: 1) Collect / import ISL data   2) Train the LSTM   3) Live translation (text + speech).
The webcam is the *viewer's browser camera* (gr.Image webcam), so it also works on cloud servers.
Each browser session gets its own Session object (MediaPipe instance, recording state, predictor, transcript).
"""
import math
import os
import time

try:
    import gradio as gr
    from signbridge.landmarks import LandmarkExtractor, draw_landmarks, hands_visible
except (ImportError, AttributeError) as e:   # AttributeError: MediaPipe without mp.solutions
    raise SystemExit(f"Startup failed: {e}\nInstall pinned dependencies (pip install -r requirements.txt) "
                     "and run `python check_env.py`.")

from signbridge import config, dataset, speech, train as trainer, video_import
from signbridge.model import ModelNotFoundError, model_exists
from signbridge.predictor import SignPredictor

SEQ = config.SEQUENCE_LENGTH
COUNTDOWN_S = 3
MIN_HAND_FRAMES = 10          # a recorded sample must show a hand in at least this many of its 30 frames
STREAM_EVERY_S = 0.1          # requested frame interval; real rate depends on server CPU/network


# ----------------------------------------------------------------------------- session
class Session:
    """Per-browser state. MediaPipe is created lazily on the first frame."""

    def __init__(self):
        self._extractor = None
        self.rec = None          # active recording: {"label", "start", "frames", "hands"}
        self.predictor = None    # SignPredictor once the model is loaded
        self.words = []

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = LandmarkExtractor()
        return self._extractor


def _sess(state) -> Session:
    return state if isinstance(state, Session) else Session()


def _nop():
    return gr.update()


def _count_rows():
    counts = dataset.class_counts()
    return [[k, v] for k, v in counts.items()] or [["(no samples yet)", 0]]


# ----------------------------------------------------------------------------- tab 1: collect
def start_record(label, state):
    s = _sess(state)
    try:
        lab = dataset.clean_label(label or "")
    except ValueError:
        return s, "⚠️ Enter a sign name first."
    s.rec = {"label": lab, "start": time.time() + COUNTDOWN_S, "frames": [], "hands": []}
    return s, f"Recording **{lab}** starts in {COUNTDOWN_S} s - get into position (camera must be on)."


def collect_stream(frame, state):
    s = _sess(state)
    if frame is None:
        return s, _nop(), _nop(), _nop()
    try:
        feats, res = s.extractor.process(frame)
        view = draw_landmarks(frame, res)
        hv = hands_visible(res)
        status, table = ("Camera on. Hands detected ✅" if hv else "Camera on. No hands detected ❌"), _nop()
        rec = s.rec
        if rec:
            wait = rec["start"] - time.time()
            if wait > 0:
                status = f"⏳ Get ready... {math.ceil(wait)}"
            else:
                rec["frames"].append(feats)
                rec["hands"].append(hv)
                n = len(rec["frames"])
                status = f"🔴 Recording {n}/{SEQ}"
                if n >= SEQ:
                    s.rec = None
                    seen = sum(rec["hands"])
                    if seen < MIN_HAND_FRAMES:
                        status = (f"❌ Discarded: a hand was visible in only {seen} of {SEQ} frames "
                                  f"(need {MIN_HAND_FRAMES}). Try again.")
                    else:
                        dataset.save_sequence(rec["label"], rec["frames"])
                        total = dataset.class_counts().get(rec["label"], 0)
                        status = f"✅ Saved sample #{total} for **{rec['label']}**."
                        table = _count_rows()
        return s, view, status, table
    except Exception as e:  # noqa: BLE001 - keep the stream alive, show the problem
        s.rec = None
        return s, _nop(), f"⚠️ Error: {e}", _nop()


def run_import(progress=gr.Progress()):
    try:
        r = video_import.import_videos(progress_cb=lambda d, t, p: progress(d / t, desc=f"Video {d}/{t}"))
    except video_import.VideoImportError as e:
        return f"⚠️ {e}", _count_rows()
    msg = (f"Imported **{r['imported']}** videos, skipped {r['skipped_already_imported']} already imported, "
           f"{len(r['failed'])} failed.")
    if r["failed"]:
        msg += "\n\n" + "\n".join(f"- `{p}`: {why}" for p, why in r["failed"][:20])
    return msg, _count_rows()


def delete_label(label, confirm):
    if not confirm:
        return "Tick the confirmation box first.", _count_rows()
    try:
        n = dataset.delete_label(label or "")
    except ValueError:
        return "⚠️ Enter the sign name to delete.", _count_rows()
    return f"Deleted {n} samples of '{label}'.", _count_rows()


# ----------------------------------------------------------------------------- tab 2: train
def _format_report(rep):
    sizes = rep["split_sizes"]
    md = (f"### Held-out test accuracy: **{rep['test_accuracy'] * 100:.1f}%** "
          f"({sizes['test']} test samples)\n"
          f"Trained on {sizes['train']} samples, validated on {sizes['validation']}, "
          f"{rep['epochs_run']} epochs run, trained at {rep['trained_at']}.\n\n")
    if sizes["test"] < 30:
        md += "⚠️ The test set is very small, so this number is a rough estimate.\n\n"
    md += f"_{rep['note']}_\n\nAfter training, press **Load model** in the Live tab to use the new model."
    rows = [[lab, round(rep["per_label"][lab]["precision"], 2), round(rep["per_label"][lab]["recall"], 2),
             round(rep["per_label"][lab]["f1-score"], 2), int(rep["per_label"][lab]["support"])]
            for lab in rep["labels"]]
    return md, rows


def run_training(epochs, batch_size, progress=gr.Progress()):
    try:
        rep = trainer.train(int(epochs), int(batch_size), progress_cb=lambda e, t, l: progress(
            e / t, desc=f"Epoch {e}/{t} - val acc {l.get('val_accuracy', 0):.2f}"))
    except trainer.TrainingError as e:
        return f"⚠️ {e}", [], _count_rows()
    md, rows = _format_report(rep)
    return md, rows, _count_rows()


def load_last_report():
    rep = trainer.load_report()
    if rep is None:
        return "No trained model yet.", []
    return _format_report(rep)


# ----------------------------------------------------------------------------- tab 3: live
def load_model(state, threshold, window):
    s = _sess(state)
    window = int(window)
    try:
        s.predictor = SignPredictor(threshold=threshold, window=window, min_agreement=max(1, math.ceil(window * 0.6)))
    except (ModelNotFoundError, ValueError) as e:
        s.predictor = None
        return s, f"⚠️ {e}"
    s.words = []
    return s, f"✅ Model loaded with {len(s.predictor.labels)} signs: {', '.join(s.predictor.labels)}. Start the camera and sign."


def set_threshold(state, value):
    s = _sess(state)
    if s.predictor is not None:
        s.predictor.set_threshold(float(value))


def live_stream(frame, state, speak_on, lang_name, show_landmarks):
    s = _sess(state)
    if frame is None:
        return s, _nop(), _nop(), _nop(), _nop()
    if s.predictor is None:
        return s, _nop(), "⚠️ Press **Load model** first.", _nop(), _nop()
    try:
        feats, res = s.extractor.process(frame)
        hv = hands_visible(res)
        p = s.predictor.push(feats, hv)
    except Exception as e:  # noqa: BLE001
        return s, _nop(), f"⚠️ Error: {e}", _nop(), _nop()

    view = draw_landmarks(frame, res) if show_landmarks else _nop()
    transcript, audio, note = _nop(), _nop(), ""
    if p.new_word:
        s.words.append(p.label)
        transcript = " ".join(w.replace("_", " ") for w in s.words)
        if speak_on:
            try:
                audio = speech.speak_to_file(p.label, speech.LANGUAGES[lang_name])
            except speech.SpeechError as e:
                note = f"\n\n🔇 {e}"

    lines = [f"**Window:** {p.frames}/{SEQ} frames · **Hands:** {'✅' if hv else '❌'}"]
    if p.top:
        lines.append("**Top 3:** " + " · ".join(f"{l.replace('_', ' ')} {c * 100:.0f}%" for l, c in p.top))
        if p.label:
            lines.append(f"### 🟢 {p.label.replace('_', ' ')} ({p.confidence * 100:.0f}%)")
        else:
            lines.append(f"Below threshold / not stable yet (threshold {s.predictor.smoother.threshold * 100:.0f}%)")
    elif p.frames < SEQ:
        lines.append("Filling the 30-frame window...")
    return s, view, "\n\n".join(lines) + note, transcript, audio


def clear_transcript(state):
    s = _sess(state)
    s.words = []
    if s.predictor is not None:
        s.predictor.reset()
    return s, ""


def speak_text(text, lang_name):
    try:
        return speech.speak_to_file(text, speech.LANGUAGES[lang_name]), ""
    except speech.SpeechError as e:
        return None, f"🔇 {e}"


# ----------------------------------------------------------------------------- UI
CSS = ".gradio-container {max-width: 1150px !important; margin: auto;} #title {text-align:center;}"


def build_ui():
    with gr.Blocks(title="SignBridge", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        state = gr.State(None)
        gr.Markdown("# 🤟 SignBridge\nReal-time **Indian Sign Language** to text and speech. "
                    "Collect or import samples → train → translate live.", elem_id="title")

        with gr.Tabs():
            # ---------------- collect
            with gr.Tab("1 · Collect data"):
                with gr.Row():
                    with gr.Column(min_width=320):
                        cam_c = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                                         mirror_webcam=False, label="Webcam")
                        label_in = gr.Textbox(label="Sign name (one word/phrase, e.g. hello, thank_you)")
                        rec_btn = gr.Button("🔴 Record one sample (3 s countdown, then 30 frames)", variant="primary")
                        status_c = gr.Markdown("Allow camera access, then enter a sign name and press Record.")
                    with gr.Column(min_width=320):
                        view_c = gr.Image(label="What the model sees (landmarks)", interactive=False)
                        counts_df = gr.Dataframe(headers=["Sign", "Samples"], value=_count_rows(),
                                                 interactive=False, label=f"Samples per sign "
                                                 f"(need ≥ {config.MIN_SAMPLES_PER_CLASS} each)")
                with gr.Accordion("Import ISL videos (data/videos/<sign>/*.mp4)", open=False):
                    gr.Markdown("Use a genuine ISL dataset (e.g. AI4Bharat INCLUDE; check its licence). "
                                "**Do not use WLASL: it is American Sign Language.**")
                    imp_btn = gr.Button("Import videos")
                    imp_out = gr.Markdown()
                with gr.Accordion("Delete a sign's samples", open=False):
                    del_label = gr.Textbox(label="Sign name")
                    del_ok = gr.Checkbox(label="I am sure")
                    del_btn = gr.Button("Delete", variant="stop")
                    del_out = gr.Markdown()

            # ---------------- train
            with gr.Tab("2 · Train"):
                with gr.Row():
                    epochs = gr.Slider(10, 300, value=100, step=10, label="Max epochs (early stopping is on)")
                    batch = gr.Slider(4, 64, value=16, step=4, label="Batch size")
                train_btn = gr.Button("🧠 Train LSTM", variant="primary")
                train_md = gr.Markdown()
                train_tbl = gr.Dataframe(headers=["Sign", "Precision", "Recall", "F1", "Test samples"],
                                         interactive=False, label="Per-sign results on the held-out test set")

            # ---------------- live
            with gr.Tab("3 · Live translate"):
                with gr.Row():
                    with gr.Column(min_width=320):
                        cam_l = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                                         mirror_webcam=False, label="Webcam")
                        with gr.Row():
                            load_btn = gr.Button("Load model", variant="primary")
                            clear_btn = gr.Button("Clear transcript")
                        thr = gr.Slider(0.30, 0.99, value=config.CONFIDENCE_THRESHOLD, step=0.01,
                                        label="Confidence threshold")
                        win = gr.Slider(1, 10, value=config.SMOOTHING_WINDOW, step=1,
                                        label="Smoothing window (applied on Load model; ~60% must agree)")
                        with gr.Row():
                            speak_cb = gr.Checkbox(value=True, label="Speak each new word")
                            lang = gr.Dropdown(list(speech.LANGUAGES), value="English", label="Speech language")
                        land_cb = gr.Checkbox(value=True, label="Show landmarks")
                    with gr.Column(min_width=320):
                        view_l = gr.Image(label="Landmarks", interactive=False)
                        status_l = gr.Markdown("Train a model, then press **Load model**." if not model_exists()
                                               else "Press **Load model**, then start the camera.")
                        transcript = gr.Textbox(label="Recognised text", lines=3, interactive=True)
                        speak_btn = gr.Button("🔊 Speak transcript")
                        audio = gr.Audio(label="Speech", type="filepath", autoplay=True)

            with gr.Tab("About"):
                gr.Markdown(
                    "- Landmarks: MediaPipe Holistic (pose, face contour, both hands, 3D), normalised to the shoulders.\n"
                    "- Classifier: LSTM on 30-frame sequences. It only knows the signs **you** collected/imported.\n"
                    "- Reported accuracy is measured on held-out samples of your own data and is optimistic if one "
                    "person recorded everything. Test live with other people/lighting before trusting it.\n"
                    "- Collect and test on the same deployment: frame rate depends on the server, and the model "
                    "sees consecutive frames.\n"
                    "- Camera frames are sent to the server running this app for processing. Speech uses gTTS "
                    "(Google's service), which sends the recognised word text to Google.")

        # ---- wiring
        demo.load(lambda: Session(), None, state)
        demo.load(load_last_report, None, [train_md, train_tbl])

        rec_btn.click(start_record, [label_in, state], [state, status_c])
        cam_c.stream(collect_stream, [cam_c, state], [state, view_c, status_c, counts_df],
                     stream_every=STREAM_EVERY_S, time_limit=900, show_progress="hidden")
        imp_btn.click(run_import, None, [imp_out, counts_df])
        del_btn.click(delete_label, [del_label, del_ok], [del_out, counts_df])

        train_btn.click(run_training, [epochs, batch], [train_md, train_tbl, counts_df])

        load_btn.click(load_model, [state, thr, win], [state, status_l])
        thr.change(set_threshold, [state, thr], None)
        cam_l.stream(live_stream, [cam_l, state, speak_cb, lang, land_cb],
                     [state, view_l, status_l, transcript, audio],
                     stream_every=STREAM_EVERY_S, time_limit=900, show_progress="hidden")
        clear_btn.click(clear_transcript, [state], [state, transcript])
        speak_btn.click(speak_text, [transcript, lang], [audio, status_l])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name=os.environ.get("HOST", "0.0.0.0"),
                              server_port=int(os.environ.get("PORT", "7860")),
                              share=os.environ.get("SIGNBRIDGE_SHARE") == "1")
build_ui().queue().launch(
    server_name=os.environ.get("HOST", "0.0.0.0"),
    server_port=int(os.environ.get("PORT", "7860"))
)