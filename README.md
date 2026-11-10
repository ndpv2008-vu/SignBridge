# SignBridge – Real-Time Sign Language to Text and Speech Translator

Translates **Indian Sign Language (ISL)** gestures into text and speech in the browser.
Pipeline: webcam → MediaPipe Holistic landmarks → 30-frame sequences → LSTM → confidence
threshold + smoothing → text + speech.

> **Status: all code is written, but it has not yet been run end-to-end.** It was syntax-checked and its
> data/split/smoothing/error-handling logic was unit-tested, but TensorFlow, MediaPipe and Gradio were not
> available where it was written. Run `python check_env.py` first and expect to fix small version issues.
> No model ships with this repo: recognition only works for signs **you** collect or import and train.
> No accuracy figure is claimed; training reports a held-out test score computed on your data.

## Folder structure

```
signbridge/
├── app.py                  # Gradio UI: Collect / Train / Live translate tabs
├── check_env.py            # verifies the environment
├── requirements.txt
├── README.md
├── signbridge/
│   ├── config.py           # paths + hyperparameters
│   ├── landmarks.py        # MediaPipe Holistic -> normalised feature vector
│   ├── dataset.py          # save/load labelled .npy sequences
│   ├── model.py            # LSTM definition + save/load
│   ├── train.py            # train + held-out evaluation
│   ├── predictor.py        # sliding window, threshold, smoothing
│   ├── speech.py           # text-to-speech (gTTS)
│   └── video_import.py     # build sequences from an ISL video dataset
├── data/                   # sequences/<label>/<n>.npy and optional videos/
└── models/                 # trained model, labels.json, training_report.json
```

## ISL data

* **Collect your own** (works with no download): record 30-frame samples per sign in the app.
* **Use a public ISL dataset** by placing videos as `data/videos/<sign_name>/*.mp4` and importing
  them. One candidate is AI4Bharat's **INCLUDE** (Indian Sign Language word videos) – check its
  current licence and download page yourself before use.
* **WLASL is American Sign Language and is not used as ISL.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10 or 3.11 recommended
pip install -r requirements.txt
python check_env.py
```

If OpenCV raises a `libGL` error on a Linux server: `sudo apt-get install -y libgl1`.

## Running in the cloud: compatibility notes

| Feature | Cloud behaviour |
|---|---|
| Webcam | The server has no camera. Gradio's webcam component uses the **viewer's browser** camera, which works on any platform serving the page over HTTPS (Codespaces forwarded ports, Hugging Face Spaces, Colab share links). `cv2.VideoCapture(0)` will **not** work and is not used. |
| MediaPipe Holistic | Runs on CPU. Needs `mediapipe==0.10.14` (newer releases dropped `mp.solutions`; I confirmed this in my sandbox with 0.10.33). Live speed depends on CPU; use `model_complexity=0/1`. |
| LSTM | Small model; trains on CPU. A GPU (e.g. Colab) only helps for large datasets. |
| Speech | Server-side audio devices don't exist in the cloud, so `pyttsx3` is unsuitable. The app synthesises audio with **gTTS** (needs outbound internet) and plays it in the browser. |

Not yet verified on a real cloud runtime – run `python check_env.py` on your chosen platform first.

## Using the app

```bash
python app.py        # open http://localhost:7860  (set PORT / HOST / SIGNBRIDGE_SHARE=1 if needed)
```

1. **Collect data** tab: allow the camera, type a sign name, press *Record* (3 s countdown, then 30 frames).
   Record at least 10 samples per sign (more is better) with varied position, speed and, ideally, several signers.
   Or import ISL videos (`data/videos/<sign>/*.mp4`) from the accordion.
2. **Train** tab: press *Train LSTM*. It reports accuracy on a held-out test split of your data.
3. **Live translate** tab: press *Load model*, start the camera, and sign. A word is shown/spoken only when
   its confidence passes the threshold and enough recent predictions agree (smoothing). Lower hands to reset
   between signs.

Command-line alternatives: `python -m signbridge.video_import`, `python -m signbridge.train`.

## Platform notes (not tested by the author - verify on yours)

* **GitHub Codespaces** (good for development): choose a Python 3.10/3.11 image, run the setup commands,
  `python app.py`, open the forwarded port 7860 (served over HTTPS, which browsers require for camera access).
* **Hugging Face Spaces** (good for hosting): Gradio SDK, `app_file: app.py`, set `python_version: 3.10` in the
  Space's README header, and add a `packages.txt` containing `libgl1`. Free CPU hardware may be slow for live use.
* **Google Colab**: good for training (optional GPU). Colab's default Python may be too new for
  `mediapipe==0.10.14`; if pip finds no wheel, use Python 3.10/3.11. Run `SIGNBRIDGE_SHARE=1 python app.py`
  to get a public link (it exposes your app, so share it carefully).
* Frames travel from the browser to the server and back, so live speed depends on network and CPU. Collect and
  test on the same deployment so frame rates match. `HOLISTIC_COMPLEXITY` in `config.py` (0 = faster) must stay
  the same for collecting and recognising.
* Server memory grows with each open browser session (one MediaPipe instance each); this is a demo/project
  app, not a multi-user production service.

## Troubleshooting

| Problem | Fix |
|---|---|
| `mediapipe has no attribute 'solutions'` | `pip install mediapipe==0.10.14` |
| `libGL.so.1` error | `sudo apt-get install -y libgl1` |
| Camera does not start | Allow camera permission; the page must be HTTPS or localhost |
| "Model and labels do not match" / feature-size error | Retrain after changing landmarks or `HOLISTIC_COMPLEXITY` |
| No speech | gTTS needs internet; browsers may block autoplay until you click something |
| Predictions flicker or nothing is accepted | Raise/lower the threshold, collect more varied samples, check hands stay in frame |

## Limitations

* Recognises only isolated signs you trained (a fixed vocabulary), not continuous sentences or grammar.
* Accuracy depends on your data; test with people and lighting not seen in training.
* Not a certified accessibility or interpreting tool.
