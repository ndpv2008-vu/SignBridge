"""Text-to-speech for recognised words.

Uses gTTS, which synthesises audio on Google's servers (needs outbound internet) and returns an mp3 file.
That fits cloud hosting, where server-side audio devices (pyttsx3) do not exist: Gradio plays the mp3 in
the user's browser. Results are cached per (language, text), so a repeated word is synthesised only once.
"""
import hashlib
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "signbridge_tts"

# Display name -> gTTS language code (all supported by gTTS).
LANGUAGES = {"English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta", "Kannada": "kn",
             "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu"}


class SpeechError(RuntimeError):
    """Raised when speech cannot be produced (empty text, gTTS missing, no internet, ...)."""


def speak_to_file(text: str, lang: str = "en") -> str:
    """Synthesise `text` and return the path of an mp3 file."""
    text = (text or "").replace("_", " ").strip()   # labels are stored with underscores
    if not text:
        raise SpeechError("Nothing to speak.")
    if lang not in LANGUAGES.values():
        raise SpeechError(f"Unsupported language code '{lang}'.")
    try:
        from gtts import gTTS
    except ImportError as e:
        raise SpeechError("gTTS is not installed. Run: pip install -r requirements.txt") from e

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{hashlib.sha1(f'{lang}|{text}'.encode('utf-8')).hexdigest()}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    try:
        gTTS(text=text, lang=lang).save(str(out))
    except Exception as e:  # noqa: BLE001 - gTTS raises network/HTTP/value errors of several types
        out.unlink(missing_ok=True)
        raise SpeechError(f"Speech synthesis failed ({e}). gTTS needs internet access.") from e
    return str(out)
