import io
import re
import wave

FILLERS = ["um", "umm", "uh", "uhh", "er", "like", "basically", "actually",
           "literally", "you know", "i mean", "kind of", "sort of"]


def wav_duration(data: bytes):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def analyze(transcript: str, seconds):
    """Word count, words-per-minute and filler-word breakdown."""
    text = transcript.lower()
    words = re.findall(r"[a-z']+", text)
    breakdown = {}
    for f in FILLERS:
        n = len(re.findall(r"\b" + re.escape(f) + r"\b", text))
        if n:
            breakdown[f] = n
    wpm = round(len(words) / (seconds / 60), 1) if seconds and seconds > 0.5 else None
    if wpm is None:
        label = "n/a"
    elif wpm < 110:
        label = "Slow (aim for 120-160 wpm)"
    elif wpm > 170:
        label = "Too fast (aim for 120-160 wpm)"
    else:
        label = "Good pace"
    return {"words": len(words), "wpm": wpm, "fillers": sum(breakdown.values()),
            "filler_breakdown": breakdown, "pace_label": label}


def tts(text: str):
    """Text -> mp3 bytes via gTTS (needs internet). Returns None on failure."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text, lang="en", tld="co.in").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None
