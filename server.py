"""
Murf AI Translation + Speech Suite — Flask Backend
═══════════════════════════════════════════════════════════════════════════════
REST API that wraps Murf's Translation and TTS APIs.
All heavy lifting (retry logic, chunking, merging) lives here so the
frontend never touches the API key.

Endpoints:
    GET  /api/health          → { "status": "ok" }
    GET  /api/voices          → available locales & voices
    POST /api/translate-file  → upload .txt, get translated text + merged MP3
    POST /api/translate-chat  → single message → Gemini AI reply → translate + TTS
"""

import os, pathlib, tempfile, time, json, base64
from typing import List
from dataclasses import dataclass, field

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pydub import AudioSegment
import httpx

# ──────────────────────────── CONFIG ────────────────────────────────────────
API_KEY          = os.getenv("MURF_KEY", "").strip()
GEMINI_KEY       = os.getenv("GEMINI_KEY", "").strip()
HTTP_TIMEOUT     = 180
MAX_RETRIES      = 3
BACKOFF_SECS     = 2

LOCALES = {
    "Hindi":       "hi-IN",
    "English US":  "en-US",
    "French":      "fr-FR",
    "German":      "de-DE",
}

VOICES = {
    "Hindi - Amit (M)":    "hi-IN-amit",
    "Hindi - Ayushi (F)":  "hi-IN-ayushi",
    "English - Ken (M)":   "en-US-ken",
    "English - Natalie":   "en-US-natalie",
    "French - Adélie":     "fr-FR-adélie",
    "French - Justine":    "fr-FR-justine",
    "German - Ruby":       "en-UK-ruby",
    "German - Lia":        "de-DE-lia",
}

SENT_LIMIT_TXT   = 10
CHAR_LIMIT_TXT   = 4_000
CHAR_LIMIT_TTS   = 3_000
WORKSPACE_BUDGET  = 100_000

# ──────────────────────────── FLASK APP ─────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ──────────────────────────── Murf Client (lazy init) ──────────────────────
_murf_client = None

def _get_client():
    """Lazy-init so the import doesn't crash when MURF_KEY is missing."""
    global _murf_client
    if _murf_client is None:
        from murf import Murf
        _murf_client = Murf(api_key=API_KEY, timeout=HTTP_TIMEOUT)
    return _murf_client


# ───────── Retry helper ────────────────────────────────────────────────────
def _retry(fn, *a, **kw):
    from murf.core.api_error import ApiError
    kw.setdefault("request_options", {"timeout": HTTP_TIMEOUT})
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*a, **kw)
        except (httpx.ReadTimeout,) as exc:
            last_exc = exc
        except ApiError as exc:
            last_exc = exc
            if not (500 <= exc.status_code < 600):
                raise
        if attempt == MAX_RETRIES:
            raise last_exc
        time.sleep(BACKOFF_SECS * 2 ** (attempt - 1))


# ───────── TEXT → TEXT helpers ──────────────────────────────────────────────
def _split_lines(txt: str) -> List[str]:
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def _wrap_tx(line: str):
    if len(line) <= CHAR_LIMIT_TXT:
        yield line
    else:
        for i in range(0, len(line), CHAR_LIMIT_TXT):
            yield line[i : i + CHAR_LIMIT_TXT]


def _batched(lines):
    buf = []
    for ln in lines:
        for p in _wrap_tx(ln):
            buf.append(p)
            if len(buf) == SENT_LIMIT_TXT:
                yield buf
                buf = []
    if buf:
        yield buf


def translate_lines(text: str, locale: str) -> List[str]:
    client = _get_client()
    out = []
    for batch in _batched(_split_lines(text)):
        r = _retry(client.text.translate, target_language=locale, texts=batch)
        out.extend(t.translated_text or "" for t in r.translations)
    return out


def translate_single(msg: str, locale: str) -> str:
    client = _get_client()
    r = _retry(client.text.translate, target_language=locale, texts=[msg])
    return r.translations[0].translated_text or ""


# ───────── TEXT → SPEECH helpers ────────────────────────────────────────────
def tts_single(text: str, voice_id: str) -> str:
    """Synthesise ≤3 000-char text and return presigned MP3 URL."""
    client = _get_client()
    if len(text) > CHAR_LIMIT_TTS:
        text = text[:CHAR_LIMIT_TTS]
    r = _retry(
        client.text_to_speech.generate,
        text=text,
        voice_id=voice_id,
        format="MP3",
        sample_rate=44_100.0,
    )
    return r.audio_file


# ─────────────────── FILE → SPEECH PIPELINE ─────────────────────────────────
@dataclass
class Segment:
    idx: int
    text: str
    url: str | None = None


def build_segments(lines: List[str]) -> List[Segment]:
    segs, buf = [], ""
    for line in lines:
        while len(line) > CHAR_LIMIT_TTS:
            segs.append(Segment(len(segs), line[:CHAR_LIMIT_TTS]))
            line = line[CHAR_LIMIT_TTS:]
        if len(buf) + len(line) + 1 > CHAR_LIMIT_TTS:
            if buf:
                segs.append(Segment(len(segs), buf))
            buf = line
        else:
            buf = f"{buf} {line}".strip() if buf else line
    if buf:
        segs.append(Segment(len(segs), buf))
    return segs


def tts_fill(segs: List[Segment], voice_id: str):
    spent = 0
    for s in segs:
        if spent + len(s.text) > WORKSPACE_BUDGET:
            break
        s.url = tts_single(s.text, voice_id)
        spent += len(s.text)


def merge_mp3(segs: List[Segment]) -> str:
    ordered = [s for s in sorted(segs, key=lambda s: s.idx) if s.url]
    if not ordered:
        raise ValueError("No audio segments were generated")

    tmpdir = tempfile.mkdtemp(prefix="murf_")
    pieces = []

    with httpx.Client(timeout=HTTP_TIMEOUT) as dl:
        for s in ordered:
            part_path = pathlib.Path(tmpdir) / f"seg_{s.idx}.mp3"
            part_path.write_bytes(dl.get(s.url).content)
            pieces.append(AudioSegment.from_file(part_path, format="mp3"))

    merged = pieces[0]
    for p in pieces[1:]:
        merged += p

    out = pathlib.Path(tmpdir) / "narration.mp3"
    merged.export(out, format="mp3")
    return str(out)


# ═══════════════════════════ API ROUTES ═════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "murf_key_set": bool(API_KEY), "gemini_key_set": bool(GEMINI_KEY)})


@app.route("/api/voices")
def voices():
    return jsonify({
        "locales": LOCALES,
        "voices": VOICES,
    })


@app.route("/api/translate-file", methods=["POST"])
def translate_file():
    """Accept .txt upload → translate → TTS → return JSON with text + audio."""
    if not API_KEY:
        return jsonify({"error": "MURF_KEY not configured on server"}), 500

    uploaded = request.files.get("file")
    target_lang = request.form.get("target_lang", "Hindi")
    voice = request.form.get("voice", "Hindi - Amit (M)")

    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        raw_text = uploaded.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "File must be UTF-8 encoded text"}), 400

    if target_lang not in LOCALES:
        return jsonify({"error": f"Unknown locale: {target_lang}"}), 400
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice: {voice}"}), 400

    try:
        # Translate
        lines_tgt = translate_lines(raw_text, LOCALES[target_lang])
        translated_text = "\n".join(lines_tgt)

        # TTS + merge
        segs = build_segments(lines_tgt)
        tts_fill(segs, VOICES[voice])
        mp3_path = merge_mp3(segs)

        # Read and base64-encode the MP3 for JSON transport
        with open(mp3_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        return jsonify({
            "translated_text": translated_text,
            "audio_base64": audio_b64,
            "audio_mime": "audio/mpeg",
            "segments": len(segs),
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ───────── Gemini AI chatbot helper ───────────────────────────────────────
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def gemini_chat(message: str, history: list) -> str:
    """Send message + history to Gemini and return the AI reply.
    Retries up to 3 times on 429 rate-limit errors."""
    # Build conversation contents from history
    contents = []
    for h in history[-10:]:  # keep last 10 turns to stay within limits
        contents.append({"role": "user", "parts": [{"text": h["user"]}]})
        contents.append({"role": "model", "parts": [{"text": h["bot"]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": (
                "You are a friendly, helpful AI assistant in the Murf AI Suite. "
                "Keep your replies concise (1-3 sentences). Be conversational and helpful. "
                "You can discuss any topic. Reply in English; the system will translate for the user."
            )}]
        },
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 0.8,
        }
    }

    last_err = None
    for attempt in range(1, 4):
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                GEMINI_URL,
                params={"key": GEMINI_KEY},
                json=payload,
            )
            if resp.status_code == 429:
                last_err = resp.text
                time.sleep(2 ** attempt)  # 2s, 4s, 8s backoff
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    raise RuntimeError(f"Gemini rate-limited after 3 retries. Try again in a minute. Details: {last_err[:200]}")


@app.route("/api/translate-chat", methods=["POST"])
def translate_chat():
    """User msg -> Gemini AI reply -> Murf translate -> Murf TTS."""
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    target_lang = data.get("target_lang", "Hindi")
    voice = data.get("voice", "Hindi - Amit (M)")
    history = data.get("history", [])  # list of {"user": ..., "bot": ...}

    if not message:
        return jsonify({"error": "Empty message"}), 400
    if target_lang not in LOCALES:
        return jsonify({"error": f"Unknown locale: {target_lang}"}), 400
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice: {voice}"}), 400

    try:
        # Step 1: Get AI reply from Gemini
        if GEMINI_KEY:
            ai_reply = gemini_chat(message, history)
        else:
            ai_reply = f"(Gemini not configured) You said: {message}"

        # Step 2: Translate the AI reply with Murf
        translated = ""
        if API_KEY:
            translated = translate_single(ai_reply, LOCALES[target_lang])
        else:
            translated = ai_reply  # fallback: no translation

        # Step 3: TTS on the translated text
        audio_url = ""
        if API_KEY:
            audio_url = tts_single(translated, VOICES[voice])

        return jsonify({
            "original": message,
            "ai_reply": ai_reply,
            "translated": translated,
            "audio_url": audio_url,
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ═══════════════════════════ MAIN ══════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"\n[*] Murf AI Suite running at http://localhost:{port}")
    print(f"    MURF_KEY set: {bool(API_KEY)}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
