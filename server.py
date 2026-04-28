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
GROQ_KEY         = os.getenv("GROQ_KEY", "").strip()
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
    """Accept file upload → extract text → translate → optional TTS."""
    if not API_KEY:
        return jsonify({"error": "MURF_KEY not configured on server"}), 500

    uploaded = request.files.get("file")
    target_lang = request.form.get("target_lang", "Hindi")
    voice = request.form.get("voice", "Hindi - Amit (M)")
    enable_tts = request.form.get("enable_tts", "true") == "true"

    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400
    if target_lang not in LOCALES:
        return jsonify({"error": f"Unknown locale: {target_lang}"}), 400
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice: {voice}"}), 400

    filename = (uploaded.filename or "").lower()

    # ── Extract text from various formats ──────────────────────────────────
    try:
        raw_text = _extract_text(uploaded, filename)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not raw_text or not raw_text.strip():
        return jsonify({"error": "File appears to be empty"}), 400

    # Limit to first 50 lines to avoid huge Murf bills / timeouts
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) > 50:
        lines = lines[:50]
        raw_text = "\n".join(lines)

    try:
        # Step 1: Translate (always)
        lines_tgt = translate_lines(raw_text, LOCALES[target_lang])
        translated_text = "\n".join(lines_tgt)

        result = {
            "translated_text": translated_text,
            "audio_base64": "",
            "audio_mime": "audio/mpeg",
            "segments": 0,
            "lines_translated": len(lines_tgt),
        }

        # Step 2: TTS + merge (optional, best-effort)
        if enable_tts and lines_tgt:
            try:
                segs = build_segments(lines_tgt)
                tts_fill(segs, VOICES[voice])
                mp3_path = merge_mp3(segs)
                with open(mp3_path, "rb") as f:
                    result["audio_base64"] = base64.b64encode(f.read()).decode("ascii")
                result["segments"] = len(segs)
            except Exception as tts_err:
                result["tts_error"] = f"TTS failed: {tts_err} (translation is still available)"

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


def _extract_text(uploaded, filename: str) -> str:
    """Extract plain text from various file formats."""

    # Plain text formats
    if filename.endswith((".txt", ".md", ".csv", ".json", ".xml", ".log", ".py", ".js", ".html")):
        raw = uploaded.read()
        # Try UTF-8 first, then latin-1 as fallback
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    # PDF
    if filename.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            raise ValueError("PDF support not installed (PyPDF2)")

    # DOCX (Word)
    if filename.endswith(".docx"):
        try:
            import docx
            # Save to temp file for python-docx
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            tmp.write(uploaded.read())
            tmp.close()
            doc = docx.Document(tmp.name)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            os.unlink(tmp.name)
            return text
        except ImportError:
            raise ValueError("DOCX support not installed (python-docx)")

    raise ValueError(f"Unsupported file format: {filename.rsplit('.', 1)[-1]}. Supported: txt, md, csv, pdf, docx, json, xml, log")


# ───────── AI Chatbot helpers (Groq primary, Gemini fallback) ─────────────
SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant in the Murf AI Suite. "
    "Keep your replies concise (1-3 sentences). Be conversational and helpful. "
    "You can discuss any topic. Reply in English; the system will translate for the user."
)

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def groq_chat(message: str, history: list) -> str:
    """Chat via Groq (Llama 3.1 8B). Free: 30 req/min, 14400 req/day."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-10:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["bot"]})
    messages.append({"role": "user", "content": message})

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 256, "temperature": 0.8},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def gemini_chat(message: str, history: list) -> str:
    """Chat via Gemini. Free: 15 req/min."""
    contents = []
    for h in history[-10:]:
        contents.append({"role": "user", "parts": [{"text": h["user"]}]})
        contents.append({"role": "model", "parts": [{"text": h["bot"]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"maxOutputTokens": 256, "temperature": 0.8},
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(GEMINI_URL, params={"key": GEMINI_KEY}, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def ai_chat(message: str, history: list) -> str:
    """Try Groq first, then Gemini, then return a fallback."""
    errors = []
    if GROQ_KEY:
        try:
            return groq_chat(message, history)
        except Exception as e:
            errors.append(f"Groq: {e}")
    if GEMINI_KEY:
        try:
            return gemini_chat(message, history)
        except Exception as e:
            errors.append(f"Gemini: {e}")
    if errors:
        raise RuntimeError(" | ".join(errors))
    return f"(No AI key configured) You said: {message}"


@app.route("/api/translate-chat", methods=["POST"])
def translate_chat():
    """User msg -> AI reply -> Murf translate -> Murf TTS."""
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    target_lang = data.get("target_lang", "Hindi")
    voice = data.get("voice", "Hindi - Amit (M)")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400
    if target_lang not in LOCALES:
        return jsonify({"error": f"Unknown locale: {target_lang}"}), 400
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice: {voice}"}), 400

    try:
        # Step 1: Get AI reply
        ai_reply = ai_chat(message, history)

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
