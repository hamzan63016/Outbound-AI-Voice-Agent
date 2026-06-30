"""
MYTM Outbound Voice Agent — Voice Cloning Edition
===================================================
Drop next to front.html. Run:   python backend.py

Voice cloning:
  Place a clean 6–30 second WAV of the target voice at:
      reference_voice.wav   (same folder as this script)
  Coqui will clone that voice automatically.
  If reference_voice.wav is missing, Coqui uses its default voice.
  If Coqui fails entirely, falls back to edge-tts (Emma Neural).

Recording:
  Agent (cloned voice) + Customer (microphone) are captured
  separately and merged chronologically into one WAV file:
      recordings/call_session_<id>_<timestamp>.wav

Requirements (auto-installed):
  pip install flask pandas openai pydub SpeechRecognition PyAudio
  pip install RealtimeTTS edge-tts pygame
  pip install TTS   ← Coqui TTS (large download ~1 GB)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SELF-INSTALLER
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, subprocess, importlib, importlib.util

_REQUIRED = [
    ("flask",              "flask"),
    ("pandas",             "pandas"),
    ("openai",             "openai"),
    ("pydub",              "pydub"),
    ("speech_recognition", "SpeechRecognition"),
    ("pyaudio",            "PyAudio"),
    ("RealtimeTTS",        "RealtimeTTS"),
    ("edge_tts",           "edge-tts"),
    ("pygame",             "pygame"),
    ("TTS",                "TTS"),
    ("numpy",              "numpy"),
]

def _bootstrap():
    missing = [pip for mod, pip in _REQUIRED if not importlib.util.find_spec(mod)]
    if missing:
        print(f"[Setup] Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("[Setup] Done.\n")

if __name__ == "__main__":
    _bootstrap()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import time, threading, sqlite3, io, asyncio, smtplib, tempfile, wave, struct
import logging, functools
import pyaudio as _pyaudio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

import speech_recognition as sr
import pandas as pd
import edge_tts
import pygame
from openai import OpenAI
from flask import Flask, render_template, jsonify, request
from pydub import AudioSegment

# ── Torch thread cap ─────────────────────────────────────────────────────────
# XTTS-v2 uses PyTorch internally.  Leaving Torch at its default (all logical
# cores) causes kernel-level thread contention with the VAD/audio threads and
# actually makes CPU inference SLOWER.  4–6 threads is the sweet-spot for
# CPU-only XTTS on most desktop/laptop hardware.
try:
    import torch
    _TORCH_THREADS = int(os.environ.get("MYTM_TORCH_THREADS", "5"))
    torch.set_num_threads(_TORCH_THREADS)
    torch.set_num_interop_threads(2)   # inter-op parallelism (graph-level) can stay low
except ImportError:
    pass  # Torch not installed yet — will be available after Coqui installs it

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR  = os.path.join(SCRIPT_DIR, "templates")
DB_PATH        = os.path.join(SCRIPT_DIR, "mytm_terminal.db")
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
SUMMARIES_DIR  = os.path.join(SCRIPT_DIR, "summaries")
REFERENCE_WAV  = os.path.join(SCRIPT_DIR, "reference_voice.wav")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_LOG_FILE = os.path.join(LOGS_DIR, f"mytm_{datetime.now():%Y%m%d_%H%M%S}.log")

# Root logger — writes to both console and rotating log file
logger = logging.getLogger("MYTM")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d  [%(levelname)-5s]  %(message)s",
    datefmt="%H:%M:%S",
)

# Console handler (INFO and above — keeps terminal readable)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

# File handler (DEBUG and above — captures everything for review)
_fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

logger.info(f"Logger initialised → {_LOG_FILE}")


def _timed(fn):
    """Decorator: logs function entry/exit and elapsed wall-clock time."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        logger.debug(f"→ ENTER  {fn.__name__}()")
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"← EXIT   {fn.__name__}()  [{elapsed:.1f} ms]")
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(f"✗ ERROR  {fn.__name__}()  [{elapsed:.1f} ms]  — {exc}")
            raise
    return wrapper

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS  (loaded from environment or hardcoded fallback)
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY",
                      "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")
SENDER_EMAIL        = os.environ.get("SENDER_EMAIL",        "hamza.n63016@gmail.com")
SENDER_APP_PASSWORD = os.environ.get("SENDER_APP_PASSWORD", "rexi usjs wsqu ukhl")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
SAMPLE_WIDTH = 2       # 16-bit
CHANNELS     = 1

# ─────────────────────────────────────────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────────────────────────────────────────
pygame.mixer.pre_init(44100, -16, 2, 256)   # 256-sample buffer ≈ 5 ms latency (was 512 ≈ 11 ms)
pygame.mixer.init()

# ─────────────────────────────────────────────────────────────────────────────
# VOICE ENGINE SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Strategy:
#   1. Try CoquiEngine (voice cloning from reference_voice.wav)
#   2. Fall back to edge-tts (Emma Neural, Microsoft Azure, free)
#
# IMPORTANT: We never use RealtimeTTS streaming for playback because it gives
# no handle on the raw audio bytes needed for recording.  Instead we always
# synthesise to a WAV buffer first, record it, then play it via pygame.
# Coqui is used here purely as a TTS synthesiser into a temp file.

_USE_COQUI        = False   # flipped to True below if Coqui loads successfully
_coqui_tts        = None    # TTS object (TTS library, not RealtimeTTS wrapper)
_speaker_embedding = None   # cached gpt_cond_latent + speaker_embedding tensors

# _vad_mute_count: number of active XTTS synthesis jobs currently running.
# VAD is suppressed whenever this is > 0.  Using a counter (not a bool) means
# parallel pre-buffer threads can each increment/decrement independently without
# one thread prematurely re-enabling the VAD while others are still synthesising.
_vad_mute_count   = 0
_vad_mute_lock    = threading.Lock()

def _vad_muted() -> bool:
    """Return True if any XTTS synthesis thread is currently active."""
    with _vad_mute_lock:
        return _vad_mute_count > 0

def _vad_mute_inc():
    global _vad_mute_count
    with _vad_mute_lock:
        _vad_mute_count += 1

def _vad_mute_dec():
    global _vad_mute_count
    with _vad_mute_lock:
        _vad_mute_count = max(0, _vad_mute_count - 1)

# ─────────────────────────────────────────────────────────────────────────────
# PRE-CLONED PHRASE CACHE
# ─────────────────────────────────────────────────────────────────────────────
# Every agent utterance is broken into the smallest reusable static segments.
# ALL segments are voice-cloned once at startup and served from RAM during
# calls — zero XTTS synthesis latency at call-time.
#
# Dynamic phrases (greeting/email/payment/date) contain one variable token
# (customer name, email, balance, date).  We pre-clone the surrounding static
# parts and stitch them with a fast edge-tts render of the variable token
# (~80 ms) instead of running full XTTS on the whole sentence (27–56 s).
#
# HOW speak_parts() WORKS
#   speak_parts("Hello, I am calling for", name, "?")
#     → plays pre-cloned WAV of "Hello, I am calling for"
#     → plays edge-tts render of `name`          (≈80 ms)
#     → plays pre-cloned WAV of "?"  (short pause segment)
#   Total first-word latency: <100 ms vs 40 s with full XTTS.
#
# To add a new phrase: append every static segment to _STATIC_SEGMENTS.
# The variable token is never in this list — it is rendered live.

_FIXED_PHRASES: dict[str, bytes] = {}   # segment_text → WAV bytes (RAM cache)

_STATIC_SEGMENTS = [

    # ── GREETING (Step 1 — split around {name}) ──────────────────────────────
    "Hello, this is Hamza calling from MYTM. I hope you're doing well today. Am I speaking with",
    # variable: customer name
    "?",

    # ── IDENTITY NOT CONFIRMED ────────────────────────────────────────────────
    "Understood. Thanks for your time, have a great day.",

    # ── EMAIL VERIFICATION (Step 2 — split around {email}) ───────────────────
    "Thank you for confirming. For verification, is your registered email address",
    # variable: customer email
    # "?" already in list above

    # ── EMAIL MISMATCH CLOSING ────────────────────────────────────────────────
    "I apologize for the inconvenience. Our team will update the records. Thanks for your time, have a great day.",

    # ── PAYMENT REQUEST (Step 3 — split around {balance}) ────────────────────
    "Thank you. I am calling regarding your outstanding balance of",
    # variable: balance string  e.g. "5,000 rupees"
    ". Could you let me know when we can expect this payment?",

    # ── INSTALLMENT OFFER (Step 3 — fully static) ────────────────────────────
    "I understand. We can split this into two installments. On which date can you make your first payment?",

    # ── DATE COLLECTION (Step 3 — split around {date1}) ──────────────────────
    "I have noted your first installment for",
    # variable: date1
    ". On which date can you make your second and final payment?",

    # ── DATE INVALID / AUTO-ADJUSTED (split around {date2}) ──────────────────
    "Company policy requires the second installment within 30 days of the first. I have updated your second installment date to",
    # variable: date2
    ". Both dates are now logged. Thanks for your time, have a great day.",

    # ── INSTALLMENT CONFIRMED (split around {date1} then {date2}) ────────────
    "Perfect. Your first installment is set for",
    # variable: date1
    "and your second for",
    # variable: date2
    ". Both are logged in our system. Thanks for your time, have a great day.",

    # ── FULL PAYMENT CLOSING (split around {promised}) ────────────────────────
    "I have logged your promise to settle the full balance on",
    # variable: promised date
    ". Thanks for your time, have a great day.",

    # ── GENERIC CLOSING (standalone) ─────────────────────────────────────────
    "Thanks for your time, have a great day.",
]

# De-duplicate while preserving order
_seen = set()
_STATIC_SEGMENTS_DEDUPED = []
for _s in _STATIC_SEGMENTS:
    if _s not in _seen:
        _seen.add(_s)
        _STATIC_SEGMENTS_DEDUPED.append(_s)
_STATIC_SEGMENTS = _STATIC_SEGMENTS_DEDUPED


@_timed
def _init_voice_engine():
    global _USE_COQUI, _coqui_tts, _speaker_embedding, _FIXED_PHRASES

    if not os.path.exists(REFERENCE_WAV):
        logger.warning("[VoiceEngine] reference_voice.wav not found — Coqui skipped, using edge-tts.")
        return

    try:
        from TTS.api import TTS as CoquiTTS
        logger.info("[VoiceEngine] Loading Coqui XTTS-v2 model (first run downloads ~1 GB)…")
        t0 = time.perf_counter()
        _coqui_tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        logger.info(f"[VoiceEngine] Coqui model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

        # ── OPT 1: Cache speaker embeddings ──────────────────────────────────
        # tts_to_file() recomputes gpt_cond_latent + speaker_embedding from the
        # reference WAV on every single call — that's ~300–800 ms wasted every
        # utterance.  Extract them once, keep in RAM, inject via synthesize().
        logger.info("[VoiceEngine] Caching speaker embeddings from reference WAV…")
        t1 = time.perf_counter()
        try:
            model = _coqui_tts.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[REFERENCE_WAV]
            )
            _speaker_embedding = {
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
            }
            logger.info(f"[VoiceEngine] Speaker embeddings cached in {(time.perf_counter()-t1)*1000:.0f} ms — "
                        "all future synthesis will skip reference WAV processing.")
        except Exception as e:
            logger.warning(f"[VoiceEngine] Embedding cache failed ({e}) — will recompute each call.")
            _speaker_embedding = None

        _USE_COQUI = True
        logger.info("[VoiceEngine] Coqui XTTS-v2 ready — voice cloning ACTIVE.")

        # ── OPT 2: Pre-clone ALL static phrase segments ──────────────────────
        # Every segment in _STATIC_SEGMENTS is synthesised now, at startup,
        # with no time pressure.  speak_parts() then serves them from RAM
        # instantly — zero XTTS latency at call-time for any agent utterance.
        total = len(_STATIC_SEGMENTS)
        logger.info(f"[VoiceEngine] Pre-cloning {total} static phrase segments…")
        t2 = time.perf_counter()
        for i, seg_text in enumerate(_STATIC_SEGMENTS, 1):
            try:
                wav = _synth_coqui_wav(seg_text)
                if wav:
                    _FIXED_PHRASES[seg_text] = wav
                    logger.info(f"[VoiceEngine] [{i}/{total}] Cloned: '{seg_text[:60]}'")
                else:
                    logger.warning(f"[VoiceEngine] [{i}/{total}] Synthesis returned None for: '{seg_text[:60]}'")
            except Exception as e:
                logger.warning(f"[VoiceEngine] [{i}/{total}] Clone failed — '{seg_text[:40]}': {e}")
        logger.info(
            f"[VoiceEngine] Pre-cloning complete in {(time.perf_counter()-t2):.1f}s — "
            f"{len(_FIXED_PHRASES)}/{total} segments ready in RAM."
        )

    except Exception as e:
        logger.warning(f"[VoiceEngine] Coqui unavailable ({e}) — falling back to edge-tts.")

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT ASYNC EVENT LOOP  (started once, reused for every speak() call)
# ─────────────────────────────────────────────────────────────────────────────
# Problem: asyncio.run() creates + destroys an event loop and a new HTTPS
# connection on EVERY call.  That overhead is 200–400 ms of silence before
# the agent says a single word, on top of whatever edge-tts takes.
#
# Fix: one event loop running in a dedicated background thread for the whole
# process lifetime.  speak() submits a coroutine to it with
# asyncio.run_coroutine_threadsafe() and waits for the result.
# The HTTPS connection to edge-tts stays alive between calls (aiohttp
# keepalive), so subsequent speak() calls skip the TCP+TLS handshake too.

_LOOP = asyncio.new_event_loop()

def _start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_start_loop, args=(_LOOP,), daemon=True).start()


def _run_async(coro):
    """Submit a coroutine to the persistent loop and block until it completes."""
    future = asyncio.run_coroutine_threadsafe(coro, _LOOP)
    return future.result()


# ─────────────────────────────────────────────────────────────────────────────
# PYAUDIO CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_PA         = _pyaudio.PyAudio()
_RATE       = 24000   # edge-tts native sample rate — no resampling needed
_WIDTH      = 2       # 16-bit PCM
_CH         = 1       # mono
_PLAY_CHUNK = 1024    # bytes per pyaudio write ≈ 21 ms — small = low latency


# ─────────────────────────────────────────────────────────────────────────────
# COQUI HELPER (no streaming in XTTS-v2)
# ─────────────────────────────────────────────────────────────────────────────
@_timed
def _synth_coqui_wav(text: str) -> bytes | None:
    """
    Synthesise text → WAV bytes via XTTS-v2.

    Optimisations applied here:
      • Uses cached speaker embeddings (skips ~300–800 ms reference WAV processing).
      • Sets _vad_muted=True for the duration so the VAD thread does not try to
        transcribe XTTS's own output or internal model audio artefacts.
      • Falls back to tts_to_file() if cached embeddings are unavailable.
    """
    global _vad_mute_count
    logger.debug(f"[CoquiSynth] Synthesising {len(text)} chars via XTTS-v2")
    t0 = time.perf_counter()

    # Increment mute counter — VAD suppressed while any synthesis job is active
    _vad_mute_inc()
    try:
        # ── Fast path: use cached speaker embeddings ──────────────────────────
        if _speaker_embedding is not None:
            try:
                model = _coqui_tts.synthesizer.tts_model
                t_synth = time.perf_counter()
                out = model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=_speaker_embedding["gpt_cond_latent"],
                    speaker_embedding=_speaker_embedding["speaker_embedding"],
                    temperature=0.7,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=50,
                    top_p=0.85,
                )
                logger.debug(f"[CoquiSynth] model.inference() took {(time.perf_counter()-t_synth)*1000:.0f} ms")

                # out["wav"] is a list of floats (float32, 24kHz mono)
                import numpy as np
                pcm_f32 = np.array(out["wav"], dtype=np.float32)
                pcm_i16 = (pcm_f32 * 32767).clip(-32768, 32767).astype(np.int16)

                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)          # 16-bit
                    wf.setframerate(24000)      # XTTS-v2 native rate
                    wf.writeframes(pcm_i16.tobytes())
                data = buf.getvalue()
                logger.info(f"[CoquiSynth] Fast-path done in {(time.perf_counter()-t0)*1000:.0f} ms — {len(data)/1024:.1f} KB")
                return data
            except Exception as e:
                logger.warning(f"[CoquiSynth] Fast-path failed ({e}) — falling back to tts_to_file()")

        # ── Slow path: recompute embeddings from reference WAV each call ──────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        _coqui_tts.tts_to_file(text=text, speaker_wav=REFERENCE_WAV,
                                language="en", file_path=tmp)
        with open(tmp, "rb") as f:
            data = f.read()
        os.remove(tmp)
        logger.info(f"[CoquiSynth] Slow-path done in {(time.perf_counter()-t0)*1000:.0f} ms — {len(data)/1024:.1f} KB")
        return data

    except Exception as e:
        logger.error(f"[CoquiSynth] Error after {(time.perf_counter()-t0)*1000:.0f} ms — {e}")
        return None
    finally:
        _vad_mute_dec()   # decrement — VAD re-enables when all jobs finish


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

call_ui_logs           = []
system_ui_status       = "IDLE"
current_ui_threshold   = "300 (Normal)"
active_customer_record = None

is_agent_speaking        = False
interrupted              = False
latest_customer_text     = ""
customer_responded_event = threading.Event()

# Recording timeline: [{"speaker": "AGENT"|"CUSTOMER", "wav_bytes": bytes, "ts": float}]
_audio_segments     = []
is_recording_active = False

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            email       TEXT NOT NULL,
            due_amount  REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def _db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

# ─────────────────────────────────────────────────────────────────────────────
# RECORDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _record_segment(speaker: str, wav_bytes: bytes):
    """Append a timestamped WAV chunk to the in-memory recording timeline."""
    with _lock:
        if is_recording_active and wav_bytes:
            _audio_segments.append({
                "speaker":   speaker,
                "wav_bytes": wav_bytes,
                "ts":        time.time(),
            })

def _wav_from_pcm(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container (matches SAMPLE_RATE/WIDTH/CHANNELS)."""
    seg = AudioSegment(data=pcm_bytes, sample_width=SAMPLE_WIDTH,
                       frame_rate=SAMPLE_RATE, channels=CHANNELS)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()

def _normalise(wav_bytes: bytes, speaker: str) -> AudioSegment:
    seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    seg = (seg.set_frame_rate(SAMPLE_RATE)
              .set_channels(CHANNELS)
              .set_sample_width(SAMPLE_WIDTH))
    return seg + 6 if speaker == "CUSTOMER" else seg

@_timed
def save_conversation_wav(customer_id: int) -> str | None:
    """
    Merge all recorded segments (agent + customer) chronologically into a
    single WAV file.  Returns the file path, or None on failure.
    """
    global is_recording_active
    with _lock:
        is_recording_active = False
        segs = list(_audio_segments)
        _audio_segments.clear()

    logger.info(f"[Recording] Merging {len(segs)} audio segments for CID={customer_id}")

    if not segs:
        logger.warning("[Recording] No audio captured — skipping save.")
        return None

    segs.sort(key=lambda s: s["ts"])
    combined = AudioSegment.empty()
    for s in segs:
        try:
            combined += _normalise(s["wav_bytes"], s["speaker"])
        except Exception as e:
            logger.warning(f"[Recording] Skipping {s['speaker']} segment: {e}")

    if len(combined) == 0:
        logger.error("[Recording] Combined audio is empty after merge.")
        return None

    try:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(RECORDINGS_DIR, f"call_session_{customer_id}_{ts}.wav")
        combined.export(fname, format="wav")
        logger.info(f"[Recording] Saved {len(combined)/1000:.1f}s WAV → {fname}")
        return fname
    except Exception as e:
        logger.error(f"[Recording] Save failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SPEAK  — synthesise → record → play → handle barge-in
# ─────────────────────────────────────────────────────────────────────────────
def speak(text: str, allow_barge_in: bool = True):
    """
    Speak with minimal pre-speech delay and zero inter-sentence gaps.

    Latency breakdown (edge-tts path):
      ~0   ms  coroutine submitted to already-running loop (no loop startup)
      ~0   ms  HTTPS connection reused (no TCP+TLS handshake after first call)
      ~80  ms  first MP3 chunk arrives from edge-tts
      ~80  ms  FIRST WORD HEARD  ← pyaudio stream opened, PCM written
      ...      subsequent chunks written gaplessly to the same open stream

    Previous delays eliminated:
      • asyncio.run()       → 200–400 ms per call (loop create/destroy)
      • New HTTPS each call → 50–150 ms TCP+TLS handshake
      • Sentence splitting  → N×800 ms gaps between fragments
      • Full-buffer decode  → waited for entire MP3 before playing
    """
    global is_agent_speaking, interrupted, system_ui_status, current_ui_threshold

    with _lock:
        interrupted       = False
        is_agent_speaking = True

    system_ui_status     = "SPEAKING"
    current_ui_threshold = "500 (Echo Protection Masking)"
    call_ui_logs.append(f"Agent: {text}")
    logger.info(f"[Speak] Agent: {text}")
    _speak_t0 = time.perf_counter()

    def _is_interrupted():
        if not allow_barge_in:
            return False
        with _lock:
            return interrupted

    # ── Coqui path (voice cloning) ────────────────────────────────────────────
    if _USE_COQUI and _coqui_tts is not None:
        # Check pre-recorded phrase cache first — zero synthesis latency for
        # any phrase that was baked at startup.
        if text in _FIXED_PHRASES:
            logger.debug(f"[Speak] Phrase cache HIT — serving from RAM, skipping XTTS synthesis.")
            wav = _FIXED_PHRASES[text]
        else:
            logger.debug(f"[Speak] Phrase cache MISS — invoking XTTS synthesis.")
            wav = _synth_coqui_wav(text)
        if wav:
            _record_segment("AGENT", wav)
            try:
                snd     = pygame.mixer.Sound(io.BytesIO(wav))
                channel = snd.play()
                while channel and channel.get_busy():
                    if _is_interrupted():
                        channel.stop()
                        break
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"[CoquiPlayback] Error: {e}")

    # ── edge-tts streaming path ───────────────────────────────────────────────
    else:
        pa_stream  = None
        pcm_record = bytearray()
        leftover   = b""

        async def _stream_coro():
            nonlocal pa_stream, leftover

            comm = edge_tts.Communicate(text, "en-US-EmmaNeural")
            async for chunk in comm.stream():
                if chunk["type"] != "audio":
                    continue

                block = leftover + chunk["data"]
                try:
                    seg      = AudioSegment.from_mp3(io.BytesIO(block))
                    leftover = b""
                except Exception:
                    leftover = block
                    continue

                pcm = (seg.set_frame_rate(_RATE)
                          .set_channels(_CH)
                          .set_sample_width(_WIDTH)
                          .raw_data)

                if pa_stream is None:
                    pa_stream = _PA.open(
                        format            = _pyaudio.paInt16,
                        channels          = _CH,
                        rate              = _RATE,
                        output            = True,
                        frames_per_buffer = _PLAY_CHUNK,
                    )

                for i in range(0, len(pcm), _PLAY_CHUNK):
                    if _is_interrupted():
                        return
                    sl = pcm[i : i + _PLAY_CHUNK]
                    pa_stream.write(sl)
                    pcm_record.extend(sl)

        _stream_t0 = time.perf_counter()
        try:
            _run_async(_stream_coro())
            logger.debug(f"[Speak] edge-tts stream completed in {(time.perf_counter()-_stream_t0)*1000:.0f} ms")
        except Exception as e:
            logger.error(f"[Speak] edge-tts stream error after {(time.perf_counter()-_stream_t0)*1000:.0f} ms — {e}")
        finally:
            if pa_stream:
                try:
                    pa_stream.stop_stream()
                    pa_stream.close()
                except Exception:
                    pass

        if pcm_record:
            try:
                seg = AudioSegment(data=bytes(pcm_record), sample_width=_WIDTH,
                                   frame_rate=_RATE, channels=_CH)
                buf = io.BytesIO()
                seg.export(buf, format="wav")
                _record_segment("AGENT", buf.getvalue())
            except Exception as e:
                print(f"[Record Error]: {e}")

    with _lock:
        is_agent_speaking = False

    _speak_elapsed = (time.perf_counter() - _speak_t0) * 1000
    logger.info(f"[Speak] Playback finished in {_speak_elapsed:.0f} ms — status → {'INTERRUPTED' if interrupted else 'LISTENING'}")

    system_ui_status     = "INTERRUPTED" if interrupted else "LISTENING"
    current_ui_threshold = "220 (Active Listening Mode)"


# ─────────────────────────────────────────────────────────────────────────────
# SPEAK_PARTS  — stitch pre-cloned segments + fast edge-tts variable tokens
# ─────────────────────────────────────────────────────────────────────────────
def _edgetts_wav(token: str) -> bytes | None:
    """
    Render a short variable token to WAV bytes using edge-tts (fallback voice).
    Used only when Coqui is unavailable.  Takes ~80 ms.
    """
    async def _coro():
        buf = io.BytesIO()
        comm = edge_tts.Communicate(token, "en-US-EmmaNeural")
        mp3_data = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3_data.extend(chunk["data"])
        if mp3_data:
            seg = (AudioSegment.from_mp3(io.BytesIO(bytes(mp3_data)))
                   .set_frame_rate(24000).set_channels(1).set_sample_width(2))
            seg.export(buf, format="wav")
            return buf.getvalue()
        return None
    try:
        t0  = time.perf_counter()
        wav = _run_async(_coro())
        logger.debug(f"[EdgeTTS] Token '{token}' rendered in {(time.perf_counter()-t0)*1000:.0f} ms")
        return wav
    except Exception as e:
        logger.warning(f"[EdgeTTS] Token render failed for '{token}': {e}")
        return None


def _render_token_wav(token: str) -> bytes | None:
    """
    Render a dynamic token (customer name, email, date, balance) in the CLONED
    voice when Coqui is active, or fall back to edge-tts otherwise.

    Strategy
    --------
    • Coqui active  → synthesise token via XTTS-v2 using cached speaker
                       embeddings so the variable token sounds identical to the
                       pre-cloned static segments.  A short token takes ~1–3 s
                       on CPU — acceptable since static phrases play first.
    • Coqui absent  → delegate to _edgetts_wav() as before (~80 ms, Emma Neural).

    Cache
    -----
    Rendered tokens are stored in _FIXED_PHRASES under their text key so that
    repeated calls (e.g. the same customer name used twice in a call) are served
    from RAM with zero re-synthesis.
    """
    # Return from phrase cache if already synthesised this session
    if token in _FIXED_PHRASES:
        logger.debug(f"[TokenRender] Cache HIT for dynamic token '{token}'")
        return _FIXED_PHRASES[token]

    if _USE_COQUI and _coqui_tts is not None:
        logger.debug(f"[TokenRender] Cloning dynamic token via XTTS: '{token}'")
        t0  = time.perf_counter()
        wav = _synth_coqui_wav(token)
        if wav:
            _FIXED_PHRASES[token] = wav   # cache for reuse within this call
            logger.info(
                f"[TokenRender] Cloned '{token}' in {(time.perf_counter()-t0)*1000:.0f} ms"
            )
            return wav
        # XTTS synthesis failed — fall through to edge-tts
        logger.warning(f"[TokenRender] XTTS failed for '{token}' — falling back to edge-tts")

    return _edgetts_wav(token)


def _concat_wavs(wav_list: list[bytes]) -> bytes | None:
    """Concatenate a list of WAV byte-strings into a single WAV."""
    combined = AudioSegment.empty()
    for w in wav_list:
        if not w:
            continue
        try:
            combined += AudioSegment.from_wav(io.BytesIO(w))
        except Exception as e:
            logger.warning(f"[Concat] Skipping segment: {e}")
    if len(combined) == 0:
        return None
    buf = io.BytesIO()
    combined.export(buf, format="wav")
    return buf.getvalue()


def speak_parts(*parts, allow_barge_in: bool = True, labels: tuple = ()):
    """
    Speak a phrase built from alternating static / variable parts.

    Call convention — pass strings or pre-buffered WAV bytes in spoken order:
        speak_parts("Hello, am I speaking with", wav_name_bytes, "?",
                    labels=("Hello, am I speaking with", name, "?"))
        speak_parts("Balance of", wav_balance_bytes, ". When can you pay?",
                    labels=("Balance of", balance, ". When can you pay?"))

    Part types:
      • bytes  → pre-buffered WAV from pre_buffer_tokens(); played instantly (0 ms).
      • str found in _FIXED_PHRASES → served from RAM (pre-cloned, ~0 ms).
      • str not in cache → rendered live via _render_token_wav().

    The `labels` kwarg provides a parallel tuple of str labels for every part so
    that bytes parts (which have no text) still appear correctly in the UI
    transcript and call log.  If omitted, only str parts are logged.
    """
    # Build the full display text:
    #   • If caller passed labels=(...) use those (correct for bytes tokens)
    #   • Otherwise fall back to str parts only (bytes silently omitted)
    if labels:
        full_text = " ".join(str(l).strip() for l in labels if str(l).strip())
    else:
        full_text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())

    if not _USE_COQUI or not _coqui_tts:
        # edge-tts fallback: play bytes parts directly, speak() for the rest
        for part in parts:
            if isinstance(part, bytes) and part:
                _record_segment("AGENT", part)
                try:
                    snd = pygame.mixer.Sound(io.BytesIO(part))
                    channel = snd.play()
                    while channel and channel.get_busy():
                        time.sleep(0.005)
                except Exception as e:
                    logger.warning(f"[SpeakParts] bytes playback error: {e}")
        if full_text:
            speak(full_text, allow_barge_in=allow_barge_in)
        return

    global is_agent_speaking, interrupted, system_ui_status, current_ui_threshold

    with _lock:
        interrupted       = False
        is_agent_speaking = True

    system_ui_status     = "SPEAKING"
    current_ui_threshold = "500 (Echo Protection Masking)"
    call_ui_logs.append(f"Agent: {full_text}")
    logger.info(f"[Speak] Agent: {full_text}")
    _t0 = time.perf_counter()

    def _is_interrupted():
        if not allow_barge_in:
            return False
        with _lock:
            return interrupted

    wav_parts = []
    for part in parts:
        if isinstance(part, bytes):
            if part:
                logger.debug("[SpeakParts] Pre-buffered WAV bytes → instant playback")
                wav_parts.append(part)
        else:
            part = part.strip()
            if not part:
                continue
            if part in _FIXED_PHRASES:
                logger.debug(f"[SpeakParts] Cache HIT  → '{part[:50]}'")
                wav_parts.append(_FIXED_PHRASES[part])
            else:
                logger.debug(f"[SpeakParts] Dynamic token → rendering: '{part[:50]}'")
                wav_parts.append(_render_token_wav(part))

    stitched = _concat_wavs(wav_parts)
    if not stitched:
        logger.error("[SpeakParts] All parts failed — nothing to play.")
    else:
        _record_segment("AGENT", stitched)
        try:
            snd     = pygame.mixer.Sound(io.BytesIO(stitched))
            channel = snd.play()
            while channel and channel.get_busy():
                if _is_interrupted():
                    channel.stop()
                    break          # ← FIXED: must break after stop() or loop spins
                time.sleep(0.005)
        except Exception as e:
            logger.error(f"[SpeakParts] Playback error: {e}")

    with _lock:
        is_agent_speaking = False

    logger.info(
        f"[SpeakParts] Done in {(time.perf_counter()-_t0)*1000:.0f} ms — "
        f"{'INTERRUPTED' if interrupted else 'LISTENING'}"
    )
    system_ui_status     = "INTERRUPTED" if interrupted else "LISTENING"
    current_ui_threshold = "220 (Active Listening Mode)"


# ─────────────────────────────────────────────────────────────────────────────
# PRE-CALL TOKEN BUFFER
# ─────────────────────────────────────────────────────────────────────────────
# When "Dial" is clicked the API thread pre-synthesises the three dynamic
# tokens (name / email / balance) via XTTS-v2 IN PARALLEL before the call
# thread even starts.  The state machine picks them up from this dict and plays
# them instantly — zero mid-call synthesis pause.
#
# Layout: { "name": bytes|None, "email": bytes|None, "balance": bytes|None }
_precall_token_buffer: dict = {}


def pre_buffer_tokens(name: str, email: str, balance: str):
    """
    Synthesise the three per-customer dynamic tokens SEQUENTIALLY and store
    their WAV bytes in _precall_token_buffer.

    FIX (was parallel): Three concurrent model.inference() calls on XTTS-v2
    trigger PyTorch thread-safety issues — the internal GIL contention between
    simultaneous inference calls can cause one thread to deadlock indefinitely,
    blocking the .join() in the old parallel implementation and freezing the
    system in PRE-BUFFERING forever.  Sequential synthesis eliminates this
    entirely.  On CPU, total cost ≈ sum of the three tokens (15–45 s) vs the
    old parallel cost ≈ slowest token (7–25 s) — an acceptable tradeoff for
    complete stability.

    If Coqui is unavailable the tokens fall back to edge-tts (~80 ms each).
    """
    global _precall_token_buffer
    _precall_token_buffer = {"name": None, "email": None, "balance": None}

    t_start = time.perf_counter()
    logger.info(
        f"[PreBuffer] Starting sequential token synthesis: "
        f"name='{name}'  email='{email}'  balance='{balance}'"
    )

    for key, token in [("name", name), ("email", email), ("balance", balance)]:
        try:
            t0 = time.perf_counter()
            if _USE_COQUI and _coqui_tts is not None:
                wav = _synth_coqui_wav(token)
                src = "XTTS"
            else:
                wav = _edgetts_wav(token)
                src = "edge-tts"
            elapsed = (time.perf_counter() - t0) * 1000
            if wav:
                _precall_token_buffer[key] = wav
                logger.info(
                    f"[PreBuffer] [{src}] '{key}' done in {elapsed:.0f} ms "
                    f"({len(wav) // 1024} KB)"
                )
            else:
                logger.warning(
                    f"[PreBuffer] '{key}' synthesis returned None after {elapsed:.0f} ms"
                )
        except Exception as e:
            # _precall_token_buffer[key] stays None; speak_parts() skips None silently
            logger.error(
                f"[PreBuffer] '{key}' synthesis failed: {e}", exc_info=True
            )

    elapsed_total = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[PreBuffer] All tokens ready in {elapsed_total:.0f} ms — "
        f"name={'OK' if _precall_token_buffer['name'] else 'FAIL'}  "
        f"email={'OK' if _precall_token_buffer['email'] else 'FAIL'}  "
        f"balance={'OK' if _precall_token_buffer['balance'] else 'FAIL'}"
    )



def background_vad_listener():
    """
    VAD listener with a trailing-silence window to handle mid-sentence pauses.

    Instead of cutting off the moment SpeechRecognition's pause_threshold fires,
    we drop down to raw PyAudio frames and accumulate audio ourselves.  We only
    hand audio to Google STT once we have seen SILENCE_TRAILING_SECONDS of
    continuous silence AFTER speech was detected.  This means a customer can
    breathe, hesitate, or think mid-sentence for up to ~1.8 s without the system
    cutting them off — while still capturing the complete utterance in one pass.

    Key tuning knobs (all near the top of the function):
      SPEAKING_THRESHOLD        – RMS gate to suppress agent-speaker echo
      SPEECH_ENERGY_THRESHOLD   – RMS level above which a frame counts as speech
      SILENCE_TRAILING_SECONDS  – how long silence must persist before we commit
                                   (1.5–2.0 s is a good conversational window)
      MAX_UTTERANCE_SECONDS     – safety cap so a stuck mic cannot block forever
    """
    global is_agent_speaking, interrupted, latest_customer_text
    global system_ui_status, current_ui_threshold, is_recording_active

    # ── Thresholds ────────────────────────────────────────────────────────────
    SPEAKING_THRESHOLD       = 300    # RMS: suppress echo while agent speaks
    SPEECH_ENERGY_THRESHOLD  = 150    # RMS: frames above this count as speech
    SILENCE_TRAILING_SECONDS = 1.8    # seconds of silence = end of utterance
    MAX_UTTERANCE_SECONDS    = 20.0   # hard cap per utterance
    PRE_ROLL_FRAMES          = 5      # frames kept before speech onset

    # ── PyAudio / frame constants ─────────────────────────────────────────────
    PA_RATE       = 16000   # 16 kHz mono — Google STT sweet-spot
    PA_CHUNK      = 1024    # ~64 ms per frame at 16 kHz
    PA_FORMAT     = _pyaudio.paInt16
    PA_CHANNELS   = 1

    frames_per_second    = PA_RATE / PA_CHUNK          # ~15.6 frames / s
    max_silence_frames   = int(SILENCE_TRAILING_SECONDS * frames_per_second)
    max_utterance_frames = int(MAX_UTTERANCE_SECONDS    * frames_per_second)

    # ── SpeechRecognition (used only for Google STT, not for listening) ───────
    recognizer = sr.Recognizer()

    # ── Open a persistent PyAudio stream ─────────────────────────────────────
    pa = _pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=PA_FORMAT,
            channels=PA_CHANNELS,
            rate=PA_RATE,
            input=True,
            frames_per_buffer=PA_CHUNK,
        )
        logger.info("[VAD] PyAudio stream opened (trailing-silence mode).")
    except OSError as e:
        logger.critical(f"[VAD] Cannot open microphone: {e}")
        pa.terminate()
        sys.exit(1)

    # ── Ambient-noise calibration (read raw frames, compute baseline RMS) ─────
    logger.info("[VAD] Calibrating ambient noise (1 s)…")
    calib_frames = int(1.0 * frames_per_second)
    rms_sum = 0.0
    for _ in range(calib_frames):
        raw = stream.read(PA_CHUNK, exception_on_overflow=False)
        samples = struct.unpack_from(f"{PA_CHUNK}h", raw)
        rms_sum += (sum(s * s for s in samples) / PA_CHUNK) ** 0.5
    ambient_rms = rms_sum / calib_frames
    # Set speech threshold to max(SPEECH_ENERGY_THRESHOLD, 1.5× ambient)
    SPEECH_ENERGY_THRESHOLD = max(SPEECH_ENERGY_THRESHOLD, ambient_rms * 1.5)
    logger.info(f"[VAD] Calibration done — ambient RMS={ambient_rms:.0f}  "
                f"speech threshold={SPEECH_ENERGY_THRESHOLD:.0f}")

    # ── Ring buffer for pre-roll (frames captured before speech onset) ────────
    pre_roll: list[bytes] = []

    def _frame_rms(raw: bytes) -> float:
        n       = len(raw) // 2
        samples = struct.unpack_from(f"{n}h", raw)
        return (sum(s * s for s in samples) / n) ** 0.5 if n else 0.0

    def _raw_to_audio_data(frames: list[bytes]) -> sr.AudioData:
        """Wrap a list of raw PCM frames into an sr.AudioData object."""
        raw_bytes = b"".join(frames)
        return sr.AudioData(raw_bytes, PA_RATE, 2)  # 2 bytes = 16-bit

    logger.info("[VAD] Entering trailing-silence capture loop.")

    while True:
        try:
            # ── Skip while XTTS is synthesising ──────────────────────────────
            if _vad_muted():
                stream.read(PA_CHUNK, exception_on_overflow=False)  # drain
                continue

            with _lock:
                speaking = is_agent_speaking

            # Dynamic energy gate: tighter while agent plays to suppress echo
            effective_threshold = (SPEAKING_THRESHOLD
                                   if speaking
                                   else SPEECH_ENERGY_THRESHOLD)
            current_ui_threshold = (
                f"{int(effective_threshold)} (Echo Protection Masking)"
                if speaking
                else f"{int(effective_threshold)} (Active Listening Mode)"
            )

            # ── Read one frame ────────────────────────────────────────────────
            raw = stream.read(PA_CHUNK, exception_on_overflow=False)
            rms = _frame_rms(raw)

            # Maintain a short pre-roll buffer (captures onset consonants)
            pre_roll.append(raw)
            if len(pre_roll) > PRE_ROLL_FRAMES:
                pre_roll.pop(0)

            if rms < effective_threshold:
                # Silence — nothing to accumulate
                continue

            # ── Speech onset detected — accumulate until trailing silence ─────
            logger.debug(f"[VAD] Speech onset (RMS={rms:.0f}) — accumulating…")
            utterance_frames: list[bytes] = list(pre_roll)  # include pre-roll
            silence_frame_count = 0
            total_frames        = len(utterance_frames)

            while True:
                # Safety: check mute / agent-speaking state each frame
                if _vad_muted():
                    break

                try:
                    raw = stream.read(PA_CHUNK, exception_on_overflow=False)
                except OSError:
                    break

                utterance_frames.append(raw)
                total_frames += 1
                rms           = _frame_rms(raw)

                with _lock:
                    speaking = is_agent_speaking
                effective_threshold = (SPEAKING_THRESHOLD
                                       if speaking
                                       else SPEECH_ENERGY_THRESHOLD)

                if rms < effective_threshold:
                    silence_frame_count += 1
                else:
                    # Mid-sentence speech resumes — reset the silence counter
                    silence_frame_count = 0
                    logger.debug(f"[VAD] Mid-sentence speech resumed (RMS={rms:.0f}) — silence counter reset")

                if silence_frame_count >= max_silence_frames:
                    logger.debug(
                        f"[VAD] Trailing silence reached ({SILENCE_TRAILING_SECONDS}s) — "
                        f"finalising utterance ({total_frames} frames)"
                    )
                    break

                if total_frames >= max_utterance_frames:
                    logger.warning("[VAD] Max utterance duration reached — forcing STT commit.")
                    break

            if len(utterance_frames) < PRE_ROLL_FRAMES + 2:
                # Too short — likely a click or noise burst, not speech
                pre_roll.clear()
                continue

            # ── Send accumulated audio to Google STT ─────────────────────────
            audio_data = _raw_to_audio_data(utterance_frames)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                logger.debug("[VAD] Audio captured but speech not recognised — continuing.")
                pre_roll.clear()
                continue
            except sr.RequestError as e:
                logger.error(f"[VAD] Google Speech API error: {e}")
                time.sleep(0.5)
                pre_roll.clear()
                continue

            if not text.strip():
                pre_roll.clear()
                continue

            with _lock:
                speaking_now = is_agent_speaking
                rec          = is_recording_active

            # Record customer audio segment
            if rec:
                try:
                    pcm = audio_data.get_raw_data(
                        convert_rate=SAMPLE_RATE, convert_width=SAMPLE_WIDTH
                    )
                    _record_segment("CUSTOMER", _wav_from_pcm(pcm))
                except Exception as e:
                    logger.warning(f"[VAD] Customer audio record error: {e}")

            if speaking_now:
                logger.info(f"[VAD] BARGE-IN detected: '{text}'")
                with _lock:
                    interrupted          = True
                    latest_customer_text = text
                system_ui_status = "INTERRUPTED"
                call_ui_logs.append(f"Customer [Interrupt]: {text}")
                customer_responded_event.set()
            else:
                logger.info(f"[VAD] Customer speech (full): '{text}'")
                with _lock:
                    latest_customer_text = text
                    interrupted          = False
                call_ui_logs.append(f"Customer: {text}")
                customer_responded_event.set()

            pre_roll.clear()

        except Exception as e:
            logger.error(f"[VAD] Unexpected error: {e}")
            time.sleep(0.1)

# ─────────────────────────────────────────────────────────────────────────────
# WAIT FOR CUSTOMER RESPONSE
# ─────────────────────────────────────────────────────────────────────────────
@_timed
def wait_for_customer_response(timeout: int = 20) -> str:
    """Wait up to `timeout` seconds for the VAD thread to deliver customer speech."""
    global latest_customer_text, interrupted

    with _lock:
        barge_pending        = interrupted
        barge_text           = latest_customer_text
        latest_customer_text = ""
        interrupted          = False

    customer_responded_event.clear()

    if barge_pending and barge_text.strip():
        logger.info(f"[WaitResponse] Re-using barge-in payload: '{barge_text}'")
        return barge_text

    logger.debug(f"[WaitResponse] Waiting up to {timeout}s for customer speech…")
    t0  = time.perf_counter()
    got = customer_responded_event.wait(timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000

    with _lock:
        result               = latest_customer_text
        latest_customer_text = ""

    if got:
        logger.info(f"[WaitResponse] Received in {elapsed:.0f} ms: '{result}'")
    else:
        logger.warning(f"[WaitResponse] TIMEOUT after {elapsed:.0f} ms — no speech detected.")

    return result if got else ""

# ─────────────────────────────────────────────────────────────────────────────
# INTENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
@_timed
def analyze_intent(reply: str, prompt: str) -> str:
    """
    Pure LLM semantic intent classifier.

    Every customer utterance — regardless of language (English, Urdu, Roman Urdu,
    mixed code-switching), length, slang, or phrasing — is passed verbatim to
    GPT-4o-mini with a rich multilingual system prompt.  No local keyword lists
    intercept the call.  The LLM understands the full semantic meaning of what
    the customer said and returns a single clean classification token.

    Examples of what the LLM handles that keywords cannot:
      "Haan ji, bilkul mein hi bol raha hoon"   → YES  (identity)
      "Yeh mera email nahi hai"                  → NO   (email mismatch)
      "Abhi paise nahi hain, agle hafte dunga"   → CANNOT_PAY
      "Things are tight right now, can I pay next week?" → CANNOT_PAY
      "I need a breakdown first before I commit" → CANNOT_PAY
      "Haan, 15 July ko de dunga"                → WILL_PAY + date
    """
    if not reply.strip():
        logger.debug("[Intent] Empty reply — returning NO")
        return "NO"

    # ── Multilingual in-context system prompt ────────────────────────────────
    # This is not fine-tuning. GPT-4o-mini already knows Urdu, Roman Urdu,
    # and English. The system prompt acts as in-context learning: we give the
    # model clear decision rules + labelled examples so it knows exactly which
    # token to output for every real-world response pattern.
    SYSTEM_PROMPT = (
        "You are a semantic intent classifier for a Pakistani debt-collection voice agent. "
        "Your job is to read EXACTLY what the customer said — in any language or style — "
        "and return a single classification token. Nothing else. No explanation.\n\n"

        "## Language understanding\n"
        "Customers may speak in:\n"
        "  - English (formal or casual)\n"
        "  - Roman Urdu (Urdu written in English letters, e.g. 'Agle hafte dunga', 'Abhi nahi hai')\n"
        "  - Mixed code-switching (e.g. 'Bhai I cannot pay right now yaar')\n"
        "  - Indirect or polite refusals (e.g. 'Things are tight', 'I need a little more time')\n"
        "Understand the semantic meaning, not just the surface words.\n\n"

        "## Classification rules\n\n"

        "### Identity confirmation (output YES or NO)\n"
        "Output YES if the customer confirms they are the person being called:\n"
        "  - English: 'yes', 'yeah', 'speaking', 'that's me', 'this is him/her', 'correct'\n"
        "  - Roman Urdu: 'haan', 'haan ji', 'bilkul', 'mein hi hoon', 'ji haan', 'theek hai'\n"
        "  - Indirect: 'who else would it be', 'you've reached me'\n"
        "Output NO if they deny, redirect, or are unclear:\n"
        "  - English: 'no', 'wrong number', 'not me', 'he's not here'\n"
        "  - Roman Urdu: 'nahi', 'galat number', 'woh nahi hain', 'mein nahi hoon'\n\n"

        "### Email confirmation (output YES or NO)\n"
        "Output YES if the customer confirms the email address is theirs:\n"
        "  - 'yes', 'that's correct', 'haan mera hai', 'ji bilkul', 'correct hai'\n"
        "Output NO if they deny it:\n"
        "  - 'no', 'that's not mine', 'mera nahi hai', 'yeh galat hai', 'wrong email'\n\n"

        "### Payment intent (output CANNOT_PAY or WILL_PAY)\n"
        "Output CANNOT_PAY if the customer expresses ANY of these:\n"
        "  - Direct refusal: 'I can't pay', 'I don't have money', 'not possible right now'\n"
        "  - Installment/split request: 'can I pay in parts', 'installment chahiye', 'split kar sakte hain'\n"
        "  - Financial hardship (any phrasing): 'things are tight', 'I'm short on cash',\n"
        "    'abhi paise nahi hain', 'mushkil hai', 'meri situation thodi tight hai'\n"
        "  - Delay/extension request: 'can I pay next week', 'give me more time',\n"
        "    'agle hafte dunga', 'thoda time chahiye', 'baad mein kar sakta hoon'\n"
        "  - Asking for breakdown first: 'I need a breakdown', 'show me the details first'\n"
        "  - Any indirect signal of inability: 'I'll see what I can do', 'not sure if I can manage'\n"
        "Output WILL_PAY if the customer agrees to pay the full amount:\n"
        "  - 'yes I'll pay', 'I'll transfer it', 'on [date]', 'sure', 'okay'\n"
        "  - 'haan de dunga', 'kal tak dunga', '[date] ko dunga', 'bilkul'\n\n"

        "### Date extraction (output the date string or UNKNOWN)\n"
        "If asked to extract a date, return it in 'DD Month' format (e.g. '15 July').\n"
        "Understand date expressions in any language:\n"
        "  - 'agle hafte' → calculate and return the actual date\n"
        "  - 'next Friday' → calculate and return the actual date\n"
        "  - 'mahine ki 15 tarikh' → '15 [current month]'\n"
        "  - 'kal' → tomorrow's date\n"
        "If no date is discernible, return UNKNOWN.\n\n"

        f"## Your task\n{prompt}\n\n"
        "Output ONLY the token. No explanation, no punctuation, no extra words."
    )

    logger.debug(f"[Intent] → LLM  reply='{reply[:80]}'")
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": reply},
            ],
            temperature=0.0,
        )
        result = r.choices[0].message.content.strip().upper().rstrip(".")
        logger.info(f"[Intent] ← LLM  result='{result}'  ({(time.perf_counter()-t0)*1000:.0f} ms)")
        return result
    except Exception as e:
        # Emergency offline fallback — only fires if the API is unreachable
        logger.error(f"[Intent] LLM call failed ({(time.perf_counter()-t0)*1000:.0f} ms) — {e}")
        cleaned = reply.strip().lower()
        return "YES" if any(w in cleaned for w in ["yes","yeah","haan","speaking","correct","bilkul"]) else "NO"

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY + EMAIL
# ─────────────────────────────────────────────────────────────────────────────
@_timed
def generate_summary_file(wav_path: str, logs: list, name: str):
    logger.info(f"[Summary] Generating audit summary for '{name}' ({len(logs)} log lines)…")
    t0 = time.perf_counter()
    try:
        os.makedirs(SUMMARIES_DIR, exist_ok=True)
        base = (os.path.basename(wav_path)
                .replace("call_session_", "summary_")
                .replace(".wav", ".txt"))
        out  = os.path.join(SUMMARIES_DIR, base)
        dump = "\n".join(logs)

        r = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": ("Debt collection audit system. Write a structured summary: "
                             "Customer Name, Call Status, Agreements, Payment Plan, Disposition.")},
                {"role": "user", "content": f"Transcript:\n\n{dump}"}
            ],
            temperature=0.2,
        )
        body = r.choices[0].message.content.strip()

        with open(out, "w", encoding="utf-8") as f:
            f.write(f"MYTM AUDIT REPORT\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                    f"Recording: {wav_path}\n{'─'*50}\n\n{body}")
        logger.info(f"[Summary] Written to {out} in {(time.perf_counter()-t0)*1000:.0f} ms")
        return out, body
    except Exception as e:
        logger.error(f"[Summary] Failed after {(time.perf_counter()-t0)*1000:.0f} ms — {e}")
        return None, ""

@_timed
def dispatch_summary_email(to_email: str, to_name: str, body: str):
    logger.info(f"[Email] Dispatching summary to {to_email}…")
    t0 = time.perf_counter()
    try:
        msg            = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email
        msg["Subject"] = "MYTM Account Payment Plan & Summary"
        html = (
            f"<html><body style='font-family:sans-serif;max-width:600px;"
            f"margin:auto;padding:25px;border:1px solid #e2e8f0;border-radius:12px'>"
            f"<div style='background:#4f46e5;color:white;padding:15px;font-weight:bold;"
            f"border-radius:8px 8px 0 0;text-align:center;font-size:18px'>"
            f"MYTM CALL DISPOSITION AUDIT</div>"
            f"<p>Dear {to_name},</p>"
            f"<pre style='background:#f8fafc;padding:15px;border-radius:6px;"
            f"white-space:pre-wrap;font-family:monospace'>{body}</pre>"
            f"</body></html>"
        )
        msg.attach(MIMEText(html, "html"))
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls(); s.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        s.sendmail(SENDER_EMAIL, to_email, msg.as_string()); s.quit()
        logger.info(f"[Email] Sent to {to_email} in {(time.perf_counter()-t0)*1000:.0f} ms")
    except Exception as e:
        logger.error(f"[Email] Failed after {(time.perf_counter()-t0)*1000:.0f} ms — {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CALL STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
@_timed
def core_call_state_machine(customer: dict):
    global system_ui_status, is_recording_active, call_ui_logs, _audio_segments

    cid     = customer["customer_id"]
    name    = customer["name"]
    email   = customer["email"]
    balance = f"{int(customer['due_amount']):,} rupees"

    _call_t0 = time.perf_counter()
    logger.info(f"[StateMachine] ── CALL START ── CID={cid}  name='{name}'  balance={balance}")

    # ── Pull pre-buffered token WAVs synthesised during dial setup ───────────
    # pre_buffer_tokens() already ran in the dial API thread.  We pull the
    # bytes here.  If a token failed to buffer (None), speak_parts() will
    # skip it silently; _render_token_wav() would be the live fallback path
    # but it's not called here — we rely on the pre-buffer for all three core
    # tokens so there is zero mid-call synthesis pause.
    wav_name    = _precall_token_buffer.get("name")
    wav_email   = _precall_token_buffer.get("email")
    wav_balance = _precall_token_buffer.get("balance")

    missing = [k for k, v in [("name", wav_name), ("email", wav_email), ("balance", wav_balance)] if not v]
    if missing:
        logger.warning(f"[StateMachine] Pre-buffer MISS for: {missing} — those tokens will be silent")
    else:
        logger.info("[StateMachine] All 3 tokens loaded from pre-buffer — 0 ms synthesis expected mid-call")

    with _lock:
        _audio_segments.clear()
        is_recording_active = True

    try:
        # Step 1 — identity check
        logger.info("[StateMachine] Step 1: Identity confirmation")
        speak_parts(
            "Hello, this is Hamza calling from MYTM. I hope you're doing well today. Am I speaking with",
            wav_name,           # ← pre-buffered WAV bytes, instant playback
            # "?",
            labels=(
                "Hello, this is Hamza calling from MYTM. I hope you're doing well today. Am I speaking with",
                name,
                "?",
            ),
        )
        reply = wait_for_customer_response()
        intent = analyze_intent(reply,
            "Identity confirmation step. The agent asked if they are speaking with the customer. "
            "Classify the customer's response as YES (they confirm they are the person) or NO (they deny, "
            "say wrong number, say the person is unavailable, or give any unclear/negative response). "
            "Accept confirmations in English, Urdu, Roman Urdu, or mixed language. "
            "Examples of YES: 'haan', 'ji haan', 'mein hi hoon', 'bilkul', 'yes speaking', 'that is me'. "
            "Examples of NO: 'nahi', 'wrong number', 'woh nahi hain', 'he is not available'. "
            "Output only YES or NO.")
        if "YES" not in intent:
            logger.info(f"[StateMachine] Step 1 FAILED — identity not confirmed (intent='{intent}')")
            speak_parts("Understood. Thanks for your time, have a great day.",
                        allow_barge_in=False)
            system_ui_status = "COMPLETED (ID NOT CONFIRMED)"
            return

        logger.info("[StateMachine] Step 1 PASSED — identity confirmed")

        # Step 2 — email verification
        logger.info("[StateMachine] Step 2: Email verification")
        speak_parts(
            "Thank you for confirming. For verification, is your registered email address",
            wav_email,          # ← pre-buffered WAV bytes, instant playback
            # "?",
            labels=(
                "Thank you for confirming. For verification, is your registered email address",
                email,
                "?",
            ),
        )
        reply = wait_for_customer_response()
        intent = analyze_intent(reply,
            "Email verification step. The agent read out the customer's registered email address and asked "
            "if it is correct. Classify as YES (customer confirms the email is theirs) or NO (customer "
            "says it is wrong, not theirs, or gives any negative/unclear response). "
            "Accept answers in English, Urdu, Roman Urdu, or mixed language. "
            "Examples of YES: 'yes', 'correct', 'haan mera hai', 'ji bilkul', 'that is right'. "
            "Examples of NO: 'no', 'that is not mine', 'mera nahi hai', 'yeh galat hai', 'wrong'. "
            "Output only YES or NO.")
        if "YES" not in intent:
            logger.info(f"[StateMachine] Step 2 FAILED — email mismatch (intent='{intent}')")
            speak_parts(
                "I apologize for the inconvenience. Our team will update the records. Thanks for your time, have a great day.",
                allow_barge_in=False,
            )
            system_ui_status = "COMPLETED (EMAIL MISMATCH)"
            return

        logger.info("[StateMachine] Step 2 PASSED — email confirmed")

        # Step 3 — payment discussion
        logger.info("[StateMachine] Step 3: Payment discussion")
        speak_parts(
            "Thank you. I am calling regarding your outstanding balance of",
            wav_balance,        # ← pre-buffered WAV bytes, instant playback
            ". Could you let me know when we can expect this payment?",
            labels=(
                "Thank you. I am calling regarding your outstanding balance of",
                balance,
                ". Could you let me know when we can expect this payment?",
            ),
        )
        reply  = wait_for_customer_response()
        intent = analyze_intent(reply,
            "Payment intent step. The agent told the customer their outstanding balance and asked when "
            "they can pay. Classify the customer's response as CANNOT_PAY or WILL_PAY. "
            "Output CANNOT_PAY if the customer says or implies ANY of: they cannot pay now, they need "
            "more time, they want installments or a split, they have financial difficulty, they ask for "
            "a breakdown first, or they use indirect delay language. This includes: "
            "'I cannot pay', 'abhi nahi', 'paise nahi hain', 'mushkil hai', 'thoda time chahiye', "
            "'agle hafte dunga' (but asking for split, not committing to full), 'meri situation theek nahi', "
            "'can I pay in parts', 'things are tight', 'I need more time', 'not right now'. "
            "Output WILL_PAY if the customer agrees to pay the full amount, commits to a full payment "
            "date, or says yes with a date: 'yes I will pay', 'haan de dunga', '[date] ko dunga', "
            "'okay I will transfer it', 'sure on Friday'. "
            "Output only CANNOT_PAY or WILL_PAY.")

        # Installment plan is offered ONLY when the customer explicitly states
        # they cannot or are unable to pay the full amount.  A willingness to
        # pay — even if they mention a date — always goes to the FULL PAYMENT path.
        wants_installment = "CANNOT_PAY" in intent

        logger.info(f"[StateMachine] Payment intent='{intent}' → {'INSTALLMENT PLAN' if wants_installment else 'FULL PAYMENT'}")

        if wants_installment:
            # ── Collect date1 — ask up to 2 times before falling back ──────────
            speak_parts(
                "I understand. We can split this into two installments. On which date can you make your first payment?",
            )
            date1         = None
            _DATE1_TRIES  = 2
            for _attempt in range(_DATE1_TRIES):
                r1    = wait_for_customer_response()
                _d1   = analyze_intent(r1,
                    f"Date extraction step. Extract the first installment date the customer mentioned. "
                    f"Today is {datetime.now():%d %B %Y}. "
                    f"Return the date in 'DD Month' format (e.g. '20 June'). "
                    f"Understand date expressions in any language: "
                    f"'agle hafte' or 'next week' → calculate actual date 7 days from today; "
                    f"'kal' or 'tomorrow' → tomorrow's date; "
                    f"'mahine ki 15' → '15 {datetime.now():%B}'; "
                    f"'agle mahine' or 'next month' → 1st of next month; "
                    f"'is hafte ke akhir' or 'end of this week' → coming Sunday's date. "
                    f"If no date is discernible at all, return UNKNOWN.")
                if "UNKNOWN" not in _d1.upper() and r1.strip():
                    date1 = _d1
                    logger.info(f"[StateMachine] date1 confirmed by customer: '{date1}'")
                    break
                # Date not understood — ask once more (only on first miss)
                if _attempt < _DATE1_TRIES - 1:
                    logger.info(f"[StateMachine] date1 not heard (attempt {_attempt+1}) — re-asking")
                    speak_parts(
                        "I'm sorry, I didn't catch that. Could you please tell me the date for your first installment?",
                    )

            if date1 is None:
                # Customer never gave a clear date — use 7-day default as last resort
                date1 = (datetime.now() + timedelta(days=7)).strftime("%d %B")
                logger.warning(f"[StateMachine] date1 not provided after {_DATE1_TRIES} attempts — auto-set to {date1}")
                call_ui_logs.append(f"System: date1 auto-set to {date1} (customer did not provide)")

            # ── Collect date2 — ask up to 2 times, enforce 30-day policy ──────
            speak_parts(
                "I have noted your first installment for",
                date1,
                ". On which date can you make your second and final payment?",
            )
            date2        = None
            _DATE2_TRIES = 2
            for _attempt in range(_DATE2_TRIES):
                r2    = wait_for_customer_response()
                eval2 = analyze_intent(r2,
                    f"Date extraction step. Extract the second installment date the customer mentioned. "
                    f"Today is {datetime.now():%d %B %Y}. First installment date is {date1}. "
                    f"The second installment MUST be within 30 days of {date1}. "
                    f"Understand date expressions in any language: "
                    f"'agle hafte' or 'next week' → calculate actual date 7 days from today; "
                    f"'kal' → tomorrow; 'mahine ki 15' → '15 {datetime.now():%B}'; "
                    f"'agle mahine ki pehli' → 1st of next month. "
                    f"If the extracted date IS within 30 days of {date1}, return it as 'DD Month_VALID' "
                    f"(e.g. '15 July_VALID'). "
                    f"If the date is more than 30 days after {date1}, or is unclear, return INVALID_DATE.")
                if "INVALID" not in eval2 and "UNKNOWN" not in eval2.upper() and r2.strip():
                    date2 = eval2.replace("_VALID", "").strip()
                    logger.info(f"[StateMachine] date2 confirmed by customer: '{date2}'")
                    break
                # Invalid or not heard — explain policy and ask once more
                if _attempt < _DATE2_TRIES - 1:
                    logger.info(f"[StateMachine] date2 invalid/missing (attempt {_attempt+1}) — re-asking with policy reminder")
                    speak_parts(
                        "I'm sorry, the second installment must be within 30 days of the first. Could you please provide a date within that window?",
                    )

            if date2 is None:
                # Customer still didn't give a valid date — enforce policy automatically
                try:
                    base = datetime.strptime(f"{date1} {datetime.now().year}", "%d %B %Y")
                except ValueError:
                    base = datetime.now()
                date2 = (base + timedelta(days=28)).strftime("%d %B")
                logger.warning(f"[StateMachine] date2 not provided after {_DATE2_TRIES} attempts — auto-set to {date2}")
                speak_parts(
                    "Company policy requires the second installment within 30 days of the first. I have updated your second installment date to",
                    date2,
                    ". Both dates are now logged. Thanks for your time, have a great day.",
                    allow_barge_in=False,
                )
                call_ui_logs.append(
                    f"System: Installment 1 → {date1}  |  Installment 2 → {date2} (auto-adjusted)")
            else:
                speak_parts(
                    "Perfect. Your first installment is set for",
                    date1,
                    "and your second for",
                    date2,
                    ". Both are logged in our system. Thanks for your time, have a great day.",
                    allow_barge_in=False,
                )
                call_ui_logs.append(
                    f"System: Installment 1 → {date1}  |  Installment 2 → {date2}")
        else:
            promised = analyze_intent(reply,
                f"Date extraction step. The customer agreed to pay the full balance. "
                f"Extract the payment date they mentioned. Today is {datetime.now():%d %B %Y}. "
                f"Understand date expressions in any language: "
                f"'kal' → tomorrow's date; 'agle hafte' or 'next week' → 7 days from today; "
                f"'is weekend' or 'is hafte ke akhir' → coming Sunday; "
                f"'mahine ki pehli' → 1st of next month; specific dates like '15 July' as-is. "
                f"Return the date in 'DD Month' format. If no date was mentioned, return UNKNOWN.")
            if "UNKNOWN" in promised.upper() or not reply.strip():
                promised = "the agreed date"
            speak_parts(
                "I have logged your promise to settle the full balance on",
                promised,
                ". Thanks for your time, have a great day.",
                allow_barge_in=False,
            )

        system_ui_status = "CALL COMPLETED"
        logger.info(f"[StateMachine] ── CALL COMPLETE ── CID={cid} in {(time.perf_counter()-_call_t0):.1f}s")

    except Exception as e:
        logger.error(f"[StateMachine] ── CALL ERROR ── CID={cid} after {(time.perf_counter()-_call_t0):.1f}s — {e}", exc_info=True)
        system_ui_status = "ERROR DISCONNECTED"

    finally:
        logger.info("[StateMachine] Finalising: saving WAV + generating summary + emailing…")
        t_final = time.perf_counter()
        wav_path = save_conversation_wav(cid)
        if not wav_path:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(RECORDINGS_DIR,
                                    f"call_session_{cid}_{ts}.wav")

        summary_path, summary_text = generate_summary_file(wav_path, call_ui_logs, name)
        if summary_path:
            dispatch_summary_email(email, name, summary_text)
        logger.info(f"[StateMachine] Post-call tasks done in {(time.perf_counter()-t_final)*1000:.0f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("front.html")

@app.route("/api/customers")
def get_customers_list():
    t0   = time.perf_counter()
    conn = _db()
    rows = conn.execute("SELECT * FROM customers").fetchall()
    conn.close()
    logger.debug(f"[API] GET /api/customers → {len(rows)} records in {(time.perf_counter()-t0)*1000:.1f} ms")
    return jsonify([dict(r) for r in rows])

@app.route("/api/add_customer", methods=["POST"])
def add_customer():
    try:
        data   = request.json
        name   = data.get("name")
        email  = data.get("email")
        amount = float(data.get("due_amount", 0))
        if not name or not email:
            return jsonify({"success": False, "message": "Missing name or email"}), 400
        conn = _db()
        conn.execute("INSERT INTO customers (name,email,due_amount) VALUES (?,?,?)",
                     (name, email, amount))
        conn.commit(); conn.close()
        return jsonify({"success": True, "message": f"Customer {name} added."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/delete_customer/<int:customer_id>", methods=["DELETE"])
def delete_customer(customer_id):
    try:
        conn = _db()
        conn.execute("DELETE FROM customers WHERE customer_id=?", (customer_id,))
        conn.commit(); conn.close()
        return jsonify({"success": True, "message": f"Customer {customer_id} deleted."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/status")
def get_state_metrics():
    return jsonify({
        "status":           system_ui_status,
        "threshold":        current_ui_threshold,
        "logs":             call_ui_logs,
        "current_customer": active_customer_record,
    })

@app.route("/api/dial/<int:customer_id>", methods=["POST"])
def trigger_outbound_dial(customer_id: int):
    global system_ui_status, active_customer_record, call_ui_logs

    logger.info(f"[API] POST /api/dial/{customer_id} — current status: '{system_ui_status}'")

    ready = (system_ui_status == "IDLE"
             or "COMPLETED" in system_ui_status
             or system_ui_status == "ERROR DISCONNECTED")
    if not ready:
        logger.warning(f"[API] Dial rejected — system busy (status='{system_ui_status}')")
        return jsonify({"message": "System is busy."}), 400

    conn = _db()
    row  = conn.execute("SELECT * FROM customers WHERE customer_id=?",
                        (customer_id,)).fetchone()
    conn.close()
    if not row:
        logger.warning(f"[API] Dial rejected — customer_id={customer_id} not found in DB")
        return jsonify({"message": "Customer not found."}), 404

    active_customer_record = dict(row)
    call_ui_logs.clear()

    # ── Snapshot the customer record NOW so the thread cannot be affected by a
    #    later write to active_customer_record (e.g. browser double-click).
    customer_snapshot = dict(row)

    name    = customer_snapshot["name"]
    email   = customer_snapshot["email"]
    balance = f"{int(customer_snapshot['due_amount']):,} rupees"

    logger.info(f"[API] Pre-buffering tokens for CID={customer_id} name='{name}' — XTTS running sequentially in background…")
    system_ui_status = "PRE-BUFFERING"

    def _launch_after_buffer():
        """Pre-buffer all dynamic tokens, then start the state machine."""
        global system_ui_status
        try:
            pre_buffer_tokens(name, email, balance)
            logger.info(f"[API] Pre-buffer complete — launching core_call_state_machine for CID={customer_id}")
            # FIX: pass customer_snapshot (not the global active_customer_record) so
            # a concurrent dial request cannot swap the customer under this thread.
            core_call_state_machine(customer_snapshot)
        except Exception as e:
            # core_call_state_machine sets ERROR DISCONNECTED internally, but if it
            # crashes AFTER its own except block (e.g. the finally block raises and
            # @_timed re-raises), this guard catches that and ensures the UI never
            # gets stuck in PRE-BUFFERING.
            logger.error(
                f"[LaunchAfterBuffer] Unhandled exception for CID={customer_id}: {e}",
                exc_info=True,
            )
            system_ui_status = "ERROR DISCONNECTED"

    threading.Thread(target=_launch_after_buffer, daemon=True).start()
    return jsonify({"message": f"Dialing customer {customer_id}… (pre-buffering voice tokens)"})

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    src = os.path.join(SCRIPT_DIR, "front.html")
    dst = os.path.join(TEMPLATES_DIR, "front.html")
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil; shutil.copy(src, dst)
        print("[Setup] Copied front.html → templates/front.html")

    init_db()
    _init_voice_engine()   # load Coqui XTTS-v2 (or fall back to edge-tts)

    threading.Thread(target=background_vad_listener, daemon=True).start()

    try:
        import torch
        logger.info(f"[Startup] Torch intra-op threads: {torch.get_num_threads()}  "
                    f"inter-op: {torch.get_num_interop_threads()}")
    except ImportError:
        pass
    logger.info("=" * 60)
    logger.info("  MYTM Outbound Voice Agent — ONLINE")
    logger.info(f"  Dashboard  → http://127.0.0.1:5000")
    logger.info(f"  Log file   → {_LOG_FILE}")
    logger.info("=" * 60)
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        sys.exit(0)
