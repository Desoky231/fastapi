"""FastAPI service: Gemini + OpenAI Whisper (cloud)

This version removes the local *faster‑whisper* dependency and
calls OpenAI’s managed Whisper‑1 model via the Audio API.

Prerequisites
-------------
    pip install fastapi uvicorn python-multipart openai google-generativeai python-dotenv

Environment variables required
------------------------------
* **OPENAI_API_KEY** – your OpenAI key with Audio API access
* **GEMINI_API_KEY** – (optional) Google Generative AI key for Gemini

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile
import subprocess
import pathlib

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import google.generativeai as genai
import openai

# ────────── Gemini configuration ──────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# ────────── OpenAI Whisper (cloud) ──────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
WHISPER_MODEL = "whisper-1"

# ────────── FastAPI app ──────────
app = FastAPI(title="Gemini + OpenAI Whisper service")

# ---------- helpers --------------------------------------------------------

def ffmpeg_resample(src: str, dst: str) -> None:
    """Convert any input to 16 kHz / mono WAV – improves WER."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        src,
        "-ac",
        "1",
        "-ar",
        "16000",
        dst,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def openai_whisper_from_upload(upload: UploadFile) -> str:
    """Save upload → resample → Whisper API → return transcript string."""
    if not upload.content_type or not upload.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    raw_path = wav_path = None
    try:
        # 1️⃣  save original upload
        suffix = pathlib.Path(upload.filename).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw_f:
            raw_f.write(upload.file.read())
            raw_path = raw_f.name

        # 2️⃣  resample to 16 kHz mono WAV (recommended by OpenAI)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_f:
            wav_path = wav_f.name
        ffmpeg_resample(raw_path, wav_path)

        # 3️⃣  call Whisper‑1
        with open(wav_path, "rb") as audio_f:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_f,
                language="ar",            # Modern Standard Arabic / dialects
                response_format="text",
            )
        return resp.strip()

    except (subprocess.CalledProcessError, openai.OpenAIError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # remove temp files
        for p in (raw_path, wav_path):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

# ---------- API schemas ----------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str

# ---------- endpoints ------------------------------------------------------

@app.post("/ask")
async def ask_ai(req: PromptRequest):
    """Gemini text‑only endpoint."""
    try:
        result = GEMINI_MODEL.generate_content(req.prompt)
        return {"response": result.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Speech‑to‑text endpoint using Whisper‑1."""
    transcript = openai_whisper_from_upload(file)
    return {"transcript": transcript}


@app.post("/chat-with-audio")
async def chat_with_audio(file: UploadFile = File(...)):
    """One‑shot: audio → Whisper‑1 → Gemini answer."""
    transcript = openai_whisper_from_upload(file)
    try:
        gemini_resp = GEMINI_MODEL.generate_content(transcript)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"transcript": transcript, "response": gemini_resp.text}
