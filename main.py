"""
FastAPI service: Gemini + OpenAI Whisper-1 (no FFmpeg, fixed upload bug)

This version sends the uploaded file to Whisper-1 exactly as FastAPI receives
it, but **passes a proper filename + MIME type** so the OpenAI SDK won’t throw
the “Unrecognized file format” 400 error.

Install
-------
pip install fastapi uvicorn python-multipart openai google-generativeai python-dotenv

Environment
-----------
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...

Run
---
uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
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
app = FastAPI(title="Gemini + OpenAI Whisper service (no-ffmpeg)")

# ---------- helpers --------------------------------------------------------


def openai_whisper_from_upload(upload: UploadFile) -> str:
    """
    Forward the uploaded audio file to OpenAI Whisper-1 and
    return a plain-text transcript.

    We pass a (filename, file-obj, content-type) tuple so the OpenAI SDK
    recognises the format.  If the original filename lacks an extension,
    we supply '.wav' by default.
    """
    if not upload.content_type or not upload.content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload an audio/* file.")

    # Make sure there is a valid extension; else fallback to .wav
    filename = upload.filename or "upload.wav"
    if not pathlib.Path(filename).suffix:
        filename += ".wav"

    try:
        upload.file.seek(0)  # rewind in case other middleware read it

        resp = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            # tuple form: (filename, file-like-object, mime-type)
            file=(filename, upload.file, upload.content_type),
            language="ar",          # Modern Standard Arabic & dialects
            response_format="text"  # get plain string back
        )
        return resp.strip()

    except openai.OpenAIError as exc:
        raise HTTPException(500, f"OpenAI API error: {exc}") from exc


# ---------- API schemas ----------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str


# ---------- endpoints ------------------------------------------------------
@app.post("/ask")
async def ask_ai(req: PromptRequest):
    """Gemini text-only endpoint."""
    try:
        result = GEMINI_MODEL.generate_content(req.prompt)
        return {"response": result.text}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text endpoint using Whisper-1 (no FFmpeg)."""
    transcript = openai_whisper_from_upload(file)
    return {"transcript": transcript}


@app.post("/chat-with-audio")
async def chat_with_audio(file: UploadFile = File(...)):
    """One-shot: audio → Whisper-1 → Gemini answer."""
    transcript = openai_whisper_from_upload(file)
    try:
        gemini_resp = GEMINI_MODEL.generate_content(transcript)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return {"transcript": transcript, "response": gemini_resp.text}
