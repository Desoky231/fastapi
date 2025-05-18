import os, tempfile, subprocess, pathlib, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from faster_whisper import WhisperModel

# ────────── Gemini configuration ──────────
genai.configure(api_key="AIzaSyDISh2I2N54owowdHykqUtwfAyMh8fosKk")   # set env var before launch!
gemini = genai.GenerativeModel("gemini-1.5-flash")

# ────────── Whisper-tiny (faster-whisper) ──────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
COMP_TYP = "int8"                                     # 8-bit on both GPU & CPU
whisper  = WhisperModel("tiny", device=DEVICE, compute_type=COMP_TYP)

# ────────── FastAPI app ──────────
app = FastAPI(title="Gemini + Whisper service")

# ---------- helpers --------------------------------------------------------
def ffmpeg_resample(src: str, dst: str):
    """Convert any input to 16 kHz / mono WAV – required for best WER."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", src,
           "-ac", "1", "-ar", "16000", dst]
    subprocess.run(cmd, capture_output=True, check=True)


def whisper_from_upload(upload: UploadFile) -> str:
    """Save upload → resample → return transcript string."""
    if not upload.content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload an audio file.")

    try:
        # 1) save
        suffix   = pathlib.Path(upload.filename).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw_f:
            raw_f.write(upload.file.read())
            raw_path = raw_f.name

        # 2) resample
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_f:
            wav_path = wav_f.name
        ffmpeg_resample(raw_path, wav_path)

        # 3) transcribe
        segments, _ = whisper.transcribe(wav_path)
        return "".join(s.text for s in segments).strip()

    finally:
        for p in (locals().get("raw_path"), locals().get("wav_path")):
            try: os.remove(p)
            except Exception: pass

# ---------- existing endpoints (unchanged) ---------------------------------
class PromptRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_ai(req: PromptRequest):
    try:
        result = gemini.generate_content(req.prompt)
        return {"response": result.text}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        transcript = whisper_from_upload(file)
        return {"transcript": transcript}
    except subprocess.CalledProcessError as ffmpeg_err:
        raise HTTPException(500, f"FFmpeg failed: {ffmpeg_err.stderr.decode()}")
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- NEW endpoint: audio → transcript → LLM -------------------------
@app.post("/chat-with-audio")
async def chat_with_audio(file: UploadFile = File(...)):
    """
    One-shot pipeline: upload audio → Whisper transcription → Gemini answer.
    Returns both pieces so the client can show them together.
    """
    try:
        transcript = whisper_from_upload(file)
        gemini_resp = gemini.generate_content(transcript)
        return {
            "transcript": transcript,
            "response":   gemini_resp.text
        }
    except subprocess.CalledProcessError as ffmpeg_err:
        raise HTTPException(500, f"FFmpeg failed: {ffmpeg_err.stderr.decode()}")
    except Exception as e:
        raise HTTPException(500, str(e))
