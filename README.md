# 🎙️ Gemini + Whisper FastAPI Service

A lightweight FastAPI server that:
- Uses **Faster-Whisper** for transcription of audio files.
- Sends the transcript to **Gemini (Google Generative AI)** for intelligent responses.
- Supports endpoints for raw prompts, transcription, and end-to-end audio-based Q&A.

---

## 📦 Requirements

### ✅ System Prerequisites:
- Python 3.8+
- `ffmpeg` installed and available in system PATH
- Optional: GPU with CUDA for faster inference

### ✅ Python Dependencies:
Install using `pip`:

```bash
pip install fastapi uvicorn google-generativeai faster-whisper torch pydantic
```

> If you're using a GPU, ensure `torch` matches your CUDA version. For CPU-only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🔐 Setup Gemini API Key

Before launching, set the environment variable for your **Gemini API Key**:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or directly replace in the script (not recommended for production):

```python
genai.configure(api_key="your-api-key-here")
```

---

## 🧠 What This App Does

| Endpoint | Function |
|----------|----------|
| `POST /ask` | Takes a prompt and returns Gemini's response. |
| `POST /transcribe` | Uploads an audio file and returns transcript from Whisper. |
| `POST /chat-with-audio` | Uploads audio, transcribes it, and sends it to Gemini – returns both transcript and LLM answer. |

---

## ▶️ Running the Server

Run with Uvicorn:

```bash
uvicorn your_script_name:app --reload
```

Replace `your_script_name` with the Python filename (without `.py`).

Example:

```bash
uvicorn main:app --reload
```

---

## 🎧 Audio Requirements

- Audio files must be in a standard format (e.g., MP3, WAV, M4A).
- Internally, the server uses `ffmpeg` to resample to:
  - **16 kHz**
  - **Mono**
  - **WAV**

---

## 🔐 .gitignore Suggestion

If you're using Git, create a `.gitignore` file to avoid unnecessary or sensitive files:

```gitignore
__pycache__/
*.pyc
*.pyo
.env
*.log
temp/
```

---

## 💬 Example Usage (with `curl`)

### Ask Gemini:
```bash
curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"prompt": "Explain quantum computing in simple terms."}'
```

### Transcribe Audio:
```bash
curl -X POST http://localhost:8000/transcribe   -F "file=@your_audio_file.wav"
```

### Full Pipeline (chat with audio):
```bash
curl -X POST http://localhost:8000/chat-with-audio   -F "file=@your_audio_file.wav"
```

---

## 🛠 Troubleshooting

- **FFmpeg not found**: Make sure it's installed and available in PATH.
- **CUDA errors**: If no GPU, change the model to CPU by modifying:
  ```python
  DEVICE = "cpu"
  COMP_TYP = "int8"  # Or "float32" if needed
  ```
- **Gemini quota exceeded**: Check API usage and limits.

---

## 📃 License

MIT – free to use, adapt, and modify.

---

Enjoy your multimodal AI assistant! 🚀
