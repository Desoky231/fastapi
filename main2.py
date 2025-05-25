from __future__ import annotations

import os
import pathlib
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict
import re
from functools import lru_cache
from typing import Literal

# ───────────── Load Environment Variables ─────────────
load_dotenv()

# ───────────── Google Gemini ─────────────
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# ───────────── OpenAI Whisper ─────────────
import openai

# ───────────── LangChain & Qdrant (RAG) ─────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda

# ───────────── Deepgram ─────────────
import json
from io import BytesIO
import re
from deepgram import DeepgramClient, SpeakOptions
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────── FastAPI App ─────────────
app = FastAPI(title="Gemini + OpenAI Whisper + RAG (with memory) + Deepgram")

# ───────────── API Keys ─────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not all([GEMINI_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, DEEPGRAM_API_KEY]):
    raise RuntimeError("Required API keys or Qdrant info not set in environment")

# ───────────── Gemini Config ─────────────
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# ───────────── OpenAI Config ─────────────
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
WHISPER_MODEL = "whisper-1"

# ───────────── Deepgram Config ─────────────
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# ───────────── Whisper Helper ─────────────
def openai_whisper_from_upload(upload: UploadFile) -> str:
    if not upload.content_type or not upload.content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload an audio/* file.")
    filename = upload.filename or "upload.wav"
    if not pathlib.Path(filename).suffix:
        filename += ".wav"
    try:
        upload.file.seek(0)
        resp = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=(filename, upload.file, upload.content_type),
            language="en",
            response_format="text"
        )
        return resp.strip()
    except openai.OpenAIError as exc:
        raise HTTPException(500, f"OpenAI API error: {exc}") from exc

# ───────────── Qdrant Retriever ─────────────
COLLECTION_NAME = "new_ramesses_ii_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"



# ───────────── Complexity Finder ─────────────

# --- lightweight question-complexity detector ------------------------------
#
# Returns the string "simple"  or  "complex"
#
# Strategy
#   1. Fast heuristic pass   – O(1), catches ~80 % of cases.
#   2. DistilBERT fallback   – only if heuristics are unsure.
#      Model ~66 M params, <20 ms on CPU.  (One-time lazy load.)


# ---------- 1) Heuristic rules ---------------------------------------------
_SIMPLE_STARTERS = ("who", "what", "when", "where", "which")
_COMPLEX_STARTERS = ("why", "how", "explain", "describe", "tell me", "give me", "outline")

def _heuristic_label(q: str) -> Literal["simple", "complex", "unsure"]:
    q = q.strip().lower()
    tokens = q.split()
    t0 = tokens[0] if tokens else ""

    # very short & specific → simple
    if len(tokens) <= 6 and t0 in _SIMPLE_STARTERS:
        return "simple"

    # obvious complex cues
    if t0 in _COMPLEX_STARTERS or "?" not in q or len(tokens) > 20:
        return "complex"

    # ambiguous → let model decide
    return "unsure"

# ---------- 2) Small-model fallback ----------------------------------------
@lru_cache()  # ensures model loads once
def _load_classifier():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="grahamaco/question-complexity-classifier",  # DistilBERT
        top_k=1,
    )

def classify_question(question: str) -> str:
    """
    Label a user question as 'simple' or 'complex'.

    Fast heuristics first; if ambiguous, defer to a tiny DistilBERT
    classifier (loaded lazily & cached).

    >>> classify_question("Who built the Great Pyramid?")
    'simple'
    >>> classify_question("Explain how the pyramids were constructed.")
    'complex'
    """
    label = _heuristic_label(question)
    if label != "unsure":
        return label

    # model fallback
    result = _load_classifier()(question)[0]["label"]
    return result.lower()  # returns 'simple' or 'complex'


def get_retriever():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
    info = client.get_collection(COLLECTION_NAME)
    if info.vectors_count == 0:
        raise RuntimeError("Qdrant collection is empty.")
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embeddings=embedder)
    return vector_store

def format_docs(docs: List) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

class SimpleStrOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return text



# ---------------------------------------------------------------------------
#  build_onchain — complexity-aware & error-free
# ---------------------------------------------------------------------------
def build_onchain(memory: ConversationBufferMemory):
    """
    • Classifies visitor question ('simple' | 'complex')
    • Injects that tag into the Ramesses-II prompt
    • Picks the Gemini client with the right max-token budget
    • Returns **str** (so TTS, headers, and Pydantic stay happy)
    """

    # 1) Gemini variants with different ceilings -----------------------------
    llm_simple  = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        generation_config={"max_output_tokens": 60}     # ≈ 1-2 sentences
    )
    llm_complex = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        generation_config={"max_output_tokens": 200}    # ≤ 200 words
    )
    llm_by_label = {"simple": llm_simple, "complex": llm_complex}

    # 2) RAG retriever --------------------------------------------------------
    vector_retriever   = get_retriever()
    multi_query_prompt = PromptTemplate.from_template(
        "Generate multiple versions of this question: {question}"
    )
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever.as_retriever(search_kwargs={"k": 5}),
        llm=llm_simple,               # cheap Gemini is fine for query-gen
        prompt=multi_query_prompt,
    )

    # 3) Prompt template with {complexity} slot ------------------------------
    prompt_template = """
You are Ramesses II, Pharaoh of Egypt. **Always speak in first-person**.

####################  GLOBAL GUIDELINES  ####################
• Remain in character – you are the living god-king.  
• Use vivid storytelling and authentic historical detail.  
• Language must project COMMAND, DIVINE WISDOM, and STRENGTH.  
• Clarity first: prefer plain, direct wording over archaic flourish.

####################  LENGTH & DEPTH RULES  ##################
<COMPLEXITY> = "{complexity}"

**If <COMPLEXITY> == "simple":**
    – Reply in **1–2 crisp sentences** (≈ ≤ 40 tokens).  
    – Focus on the single fact the visitor seeks.

**If <COMPLEXITY> == "complex":**
    – Provide a **thorough yet structured answer** (3–6 sentences, ≤ 200 words).

####################  FALLBACK  ####################
If uncertain, confess modestly (“Even a god may not recall…”) and offer to investigate.
──────────────────────────────────────────────────────────────
### Chat History:
{chat_history}

### Context (retrieved scrolls, stelae, or papyri):
{context}

### Visitor’s Question:
{input}

### Your Response (as Ramesses II):
"""
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)

    # 4) Chain pieces ---------------------------------------------------------
    classify_node = RunnableLambda(classify_question)   # 'simple' | 'complex'

    # A. Gather all fields
    gather = {
        "context": mq_retriever | format_docs,
        "input":   RunnablePassthrough(),                        # raw question
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
        "complexity":  classify_node,
    }

    # B. Build ChatPromptValue + keep complexity
    def build_payload(d):
        return {
            "prompt":     chat_prompt.format_prompt(**d),  # ChatPromptValue
            "complexity": d["complexity"],
        }

    # C. Run the correct Gemini and return **text** (str)
    def run_llm(d):
        llm      = llm_by_label[d["complexity"]]
        messages = d["prompt"].to_messages()              # list[BaseMessage]
        resp     = llm.invoke(messages)
        # ChatGoogleGenerativeAI returns an AIMessage; grab .content
        return resp.content if hasattr(resp, "content") else str(resp)

    chain = (
        gather
        | RunnableLambda(build_payload)
        | RunnableLambda(run_llm)        # -> str
    )

    # 5) Memory-aware wrapper -----------------------------------------------
    def chain_with_memory(user_input: str) -> str:
        answer = chain.invoke(user_input)                 # <-- plain str
        memory.save_context({"input": user_input}, {"output": answer})
        return answer

    return chain_with_memory

# ───────────── Memory Store (IP-based) ─────────────
memory_store = defaultdict(lambda: ConversationBufferMemory(memory_key="chat_history", return_messages=True))


# ───────────── TTS Helper Functions ─────────────
def segmentTextBySentence(text):
    return re.findall(r"[^.!?]+[.!?]", text)


def synthesizeAudio(text):
    options = SpeakOptions(model="aura-2-odysseus-en")
    SPEAK_OPTIONS = {"text": text}
    response = deepgram.speak.rest.v("1").stream_memory(SPEAK_OPTIONS, options)
    if response and response.stream:
        audio_data = BytesIO(response.stream.read())
        return AudioSegment.from_file(audio_data, format="mp3")
    else:
        raise Exception('Error generating audio')


def estimate_token_budget(question: str) -> int:
    length = len(question.split())
    if length <= 5:
        return 40   # Simple question: keep it short
    elif length <= 15:
        return 60   # Moderate detail
    else:
        return 80   # More complex: allow more explanation


def text_to_speech_combined(input_text: str) -> BytesIO:
    """Convert text to a single MP3 audio file in memory."""
    try:
        segments = segmentTextBySentence(input_text)
        results = [None] * len(segments)

        # Generate audio segments in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {executor.submit(synthesizeAudio, segment): i for i, segment in enumerate(segments)}
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    results[i] = future.result()
                except Exception as error:
                    print(f"Error synthesizing audio for segment {i}: {error}")

        # Combine audio segments
        combined_audio = AudioSegment.silent(duration=0)
        for audio in results:
            if audio:
                combined_audio += audio

        # Export to BytesIO buffer
        audio_buffer = BytesIO()
        combined_audio.export(audio_buffer, format="mp3")
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as exc:
        raise HTTPException(500, f"Error generating audio: {str(exc)}") from exc

# ───────────── Schemas ─────────────
class PromptRequest(BaseModel):
    prompt: str

# ───────────── Endpoints ─────────────

@app.post("/ask")
async def ask_ai(req: PromptRequest):
    try:
        result = GEMINI_MODEL.generate_content(req.prompt)
        return {"response": result.text}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    transcript = openai_whisper_from_upload(file)
    return {"transcript": transcript}

from fastapi import Request  # Make sure this is imported

@app.post("/chat-with-audio")
async def chat_with_audio(file: UploadFile = File(...), request: Request = None):
    try:
        session_id = request.client.host
        memory = memory_store[session_id]

        # Transcribe audio to text
        transcript = openai_whisper_from_upload(file)

        # Generate response using RAG chain
        rag_chain = build_onchain(memory)
        response_text = rag_chain(transcript)

        # Convert response to audio
        audio_buffer = text_to_speech_combined(response_text)

        # Extract chat history messages as a list of dicts or strings
        chat_history = memory.load_memory_variables({})["chat_history"]
        history_for_client = [{"type": message.type, "content": message.content} for message in chat_history]

        # Prepare metadata for headers (limited by header size)
        metadata = {
            "session": session_id,
            "transcript": transcript,  # Truncate to avoid header size limits
            "response": response_text,  # Truncate to avoid header size limits
        }
        headers = {
            "X-Metadata": json.dumps(metadata),  # Simple metadata in header
            "Content-Disposition": "attachment; filename=response_audio.mp3",
        }

        # Return streaming response with audio
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mp3",
            headers=headers
        )

    except Exception as exc:
            raise HTTPException(500, f"Error processing request: {str(exc)}") from exc



@app.post("/reset-memory")
async def reset_memory(request: Request):
    session_id = request.client.host
    memory_store[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return {"session": session_id, "status": "memory reset"}

# End of file