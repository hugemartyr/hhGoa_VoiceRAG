"""FastAPI application entrypoint.

Serves the frontend UI and exposes the `/ask` endpoint which
triggers the STT and the RAG pipeline.
"""

import time
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from app.pipeline.stt import SarvamSTT
from app.pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HH Goa 2026 Voice RAG")

# Lazy singletons to avoid heavy import/cold-start costs on module import
_stt_client = None
_orchestrator = None


def get_stt_client():
    global _stt_client
    if _stt_client is None:
        _stt_client = SarvamSTT()
    return _stt_client


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


@app.post("/ask")
async def ask_question(audio: UploadFile = File(...)):
    """Accepts voice audio, transcribes it, runs RAG, and returns the answer."""
    start_time = time.perf_counter()
    
    # 1. Read Audio
    audio_bytes = await audio.read()
    
    # 2. STT (Voice to English Text)
    stt = get_stt_client()
    stt_res = await stt.transcribe_and_translate(audio_bytes, filename=audio.filename)
    
    # Fallback if STT fails completely
    if not stt_res.text:
        return JSONResponse({
            "answer": "Sorry, I could not hear or transcribe what you said.",
            "transcribed_text": "",
            "latency": {"e2e_total_ms": (time.perf_counter() - start_time) * 1000}
        })
        
    # 3. RAG Pipeline
    pipeline_res = get_orchestrator().process_query(stt_res.text, stt_latency=stt_res.latency_ms)
    
    # Force E2E latency to include the time taken for audio reading, etc.
    pipeline_res.latency.e2e_total_ms = (time.perf_counter() - start_time) * 1000
    
    return pipeline_res.model_dump()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the basic HTML frontend."""
    # Read index.html from static folder
    import os
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Frontend not found. Please create app/static/index.html</h1>"



@app.get("/health")
async def health():
    """Lightweight health check that does not initialize heavy ML models."""
    return JSONResponse({"status": "ok"})
