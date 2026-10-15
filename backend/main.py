"""AURA FastAPI entrypoint — chat agent + uploads + voice + static UI."""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.chat_agent import AuraChatAgent
from backend.config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    CORS_ORIGINS,
    FRONTEND_DIR,
    MAX_UPLOAD_BYTES,
    SARVAM_API_KEY,
    UPLOAD_DIR,
)
from backend.models.schemas import AskRequest, ChatRequest

agent: Optional[AuraChatAgent] = None
initialization_error: Optional[str] = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global agent, initialization_error
    print("Initializing AURA chat agent...")
    try:
        agent = AuraChatAgent()
        initialization_error = None
        print("AURA agent ready.")
    except Exception as exc:
        agent = None
        initialization_error = str(exc)
        logger.exception("AURA initialization failed")
        print("ERROR: AURA initialization failed:", exc)

    print("=" * 64)
    print("AURA CHAT AGENT READY")
    print("POST /api/chat")
    print("POST /api/upload")
    print("POST /api/ask")
    print("POST /api/voice")
    print("GET  /health")
    print("Sarvam:", "CONFIGURED" if SARVAM_API_KEY else "NOT CONFIGURED")
    print("=" * 64)
    yield


app = FastAPI(
    title="AURA RAG Chat Agent",
    description="Offline ChatGPT-style RAG agent with file attachments",
    version="3.0.0",
    lifespan=lifespan,
)

origins = (
    ["*"]
    if CORS_ORIGINS.strip() == "*"
    else [x.strip() for x in CORS_ORIGINS.split(",") if x.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=CORS_ORIGINS.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_agent() -> AuraChatAgent:
    if agent is None:
        raise HTTPException(status_code=503, detail="AURA is not initialized.")
    return agent


@app.get("/health")
def health():
    if agent is None:
        return {
            "status": "degraded",
            "aura_ready": False,
            "error": initialization_error,
            "voice_provider": "sarvam" if SARVAM_API_KEY else "not_configured",
            "version": "3.0.0",
        }
    payload = agent.health()
    payload["voice_provider"] = "sarvam" if SARVAM_API_KEY else "not_configured"
    payload["version"] = "3.0.0"
    return payload


@app.get("/api")
def api_info():
    return {
        "name": "AURA",
        "version": "3.0.0",
        "mode": "offline_chat_rag_agent",
        "endpoints": [
            "POST /api/chat",
            "POST /api/upload",
            "POST /api/ask",
            "POST /api/voice",
            "GET /api/sessions",
            "GET /api/sessions/{id}",
            "DELETE /api/sessions/{id}",
            "GET /api/documents",
            "GET /health",
        ],
        "pipelines": [
            "listen",
            "analyze",
            "retrieve(base+uploads)",
            "rerank",
            "generate(local_ollama_or_extractive)",
            "guardrail",
        ],
    }


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": require_agent().list_sessions()}


@app.post("/api/sessions")
def create_session():
    return require_agent().create_session()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = require_agent().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    ok = require_agent().delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


@app.get("/api/documents")
def list_documents():
    aura = require_agent()
    return {"documents": aura.rag.user_store.list_documents()}


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        result = require_agent().chat(
            message=request.message,
            session_id=request.session_id,
            use_base_corpus=request.use_base_corpus,
            use_uploads=request.use_uploads,
        )
    except RuntimeError as exc:
        logger.exception("AURA chat pipeline failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result


@app.post("/api/ask")
def ask(request: AskRequest):
    """Legacy single-turn endpoint (still powered by the same backend harness)."""
    try:
        result = require_agent().rag.ask(request.query)
    except RuntimeError as exc:
        logger.exception("AURA ask pipeline failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "query": result.get("query", request.query),
        "answer": result.get("answer", ""),
        "grounded": bool(result.get("grounded", False)),
        "confidence": float(result.get("confidence", 0.0)),
        "reason": result.get("reason", "unknown"),
        "generation_mode": result.get("generation_mode", "extractive"),
        "analysis": result.get("analysis", {}),
        "sources": result.get("sources", []),
        "stages_ms": result.get("stages_ms", {}),
        "total_ms": float(result.get("total_ms", 0.0)),
        "index_size": int(result.get("index_size", 0)),
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    aura = require_agent()
    filename = file.filename or "upload.bin"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Allowed: "
                + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
            ),
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:10]}_{Path(filename).name}"
    dest = UPLOAD_DIR / safe_name

    try:
        size = 0
        with dest.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File too large. Max "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                target.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file.")

        info = await run_in_threadpool(
            aura.ingest_upload,
            dest,
            original_name=filename,
            session_id=session_id,
        )
    except HTTPException:
        if dest.exists():
            dest.unlink()
        raise
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise HTTPException(status_code=422, detail=f"Ingest failed: {exc}") from exc
    finally:
        await file.close()

    return {
        "ok": True,
        "message": f"Indexed {filename}",
        **info,
    }


@app.post("/api/voice")
async def voice_aura(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    aura = require_agent()
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is not configured.",
        )

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    content_type = (file.content_type or "audio/webm").split(";")[0].strip().lower()
    stt_start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Accept": "application/json",
                },
                files={
                    "file": (
                        file.filename or "aura-voice.webm",
                        io.BytesIO(audio),
                        content_type,
                    )
                },
                data={"model": "saaras:v3", "mode": "transcribe"},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Sarvam STT request failed: {exc}",
        ) from exc

    stt_ms = (time.perf_counter() - stt_start) * 1000
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Sarvam STT error ({response.status_code}): {response.text[:500]}",
        )

    payload = response.json()
    transcript = (payload.get("transcript") or "").strip()
    language_code = payload.get("language_code") or "unknown"
    if not transcript:
        raise HTTPException(status_code=422, detail="Empty transcript from Sarvam.")

    try:
        result = await run_in_threadpool(
            aura.chat,
            message=transcript,
            session_id=session_id,
        )
    except RuntimeError as exc:
        logger.exception("AURA voice chat pipeline failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["voice"] = {
        "provider": "Sarvam Saaras v3",
        "transcript": transcript,
        "language_code": language_code,
        "stt_ms": round(stt_ms, 3),
    }
    return result


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/{path:path}")
def frontend_fallback(path: str):
    if path.startswith("api") or path in {"health", "docs", "openapi.json", "redoc"}:
        raise HTTPException(status_code=404, detail="Not found")

    candidate = os.path.join(FRONTEND_DIR, path)
    if (
        os.path.isfile(candidate)
        and os.path.realpath(candidate).startswith(os.path.realpath(FRONTEND_DIR))
    ):
        return FileResponse(candidate)

    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
