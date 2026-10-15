"""Central configuration for AURA offline RAG agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
USER_INDEX_DIR = DATA_DIR / "user_index"
UPLOAD_DIR = DATA_DIR / "uploads"
SESSION_DIR = DATA_DIR / "sessions"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

BASE_INDEX_FILE = INDEX_DIR / "aura.index"
BASE_METADATA_FILE = INDEX_DIR / "metadata.jsonl"
USER_INDEX_FILE = USER_INDEX_DIR / "user.index"
USER_METADATA_FILE = USER_INDEX_DIR / "user_metadata.jsonl"

EMBEDDING_MODEL = os.getenv(
    "AURA_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
LOCAL_LLM_URL = os.getenv("AURA_LOCAL_LLM_URL", "http://127.0.0.1:11434").rstrip("/")
LOCAL_LLM_MODEL = os.getenv("AURA_LOCAL_LLM_MODEL", "").strip()
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("AURA_LOCAL_LLM_TIMEOUT_SECONDS", "90"))

MAX_UPLOAD_BYTES = int(os.getenv("AURA_MAX_UPLOAD_MB", "25")) * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".docx",
    ".log",
}

for path in (UPLOAD_DIR, USER_INDEX_DIR, SESSION_DIR):
    path.mkdir(parents=True, exist_ok=True)
