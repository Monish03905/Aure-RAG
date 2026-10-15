# AURA — Adaptive Universal RAG Assistant

VOICE-FIRST • MULTILINGUAL • GROUNDED • MEASURABLE • FUTURISTIC

One deployable project: FastAPI serves the API **and** the interactive frontend from a single process.

## Official task alignment

AURA implements the required flow:

Microphone → Sarvam Saaras v3 STT → query analysis → adaptive retrieval → FAISS → reranking → grounded answer → guardrails → futuristic UI.

The official brief requires a voice-enabled RAG system, Sarvam or ElevenLabs STT, non-trivial chunking, a full RAG-path target under 200 ms, P50/P70/P100 analytics, a structured harness, and guardrails for unsupported/unsafe/off-topic answers.

## Technical design

### 1. STT
Uses Sarvam Saaras v3 through `POST /api/voice`.

### 2. Chunking
The corpus uses three strategies:
- sentence chunks
- sliding semantic chunks with overlap
- metadata-aware chunks preserving language/query-type/selection metadata

### 3. Retrieval
- multilingual sentence-transformer embeddings
- FAISS dense retrieval
- adaptive top-N candidate count based on query complexity/type
- deduplication
- lightweight metadata + lexical reranking

### 4. Harness
`AuraRAG.ask()` is the structured controller. It provides:
- input validation
- retries
- structured JSON output
- stage-level latency instrumentation
- fallback/abstention paths

### 5. Guardrails
AURA can refuse:
- unsafe queries
- insufficiently supported queries
- empty/invalid answers

Every response includes a confidence/grounding state and supporting evidence.

### 6. Voice
The browser records a short audio clip and sends it to `/api/voice`. Sarvam returns transcript + language code, then AURA runs the RAG pipeline on the transcript.

## Project layout

```text
AURA_FINAL_PATCH/
├── backend/                 # FastAPI + RAG harness
│   ├── main.py              # API + static UI hosting
│   └── rag/                 # analyzer, retrieve, rerank, generate, guardrails
├── frontend/                # Interactive UI (served by FastAPI)
├── data/index/              # Prebuilt FAISS index + metadata.jsonl
├── scripts/                 # Benchmark + guardrail tests
├── requirements.txt
├── .env.example
├── run.ps1                  # One-command local launch (Windows)
├── Dockerfile               # Container deploy
└── Procfile                 # PaaS deploy (Railway/Render/Heroku-style)
```

## Setup

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

Put your Sarvam key in `.env`:

```text
SARVAM_API_KEY=...
```

The prebuilt FAISS assets must already exist:

```text
data/index/aura.index
data/index/metadata.jsonl
```

## Run (recommended — single port)

```powershell
.\run.ps1
```

Or manually:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

`http://127.0.0.1:8000`

API docs:

`http://127.0.0.1:8000/docs`

Health:

`http://127.0.0.1:8000/health`

> Optional separate frontend server is no longer required. FastAPI already hosts `frontend/`.

## Deploy

### Docker

```powershell
docker build -t aura-rag .
docker run -p 8000:8000 --env-file .env aura-rag
```

### PaaS (Railway / Render / Fly)

1. Set `SARVAM_API_KEY` in the platform environment.
2. Use the `Procfile` start command, or:
   `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Ensure `data/index/` is included in the deploy artifact.

First boot downloads the sentence-transformer model (≈120MB). Keep that warm or bake it into the image for faster cold starts.

## Benchmark

Run at least 100 queries:

```powershell
python -m scripts.benchmark_e2e 100
```

The final report is written to:

`data/benchmark/e2e_benchmark_report.txt`

It reports P50/P70/P100 for the complete RAG path and separate stage timings.

## Demo checklist

1. English typed query
2. Telugu typed query
3. Voice query using Sarvam
4. Unsupported/off-topic query → abstention
5. Unsafe query → refusal
6. Show sources and confidence
7. Show measured latency telemetry

## Important wording for the demo

By default, answers are intentionally extractive and evidence-preserving. To draft richer responses, set `AURA_LOCAL_LLM_MODEL` to a model already installed in local Ollama. AURA sends retrieved evidence only to a loopback Ollama URL and still applies the same evidence threshold and safety guardrails. With no local model configured, no LLM server is contacted.

## Chat and document workflow

The browser interface and API are served by the same FastAPI process. Each chat has its own persistent session, message history, and attachment list:

1. Create a chat or select a saved chat in the sidebar.
2. Attach one or more PDF, TXT, Markdown, CSV, JSON, DOCX, or LOG files. A file is validated, saved locally, parsed, chunked with overlap, embedded, and added to the persistent upload FAISS index before it is marked indexed.
3. Ask a question. The chat endpoint scopes upload retrieval to documents attached to that chat, optionally searches the bundled base corpus, reranks the combined evidence, generates a grounded answer, and runs the unsupported/unsafe guardrails.
4. Inspect the cited evidence, query analysis, per-stage timings, and grounded/abstained status in the chat and Details panel.

The Files and Corpus switches control retrieval independently. Chat attachments are isolated between chat sessions; the legacy `/api/ask` endpoint remains available for one-off corpus-wide queries. Upload size is limited by `AURA_MAX_UPLOAD_MB` (25 MB by default). Uploaded originals, session JSON, and the mutable user index are local runtime data and are excluded from Git.

### Pipeline stages

* **Listen:** Accept a typed message, voice transcript, or attachment; associate it with a persistent session.
* **Analyze:** Infer language, query type, complexity, and normalized search text; return these in `analysis`.
* **Retrieve:** Embed and search the enabled FAISS indexes; restrict upload hits to the current chat.
* **Rerank:** Deduplicate candidates, then blend semantic, lexical, and metadata relevance; return ranked chunks and scores.
* **Generate:** Use a matching dataset answer/evidence extract by default, or a local Ollama model when configured; report `generation_mode`.
* **Guardrail:** Skip generation for unsafe or weakly supported retrieval, then report grounded state, reason, and confidence.

With the optional local model configured, AURA can draft longer explanations and project/code content grounded in retrieved documents. It does not run shell commands or write generated project files. To enable richer generation, install Ollama, download a model once (for example `ollama pull qwen2.5:3b`), then set `AURA_LOCAL_LLM_MODEL=qwen2.5:3b` in `.env`; keep the Ollama server on `127.0.0.1`. Without that setting AURA remains fast and extractive. When the embedding model is not already cached, first startup needs internet access to download it; subsequent runs can use the local cache. Voice transcription via Sarvam is optional and requires internet.

## Validation

Install dependencies, then run the focused workflow tests:

```powershell
.\.venv\Scripts\python.exe -m unittest scripts.test_rag_workflow scripts.test_guardrails
```

To check the running service and endpoints, start AURA with `.\run.ps1`, then visit `/health` and `/docs`. The workflow tests verify chat attachment isolation, upload indexing, and persistence across a store reload.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AURA_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local multilingual embedding model |
| `AURA_MAX_UPLOAD_MB` | `25` | Per-file upload size limit |
| `CORS_ORIGINS` | `*` | Comma-separated browser origins |
| `SARVAM_API_KEY` | empty | Optional voice transcription key |
| `AURA_LOCAL_LLM_URL` | `http://127.0.0.1:11434` | Local Ollama URL (must be loopback) |
| `AURA_LOCAL_LLM_MODEL` | empty | Ollama model name; blank keeps extractive generation |
| `AURA_LOCAL_LLM_TIMEOUT_SECONDS` | `90` | Local generation request timeout |
