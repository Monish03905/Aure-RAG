"""ChatGPT-style offline RAG agent orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.schemas import PipelineStage
from backend.config import LOCAL_LLM_MODEL, UPLOAD_DIR
from backend.rag.pipeline import AuraRAG
from backend.services.sessions import SessionStore


class AuraChatAgent:
    """
    Backend agent that:
    1) manages chat sessions
    2) ingests uploaded files into an offline FAISS store
    3) runs the structured RAG pipeline
    4) returns grounded answers + pipeline telemetry
    """

    def __init__(self):
        self.rag = AuraRAG()
        self.sessions = SessionStore()

    @property
    def ready(self) -> bool:
        return self.rag is not None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "aura_ready": True,
            "base_index_size": self.rag.chunk_count,
            "upload_index_size": self.rag.user_store.size,
            "documents": self.rag.user_store.list_documents(),
            "sessions": len(self.sessions.list_sessions()),
            "embedding_model": self.rag.model_name,
            "generation_mode": "ollama" if LOCAL_LLM_MODEL else "extractive",
            "mode": "offline_rag_agent",
        }

    def list_sessions(self) -> List[Dict]:
        return self.sessions.list_sessions()

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def create_session(self, title: str = "New chat") -> Dict:
        return self.sessions.create(title=title)

    def delete_session(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False

        attachments = session.get("attachments", [])
        doc_ids = [item["doc_id"] for item in attachments if item.get("doc_id")]
        self.rag.user_store.remove_documents(doc_ids)
        upload_root = UPLOAD_DIR.resolve()
        for item in attachments:
            stored_name = item.get("stored_name")
            if not stored_name:
                continue
            stored_path = (UPLOAD_DIR / Path(stored_name).name).resolve()
            if stored_path.parent == upload_root and stored_path.is_file():
                stored_path.unlink()
        return self.sessions.delete(session_id)

    def ingest_upload(
        self,
        path: Path,
        original_name: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = self.sessions.ensure(session_id)
        info = self.rag.ingest_document(path, original_name=original_name)
        info["session_id"] = session["session_id"]
        info["stored_name"] = path.name
        self.sessions.attach_file(session["session_id"], info)
        self.sessions.add_message(
            session["session_id"],
            "system",
            f"Attached and indexed `{original_name}` ({info['chunks']} chunks).",
            meta={"type": "upload", **info},
        )
        return info

    def _pipeline_view(self, stages_ms: Dict[str, float], grounded: bool) -> List[PipelineStage]:
        catalog = [
            ("listen", "Accept user message / voice transcript"),
            ("analyze", "Detect language, query type, complexity"),
            ("retrieve", "Dense FAISS search over base + uploads"),
            ("rerank", "Metadata + lexical reranking / dedupe"),
            ("generate", "Local Ollama synthesis when configured; otherwise extractive"),
            ("guardrail", "Unsafe / unsupported answer checks"),
        ]
        mapping = {
            "listen": 0.0,
            "analyze": stages_ms.get("analyze_ms", 0.0),
            "retrieve": stages_ms.get("embedding_ms", 0.0)
            + stages_ms.get("faiss_ms", 0.0)
            + stages_ms.get("user_faiss_ms", 0.0),
            "rerank": stages_ms.get("rerank_ms", 0.0),
            "generate": stages_ms.get("generation_ms", 0.0),
            "guardrail": stages_ms.get("guardrail_ms", 0.0),
        }
        stages: List[PipelineStage] = []
        for name, description in catalog:
            ms = float(mapping.get(name, 0.0))
            status = "done" if ms > 0 or name == "listen" else ("done" if grounded else "skipped")
            if name == "guardrail":
                status = "done"
            stages.append(
                PipelineStage(
                    name=name,
                    description=description,
                    status=status,
                    ms=round(ms, 3),
                )
            )
        return stages

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_base_corpus: bool = True,
        use_uploads: bool = True,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session = self.sessions.ensure(session_id)
        sid = session["session_id"]

        self.sessions.add_message(sid, "user", message)

        result = self.rag.ask(
            message,
            use_base_corpus=use_base_corpus,
            use_uploads=use_uploads,
            session_attachments=session.get("attachments", []),
        )

        attachments_used = sorted(
            {
                s.get("source_file")
                for s in result.get("sources", [])
                if s.get("origin") == "upload" and s.get("source_file")
            }
        )

        payload = {
            "session_id": sid,
            "answer": result["answer"],
            "grounded": result["grounded"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "generation_mode": result.get("generation_mode", "extractive"),
            "analysis": result.get("analysis", {}),
            "sources": result.get("sources", []),
            "stages_ms": result.get("stages_ms", {}),
            "total_ms": result.get("total_ms", round((time.perf_counter() - started) * 1000, 3)),
            "pipeline": [
                stage.model_dump()
                for stage in self._pipeline_view(result.get("stages_ms", {}), result["grounded"])
            ],
            "attachments_used": attachments_used,
            "index_size": result.get("index_size", self.rag.chunk_count),
        }

        self.sessions.add_message(
            sid,
            "assistant",
            payload["answer"],
            meta={
                "grounded": payload["grounded"],
                "confidence": payload["confidence"],
                "reason": payload["reason"],
                "generation_mode": payload["generation_mode"],
                "analysis": payload["analysis"],
                "total_ms": payload["total_ms"],
                "sources": payload["sources"][:3],
                "pipeline": payload["pipeline"],
            },
        )
        return payload
