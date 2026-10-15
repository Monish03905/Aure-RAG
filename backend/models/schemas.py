"""Pydantic API schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    use_base_corpus: bool = True
    use_uploads: bool = True
    history: List[ChatMessageIn] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class PipelineStage(BaseModel):
    name: str
    description: str
    status: str = "idle"
    ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    grounded: bool
    confidence: float
    reason: str
    generation_mode: str = "extractive"
    analysis: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    stages_ms: Dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0
    pipeline: List[PipelineStage] = Field(default_factory=list)
    attachments_used: List[str] = Field(default_factory=list)
    index_size: int = 0
