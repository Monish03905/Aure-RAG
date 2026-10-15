"""Lightweight offline chat session store."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import SESSION_DIR


class SessionStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict] = {}
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def _path(self, session_id: str) -> Path:
        return SESSION_DIR / f"{session_id}.json"

    def _load_all(self) -> None:
        for path in SESSION_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sid = data.get("session_id") or path.stem
                self._sessions[sid] = data
            except Exception:
                continue

    def _persist(self, session_id: str) -> None:
        path = self._path(session_id)
        path.write_text(
            json.dumps(self._sessions[session_id], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(self, title: str = "New chat") -> Dict:
        session_id = uuid.uuid4().hex[:12]
        now = time.time()
        session = {
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "attachments": [],
        }
        with self._lock:
            self._sessions[session_id] = session
            self._persist(session_id)
        return session

    def list_sessions(self) -> List[Dict]:
        with self._lock:
            items = sorted(
                self._sessions.values(),
                key=lambda s: s.get("updated_at", 0),
                reverse=True,
            )
            return [
                {
                    "session_id": s["session_id"],
                    "title": s.get("title", "New chat"),
                    "updated_at": s.get("updated_at"),
                    "message_count": len(s.get("messages", [])),
                    "attachments": s.get("attachments", []),
                }
                for s in items
            ]

    def get(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            return self._sessions.get(session_id)

    def ensure(self, session_id: Optional[str] = None) -> Dict:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        return self.create()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: Optional[Dict] = None,
    ) -> Dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise KeyError(session_id)
            message = {
                "id": uuid.uuid4().hex[:10],
                "role": role,
                "content": content,
                "ts": time.time(),
                "meta": meta or {},
            }
            session["messages"].append(message)
            session["updated_at"] = time.time()
            if role == "user" and session.get("title") in {"New chat", ""}:
                session["title"] = content.strip()[:48] or "New chat"
            self._persist(session_id)
            return message

    def attach_file(self, session_id: str, file_info: Dict) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise KeyError(session_id)
            session.setdefault("attachments", []).append(file_info)
            session["updated_at"] = time.time()
            self._persist(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            path = self._path(session_id)
            if path.exists():
                path.unlink()
            return True
