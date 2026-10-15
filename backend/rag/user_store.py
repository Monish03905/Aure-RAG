"""Persistent FAISS store for user-uploaded documents (offline)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from backend.config import USER_INDEX_FILE, USER_METADATA_FILE, USER_INDEX_DIR
from backend.rag.document_loader import chunk_document, extract_text


class UserDocumentStore:
    def __init__(self, embedding_model, dim: int = 384):
        self.embedding_model = embedding_model
        self.dim = dim
        self._lock = threading.RLock()
        USER_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        self.metadata: List[Dict] = []
        self.documents: Dict[str, Dict] = {}
        self.index = faiss.IndexFlatIP(self.dim)

        self._load()

    def _load(self) -> None:
        if USER_METADATA_FILE.exists():
            with USER_METADATA_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    self.metadata.append(item)
                    doc_id = item.get("doc_id")
                    if doc_id and doc_id not in self.documents:
                        self.documents[doc_id] = {
                            "doc_id": doc_id,
                            "filename": item.get("source_file", "unknown"),
                            "chunks": 0,
                        }

            for item in self.metadata:
                doc_id = item.get("doc_id")
                if doc_id in self.documents:
                    self.documents[doc_id]["chunks"] += 1

        if USER_INDEX_FILE.exists() and self.metadata:
            self.index = faiss.read_index(str(USER_INDEX_FILE))
            if self.index.ntotal != len(self.metadata):
                # Rebuild if mismatched.
                self._rebuild_from_metadata()
        elif self.metadata:
            self._rebuild_from_metadata()

    def _rebuild_from_metadata(self) -> None:
        texts = [m.get("text", "") for m in self.metadata]
        if not texts:
            self.index = faiss.IndexFlatIP(self.dim)
            return
        vectors = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self._persist()

    def _persist(self) -> None:
        USER_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(USER_INDEX_FILE))
        with USER_METADATA_FILE.open("w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @property
    def size(self) -> int:
        return len(self.metadata)

    def list_documents(self) -> List[Dict]:
        with self._lock:
            return list(self.documents.values())

    def ingest_file(self, path: Path, original_name: Optional[str] = None) -> Dict:
        filename = original_name or path.name
        text = extract_text(path)
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_document(text, source_name=filename, doc_id=doc_id)
        if not chunks:
            raise ValueError("Document produced no chunks.")

        vectors = self.embedding_model.encode(
            [c["text"] for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        with self._lock:
            self.index.add(vectors)
            self.metadata.extend(chunks)
            self.documents[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(chunks),
                "chars": len(text),
                "added_at": time.time(),
            }
            self._persist()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "chars": len(text),
            "index_size": self.size,
        }

    def search(
        self,
        query: str,
        top_k: int = 12,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        with self._lock:
            if self.index.ntotal == 0:
                return []

            vector = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            k = min(max(top_k * 3, top_k), self.index.ntotal)
            scores, indices = self.index.search(vector, k)

            results: List[Dict] = []
            allowed = set(doc_ids) if doc_ids else None

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = self.metadata[idx].copy()
                if allowed is not None and item.get("doc_id") not in allowed:
                    continue
                item["similarity"] = float(score)
                results.append(item)
                if len(results) >= top_k:
                    break

            return results

    def clear(self) -> None:
        with self._lock:
            self.metadata = []
            self.documents = {}
            self.index = faiss.IndexFlatIP(self.dim)
            if USER_INDEX_FILE.exists():
                USER_INDEX_FILE.unlink()
            if USER_METADATA_FILE.exists():
                USER_METADATA_FILE.unlink()

    def remove_documents(self, doc_ids: List[str]) -> int:
        requested = set(doc_ids)
        if not requested:
            return 0

        with self._lock:
            remove_indices = np.array(
                [
                    index
                    for index, item in enumerate(self.metadata)
                    if item.get("doc_id") in requested
                ],
                dtype="int64",
            )
            if not len(remove_indices):
                return 0

            removed_count = int(self.index.remove_ids(remove_indices))
            self.metadata = [
                item
                for item in self.metadata
                if item.get("doc_id") not in requested
            ]
            for doc_id in requested:
                self.documents.pop(doc_id, None)
            self._persist()
            return removed_count
