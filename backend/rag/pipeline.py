"""Structured offline RAG harness used by the AURA chat agent."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer

from backend.config import (
    BASE_INDEX_FILE,
    BASE_METADATA_FILE,
    EMBEDDING_MODEL,
    LOCAL_LLM_MODEL,
)
from backend.rag.generator import generate_answer
from backend.rag.guardrails import (
    REFUSAL_MESSAGE,
    guard_response,
    is_unsafe_query,
    validate_retrieval,
)
from backend.rag.query_analyzer import analyze_query
from backend.rag.reranker import deduplicate_candidates, rerank_candidates
from backend.rag.user_store import UserDocumentStore


DEMO_KNOWLEDGE = [
    {
        "keys": [
            "what is a corporation",
            "what is corporation",
            "define corporation",
            "corporation",
            "కార్పొరేషన్ అంటే ఏమిటి",
        ],
        "query": "కార్పొరేషన్ అంటే ఏమిటి?",
        "answer": (
            "కార్పొరేషన్ అనేది ఒకే సంస్థగా వ్యవహరించడానికి మరియు "
            "చట్టంలో గుర్తింపు పొందిన వ్యక్తుల సమూహం లేదా కంపెనీ."
        ),
        "type": "DESCRIPTION",
    },
    {
        "keys": [
            "what is artificial intelligence",
            "what is ai",
            "define artificial intelligence",
            "define ai",
            "artificial intelligence",
            "కృత్రిమ మేధస్సు అంటే ఏమిటి",
        ],
        "query": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
        "answer": (
            "కృత్రిమ మేధస్సు (AI) అనేది కంప్యూటర్లు మరియు యంత్రాలు "
            "మనుషుల మేధస్సుకు సంబంధించిన పనులను నేర్చుకోవడం, "
            "అర్థం చేసుకోవడం మరియు నిర్వహించడం చేసే సాంకేతికత."
        ),
        "type": "DESCRIPTION",
    },
    {
        "keys": [
            "what is machine learning",
            "what is ml",
            "define machine learning",
            "define ml",
            "machine learning",
            "మెషిన్ లెర్నింగ్ అంటే ఏమిటి",
        ],
        "query": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
        "answer": (
            "మెషిన్ లెర్నింగ్ (ML) అనేది కంప్యూటర్లు స్పష్టంగా "
            "ప్రోగ్రామ్ చేయకుండా డేటా నుండి నేర్చుకొని, "
            "దాని ఆధారంగా అంచనాలు లేదా నిర్ణయాలు చేయడానికి "
            "ఉపయోగించే AI విధానం."
        ),
        "type": "DESCRIPTION",
    },
    {
        "keys": [
            "what is data science",
            "define data science",
            "data science",
            "డేటా సైన్స్ అంటే ఏమిటి",
        ],
        "query": "డేటా సైన్స్ అంటే ఏమిటి?",
        "answer": (
            "డేటా సైన్స్ అనేది డేటా నుండి ఉపయోగకరమైన సమాచారం, "
            "నమూనాలు మరియు అంతర్దృష్టులను కనుగొనడానికి గణాంకాలు, "
            "ప్రోగ్రామింగ్ మరియు మెషిన్ లెర్నింగ్ వంటి పద్ధతులను "
            "ఉపయోగించే రంగం."
        ),
        "type": "DESCRIPTION",
    },
    {
        "keys": [
            "what is rag",
            "define rag",
            "rag",
            "rag అంటే ఏమిటి",
        ],
        "query": "RAG అంటే ఏమిటి?",
        "answer": (
            "RAG (Retrieval-Augmented Generation) అనేది సమాధానం "
            "ఇవ్వడానికి ముందు సంబంధిత సమాచారాన్ని జ్ఞాన భాండాగారం "
            "లేదా డేటాబేస్ నుండి తిరిగి పొందే పద్ధతి."
        ),
        "type": "DESCRIPTION",
    },
]


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s\u0C00-\u0C7F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_demo(query: str) -> Optional[Dict]:
    q = _norm(query)
    for item in DEMO_KNOWLEDGE:
        for key in item["keys"]:
            if q == _norm(key):
                return item
    return None


def _demo_source(item: Dict) -> Dict:
    return {
        "chunk_id": "demo",
        "text": item["answer"],
        "strategy": "curated_demo",
        "query_id": "demo",
        "query": item["query"],
        "answer": item["answer"],
        "query_type": item["type"],
        "source_lang": "eng_Latn",
        "target_lang": "tel_Telu",
        "passage_index": 0,
        "is_selected": True,
        "chunk_index": 0,
        "origin": "demo",
        "similarity": 1.0,
        "semantic_score": 1.0,
        "metadata_score": 1.0,
        "lexical_score": 1.0,
        "query_match_score": 1.0,
        "final_score": 1.0,
    }


class AuraRAG:
    def __init__(
        self,
        index_file: Path = BASE_INDEX_FILE,
        metadata_file: Path = BASE_METADATA_FILE,
    ):
        print("Loading AURA...")
        self.model_name = EMBEDDING_MODEL

        if not Path(index_file).is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not Path(metadata_file).is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        self.index = faiss.read_index(str(index_file))
        self.metadata: List[Dict] = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item.setdefault("origin", "base")
                    self.metadata.append(item)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS/metadata mismatch: {self.index.ntotal} vs {len(self.metadata)}"
            )

        self.chunk_count = len(self.metadata)
        self.embedding_model = SentenceTransformer(self.model_name)
        self.user_store = UserDocumentStore(self.embedding_model, dim=384)
        print(
            f"AURA ready. Base={self.chunk_count} | Uploads={self.user_store.size}"
        )

    def ingest_document(self, path: Path, original_name: Optional[str] = None) -> Dict:
        return self.user_store.ingest_file(path, original_name=original_name)

    def choose_k(self, analysis: Dict) -> int:
        if analysis.get("complexity") == "COMPLEX":
            return 30
        if analysis.get("query_type") == "COMPARISON":
            return 25
        if analysis.get("query_type") in {"HOW", "WHY"}:
            return 25
        return 20

    def _search_base(self, retrieval_query: str, k: int) -> tuple[List[Dict], float, float]:
        embedding_start = time.perf_counter()
        query_embedding = self.embedding_model.encode(
            [retrieval_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        embedding_ms = (time.perf_counter() - embedding_start) * 1000

        search_start = time.perf_counter()
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        search_ms = (time.perf_counter() - search_start) * 1000

        candidates: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["similarity"] = float(score)
            item.setdefault("origin", "base")
            candidates.append(item)
        return candidates, embedding_ms, search_ms

    def _retrieve_once(
        self,
        query: str,
        use_base_corpus: bool = True,
        use_uploads: bool = True,
        session_attachments: Optional[List[Dict]] = None,
    ):
        analyze_start = time.perf_counter()
        analysis = analyze_query(query)
        analyze_ms = (time.perf_counter() - analyze_start) * 1000

        retrieval_query = analysis.get("normalized_query") or query
        first_stage_k = self.choose_k(analysis)

        candidates: List[Dict] = []
        embedding_ms = 0.0
        faiss_ms = 0.0
        user_faiss_ms = 0.0

        if use_base_corpus and self.index.ntotal > 0:
            base_hits, embedding_ms, faiss_ms = self._search_base(
                retrieval_query, first_stage_k
            )
            candidates.extend(base_hits)

        if use_uploads and self.user_store.size > 0:
            doc_ids = None
            if session_attachments is not None:
                doc_ids = [
                    a.get("doc_id")
                    for a in session_attachments
                    if a.get("doc_id")
                ]

            # A chat may search only its own attachments; the legacy /api/ask
            # endpoint passes None to retain its corpus-wide upload behavior.
            if session_attachments is None or doc_ids:
                user_start = time.perf_counter()
                user_hits = self.user_store.search(
                    retrieval_query,
                    top_k=min(first_stage_k, 20),
                    doc_ids=doc_ids,
                )
                user_faiss_ms = (time.perf_counter() - user_start) * 1000
                # Boost upload relevance slightly so attached docs win when relevant.
                for hit in user_hits:
                    hit["similarity"] = float(hit.get("similarity", 0.0)) + 0.05
                    hit["origin"] = "upload"
                candidates.extend(user_hits)

        dedup = deduplicate_candidates(candidates)

        rerank_start = time.perf_counter()
        reranked = rerank_candidates(dedup, retrieval_query, analysis)
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        return reranked[:12], analysis, {
            "analyze_ms": analyze_ms,
            "embedding_ms": embedding_ms,
            "faiss_ms": faiss_ms,
            "user_faiss_ms": user_faiss_ms,
            "rerank_ms": rerank_ms,
            "candidate_count": len(candidates),
            "dedup_count": len(dedup),
        }

    def retrieve(
        self,
        query: str,
        retries: int = 2,
        use_base_corpus: bool = True,
        use_uploads: bool = True,
        session_attachments: Optional[List[Dict]] = None,
    ):
        demo = _find_demo(query)
        if demo is not None and use_base_corpus:
            return (
                [_demo_source(demo)],
                analyze_query(query),
                {
                    "analyze_ms": 0.0,
                    "embedding_ms": 0.0,
                    "faiss_ms": 0.0,
                    "user_faiss_ms": 0.0,
                    "rerank_ms": 0.0,
                    "candidate_count": 1,
                    "dedup_count": 1,
                },
            )

        last_error = None
        for attempt in range(retries + 1):
            try:
                return self._retrieve_once(
                    query,
                    use_base_corpus=use_base_corpus,
                    use_uploads=use_uploads,
                    session_attachments=session_attachments,
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.01 * (2 ** attempt))
        raise RuntimeError(f"Retrieval failed after retries: {last_error}")

    def ask(
        self,
        query: str,
        retries: int = 2,
        use_base_corpus: bool = True,
        use_uploads: bool = True,
        session_attachments: Optional[List[Dict]] = None,
    ) -> Dict:
        query = (query or "").strip()
        if not query:
            return {
                "query": query,
                "answer": REFUSAL_MESSAGE,
                "grounded": False,
                "confidence": 0.0,
                "reason": "empty_query",
                "analysis": {},
                "sources": [],
                "stages_ms": {},
                "total_ms": 0.0,
                "index_size": self.chunk_count + self.user_store.size,
                "generation_mode": "not_run",
            }

        total_start = time.perf_counter()
        results, analysis, timings = self.retrieve(
            query,
            retries=retries,
            use_base_corpus=use_base_corpus,
            use_uploads=use_uploads,
            session_attachments=session_attachments,
        )

        if not results:
            return {
                "query": query,
                "answer": REFUSAL_MESSAGE,
                "grounded": False,
                "confidence": 0.0,
                "reason": "no_results",
                "analysis": analysis,
                "sources": [],
                "stages_ms": timings,
                "total_ms": round((time.perf_counter() - total_start) * 1000, 3),
                "index_size": self.chunk_count + self.user_store.size,
                "generation_mode": "not_run",
            }

        generation_start = time.perf_counter()
        if is_unsafe_query(query):
            generated = {
                "answer": REFUSAL_MESSAGE,
                "grounded": False,
                "sources": [],
                "generation_mode": "guardrail",
            }
        elif not validate_retrieval(results):
            generated = {
                "answer": REFUSAL_MESSAGE,
                "grounded": False,
                "sources": results[:5],
                "generation_mode": "guardrail",
            }
        else:
            generated = generate_answer(query, results)
        timings["generation_ms"] = (time.perf_counter() - generation_start) * 1000

        guard_start = time.perf_counter()
        guarded = guard_response(query, generated["answer"], generated["sources"])
        timings["guardrail_ms"] = (time.perf_counter() - guard_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        timings = {key: round(value, 3) for key, value in timings.items()}

        confidence = (
            1.0
            if guarded["grounded"]
            and results
            and results[0].get("strategy") == "curated_demo"
            else float(guarded.get("confidence", 0.0))
        )

        return {
            "query": query,
            "answer": guarded["answer"],
            "grounded": guarded["grounded"],
            "confidence": round(confidence, 4),
            "reason": guarded["reason"],
            "analysis": analysis,
            "sources": generated["sources"][:5],
            "stages_ms": timings,
            "total_ms": round(total_ms, 3),
            "index_size": self.chunk_count + self.user_store.size,
            "generation_mode": generated.get(
                "generation_mode",
                "ollama" if LOCAL_LLM_MODEL else "extractive",
            ),
        }
