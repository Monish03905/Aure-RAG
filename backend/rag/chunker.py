import re
from typing import Dict, List


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?।])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def sentence_chunking(text: str, sentences_per_chunk: int = 2) -> List[str]:
    sentences = split_sentences(text)
    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), max(1, sentences_per_chunk))
        if sentences[i:i + sentences_per_chunk]
    ]


def sliding_window_chunking(text: str, window_size: int = 4, overlap: int = 1) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    step = max(1, window_size - overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        current = sentences[start:start + window_size]
        if not current:
            break
        chunks.append(" ".join(current))
        if start + window_size >= len(sentences):
            break
    return chunks


def metadata_aware_chunking(text: str, metadata: Dict) -> List[Dict]:
    chunks = sliding_window_chunking(text, window_size=4, overlap=1)
    results = []
    for idx, chunk in enumerate(chunks):
        results.append({
            "text": chunk,
            "chunk_index": idx,
            "query_id": metadata.get("query_id"),
            "query_type": metadata.get("query_type"),
            "source_lang": metadata.get("source_lang"),
            "target_lang": metadata.get("target_lang"),
            "is_selected": metadata.get("is_selected"),
            "strategy": "metadata_aware",
        })
    return results


def create_chunks(text: str, metadata: Dict) -> Dict:
    return {
        "sentence": sentence_chunking(text, 2),
        "sliding": sliding_window_chunking(text, 4, 1),
        "metadata": metadata_aware_chunking(text, metadata),
    }
