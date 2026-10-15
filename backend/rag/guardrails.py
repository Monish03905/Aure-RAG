from typing import Dict, List
import re

REFUSAL_MESSAGE = (
    "I don't have enough evidence in my knowledge base to answer that reliably."
)

UNSAFE_PATTERNS = [
    r"\bhow to make a bomb\b",
    r"\bbuild a bomb\b",
    r"\bmake explosives\b",
    r"\bcreate malware\b",
    r"\bransomware\b",
    r"\bsteal (?:a|the) password\b",
    r"\bhow to hack (?:someone|an account|a system)\b",
    r"\bhow to hurt myself\b",
    r"\bhow to kill myself\b",
]


def is_unsafe_query(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(pattern, q) for pattern in UNSAFE_PATTERNS)


def validate_retrieval(results: List[Dict], minimum_score: float = 0.58) -> bool:
    if not results:
        return False
    best = results[0]
    if best.get("strategy") == "curated_demo":
        return True
    final_score = float(best.get("final_score", 0.0))
    semantic = float(best.get("semantic_score", 0.0))
    lexical = float(best.get("lexical_score", 0.0))
    if best.get("origin") == "upload":
        return final_score >= 0.45 and (semantic >= 0.58 or lexical >= 0.30)
    # Require a strong semantic signal plus at least some lexical/metadata support.
    return final_score >= minimum_score and (semantic >= 0.68 or lexical >= 0.15)


def validate_answer(answer: str) -> bool:
    return bool(answer and len(answer.strip()) >= 5)


def guard_response(query: str, answer: str, results: List[Dict]) -> Dict:
    if is_unsafe_query(query):
        return {
            "answer": "I can't help with that request.",
            "grounded": False,
            "reason": "unsafe_query",
            "confidence": 0.0,
        }
    if not validate_retrieval(results):
        return {
            "answer": REFUSAL_MESSAGE,
            "grounded": False,
            "reason": "insufficient_retrieval",
            "confidence": 0.0,
        }
    if not validate_answer(answer):
        return {
            "answer": REFUSAL_MESSAGE,
            "grounded": False,
            "reason": "invalid_answer",
            "confidence": 0.0,
        }
    confidence = min(1.0, max(0.0, float(results[0].get("final_score", 0.0))))
    return {
        "answer": answer,
        "grounded": True,
        "reason": "supported",
        "confidence": round(confidence, 4),
    }
