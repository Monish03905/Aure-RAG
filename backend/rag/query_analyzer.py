import re
from typing import Dict

TELUGU_PATTERN = re.compile(r"[\u0C00-\u0C7F]")

ENGLISH_TO_TELUGU = {
    "what is a corporation?": "కార్పొరేషన్ అంటే ఏమిటి?",
    "what is corporation?": "కార్పొరేషన్ అంటే ఏమిటి?",
    "define corporation": "కార్పొరేషన్ అంటే ఏమిటి?",
    "what is artificial intelligence?": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
    "what is ai?": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
    "define artificial intelligence": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
    "define ai": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
    "what is machine learning?": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
    "what is ml?": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
    "define machine learning": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
    "define ml": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
    "what is data science?": "డేటా సైన్స్ అంటే ఏమిటి?",
    "define data science": "డేటా సైన్స్ అంటే ఏమిటి?",
    "what is rag?": "RAG అంటే ఏమిటి?",
    "what is rag": "RAG అంటే ఏమిటి?",
    "define rag": "RAG అంటే ఏమిటి?",
}


def detect_language(query: str) -> str:
    if not query or not query.strip():
        return "unknown"

    compact = re.sub(r"\s+", "", query)
    telugu_count = len(TELUGU_PATTERN.findall(query))

    return (
        "tel_Telu"
        if telugu_count / max(len(compact), 1) > 0.30
        else "eng_Latn"
    )


def normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", (query or "").strip())
    if not query:
        return ""

    key = query.lower()

    if key in ENGLISH_TO_TELUGU:
        return ENGLISH_TO_TELUGU[key]

    return query


def detect_query_type(query: str) -> str:
    q = (query or "").lower().strip()

    if (
        any(x in query for x in ["ఏమిటి", "ఏమి", "అంటే ఏమిటి"])
        or any(
            x in q
            for x in [
                "what is",
                "what are",
                "define",
                "what does",
                "who is",
                "who are",
            ]
        )
    ):
        return "DESCRIPTION"

    if "why" in q or "ఎందుకు" in query:
        return "WHY"

    if "how" in q or "ఎలా" in query:
        return "HOW"

    if (
        "difference" in q
        or "compare" in q
        or "versus" in q
        or " vs " in q
        or "తేడా" in query
        or "పోల్చి" in query
    ):
        return "COMPARISON"

    if "when" in q or "ఎప్పుడు" in query:
        return "TEMPORAL"

    return "GENERAL"


def estimate_complexity(query: str) -> str:
    words = [w for w in query.split() if w]

    if len(words) <= 6:
        return "SIMPLE"
    if len(words) <= 15:
        return "MEDIUM"
    return "COMPLEX"


def analyze_query(query: str) -> Dict:
    original = (query or "").strip()

    return {
        "language": detect_language(original),
        "query_type": detect_query_type(original),
        "complexity": estimate_complexity(original),
        "normalized_query": normalize_query(original),
        "original_query": original,
    }