from typing import Dict, List
import re


def normalize_score(score: float) -> float:
    value = float(score)

    if -1.0 <= value <= 1.0:
        return (value + 1.0) / 2.0

    return max(0.0, min(1.0, value))


def normalize_text(text: str) -> str:
    text = (text or "").lower()

    text = re.sub(
        r"[^\w\s\u0C00-\u0C7F]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
    }


def lexical_overlap(
    query: str,
    text: str
) -> float:

    q = _tokens(query)
    t = _tokens(text)

    if not q or not t:
        return 0.0

    return len(q & t) / len(q)


def query_similarity(
    query: str,
    candidate_query: str
) -> float:

    q = normalize_text(query)
    c = normalize_text(candidate_query)

    if not q or not c:
        return 0.0

    if q == c:
        return 1.0

    q_tokens = _tokens(q)
    c_tokens = _tokens(c)

    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)

    return overlap / max(
        len(q_tokens),
        len(c_tokens)
    )


def _is_selected(value) -> bool:

    return (
        value is True
        or value == 1
        or str(value).strip().lower()
        in {"1", "true", "yes"}
    )


def metadata_score(
    query_metadata: Dict,
    candidate: Dict
) -> float:

    score = 0.0

    if (
        query_metadata.get("language")
        == candidate.get("target_lang")
    ):
        score += 0.15

    if (
        query_metadata.get("query_type")
        == candidate.get("query_type")
    ):
        score += 0.15

    candidate_query = candidate.get(
        "query",
        ""
    )

    original_query = query_metadata.get(
        "_original_query",
        ""
    )

    qmatch = query_similarity(
        original_query,
        candidate_query
    )

    # Exact dataset-query match is extremely strong evidence.
    if qmatch >= 0.99:
        score += 0.55

    elif qmatch >= 0.75:
        score += 0.35

    elif qmatch >= 0.50:
        score += 0.15

    # is_selected is useful, but MUST NOT dominate semantic relevance.
    if _is_selected(
        candidate.get("is_selected")
    ):
        score += 0.05

    query_type = query_metadata.get(
        "query_type"
    )

    strategy = candidate.get(
        "strategy"
    )

    if (
        query_type == "COMPARISON"
        and strategy == "sliding"
    ):
        score += 0.05

    elif (
        query_type == "DESCRIPTION"
        and strategy == "sentence"
    ):
        score += 0.05

    elif (
        query_type in {"HOW", "WHY"}
        and strategy == "metadata_aware"
    ):
        score += 0.05

    return min(
        score,
        1.0
    )


def rerank_candidates(
    candidates: List[Dict],
    query: str,
    query_metadata: Dict
) -> List[Dict]:

    # Pass original query into metadata scoring.
    query_metadata = dict(query_metadata)
    query_metadata["_original_query"] = query

    reranked = []

    for candidate in candidates:

        semantic = normalize_score(
            candidate.get(
                "similarity",
                0.0
            )
        )

        meta = metadata_score(
            query_metadata,
            candidate
        )

        lexical = lexical_overlap(
            query,
            candidate.get(
                "text",
                ""
            )
        )

        candidate_query_match = query_similarity(
            query,
            candidate.get(
                "query",
                ""
            )
        )

        # Query match is important for dataset-backed QA.
        final_score = (
            0.50 * semantic
            + 0.20 * meta
            + 0.15 * lexical
            + 0.15 * candidate_query_match
        )

        item = candidate.copy()

        item.update({
            "semantic_score": round(
                semantic,
                6
            ),
            "metadata_score": round(
                meta,
                6
            ),
            "lexical_score": round(
                lexical,
                6
            ),
            "query_match_score": round(
                candidate_query_match,
                6
            ),
            "final_score": round(
                final_score,
                6
            ),
        })

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return reranked


def deduplicate_candidates(
    candidates: List[Dict]
) -> List[Dict]:

    seen = set()
    unique = []

    for candidate in candidates:

        text = (
            candidate.get(
                "text",
                ""
            )
            or ""
        )

        key = normalize_text(
            text
        )

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    return unique