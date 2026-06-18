"""Query profiler — cheap, interpretable features from query string + BM25 vocab."""
from __future__ import annotations

import re
from typing import Dict, Set

from bm25 import tokenize

_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "that", "this",
    "these", "those", "it", "its", "s", "t", "and", "but", "or", "nor",
    "not", "no", "so", "if", "when", "where", "who", "whom", "which",
    "what", "how", "than", "too", "very", "just",
}

_GROUPED_NUM = re.compile(r"\d{1,3}(,\d{3})+")
_YEAR_DECADE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")
_ANY_DIGIT = re.compile(r"\d")
_WH_WORDS = ["who", "where", "when", "which", "what", "how"]

_PROTECT_GROUPED = re.compile(r"\d{1,3}(,\d{3})+")
_PROTECT_NUMSPAN = re.compile(r"\d[\d.,]*\d")
_PROTECT_QUOTED = re.compile(r'"[^"]*"')

_SYNTHESIS = re.compile(
    r"\b(links?|connect(?:s|ed)?|combine[sd]?|relate[sd]?|"
    r"tie[sd]?\s+together|fit\s+together|what\s+can\s+be\s+learned)\b",
    re.IGNORECASE,
)


def _mask_protected(q: str) -> str:
    """Mask number groups and quoted spans so commas inside don't trigger splitting."""
    q = _PROTECT_GROUPED.sub("NUM", q)
    q = _PROTECT_NUMSPAN.sub("NUM", q)
    q = _PROTECT_QUOTED.sub("QUOTED", q)
    return q


def _count_enum_items(q_masked: str) -> int:
    """Count top-level items in 'A, B, and C' or 'A and B' patterns."""
    parts = re.split(r",\s*(?:and\s+)?|,?\s+and\s+", q_masked)
    content_parts = []
    for p in parts:
        words = [w for w in p.split() if w.lower() not in _STOPWORDS and len(w) > 1]
        if len(words) >= 2:
            content_parts.append(p.strip())
    return len(content_parts)


def decompose_query(query: str) -> list[str]:
    """Split an enumerated query into clauses, protecting numbers and quotes.

    Returns the original query as a single-element list if decomposition
    doesn't produce >=2 meaningful clauses.
    """
    masked = _mask_protected(query)

    # Build a map of placeholder positions to restore later
    protected_spans: list[tuple[str, str]] = []
    temp = query
    for pat in [_PROTECT_GROUPED, _PROTECT_NUMSPAN, _PROTECT_QUOTED]:
        for m in pat.finditer(query):
            protected_spans.append((m.group(), f"__PROT{len(protected_spans)}__"))
    restore_q = query
    for original, placeholder in protected_spans:
        restore_q = restore_q.replace(original, placeholder, 1)
        masked = masked.replace("NUM", placeholder, 1) if "NUM" in masked else masked

    # Split on ', and' / ', ' / ' and ' in the masked version
    parts = re.split(r",\s*(?:and\s+)?|,?\s+and\s+", restore_q)

    clauses = []
    for p in parts:
        p = p.strip().rstrip("?").strip()
        # Restore protected spans
        for original, placeholder in protected_spans:
            p = p.replace(placeholder, original)
        words = [w for w in p.split() if w.lower() not in _STOPWORDS and len(w) > 1]
        if len(words) >= 2:
            clauses.append(p)

    if len(clauses) < 2:
        return [query]
    return clauses


def profile_query(
    query: str,
    bm25_vocab: Set[str],
    bm25_idf: Dict[str, float],
) -> Dict:
    q_tokens = tokenize(query)
    content_tokens = [t for t in q_tokens if t not in _STOPWORDS]

    if content_tokens:
        in_vocab = sum(1 for t in content_tokens if t in bm25_vocab)
        coverage = in_vocab / len(content_tokens)
    else:
        coverage = 0.0

    idfs = [bm25_idf.get(t, 0.0) for t in content_tokens if t in bm25_idf]
    max_idf = max(idfs) if idfs else 0.0
    sum_idf = sum(idfs)

    q_lower = query.lower()
    wh_type = "other"
    for w in _WH_WORDS:
        if re.search(rf"\b{w}\b", q_lower):
            wh_type = w
            break

    q_masked = _mask_protected(query)
    synthesis_trigger = bool(_SYNTHESIS.search(q_masked))
    enum_count = _count_enum_items(q_masked)
    is_enumerated = synthesis_trigger or enum_count >= 2

    return {
        "has_grouped_number": bool(_GROUPED_NUM.search(query)),
        "has_year_or_decade": bool(_YEAR_DECADE.search(query)),
        "has_any_number": bool(_ANY_DIGIT.search(query)),
        "has_quote": '"' in query,
        "wh_type": wh_type,
        "coverage": coverage,
        "max_idf": max_idf,
        "sum_idf": sum_idf,
        "synthesis_trigger": synthesis_trigger,
        "enum_count": enum_count,
        "is_enumerated": is_enumerated,
        "n_words": len(query.split()),
    }
