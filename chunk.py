"""Sentence-aware chunking with token-budget windows."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

TARGET_TOKENS = 300
HARD_CEILING = 350
OVERLAP_TOKENS = 40
MAX_CHUNKS_PER_PAGE = 8
WORDS_PER_TOKEN = 0.75  # conservative: ~1.33 tokens per word for MiniLM

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD_RE = re.compile(r'\S+')


def _est_tokens(text: str) -> int:
    """Fast word-count-based token estimate (no tokenizer call)."""
    return int(len(_WORD_RE.findall(text)) / WORDS_PER_TOKEN)


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = record.get("title", "")
    full_text = entry_text(record)

    sentences = _SENT_SPLIT.split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [Chunk(page_id=page_id, chunk_id=0, text=title or "")]

    title_prefix = f"{title}\n\n" if title else ""
    title_cost = _est_tokens(title_prefix) if title_prefix else 0
    budget = TARGET_TOKENS - title_cost

    if budget < 20:
        budget = 20

    sent_lens = [_est_tokens(s) for s in sentences]

    chunks: List[Chunk] = []
    chunk_id = 0
    i = 0

    while i < len(sentences) and chunk_id < MAX_CHUNKS_PER_PAGE:
        window_sents: List[str] = []
        window_tokens = 0

        j = i
        while j < len(sentences):
            cost = sent_lens[j]
            if window_sents and window_tokens + cost > budget:
                break
            window_sents.append(sentences[j])
            window_tokens += cost
            j += 1

        if not window_sents:
            window_sents.append(sentences[i])
            j = i + 1

        body = " ".join(window_sents)
        chunk_text = (title_prefix + body) if (chunk_id > 0 and title_prefix) else body

        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
        chunk_id += 1

        # advance with overlap
        overlap_tokens = 0
        next_start = j
        for k in range(j - 1, i, -1):
            overlap_tokens += sent_lens[k]
            if overlap_tokens >= OVERLAP_TOKENS:
                next_start = k
                break

        if next_start <= i:
            next_start = j
        i = next_start

    return chunks if chunks else [Chunk(page_id=page_id, chunk_id=0, text=full_text)]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
