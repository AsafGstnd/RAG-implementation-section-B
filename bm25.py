"""Hand-rolled BM25 page-level index (stdlib + numpy only)."""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import ARTIFACTS_DIR, entry_text, iter_entries

BM25_INDEX_NAME = "bm25_index.pkl"

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKENIZE_RE.findall(text.lower())


def build_bm25_index(artifacts_dir: Path | None = None) -> Dict:
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    page_ids: List[int] = []
    doc_lens: List[int] = []
    # term -> list of (doc_idx, tf)
    postings: Dict[str, List[Tuple[int, int]]] = {}

    for doc_idx, record in enumerate(iter_entries()):
        pid = int(record["page_id"])
        page_ids.append(pid)
        text = entry_text(record)
        tokens = tokenize(text)
        doc_lens.append(len(tokens))

        tf = Counter(tokens)
        for term, count in tf.items():
            if term not in postings:
                postings[term] = []
            postings[term].append((doc_idx, count))

    N = len(page_ids)
    avgdl = sum(doc_lens) / N if N > 0 else 1.0

    idf: Dict[str, float] = {}
    for term, posting_list in postings.items():
        df = len(posting_list)
        idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    index = {
        "page_ids": page_ids,
        "doc_lens": doc_lens,
        "postings": postings,
        "idf": idf,
        "N": N,
        "avgdl": avgdl,
    }

    with open(out_dir / BM25_INDEX_NAME, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"BM25 index built: {N} pages, {len(postings)} terms")
    return index


def load_bm25_index(artifacts_dir: Path | None = None) -> Dict:
    root = artifacts_dir or ARTIFACTS_DIR
    with open(root / BM25_INDEX_NAME, "rb") as f:
        return pickle.load(f)


def bm25_score_batch(
    queries: List[str],
    index: Dict,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    top_n: int = 200,
) -> List[List[Tuple[int, float]]]:
    """Return top-n (page_id, score) pairs per query, sorted descending."""
    page_ids = index["page_ids"]
    doc_lens = index["doc_lens"]
    postings = index["postings"]
    idf = index["idf"]
    avgdl = index["avgdl"]

    results: List[List[Tuple[int, float]]] = []

    for query in queries:
        q_tokens = tokenize(query)
        scores: Dict[int, float] = {}

        for term in q_tokens:
            if term not in postings:
                continue
            term_idf = idf[term]
            for doc_idx, tf in postings[term]:
                dl = doc_lens[doc_idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score = term_idf * tf_norm
                if doc_idx in scores:
                    scores[doc_idx] += score
                else:
                    scores[doc_idx] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        results.append([(page_ids[doc_idx], sc) for doc_idx, sc in ranked])

    return results
