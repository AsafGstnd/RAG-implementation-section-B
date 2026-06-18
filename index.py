"""Offline index build and load."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
PAGE_VECTORS_NAME = "page_vectors.npy"
PAGE_META_NAME = "page_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Build all artifacts: chunk vectors, page vectors, BM25 index."""
    from bm25 import build_bm25_index

    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    # Chunk-level vectors
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors.astype(np.float16))
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Page-level vectors (full lead, ~256 token truncation by MiniLM)
    page_texts = [entry_text(r) for r in records]
    page_pids = [int(r["page_id"]) for r in records]
    page_vectors = embed_texts(page_texts)
    np.save(out_dir / PAGE_VECTORS_NAME, page_vectors.astype(np.float16))
    (out_dir / PAGE_META_NAME).write_text(
        json.dumps({"page_ids": page_pids}), encoding="utf-8"
    )

    # BM25 index
    build_bm25_index(out_dir)

    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed chunk vectors and page_id map."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME).astype(np.float32)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids
