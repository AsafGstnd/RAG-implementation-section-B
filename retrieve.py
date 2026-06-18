"""Query-time retrieval: three-way RRF (page-dense + chunk-dense + BM25)."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bm25 import bm25_score_batch, load_bm25_index
from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

# ── Centralized config ────────────────────────────────────────────────
CONFIG = {
    # Chunk parameters (used at build time by chunk.py)
    "chunk_target_tokens": 300,
    "chunk_hard_ceiling": 350,
    "chunk_overlap_tokens": 40,
    "chunk_max_per_page": 8,
    # Chunk aggregation
    "chunk_agg_top_k": 3,          # top-3 mean for chunk→page score
    "chunk_length_damping": 0.10,  # log(n_chunks) penalty
    # Per-channel candidate pool depth
    "pool_depth": 200,
    # RRF
    "rrf_k": 10,
    "w_page": 1.5,
    "w_chunk": 1.0,
    "w_bm25": 0.75,
    # BM25 (used at build time by bm25.py, and at query time)
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}
# ──────────────────────────────────────────────────────────────────────


def _cfg(cfg: Optional[Dict[str, Any]], key: str):
    if cfg is not None and key in cfg:
        return cfg[key]
    return CONFIG[key]


def load_all(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load all artifacts. Public so the sweep harness can call once and reuse."""
    root = artifacts_dir or ARTIFACTS_DIR

    page_vectors = np.load(root / "page_vectors.npy").astype(np.float32)
    page_meta = json.loads((root / "page_meta.json").read_text(encoding="utf-8"))
    page_ids = page_meta["page_ids"]

    chunk_vectors, chunk_page_ids = load_index(artifacts_dir)
    chunk_counts = Counter(chunk_page_ids)

    bm25_index = load_bm25_index(artifacts_dir)

    return {
        "page_vectors": page_vectors,
        "page_ids": page_ids,
        "chunk_vectors": chunk_vectors,
        "chunk_page_ids": chunk_page_ids,
        "chunk_counts": chunk_counts,
        "bm25_index": bm25_index,
    }


_cache: Dict | None = None


def _get_index(index: Optional[Dict], artifacts_dir: Optional[Path]) -> Dict:
    global _cache
    if index is not None:
        return index
    if _cache is not None:
        return _cache
    _cache = load_all(artifacts_dir)
    return _cache


def _page_dense_ranking(
    qv: np.ndarray, page_vectors: np.ndarray, page_ids: List[int],
    depth: int,
) -> List[Tuple[int, float]]:
    sims = page_vectors @ qv
    order = np.argsort(-sims)[:depth]
    return [(page_ids[int(i)], float(sims[i])) for i in order]


def _chunk_dense_ranking(
    qv: np.ndarray, chunk_vectors: np.ndarray,
    chunk_page_ids: List[int], chunk_counts: Counter,
    depth: int, agg_k: int, damping: float,
) -> List[Tuple[int, float]]:
    sims = chunk_vectors @ qv
    top_idx = np.argpartition(-sims, depth)[:depth]

    page_sims: dict[int, list[float]] = defaultdict(list)
    for idx in top_idx:
        pid = chunk_page_ids[int(idx)]
        page_sims[pid].append(float(sims[idx]))

    agg = {}
    for pid, s_list in page_sims.items():
        s_list.sort(reverse=True)
        if agg_k == -1:
            raw = float(np.mean(s_list))
        else:
            raw = float(np.mean(s_list[:agg_k]))
        agg[pid] = raw - damping * math.log(chunk_counts[pid])

    return sorted(agg.items(), key=lambda x: (-x[1], x[0]))


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    cfg: Optional[Dict[str, Any]] = None,
    index: Optional[Dict[str, Any]] = None,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    data = _get_index(index, artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    k = _cfg(cfg, "rrf_k")
    w_page = _cfg(cfg, "w_page")
    w_chunk = _cfg(cfg, "w_chunk")
    w_bm25 = _cfg(cfg, "w_bm25")
    depth = _cfg(cfg, "pool_depth")
    agg_k = _cfg(cfg, "chunk_agg_top_k")
    damping = _cfg(cfg, "chunk_length_damping")
    bm25_k1 = _cfg(cfg, "bm25_k1")
    bm25_b = _cfg(cfg, "bm25_b")

    bm25_results = bm25_score_batch(
        queries, data["bm25_index"], k1=bm25_k1, b=bm25_b, top_n=depth,
    )

    ranked: List[List[int]] = []
    for i, qv in enumerate(query_vectors):
        page_r = _page_dense_ranking(
            qv, data["page_vectors"], data["page_ids"], depth,
        )
        chunk_r = _chunk_dense_ranking(
            qv, data["chunk_vectors"], data["chunk_page_ids"],
            data["chunk_counts"], depth, agg_k, damping,
        )

        scores: Dict[int, float] = {}
        for rank, (pid, _) in enumerate(page_r):
            scores[pid] = scores.get(pid, 0.0) + w_page / (k + rank + 1)
        for rank, (pid, _) in enumerate(chunk_r[:depth]):
            scores[pid] = scores.get(pid, 0.0) + w_chunk / (k + rank + 1)
        for rank, (pid, _) in enumerate(bm25_results[i]):
            scores[pid] = scores.get(pid, 0.0) + w_bm25 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        ranked.append([pid for pid, _ in fused[:top_k]])

    return ranked
