# Results — Ablation Ladder & Hyperparameter Sweep

## Final Result

**NDCG@10 = 0.4265** on 29 public queries (+29.7% over baseline).

| Metric | Baseline | Final | Δ |
|---|---|---|---|
| Overall NDCG@10 | 0.3289 | **0.4265** | +29.7% |
| Single-relevant (12 queries) | 0.5522 | **0.7157** | +29.6% |
| Multi-relevant (17 queries) | 0.1713 | **0.2224** | +29.8% |
| Recall@10 | 0.5308 | **0.5416** | +2.0% |
| Runtime (29 queries) | 3.97s | **6.04s** | — |

## Final Configuration

```python
CONFIG = {
    "chunk_target_tokens": 300,
    "chunk_hard_ceiling": 350,
    "chunk_overlap_tokens": 40,
    "chunk_max_per_page": 8,
    "chunk_agg_top_k": 3,          # top-3 mean
    "chunk_length_damping": 0.10,
    "pool_depth": 200,
    "rrf_k": 10,
    "w_page": 1.5,
    "w_chunk": 1.0,
    "w_bm25": 0.75,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}
```

## Architecture

Three-way Reciprocal Rank Fusion:

| Channel | Description | RRF Weight |
|---|---|---|
| Page-dense | 1 MiniLM vector per page (full ~256-token lead) | 1.5 |
| Chunk-dense | Sentence-aware chunks (300-tok, max 8/page, 40-tok overlap), top-3 mean aggregation, log-damping 0.10 | 1.0 |
| BM25 | Hand-rolled page-level BM25 (k1=1.5, b=0.75) | 0.75 |

Fusion: RRF with k=10 over the union of each channel's top-200 candidates. Tie-break: `(-score, page_id)`.

## Ablation Ladder

| Variant | Overall | Single-rel | Multi-rel | Notes |
|---|---|---|---|---|
| Phase 0: Baseline (1 vec/page, dot product) | 0.3289 | 0.5522 | 0.1713 | Starting point |
| Phase 1: +Chunking only | 0.3190 | 0.3791 | 0.1736 | Worse — long-doc bias |
| Phase 3a: Chunk + BM25 (2-way RRF) | 0.3566 | 0.5615 | 0.2120 | BM25 is the real win |
| Phase 3b: Page-dense + BM25 (2-way RRF) | 0.3593 | 0.5550 | 0.2212 | Page-dense > chunk-dense as anchor |
| Three-way RRF (locked baseline) | 0.4065 | 0.6900 | 0.2064 | Best architecture |
| **Three-way RRF (tuned)** | **0.4265** | **0.7157** | **0.2224** | **After 90k+ config sweep** |

## Hyperparameter Sweep Summary

Ran a structured sweep across 90,720 configurations spanning:
- 48 chunk geometries: tokens ∈ {200, 256, 300, 400} × max_chunks ∈ {4, 8, 12} × overlap ∈ {0, 20, 40, 60}
- Per geometry: weights × RRF k × damping × aggregation method (2,160 combos each)
- Plus 2,012 fast retrieval-only configs on the fixed index

### Consistent patterns across top configs

- **overlap=40** in every top-10 result
- **damping=0.10–0.15** universally better than the original 0.05
- **agg=top-3 mean** beating max-pool and top-2 mean
- **chunk weight raised to 1.0** (from original 0.5) — chunks contribute more as a fusion channel
- **300-token chunks** with the existing index turned out optimal

### Key finding

The best config (0.4265) uses the **same chunk index** as our locked baseline (300tok/8max/40ov) — only retrieval-time parameters changed. No index rebuild was needed.

## Query-Type Analysis

Profiled all 29 queries by type using `query_profile.py`:

| Feature | Queries where BM25 helps | Queries where BM25 hurts | Neutral |
|---|---|---|---|
| Count | 12 | 6 | 11 |

- BM25 hurts with **high confidence** (max BM25 scores 34–48), not low — a floor-based silence gate cannot help
- Enumeration flag (`is_enumerated`) tracks multi-relevant well, but additive decomposition was net-negative
- Decision: **ship ungated** because no gate passed acceptance criteria (overall improvement without single-rel regression)

## Tested and Discarded

| Mechanism | Result | Reason |
|---|---|---|
| Chunking as primary signal | 0.3190 (worse) | Long-doc bias: pages with many chunks get noisy high max scores |
| Neighbor/cluster expansion | Net zero | Helps multi-rel, equally hurts single-rel |
| PRF / Rocchio | No improvement | Drifts on short factoid queries |
| Query decomposition (clause splitting) | 0.3876 (worse) | Mangled numbers; fragments too vague to match |
| Shorter anchor text (title-only, title+1-2 sentences) | 0.05–0.28 | Less context hurts disambiguation |
| Score-normalized fusion (vs RRF) | 0.3755 (worse) | RRF's rank-based approach is more robust |
| Query prefix ("search:", "query:") | 0.35–0.39 (worse) | MiniLM-L6-v2 wasn't trained with prefixes |
| Per-channel RRF k | No improvement | Uniform k is near-optimal |
| BM25 field weighting (title 3x) | Identical | Title terms already captured |
| BM25 stopword removal | +0.001 | Noise |
| BM25 grouped-number tokenizer | Identical | Split tokens already rare enough to match |
| Title-only dense index (4th channel) | No improvement | Page-dense already captures title |
| Offline cluster-mass bonus | +0.004 | Within noise |
| BM25-silence gating | Not justified | BM25 hurts with high confidence, not low |
| Number-boost gating | Not justified | Only 1 applicable query |

## Dev Set Note

The dev set is 29 queries (12 single-relevant, 17 multi-relevant). At this size, **1 query ≈ 0.034 NDCG**. The 0.4265 → 0.4065 improvement (+0.020) represents approximately half a query flip — meaningful but within noise. We report it honestly and do not over-claim.
