# Section B — Retrieval Pipeline

## Result

**NDCG@10 = 0.4265** on 29 public queries, runtime ~6s (budget: 60s).

## Architecture

Three-way Reciprocal Rank Fusion over a prebuilt index:

| Channel | Description | RRF Weight |
|---|---|---|
| **Page-dense** | 1 MiniLM vector per page (full ~256-token lead) | 1.5 |
| **Chunk-dense** | Sentence-aware chunks (300-tok, max 8/page, 40-tok overlap), top-3 mean aggregation | 1.0 |
| **BM25** | Hand-rolled page-level BM25 (k1=1.5, b=0.75) | 0.75 |

Fusion: RRF with k=10 over the union of each channel's top-200 candidates.

```
OFFLINE (build_index):
  corpus JSONs → chunk_corpus() → embed_texts() → save chunk vectors (float16)
                → embed full pages → save page vectors (float16)
                → build BM25 inverted index → save as pickle

ONLINE (run → search_batch):
  embed_queries(MiniLM)
  ├─ PAGE-DENSE:   page_vectors @ query → top-200 pages
  ├─ CHUNK-DENSE:  chunk_vectors @ query → top-200 chunks → top-3 mean per page
  └─ BM25:         BM25 scoring → top-200 pages
  → Weighted RRF fusion (k=10) over union → distinct top-10 page_ids
```

## Setup

```bash
pip install -r requirements.txt
```

Corpus: `data/Wikipedia Entries/` (27,074 pages).

## Artifacts

All prebuilt under `artifacts/` — **submit these in the repo** (staff do not rebuild):

| File | Description | Size |
|---|---|---|
| `page_vectors.npy` | Page-level embeddings (27074, 384) float16 | 20 MB |
| `page_meta.json` | Page ID mapping | 180 KB |
| `index_vectors.npy` | Chunk-level embeddings (~154k, 384) float16 | 113 MB |
| `index_meta.json` | Chunk→page mapping + chunk IDs | 2.6 MB |
| `bm25_index.pkl` | BM25 inverted index (page-level, full text) | 138 MB |

## Build Index (offline, not timed)

Run once locally to create `artifacts/`:

```bash
python scripts/build_index.py
```

## Evaluate

After building (or using prebuilt artifacts):

```bash
python scripts/eval_public.py
```

Loads prebuilt artifacts only — no rebuild needed.

## Configuration

All hyperparameters are centralized in `retrieve.py:CONFIG`:

```python
CONFIG = {
    "chunk_target_tokens": 300,      # chunk window size
    "chunk_max_per_page": 8,         # max chunks per page
    "chunk_overlap_tokens": 40,      # sentence overlap between windows
    "chunk_agg_top_k": 3,            # top-3 mean for chunk→page score
    "chunk_length_damping": 0.10,    # log(n_chunks) penalty
    "pool_depth": 200,               # candidates per channel
    "rrf_k": 10,                     # RRF smoothing parameter
    "w_page": 1.5,                   # page-dense weight
    "w_chunk": 1.0,                  # chunk-dense weight
    "w_bm25": 0.75,                  # BM25 weight
    "bm25_k1": 1.5,                  # BM25 term frequency saturation
    "bm25_b": 0.75,                  # BM25 length normalization
}
```

## Pipeline Modules

| File | Role |
|---|---|
| `main.py` | Entry point: `run()` → `search_batch()`, `build_offline_index()` → `build_index()` |
| `retrieve.py` | Query-time retrieval: three-way RRF fusion with centralized CONFIG |
| `index.py` | Offline build (chunk + page vectors + BM25) and artifact loading |
| `chunk.py` | Sentence-aware chunking with word-estimate token budget |
| `embed.py` | MiniLM embedding with auto device detection (CUDA/MPS/CPU) |
| `bm25.py` | Hand-rolled BM25 scoring (page-level, stdlib + numpy only) |
| `utils.py` | Shared paths, corpus iterator, helpers |

## Development & Analysis

| File | Purpose |
|---|---|
| `scripts/eval_dev.py` | Detailed eval with single/multi-rel breakdown |
| `scripts/sweep.py` | Fast retrieval-parameter sweep harness |
| `scripts/sweep_chunks.py` | Full chunk-geometry + parameter sweep (overnight) |
| `scripts/check_best.py` | Quick view of best sweep results |
| `experiments/` | Archived analysis scripts (query profiling, gating experiments) |
| `RESULTS.md` | Full ablation ladder, sweep results, tested-and-discarded list |

## Video

[[VIDEO LINK HERE]](https://drive.google.com/file/d/1fv6sd2OBmt2VJk_q2hbf76A_ndGyk1Rz/view?usp=sharing)
