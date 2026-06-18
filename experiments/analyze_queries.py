"""Query-type profiling and diagnostic analysis."""
from __future__ import annotations

import sys
import math
from pathlib import Path
from collections import Counter, defaultdict

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import numpy as np
from eval import load_query_file, ndcg_at_k, K_EVAL
from utils import PUBLIC_QUERIES_PATH, ARTIFACTS_DIR
from embed import embed_queries
from bm25 import load_bm25_index, bm25_score_batch, tokenize
from query_profile import profile_query
import json


def load_artifacts():
    pvecs = np.load(ARTIFACTS_DIR / "page_vectors.npy").astype(np.float32)
    pmeta = json.loads((ARTIFACTS_DIR / "page_meta.json").read_text())
    pids_p = pmeta["page_ids"]

    cvecs = np.load(ARTIFACTS_DIR / "index_vectors.npy").astype(np.float32)
    cmeta = json.loads((ARTIFACTS_DIR / "index_meta.json").read_text())
    pids_c = [int(x) for x in cmeta["page_ids"]]
    cc = Counter(pids_c)

    bm25_index = load_bm25_index()
    return pvecs, pids_p, cvecs, pids_c, cc, bm25_index


def page_ranking(qv, pvecs, pids_p):
    sims = pvecs @ qv
    order = np.argsort(-sims)[:200]
    return [(pids_p[int(i)], float(sims[i])) for i in order]


def chunk_ranking(qv, cvecs, pids_c, cc):
    sims = cvecs @ qv
    top_idx = np.argpartition(-sims, 200)[:200]
    page_sims = defaultdict(list)
    for idx in top_idx:
        pid = pids_c[int(idx)]
        page_sims[pid].append(float(sims[idx]))
    agg = {}
    for pid, sl in page_sims.items():
        sl.sort(reverse=True)
        agg[pid] = float(np.mean(sl[:2])) - 0.05 * math.log(cc[pid])
    return sorted(agg.items(), key=lambda x: (-x[1], x[0]))


def fuse(page_r, chunk_r, bm25_r, w_page=2.0, w_chunk=0.5, w_bm25=1.0, k=30):
    scores = {}
    for rank, (pid, _) in enumerate(page_r):
        scores[pid] = scores.get(pid, 0) + w_page / (k + rank + 1)
    for rank, (pid, _) in enumerate(chunk_r[:200]):
        scores[pid] = scores.get(pid, 0) + w_chunk / (k + rank + 1)
    for rank, (pid, _) in enumerate(bm25_r):
        scores[pid] = scores.get(pid, 0) + w_bm25 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def main():
    pvecs, pids_p, cvecs, pids_c, cc, bm25_index = load_artifacts()
    bm25_vocab = set(bm25_index["idf"].keys())
    bm25_idf = bm25_index["idf"]

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    relevants = [r["relevant_page_ids"] for r in rows]
    qvecs = embed_queries(queries)
    bm25_results = bm25_score_batch(queries, bm25_index, top_n=200)

    results = []

    for i, (query, qv) in enumerate(zip(queries, qvecs)):
        prof = profile_query(query, bm25_vocab, bm25_idf)
        n_rel = len(relevants[i])
        subset = "single" if n_rel == 1 else "multi"

        pr = page_ranking(qv, pvecs, pids_p)
        cr = chunk_ranking(qv, cvecs, pids_c, cc)
        br = bm25_results[i]

        # Full three-way
        f_full = fuse(pr, cr, br)
        ndcg_full = ndcg_at_k([p for p, _ in f_full[:10]], relevants[i], K_EVAL)

        # BM25 off
        f_nobm25 = fuse(pr, cr, br, w_bm25=0.0)
        ndcg_nobm25 = ndcg_at_k([p for p, _ in f_nobm25[:10]], relevants[i], K_EVAL)

        # Page-only
        ndcg_pageonly = ndcg_at_k([p for p, _ in pr[:10]], relevants[i], K_EVAL)

        # BM25 max score
        bm25_max = br[0][1] if br else 0.0

        results.append({
            "qid": rows[i]["query_id"],
            "query": query,
            "n_rel": n_rel,
            "subset": subset,
            "ndcg_full": ndcg_full,
            "ndcg_nobm25": ndcg_nobm25,
            "ndcg_pageonly": ndcg_pageonly,
            "bm25_delta": ndcg_full - ndcg_nobm25,
            "bm25_max": bm25_max,
            **prof,
        })

    # === REPORT ===
    print("\n" + "=" * 100)
    print("  QUERY-TYPE ANALYSIS")
    print("=" * 100)

    # Per-query table
    print(f"\n{'QID':<16} {'#R':>3} {'Type':<6} {'Full':>6} {'NoBM':>6} {'Page':>6} {'Δbm25':>6} {'Enum':>4} {'Cov':>5} {'WhType':<6} Query")
    print("-" * 120)
    for r in results:
        print(f"{r['qid']:<16} {r['n_rel']:>3} {r['subset']:<6} {r['ndcg_full']:>6.3f} {r['ndcg_nobm25']:>6.3f} {r['ndcg_pageonly']:>6.3f} {r['bm25_delta']:>+6.3f} {'Y' if r['is_enumerated'] else 'N':>4} {r['coverage']:>5.2f} {r['wh_type']:<6} {r['query'][:50]}")

    # Bucket analyses
    def bucket_report(name, key_fn):
        buckets = defaultdict(list)
        for r in results:
            buckets[key_fn(r)].append(r)
        print(f"\n--- {name} ---")
        print(f"{'Bucket':<20} {'N':>3} {'Mean NDCG':>10} {'Mean Δbm25':>11} {'Mean Cov':>9}")
        for bk in sorted(buckets.keys(), key=str):
            items = buckets[bk]
            mn = np.mean([r["ndcg_full"] for r in items])
            md = np.mean([r["bm25_delta"] for r in items])
            mc = np.mean([r["coverage"] for r in items])
            print(f"{str(bk):<20} {len(items):>3} {mn:>10.4f} {md:>+11.4f} {mc:>9.3f}")

    bucket_report("has_grouped_number", lambda r: r["has_grouped_number"])
    bucket_report("has_year_or_decade", lambda r: r["has_year_or_decade"])
    bucket_report("has_any_number", lambda r: r["has_any_number"])
    bucket_report("is_enumerated", lambda r: r["is_enumerated"])
    bucket_report("wh_type", lambda r: r["wh_type"])

    # Coverage buckets
    def cov_bucket(r):
        c = r["coverage"]
        if c < 0.5: return "low(<0.5)"
        if c < 0.8: return "med(0.5-0.8)"
        return "high(>=0.8)"
    bucket_report("coverage", cov_bucket)

    # Cross-tab: is_enumerated × subset
    print("\n--- Cross-tab: is_enumerated × single/multi ---")
    ct = defaultdict(list)
    for r in results:
        ct[(r["is_enumerated"], r["subset"])].append(r["ndcg_full"])
    for key in sorted(ct.keys()):
        vals = ct[key]
        print(f"  enum={key[0]!s:<6} {key[1]:<6}: N={len(vals):>2}  mean NDCG={np.mean(vals):.4f}")

    # BM25-hurting queries
    print("\n--- Queries where BM25 HURTS (ndcg_nobm25 > ndcg_full) ---")
    hurting = [r for r in results if r["bm25_delta"] < -0.001]
    if hurting:
        for r in hurting:
            print(f"  {r['qid']}: Δ={r['bm25_delta']:+.4f}  bm25_max={r['bm25_max']:.2f}  cov={r['coverage']:.2f}  enum={r['is_enumerated']}  {r['query'][:60]}")
    else:
        print("  None — BM25 never hurts on this query set.")

    # Multi-rel with low NDCG + is_enumerated
    print("\n--- Multi-rel, low NDCG, is_enumerated (decomposition targets) ---")
    targets = [r for r in results if r["subset"] == "multi" and r["ndcg_full"] < 0.3 and r["is_enumerated"]]
    if targets:
        for r in targets:
            print(f"  {r['qid']}: NDCG={r['ndcg_full']:.4f}  #rel={r['n_rel']}  enum_count={r['enum_count']}  {r['query'][:60]}")
    else:
        print("  None matching criteria.")

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    overall = np.mean([r["ndcg_full"] for r in results])
    single = np.mean([r["ndcg_full"] for r in results if r["subset"] == "single"])
    multi = np.mean([r["ndcg_full"] for r in results if r["subset"] == "multi"])
    print(f"  Overall: {overall:.4f}  Single: {single:.4f}  Multi: {multi:.4f}")
    print(f"  Queries where BM25 helps: {sum(1 for r in results if r['bm25_delta'] > 0.001)}")
    print(f"  Queries where BM25 hurts: {sum(1 for r in results if r['bm25_delta'] < -0.001)}")
    print(f"  Queries where BM25 neutral: {sum(1 for r in results if abs(r['bm25_delta']) <= 0.001)}")


if __name__ == "__main__":
    main()
