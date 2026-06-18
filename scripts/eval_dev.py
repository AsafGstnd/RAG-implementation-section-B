"""Dev evaluation with single/multi-relevant breakdown and per-query table."""
from __future__ import annotations

import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import load_query_file, ndcg_at_k, K_EVAL
from main import run
from utils import PUBLIC_QUERIES_PATH, DATA_DIR


DEV_HARD_PATH = DATA_DIR / "dev_hard_queries.json"


def evaluate_dataset(name: str, path: Path) -> None:
    if not path.exists():
        print(f"\n=== {name}: SKIPPED (file not found: {path}) ===\n")
        return

    rows = load_query_file(path)
    queries = [r["query"] for r in rows]
    relevants = [r["relevant_page_ids"] for r in rows]

    t0 = time.perf_counter()
    ranked = run(queries)
    elapsed = time.perf_counter() - t0

    single_ndcg, multi_ndcg = [], []
    single_recall, multi_recall = [], []
    all_ndcg, all_recall = [], []

    print(f"\n{'='*90}")
    print(f"  {name}  ({len(queries)} queries, {elapsed:.2f}s)")
    print(f"{'='*90}")
    print(f"{'QID':<20} {'#Rel':>4} {'NDCG@10':>8} {'Hits':>5}  Query")
    print(f"{'-'*20} {'-'*4} {'-'*8} {'-'*5}  {'-'*40}")

    for row, ranked_ids, rel_set in zip(rows, ranked, relevants):
        qid = row["query_id"]
        query_text = row["query"]
        n_rel = len(rel_set)
        score = ndcg_at_k(ranked_ids, rel_set, K_EVAL)
        hits = len(set(ranked_ids[:K_EVAL]) & rel_set)
        recall = hits / n_rel if n_rel > 0 else 0.0

        all_ndcg.append(score)
        all_recall.append(recall)

        if n_rel == 1:
            single_ndcg.append(score)
            single_recall.append(recall)
        else:
            multi_ndcg.append(score)
            multi_recall.append(recall)

        print(f"{qid:<20} {n_rel:>4} {score:>8.4f} {hits:>3}/{n_rel:<2} {query_text[:50]}")

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'--- Summary ---':^90}")
    print(f"  Overall        : NDCG@10 = {mean(all_ndcg):.4f}  |  Recall@10 = {mean(all_recall):.4f}  ({len(all_ndcg)} queries)")
    print(f"  Single-relevant: NDCG@10 = {mean(single_ndcg):.4f}  |  Recall@10 = {mean(single_recall):.4f}  ({len(single_ndcg)} queries)")
    print(f"  Multi-relevant : NDCG@10 = {mean(multi_ndcg):.4f}  |  Recall@10 = {mean(multi_recall):.4f}  ({len(multi_ndcg)} queries)")
    print(f"  Wall-clock     : {elapsed:.2f}s")
    print()


def main() -> None:
    evaluate_dataset("Public Queries", PUBLIC_QUERIES_PATH)
    evaluate_dataset("Dev Hard Queries", DEV_HARD_PATH)


if __name__ == "__main__":
    main()
