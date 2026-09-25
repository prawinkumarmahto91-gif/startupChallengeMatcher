"""
src/evaluation.py

Measures ranking quality using standard information-retrieval metrics:
Precision@5, Precision@10, Recall@10, NDCG@10.

Ground truth comes from data/evaluation.csv (see generate_evaluation_data.py
for how it was created and its current limitations).
"""

import math
from typing import List, Set

import pandas as pd

from src.ranking import rank_candidates


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are actually relevant."""
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for sid in top_k if sid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int):
    """Fraction of ALL relevant startups that were found in the top-k.

    Returns None if there are no known relevant startups for this
    challenge - recall is undefined in that case, not zero.
    """
    if not relevant_ids:
        return None
    top_k = recommended_ids[:k]
    hits = sum(1 for sid in top_k if sid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain - rewards relevant items
    appearing EARLIER in the ranking, not just being present somewhere
    in the top-k.
    """
    dcg = 0.0
    for i, sid in enumerate(recommended_ids[:k]):
        relevance = 1 if sid in relevant_ids else 0
        dcg += relevance / math.log2(i + 2)  # position 1 -> log2(2), position 2 -> log2(3), ...

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_challenge(challenge_id: int, relevant_ids: Set[int]) -> dict:
    """Run the full ranking pipeline for one challenge and compute metrics
    against its ground-truth relevant startups.
    """
    ranked = rank_candidates(challenge_id, top_k_semantic=20, top_n_final=10)
    recommended_ids = ranked["startup_id"].tolist()

    return {
        "challenge_id": challenge_id,
        "precision@5": precision_at_k(recommended_ids, relevant_ids, 5),
        "precision@10": precision_at_k(recommended_ids, relevant_ids, 10),
        "recall@10": recall_at_k(recommended_ids, relevant_ids, 10),
        "ndcg@10": ndcg_at_k(recommended_ids, relevant_ids, 10),
        "num_relevant": len(relevant_ids),
        "num_recommended": len(recommended_ids),
    }


if __name__ == "__main__":
    evaluation_df = pd.read_csv("data/evaluation.csv")

    results = []
    for challenge_id in sorted(evaluation_df["challenge_id"].unique()):
        relevant_ids = set(
            evaluation_df.loc[evaluation_df["challenge_id"] == challenge_id, "startup_id"]
        )
        result = evaluate_challenge(challenge_id, relevant_ids)
        results.append(result)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 40)
    print("AVERAGE METRICS ACROSS ALL CHALLENGES")
    print("=" * 40)
    print(f"Mean Precision@5:  {results_df['precision@5'].mean():.4f}")
    print(f"Mean Precision@10: {results_df['precision@10'].mean():.4f}")
    print(f"Mean Recall@10:    {results_df['recall@10'].mean():.4f}")
    print(f"Mean NDCG@10:      {results_df['ndcg@10'].mean():.4f}")