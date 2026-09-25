"""
src/ranking.py

Combines the semantic similarity score with the five engineered features
(technology_match, sector_match, experience_score, budget_score,
location_score) into one final weighted score, using the formula:

Final Score =
    0.40 x Semantic Similarity
  + 0.15 x Technology Match
  + 0.15 x Sector Match
  + 0.10 x Experience
  + 0.10 x Budget Fit
  + 0.10 x Location Fit

Weights are configurable (not hardcoded permanently) - pass a custom
weights dict to rank_candidates() to experiment with different priorities.

This module also wires together the full pipeline for the first time:
retrieve candidates -> filter by eligibility -> compute features ->
apply hybrid formula -> sort -> return Top N.
"""

import pandas as pd

from src.retrieval import get_top_candidates, get_challenge_row
from src.eligibility import filter_eligible_startups
from src.features import add_features_to_candidates

DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "technology": 0.15,
    "sector": 0.15,
    "experience": 0.10,
    "budget": 0.10,
    "location": 0.10,
}


def compute_final_score(row: pd.Series, weights: dict = None) -> float:
    """Combine one candidate's semantic score and features into a single
    final score, expressed as a percentage (0-100) to match the spec's
    example output format (e.g. 92.4).
    """
    weights = weights or DEFAULT_WEIGHTS

    score = (
        weights["semantic"] * row["semantic_score"]
        + weights["technology"] * row["technology_match"]
        + weights["sector"] * row["sector_match"]
        + weights["experience"] * row["experience_score"]
        + weights["budget"] * row["budget_score"]
        + weights["location"] * row["location_score"]
    )
    return round(score * 100, 2)


def rank_candidates(
    challenge_id: int,
    top_k_semantic: int = 20,
    top_n_final: int = 10,
    weights: dict = None,
) -> pd.DataFrame:
    """Full pipeline for one challenge:
    1. Retrieve top_k_semantic candidates by semantic similarity
    2. Filter out ineligible candidates (hard requirements)
    3. Compute the five ranking features for the eligible candidates
    4. Compute the final hybrid score for each
    5. Sort and return the top_n_final candidates
    """
    # NOTE (Day 4 change): get_challenge_row() and get_top_candidates() now
    # query PostgreSQL + pgvector directly, instead of reading CSV files.
    # get_top_candidates() already returns every column eligibility.py and
    # features.py need, so no merge step is required anymore.
    challenge_row = get_challenge_row(challenge_id)
    candidates_full = get_top_candidates(challenge_id=challenge_id, top_k=top_k_semantic)

    checked = filter_eligible_startups(challenge_row, candidates_full)
    eligible = checked[checked["eligible"]].copy()

    with_features = add_features_to_candidates(challenge_row, eligible)
    with_features["final_score"] = with_features.apply(
        lambda row: compute_final_score(row, weights), axis=1
    )

    ranked = with_features.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return ranked.head(top_n_final)


if __name__ == "__main__":
    test_challenge_id = 102
    challenge_row = get_challenge_row(test_challenge_id)

    print(f"Challenge: {challenge_row['title']} (id={test_challenge_id})")
    print(f"Weights used: {DEFAULT_WEIGHTS}\n")

    result = rank_candidates(test_challenge_id, top_k_semantic=20, top_n_final=10)

    display_cols = ["rank", "startup_id", "name", "final_score", "semantic_score",
                     "technology_match", "sector_match", "experience_score",
                     "budget_score", "location_score"]
    print(result[display_cols].to_string(index=False))

    # --- Quick demo: show weights are configurable, not hardcoded ---
    print("\n" + "=" * 60)
    print("Same challenge, but with budget weighted much higher (0.40)")
    print("=" * 60)

    budget_focused_weights = {
        "semantic": 0.25,
        "technology": 0.10,
        "sector": 0.10,
        "experience": 0.10,
        "budget": 0.40,
        "location": 0.05,
    }
    result_alt = rank_candidates(
        test_challenge_id, top_k_semantic=20, top_n_final=10, weights=budget_focused_weights
    )
    print(result_alt[display_cols].to_string(index=False))