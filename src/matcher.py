"""
src/matcher.py

The single entry point for the entire ML system. This is the ONLY
function other code (backend, notebooks, tests) should need to import.

Usage:
    from src.matcher import match_startups
    results = match_startups(challenge_id=101)

Everything else in src/ (similarity, eligibility, features, ranking,
explanation) is an internal implementation detail behind this function.
"""

from typing import Optional

from src.ranking import rank_candidates
from src.explanation import build_recommendation_record
from src.retrieval import get_challenge_row


def match_startups(
    challenge_id: int,
    top_k_semantic: int = 20,
    top_n_final: int = 10,
    weights: Optional[dict] = None,
) -> dict:
    """Given a challenge_id, return ranked, explained startup recommendations.

    Pipeline (all internal): semantic candidate retrieval -> eligibility
    filtering -> feature engineering -> hybrid ranking -> explanation.

    Args:
        challenge_id: the ID of the government challenge to match against.
        top_k_semantic: how many candidates to retrieve by semantic
            similarity before eligibility filtering (default 20).
        top_n_final: how many final recommendations to return (default 10).
        weights: optional custom weights dict for the hybrid formula
            (see src/ranking.py DEFAULT_WEIGHTS for the expected keys).

    Returns:
        A dict:
        {
          "challenge_id": int,
          "challenge_title": str,
          "recommendations": [ {startup_id, startup_name, rank,
                                 final_score, semantic_score, ...,
                                 reasons: [...]}, ... ]
        }

    Raises:
        ValueError: if challenge_id does not exist in the dataset.
    """
    # NOTE (Day 4 change): get_challenge_row() queries PostgreSQL directly.
    # It already raises ValueError("No challenge found with challenge_id=...")
    # for an unknown ID - same error contract api.py already relies on.
    challenge_row = get_challenge_row(challenge_id)

    ranked = rank_candidates(
        challenge_id,
        top_k_semantic=top_k_semantic,
        top_n_final=top_n_final,
        weights=weights,
    )

    if ranked.empty:
        # A real, valid outcome - not an error. Every startup that passed
        # semantic retrieval failed the hard eligibility requirements.
        recommendations = []
    else:
        recommendations = [
            build_recommendation_record(row, challenge_row) for _, row in ranked.iterrows()
        ]

    return {
        "challenge_id": int(challenge_id),
        "challenge_title": challenge_row["title"],
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import json

    # --- Full example: one challenge, full structured output ---
    print("=" * 60)
    print("Example: match_startups(challenge_id=101)")
    print("=" * 60)

    result = match_startups(challenge_id=101)
    print(f"Challenge: {result['challenge_title']}")
    print(f"Number of recommendations: {len(result['recommendations'])}\n")
    print("Top recommendation:")
    print(json.dumps(result["recommendations"][0], indent=2))

    # --- Smoke test: run across all 5 named challenges ---
    print("\n" + "=" * 60)
    print("Smoke test: top pick for each named challenge")
    print("=" * 60)

    for cid in [101, 102, 103, 104, 105]:
        result = match_startups(challenge_id=cid)
        if result["recommendations"]:
            top = result["recommendations"][0]
            print(f"[{cid}] {result['challenge_title']:<35} -> "
                  f"{top['startup_name']} (score: {top['final_score']})")
        else:
            print(f"[{cid}] {result['challenge_title']:<35} -> No eligible candidates found")

    # --- Error handling test: invalid challenge_id ---
    print("\n" + "=" * 60)
    print("Error handling test: invalid challenge_id")
    print("=" * 60)
    try:
        match_startups(challenge_id=99999)
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")