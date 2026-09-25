"""
src/similarity.py

Given a government challenge, this module:
1. Generates an embedding for the challenge
2. Generates embeddings for every startup
3. Ranks startups by cosine similarity to the challenge

This is SEMANTIC-ONLY matching - no eligibility filtering, no hybrid
scoring yet. Those are added in later days. Today's goal is just to
confirm that pure semantic similarity already produces a sensible order.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import generate_embeddings
from src.text_builder import create_challenge_text

EMBEDDING_CACHE_PATH = "models/startup_embeddings.npz"


def load_ready_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the startups/challenges CSVs that already include the
    'embedding_text' column (produced in Day 5).
    """
    startups = pd.read_csv("data/startups_ready.csv")
    challenges = pd.read_csv("data/challenges_ready.csv")
    return startups, challenges


def embed_startups(startups: pd.DataFrame) -> np.ndarray:
    """Generate one embedding per startup, in the same row order as the dataframe."""
    texts = startups["embedding_text"].tolist()
    return generate_embeddings(texts)


def get_or_create_startup_embeddings(
    startups: pd.DataFrame, cache_path: str = EMBEDDING_CACHE_PATH
) -> np.ndarray:
    """Load cached startup embeddings from disk if they match the current
    dataset; otherwise compute them fresh and save to disk for next time.

    Why: embedding 40 startups takes several seconds. Embedding 1,000+
    startups (production scale) could take minutes. We should never pay
    that cost more than once per dataset version.
    """
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        cached_ids = list(cached["startup_ids"])
        current_ids = list(startups["startup_id"])

        if cached_ids == current_ids:
            print(f"Loaded cached startup embeddings from {cache_path} "
                  f"(skipped recomputation).")
            return cached["embeddings"]
        else:
            print("Cached embeddings don't match the current dataset "
                  "(startup list changed) - recomputing.")

    print("Computing startup embeddings (not cached yet, this may take a moment)...")
    start = time.time()
    embeddings = embed_startups(startups)
    elapsed = time.time() - start
    print(f"Computed embeddings for {len(startups)} startups in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, startup_ids=startups["startup_id"].values)
    print(f"Saved embeddings cache to {cache_path}")

    return embeddings


def get_top_candidates(challenge_id: int, top_k: int = 20) -> pd.DataFrame:
    """End-to-end candidate retrieval: given a challenge_id, return the
    top_k most semantically similar startups.

    This is the function later modules (eligibility, ranking, matcher)
    will call - it hides all the loading/caching/embedding details.
    """
    startups, challenges = load_ready_data()
    startup_embeddings = get_or_create_startup_embeddings(startups)

    matching_rows = challenges[challenges["challenge_id"] == challenge_id]
    if matching_rows.empty:
        raise ValueError(f"No challenge found with challenge_id={challenge_id}")
    challenge_row = matching_rows.iloc[0]

    return rank_startups_for_challenge(challenge_row, startups, startup_embeddings, top_n=top_k)


def rank_startups_for_challenge(
    challenge_row: pd.Series,
    startups: pd.DataFrame,
    startup_embeddings: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank all startups by semantic similarity to a single challenge.

    Returns a dataframe with startup_id, name, sector, and semantic_score,
    sorted by semantic_score descending.
    """
    challenge_text = create_challenge_text(challenge_row)
    challenge_embedding = generate_embeddings([challenge_text])[0]

    similarities = cosine_similarity([challenge_embedding], startup_embeddings)[0]

    results = startups[["startup_id", "name", "sector"]].copy()
    results["semantic_score"] = similarities
    results = results.sort_values("semantic_score", ascending=False).reset_index(drop=True)

    return results.head(top_n)


if __name__ == "__main__":
    # --- First call: Waste Collection Optimization ---
    print("=" * 60)
    print("Query 1: challenge_id=101 (Waste Collection Optimization)")
    print("=" * 60)

    start = time.time()
    top_matches_1 = get_top_candidates(challenge_id=101, top_k=20)
    elapsed_1 = time.time() - start

    print(f"\nTop 10 of {len(top_matches_1)} candidates:")
    print(top_matches_1.head(10).to_string(index=False))
    print(f"\nTotal time for this query: {elapsed_1:.2f}s")

    # --- Second call: different challenge, same session ---
    # Startup embeddings are now cached on disk, so this call should skip
    # re-embedding all 40 startups and only embed the new challenge text.
    print("\n" + "=" * 60)
    print("Query 2: challenge_id=104 (Urban Traffic Management)")
    print("=" * 60)

    start = time.time()
    top_matches_2 = get_top_candidates(challenge_id=104, top_k=20)
    elapsed_2 = time.time() - start

    print(f"\nTop 10 of {len(top_matches_2)} candidates:")
    print(top_matches_2.head(10).to_string(index=False))
    print(f"\nTotal time for this query: {elapsed_2:.2f}s")

    print(f"\nQuery 1 took {elapsed_1:.2f}s, Query 2 took {elapsed_2:.2f}s "
          f"(Query 2 should be noticeably faster - no startup re-embedding).")