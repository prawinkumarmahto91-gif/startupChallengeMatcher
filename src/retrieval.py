"""
src/retrieval.py

Replaces similarity.py's CSV + in-memory cosine similarity approach.
Retrieval now happens as a single SQL query using pgvector's <=>
(cosine distance) operator, directly against the live PostgreSQL
database - no vectors are loaded into Python memory manually.

Public functions mirror similarity.py's old interface so that ranking.py
only needs a one-line import change to switch over (done in Day 4):
    get_challenge_row(challenge_id) -> pd.Series
    get_top_candidates(challenge_id, top_k) -> pd.DataFrame
"""

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine


def get_challenge_row(challenge_id: int) -> pd.Series:
    """Fetch one challenge's full row from PostgreSQL, including its
    embedding as a pgvector text literal (e.g. "[0.1,0.2,...]") so it
    can be reused directly as a query parameter in get_top_candidates().
    """
    engine = get_engine()
    query = text("""
        SELECT challenge_id, title, problem, sector, technologies,
               desired_outcome, location, budget, mandatory_requirements,
               embedding::text AS embedding_vec, embedding_status
        FROM challenges
        WHERE challenge_id = :challenge_id
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"challenge_id": challenge_id})
        row = result.mappings().first()

    if row is None:
        raise ValueError(f"No challenge found with challenge_id={challenge_id}")

    if row["embedding_status"] != "ready":
        raise ValueError(
            f"Challenge {challenge_id} has embedding_status="
            f"'{row['embedding_status']}' - cannot run matching until it is 'ready'."
        )

    return pd.Series(dict(row))


def get_top_candidates(challenge_id: int, top_k: int = 20) -> pd.DataFrame:
    """Return the top_k startups most semantically similar to the given
    challenge, using pgvector's cosine distance operator (<=>) directly
    in SQL. Only considers startups whose embedding_status is 'ready' -
    a startup mid-registration or with a failed embedding is excluded.
    """
    challenge_row = get_challenge_row(challenge_id)
    challenge_vec = challenge_row["embedding_vec"]

    engine = get_engine()
    query = text("""
        SELECT startup_id, name, description, sector, technologies,
               capabilities, experience, previous_projects, location,
               budget, certifications, dpiit_recognized,
               1 - (embedding <=> CAST(:challenge_vec AS vector)) AS semantic_score
        FROM startups
        WHERE embedding_status = 'ready'
        ORDER BY embedding <=> CAST(:challenge_vec AS vector)
        LIMIT :top_k
    """)

    return pd.read_sql(query, engine, params={"challenge_vec": challenge_vec, "top_k": top_k})


if __name__ == "__main__":
    # Sanity check: same 5 named challenges we verified back on Day 13,
    # now sourced entirely from PostgreSQL instead of CSV + in-memory cosine.
    test_cases = [
        (101, "Waste Collection Optimization", "EcoRoute AI"),
        (102, "Hospital Patient Flow Optimization", "HealthQueue Technologies"),
        (103, "Crop Disease Detection", "AgriVision Labs"),
        (104, "Urban Traffic Management", "UrbanTraffic AI"),
        (105, "Water Leak Detection", "WaterWatch Technologies"),
    ]

    for challenge_id, expected_title, expected_top_startup in test_cases:
        challenge_row = get_challenge_row(challenge_id)
        candidates = get_top_candidates(challenge_id, top_k=20)

        top_name = candidates.iloc[0]["name"]
        top_score = candidates.iloc[0]["semantic_score"]

        status = "OK" if top_name == expected_top_startup else "CHECK THIS"
        print(f"[{challenge_id}] {challenge_row['title']:<35} -> "
              f"top: {top_name} ({top_score:.4f}) "
              f"[{status}, expected {expected_top_startup}]")

    print(f"\nTotal candidates returned per query: {len(candidates)} "
          f"(all sourced live from PostgreSQL, no CSV involved)")