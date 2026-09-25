"""
migrate_csv_to_postgres.py

ONE-TIME migration: reads the existing cleaned CSV data, generates
embeddings using BGE-M3 (the same frozen model we already use), and
inserts everything into PostgreSQL - startups and challenges both.

Safe to re-run: checks which IDs already exist in the database and
only inserts the ones that are missing. Will not create duplicates.

Run with:
    python migrate_csv_to_postgres.py
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine
from src.text_builder import create_startup_text, create_challenge_text
from src.embeddings import generate_embeddings


def vector_to_pgvector_literal(vec: np.ndarray) -> str:
    """Convert a numpy embedding vector into the text format pgvector
    expects for an INSERT, e.g. '[0.123,0.456,...]'.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def get_existing_ids(engine, table: str, id_column: str) -> set:
    """Return the set of IDs already present in the given table, so we
    can skip them and avoid duplicate inserts.
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT {id_column} FROM {table}"))
        return {row[0] for row in result}


def migrate_startups(engine):
    startups = pd.read_csv("data/startups_clean.csv")
    startups["embedding_text"] = startups.apply(create_startup_text, axis=1)

    existing_ids = get_existing_ids(engine, "startups", "startup_id")
    new_rows = startups[~startups["startup_id"].isin(existing_ids)]

    if new_rows.empty:
        print(f"Startups: {len(startups)} in CSV, all already in database. Nothing to migrate.")
        return

    print(f"Startups: {len(new_rows)} new rows to migrate (out of {len(startups)} total)...")
    embeddings = generate_embeddings(new_rows["embedding_text"].tolist())

    insert_sql = text("""
        INSERT INTO startups (
            startup_id, name, description, sector, technologies, capabilities,
            experience, previous_projects, location, budget, certifications,
            dpiit_recognized, embedding_text, embedding, embedding_model,
            embedding_version, embedding_status, embedding_updated_at
        ) VALUES (
            :startup_id, :name, :description, :sector, :technologies, :capabilities,
            :experience, :previous_projects, :location, :budget, :certifications,
            :dpiit_recognized, :embedding_text, CAST(:embedding AS vector), 'BAAI/bge-m3',
            1, 'ready', now()
        )
    """)

    with engine.connect() as conn:
        for (_, row), emb in zip(new_rows.iterrows(), embeddings):
            conn.execute(insert_sql, {
                "startup_id": int(row["startup_id"]),
                "name": row["name"],
                "description": row["description"],
                "sector": row["sector"],
                "technologies": row["technologies"],
                "capabilities": row["capabilities"],
                "experience": int(row["experience"]),
                "previous_projects": row["previous_projects"],
                "location": row["location"],
                "budget": float(row["budget"]),
                "certifications": row["certifications"],
                "dpiit_recognized": bool(row["dpiit_recognized"]),
                "embedding_text": row["embedding_text"],
                "embedding": vector_to_pgvector_literal(emb),
            })
        # Keep the auto-increment sequence ahead of our manually-specified IDs,
        # so future new registrations (without an explicit ID) don't collide.
        conn.execute(text(
            "SELECT setval('startups_startup_id_seq', (SELECT MAX(startup_id) FROM startups))"
        ))
        conn.commit()

    print(f"Startups: {len(new_rows)} rows inserted with embeddings.")


def migrate_challenges(engine):
    challenges = pd.read_csv("data/challenges_clean.csv")
    challenges["embedding_text"] = challenges.apply(create_challenge_text, axis=1)

    existing_ids = get_existing_ids(engine, "challenges", "challenge_id")
    new_rows = challenges[~challenges["challenge_id"].isin(existing_ids)]

    if new_rows.empty:
        print(f"Challenges: {len(challenges)} in CSV, all already in database. Nothing to migrate.")
        return

    print(f"Challenges: {len(new_rows)} new rows to migrate (out of {len(challenges)} total)...")
    embeddings = generate_embeddings(new_rows["embedding_text"].tolist())

    insert_sql = text("""
        INSERT INTO challenges (
            challenge_id, title, problem, sector, technologies, desired_outcome,
            location, budget, mandatory_requirements, embedding_text, embedding,
            embedding_model, embedding_version, embedding_status, embedding_updated_at
        ) VALUES (
            :challenge_id, :title, :problem, :sector, :technologies, :desired_outcome,
            :location, :budget, :mandatory_requirements, :embedding_text, CAST(:embedding AS vector),
            'BAAI/bge-m3', 1, 'ready', now()
        )
    """)

    with engine.connect() as conn:
        for (_, row), emb in zip(new_rows.iterrows(), embeddings):
            conn.execute(insert_sql, {
                "challenge_id": int(row["challenge_id"]),
                "title": row["title"],
                "problem": row["problem"],
                "sector": row["sector"],
                "technologies": row["technologies"],
                "desired_outcome": row["desired_outcome"],
                "location": row["location"],
                "budget": float(row["budget"]),
                "mandatory_requirements": row["mandatory_requirements"],
                "embedding_text": row["embedding_text"],
                "embedding": vector_to_pgvector_literal(emb),
            })
        conn.execute(text(
            "SELECT setval('challenges_challenge_id_seq', (SELECT MAX(challenge_id) FROM challenges))"
        ))
        conn.commit()

    print(f"Challenges: {len(new_rows)} rows inserted with embeddings.")


if __name__ == "__main__":
    engine = get_engine()

    migrate_startups(engine)
    migrate_challenges(engine)

    with engine.connect() as conn:
        startup_count = conn.execute(text("SELECT COUNT(*) FROM startups")).scalar()
        challenge_count = conn.execute(text("SELECT COUNT(*) FROM challenges")).scalar()
        ready_startups = conn.execute(
            text("SELECT COUNT(*) FROM startups WHERE embedding_status = 'ready'")
        ).scalar()

    print("\nMigration complete.")
    print(f"Total startups in database: {startup_count} ({ready_startups} with ready embeddings)")
    print(f"Total challenges in database: {challenge_count}")