"""
init_db.py

Run this ONCE (or re-run anytime - it's safe, uses IF NOT EXISTS) to set
up the PostgreSQL schema: enables pgvector, creates startups, challenges,
and matches tables.

Field names deliberately match the EXISTING codebase (sector, capabilities,
previous_projects, dpiit_recognized, budget) rather than introducing new
names - this preserves compatibility with eligibility.py, features.py,
and the rest of the working pipeline, which are not being rewritten.

Run with:
    python init_db.py
"""

from sqlalchemy import text

from src.db.connection import get_engine

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS startups (
    startup_id           SERIAL PRIMARY KEY,
    name                 TEXT NOT NULL,
    description          TEXT,
    sector               TEXT,
    technologies         TEXT,
    capabilities         TEXT,
    experience           INTEGER,
    previous_projects    TEXT,
    location             TEXT,
    budget               NUMERIC,
    certifications       TEXT,
    dpiit_recognized     BOOLEAN DEFAULT FALSE,

    embedding_text       TEXT,
    embedding            VECTOR(1024),
    embedding_model      TEXT DEFAULT 'BAAI/bge-m3',
    embedding_version    INTEGER DEFAULT 1,
    embedding_status     TEXT DEFAULT 'pending',  -- pending | ready | failed
    embedding_updated_at TIMESTAMP,

    created_at           TIMESTAMP DEFAULT now(),
    updated_at           TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS challenges (
    challenge_id            SERIAL PRIMARY KEY,
    title                   TEXT NOT NULL,
    problem                 TEXT,
    sector                  TEXT,
    technologies            TEXT,
    desired_outcome         TEXT,
    location                TEXT,
    budget                  NUMERIC,
    mandatory_requirements  TEXT,

    embedding_text          TEXT,
    embedding                VECTOR(1024),
    embedding_model          TEXT DEFAULT 'BAAI/bge-m3',
    embedding_version        INTEGER DEFAULT 1,
    embedding_status         TEXT DEFAULT 'pending',
    embedding_updated_at     TIMESTAMP,

    created_at               TIMESTAMP DEFAULT now(),
    updated_at                TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS matches (
    id                   SERIAL PRIMARY KEY,
    challenge_id         INTEGER REFERENCES challenges(challenge_id),
    startup_id           INTEGER REFERENCES startups(startup_id),

    semantic_score       NUMERIC,
    technology_match     NUMERIC,
    sector_match         NUMERIC,
    experience_score     NUMERIC,
    budget_score         NUMERIC,
    location_score       NUMERIC,
    final_score          NUMERIC,
    rank                 INTEGER,
    explanation          TEXT,  -- stored as JSON text (list of reason strings)

    created_at            TIMESTAMP DEFAULT now()
);
"""

if __name__ == "__main__":
    engine = get_engine()

    with engine.connect() as conn:
        for statement in SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()

    print("Schema created/verified successfully:")
    print("  - pgvector extension enabled")
    print("  - 'startups' table ready (embedding, embedding_status, embedding_model/version)")
    print("  - 'challenges' table ready (same tracking columns)")
    print("  - 'matches' table ready (persists prediction results)")