"""
src/db/connection.py

Sets up the SQLAlchemy engine used to talk to PostgreSQL. Every other
db module (models.py, crud.py, retrieval.py) imports get_engine() from
here rather than creating its own connection.
"""

import os

from sqlalchemy import create_engine

# In production, set this via an environment variable instead of
# hardcoding it. This default matches the local Docker container
# from Phase 1 of the setup guide.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:sih2026@localhost:5432/startup_matching",
)

_engine = None


def get_engine():
    """Return a single shared SQLAlchemy engine (created once, reused
    everywhere - avoids opening a new connection pool per call).
    """
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine