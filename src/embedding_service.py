"""
src/embedding_service.py

Generates and stores an embedding for ONE startup or challenge already
sitting in PostgreSQL. Called synchronously inside POST/PUT request
handlers in api.py - a single inference call takes milliseconds, so no
background job/queue is needed at this scale.

If embedding generation fails for any reason, the row's data is NOT
lost - only embedding_status is set to 'failed', and a retry endpoint
can be called later.
"""

from sqlalchemy import text

from src.db.connection import get_engine
from src.text_builder import create_startup_text, create_challenge_text
from src.embeddings import generate_embedding

# Fields that, if changed on PUT, require the embedding to be regenerated.
# Fields NOT in this list (budget, location, dpiit_recognized,
# certifications, mandatory_requirements) can change freely without
# touching the embedding at all - they aren't part of the embedded text.
STARTUP_EMBEDDING_FIELDS = {
    "name", "description", "sector", "technologies",
    "capabilities", "experience", "previous_projects",
}
CHALLENGE_EMBEDDING_FIELDS = {
    "title", "problem", "sector", "technologies", "desired_outcome",
}


def _vector_literal(vec) -> str:
    """Convert a numpy embedding vector into pgvector's text format."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def embed_and_store_startup(startup_id: int) -> str:
    """Fetch a startup's current fields, build its embedding text,
    generate its embedding, and store it back in PostgreSQL.

    Returns the final embedding_status: 'ready' or 'failed'.
    """
    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM startups WHERE startup_id = :id"),
            {"id": startup_id},
        ).mappings().first()

    if row is None:
        raise ValueError(f"No startup found with startup_id={startup_id}")

    try:
        embedding_text = create_startup_text(row)
        embedding = generate_embedding(embedding_text)

        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE startups
                SET embedding_text = :embedding_text,
                    embedding = CAST(:embedding AS vector),
                    embedding_status = 'ready',
                    embedding_updated_at = now(),
                    updated_at = now()
                WHERE startup_id = :id
            """), {
                "embedding_text": embedding_text,
                "embedding": _vector_literal(embedding),
                "id": startup_id,
            })
            conn.commit()

        return "ready"

    except Exception as e:
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE startups
                SET embedding_status = 'failed', updated_at = now()
                WHERE startup_id = :id
            """), {"id": startup_id})
            conn.commit()
        print(f"Embedding generation failed for startup_id={startup_id}: {e}")
        return "failed"


def embed_and_store_challenge(challenge_id: int) -> str:
    """Same pattern as embed_and_store_startup(), for challenges."""
    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM challenges WHERE challenge_id = :id"),
            {"id": challenge_id},
        ).mappings().first()

    if row is None:
        raise ValueError(f"No challenge found with challenge_id={challenge_id}")

    try:
        embedding_text = create_challenge_text(row)
        embedding = generate_embedding(embedding_text)

        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE challenges
                SET embedding_text = :embedding_text,
                    embedding = CAST(:embedding AS vector),
                    embedding_status = 'ready',
                    embedding_updated_at = now(),
                    updated_at = now()
                WHERE challenge_id = :id
            """), {
                "embedding_text": embedding_text,
                "embedding": _vector_literal(embedding),
                "id": challenge_id,
            })
            conn.commit()

        return "ready"

    except Exception as e:
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE challenges
                SET embedding_status = 'failed', updated_at = now()
                WHERE challenge_id = :id
            """), {"id": challenge_id})
            conn.commit()
        print(f"Embedding generation failed for challenge_id={challenge_id}: {e}")
        return "failed"