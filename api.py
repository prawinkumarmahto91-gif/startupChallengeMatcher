"""
api.py

FastAPI wrapper around the ML matcher module (src/matcher.py). This is
what the backend team actually calls over HTTP - they don't need Python,
pandas, or any ML libraries installed; they just send a challenge_id and
get back JSON recommendations.

Run with (from the project root, venv active):
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API documentation.
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from src.matcher import match_startups
from src.db.connection import get_engine
from src.embedding_service import (
    embed_and_store_startup,
    embed_and_store_challenge,
    STARTUP_EMBEDDING_FIELDS,
    CHALLENGE_EMBEDDING_FIELDS,
)

app = FastAPI(
    title="Startup-Challenge Matching API",
    description="ML-powered matching between government challenges and startups.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    challenge_id: int
    top_k_semantic: Optional[int] = 20
    top_n_final: Optional[int] = 10


class RecommendationResponse(BaseModel):
    startup_id: int
    startup_name: str
    rank: int
    final_score: float
    semantic_score: float
    technology_match: float
    sector_match: float
    experience_score: float
    budget_score: float
    location_score: float
    reasons: List[str]


class PredictResponse(BaseModel):
    challenge_id: int
    challenge_title: str
    recommendations: List[RecommendationResponse]


class StartupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    technologies: Optional[str] = None
    capabilities: Optional[str] = None
    experience: Optional[int] = 0
    previous_projects: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    certifications: Optional[str] = None
    dpiit_recognized: Optional[bool] = False


class StartupUpdate(BaseModel):
    # Every field optional - PUT only updates fields actually provided.
    name: Optional[str] = None
    description: Optional[str] = None
    sector: Optional[str] = None
    technologies: Optional[str] = None
    capabilities: Optional[str] = None
    experience: Optional[int] = None
    previous_projects: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    certifications: Optional[str] = None
    dpiit_recognized: Optional[bool] = None


class ChallengeCreate(BaseModel):
    title: str
    problem: Optional[str] = None
    sector: Optional[str] = None
    technologies: Optional[str] = None
    desired_outcome: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    mandatory_requirements: Optional[str] = None


class ChallengeUpdate(BaseModel):
    title: Optional[str] = None
    problem: Optional[str] = None
    sector: Optional[str] = None
    technologies: Optional[str] = None
    desired_outcome: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    mandatory_requirements: Optional[str] = None


@app.get("/health")
def health_check():
    """Simple endpoint the backend team can use to verify the service is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Main endpoint: given a challenge_id, return ranked, explained
    startup recommendations.
    """
    try:
        result = match_startups(
            challenge_id=request.challenge_id,
            top_k_semantic=request.top_k_semantic,
            top_n_final=request.top_n_final,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Day 5 - Live write endpoints: register/update startups and challenges.
# A new record becomes immediately available for matching - no restart,
# no retraining, no cache rebuild. See src/embedding_service.py.
# ---------------------------------------------------------------------------

@app.post("/startups")
def register_startup(payload: StartupCreate):
    """Register a new startup. Saves it to PostgreSQL first (so its data
    is never lost even if embedding generation fails), then generates
    and stores its embedding.
    """
    engine = get_engine()
    insert_sql = text("""
        INSERT INTO startups (
            name, description, sector, technologies, capabilities,
            experience, previous_projects, location, budget,
            certifications, dpiit_recognized, embedding_status
        ) VALUES (
            :name, :description, :sector, :technologies, :capabilities,
            :experience, :previous_projects, :location, :budget,
            :certifications, :dpiit_recognized, 'pending'
        )
        RETURNING startup_id
    """)

    with engine.connect() as conn:
        result = conn.execute(insert_sql, payload.model_dump())
        startup_id = result.scalar()
        conn.commit()

    status = embed_and_store_startup(startup_id)

    if status == "failed":
        raise HTTPException(
            status_code=502,
            detail=(
                f"Startup {startup_id} was saved, but embedding generation failed. "
                f"Its data is not lost - retry with POST /startups/{startup_id}/retry-embedding"
            ),
        )

    return {"startup_id": startup_id, "embedding_status": status}


@app.put("/startups/{startup_id}")
def update_startup(startup_id: int, payload: StartupUpdate):
    """Update an existing startup. Only regenerates the embedding if a
    field that's actually part of the embedded text changed - e.g.
    updating just `budget` or `location` skips re-embedding entirely.
    """
    update_fields = payload.model_dump(exclude_unset=True)

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    engine = get_engine()
    set_clause = ", ".join(f"{key} = :{key}" for key in update_fields.keys())
    params = {**update_fields, "startup_id": startup_id}

    with engine.connect() as conn:
        result = conn.execute(
            text(f"UPDATE startups SET {set_clause}, updated_at = now() "
                 f"WHERE startup_id = :startup_id"),
            params,
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"No startup found with startup_id={startup_id}")
        conn.commit()

    needs_reembedding = bool(STARTUP_EMBEDDING_FIELDS & update_fields.keys())
    new_status = embed_and_store_startup(startup_id) if needs_reembedding else "unchanged"

    return {
        "startup_id": startup_id,
        "updated_fields": list(update_fields.keys()),
        "embedding_regenerated": needs_reembedding,
        "embedding_status": new_status,
    }


@app.post("/startups/{startup_id}/retry-embedding")
def retry_startup_embedding(startup_id: int):
    """Manually retry embedding generation for a startup whose status is 'failed'."""
    try:
        status = embed_and_store_startup(startup_id)
        return {"startup_id": startup_id, "embedding_status": status}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/challenges")
def create_challenge(payload: ChallengeCreate):
    """Create a new government challenge. Same pattern as startup
    registration: save first, then embed.
    """
    engine = get_engine()
    insert_sql = text("""
        INSERT INTO challenges (
            title, problem, sector, technologies, desired_outcome,
            location, budget, mandatory_requirements, embedding_status
        ) VALUES (
            :title, :problem, :sector, :technologies, :desired_outcome,
            :location, :budget, :mandatory_requirements, 'pending'
        )
        RETURNING challenge_id
    """)

    with engine.connect() as conn:
        result = conn.execute(insert_sql, payload.model_dump())
        challenge_id = result.scalar()
        conn.commit()

    status = embed_and_store_challenge(challenge_id)

    if status == "failed":
        raise HTTPException(
            status_code=502,
            detail=(
                f"Challenge {challenge_id} was saved, but embedding generation failed. "
                f"Retry with POST /challenges/{challenge_id}/retry-embedding"
            ),
        )

    return {"challenge_id": challenge_id, "embedding_status": status}


@app.put("/challenges/{challenge_id}")
def update_challenge(challenge_id: int, payload: ChallengeUpdate):
    """Update an existing challenge. Only regenerates the embedding if an
    embedding-relevant field changed (e.g. updating just `budget` or
    `mandatory_requirements` skips re-embedding).
    """
    update_fields = payload.model_dump(exclude_unset=True)

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    engine = get_engine()
    set_clause = ", ".join(f"{key} = :{key}" for key in update_fields.keys())
    params = {**update_fields, "challenge_id": challenge_id}

    with engine.connect() as conn:
        result = conn.execute(
            text(f"UPDATE challenges SET {set_clause}, updated_at = now() "
                 f"WHERE challenge_id = :challenge_id"),
            params,
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"No challenge found with challenge_id={challenge_id}")
        conn.commit()

    needs_reembedding = bool(CHALLENGE_EMBEDDING_FIELDS & update_fields.keys())
    new_status = embed_and_store_challenge(challenge_id) if needs_reembedding else "unchanged"

    return {
        "challenge_id": challenge_id,
        "updated_fields": list(update_fields.keys()),
        "embedding_regenerated": needs_reembedding,
        "embedding_status": new_status,
    }


@app.post("/challenges/{challenge_id}/retry-embedding")
def retry_challenge_embedding(challenge_id: int):
    """Manually retry embedding generation for a challenge whose status is 'failed'."""
    try:
        status = embed_and_store_challenge(challenge_id)
        return {"challenge_id": challenge_id, "embedding_status": status}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))