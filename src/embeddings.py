"""
src/embeddings.py

Loads the BGE-M3 embedding model and provides functions to convert text
into numeric vectors (embeddings) that capture semantic meaning.

Why BGE-M3:
- Pretrained, general-purpose, strong semantic embedding model
- No need to train anything ourselves for the MVP
- Works well for comparing free-text descriptions (challenges vs startups)
"""

import os

# Force the Hugging Face model cache to live in D:\hf_cache, no matter
# whether the HF_HOME environment variable happens to be set in the
# current terminal session. This guarantees we always reuse the model
# we already downloaded, instead of depending on remembering to open
# a "correctly configured" terminal every time.
os.environ.setdefault("HF_HOME", "D:/hf_cache")

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and cache the BGE-M3 model so it is only loaded into memory once.

    The first call downloads ~2GB of model weights from Hugging Face
    (one-time only — cached locally afterwards). This can take a few
    minutes depending on your internet speed.
    """
    print(f"Loading model: {MODEL_NAME} (first run downloads ~2GB, please wait)...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    return model


def generate_embedding(text: str) -> np.ndarray:
    """Convert a single piece of text into a normalized embedding vector.

    normalize_embeddings=True scales each vector to length 1. This means
    a simple dot product between two vectors gives us cosine similarity
    directly, without extra math later.
    """
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Convert a list of texts into embeddings in a single batch call.

    Batching is faster than calling generate_embedding() in a loop,
    since the model processes multiple texts together on the same pass.
    """
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity

    # Two texts about the same real-world topic (waste collection routing),
    # worded completely differently, plus one unrelated text (crop disease).
    sample_texts = [
        "AI-powered route optimization for municipal waste collection.",
        "Smart routing system to optimize garbage truck collection paths.",
        "Computer vision system for detecting crop diseases in farm fields.",
    ]

    embeddings = generate_embeddings(sample_texts)

    print("\nEmbedding matrix shape:", embeddings.shape)
    print("Embedding dimension per text:", embeddings.shape[1])

    sim_matrix = cosine_similarity(embeddings)

    print("\nFull cosine similarity matrix:")
    print(np.round(sim_matrix, 4))

    print("\nSimilarity: waste-text-1 vs waste-text-2 (expect HIGH, e.g. > 0.6):",
          round(sim_matrix[0][1], 4))
    print("Similarity: waste-text-1 vs crop-text (expect LOWER, e.g. < 0.5):",
          round(sim_matrix[0][2], 4))