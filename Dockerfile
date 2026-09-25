# Dockerfile
#
# Builds a container that runs our FastAPI ML service on Hugging Face
# Spaces. The BGE-M3 model is downloaded DURING THE BUILD (not at
# runtime), so the container starts instantly every time - no waiting
# for a 2GB download on cold start.

FROM python:3.11-slim

WORKDIR /app

# Some Python ML packages need basic build tools available
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching -
# this step is skipped on rebuilds unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (api.py, src/, data/, etc.)
COPY . .

# Store the model cache inside the image itself
ENV HF_HOME=/app/hf_cache

# Pre-download BGE-M3 into the image RIGHT NOW, at build time.
# This means the deployed app never has to download anything when
# it starts up or restarts - it's already baked into the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]