# Match the Python version used in CI (.github/workflows/main.yml).
FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: don't litter the image with .pyc files
# - PYTHONUNBUFFERED: stream logs straight to the console
# - HF_HOME: where sentence-transformers caches downloaded models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# System libraries. libmagic1 provides the native libmagic that python-magic
# (used by pyxtxt for file-type detection) loads at runtime; it isn't bundled
# in the slim image, so text extraction fails without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (both the API and the Streamlit UI share this image).
COPY app/ ./app/
COPY ui/ ./ui/

# Run as an unprivileged user and pre-create the writable dirs it needs.
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/uploads "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# FastAPI (8000) and Streamlit (8501).
EXPOSE 8000 8501

# Default to the API; docker-compose overrides the command for the UI.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
