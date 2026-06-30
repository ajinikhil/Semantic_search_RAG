import os

from dotenv import load_dotenv

# This is to load environment variables from .env

load_dotenv()

# Embedding Model Configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Chunking Configuration

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ChromaDB Configuration
CHROMA_DB_PATH = "data/chroma_db"
CHROMA_RESULTS_RETURNED = 5

# LLM Configuration (local, served by Ollama)

MODEL_NAME = "qwen2.5:3b-instruct"

# Ollama server URL. When unset the client talks to the local daemon
# (http://127.0.0.1:11434); override via the OLLAMA_HOST env var, e.g.
# "http://host.docker.internal:11434" when running inside Docker.
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# Retry policy for transient Ollama server errors. RETRYABLE_STATUS lists
# the HTTP status codes that are safe to retry: rate limit (429), internal
# error (500) and "service unavailable"/model still loading (503).
RETRYABLE_STATUS = {429, 500, 503}
MAX_ATTEMPTS = 4
BASE_DELAY = 1.0  # seconds; doubled each retry with jitter

# Configuration For File Uploads

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = "uploads"
