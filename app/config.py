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

# LLM Configuration

MODEL_NAME = "gemini-3-flash-preview"

# Retry policy for transient Gemini errors. RETRYABLE_STATUS lists the
# error codes that are safe to retry: rate limit (429), internal error
# (500) and "model overloaded" (503).
RETRYABLE_STATUS = {429, 500, 503}
MAX_ATTEMPTS = 4
BASE_DELAY = 1.0  # seconds; doubled each retry with jitter

# Configuration For File Uploads

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = "uploads"
