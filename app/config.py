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
TOP_K = 5

# LLM Configuration

MODEL_NAME = "gemini-3-flash-preview"

# Configuration For File Uploads

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = "uploads"
