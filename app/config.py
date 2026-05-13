from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DB_PATH = "data/chroma_db"
CHROMA_RESULTS_RETURNED = 5
MODEL_NAME = "gemini-3-flash-preview"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = "uploads"
TOP_K = 5
