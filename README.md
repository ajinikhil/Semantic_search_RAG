# Semantic Search Engine

> Work in progress — currently in active development

A local, fully private semantic search engine that lets you upload your own documents and ask questions about them in plain English. No cloud. No data leaving your machine.

---

## How it works

1. You upload a document (PDF, DOCX, or TXT)
2. The system splits it into chunks and converts each chunk into a vector — a mathematical representation of meaning
3. Those vectors are stored in a local database (ChromaDB)
4. When you ask a question, it gets converted into a vector too
5. ChromaDB finds the chunks most similar in meaning to your question
6. Those chunks get sent to a local LLM along with your question
7. The LLM generates a grounded answer and returns it to you

No guessing. No hallucinating facts that aren't in your documents.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Embeddings | HuggingFace `sentence-transformers` (local) |
| Vector store | ChromaDB (local) |
| LLM | Google Gemini |
| Testing | pytest + pytest-cov |
| CI/CD | GitHub Actions |

---

## Project structure

```
semantic-rag/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings and environment variables
│   ├── routers/
│   │   ├── ingest.py        # Upload, list, delete document endpoints
│   │   └── query.py         # Ask question endpoint
│   ├── services/
│   │   ├── parser.py        # Extract text from PDF / DOCX / TXT
│   │   ├── embedder.py      # HuggingFace embedding model
│   │   ├── vectorstore.py   # ChromaDB read / write
│   │   └── llm.py           # LLM answer generation
│   └── models/
│       └── schemas.py       # Pydantic request / response models
├── tests/                   # pytest test suite
├── postman/                 # Postman collection for manual testing
├── requirements.txt
└── .env                     # Local config (not committed)
```

---

## Getting started

### Prerequisites

- Python 3.11+
- A Gemini API key — get one at [ai.google.dev](https://ai.google.dev)

### Installation

```bash
# Clone the repo
git clone https://github.com/ajinikhil/Semantic_search_RAG.git
cd Semantic_search_RAG

# Install dependencies
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Running the server

```bash
uvicorn app.main:app --reload
```

API docs are auto-generated and available at `http://localhost:8000/docs`

---

## API endpoints

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/upload` | Upload a PDF, DOCX, or TXT file |
| `GET` | `/ingest/documents` | List all stored documents |
| `DELETE` | `/ingest/documents/{filename}` | Delete a document and its chunks |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question against your documents |

---

## Running tests

```bash
pytest tests/ -v --cov=app --cov-fail-under=80
```

---

## CI/CD

The pipeline runs on every push to `main` or `develop` and on all pull requests to `main`:

```
Lint (ruff + black) → Tests (pytest + coverage) → Security (pip-audit) → Build check
```

Each stage only runs if the previous one passes.

---

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `EMBEDDING_MODEL` | HuggingFace model name (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `CHROMA_PATH` | Path to persist ChromaDB (default: `data/chroma_db`) |
| `TOP_K` | Number of chunks to retrieve per query (default: `5`) |
