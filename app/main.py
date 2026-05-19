from fastapi import FastAPI

from app.routers import ingest, query

app = FastAPI(
    title="Semantic Search RAG",
    description="Upload documents and ask questions and receive"
    "answers based on the documents",
    version="1.0.0",
)


app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"health": "ok"}
