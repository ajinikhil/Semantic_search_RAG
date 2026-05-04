from typing import Optional

from pydantic import BaseModel, Field

# INGESTION


class IngestResponse(BaseModel):
    "Returened after a document is successfully uploaded"

    filename: str
    chunks_created: int
    message: str


class DeleteResponse(BaseModel):
    "Returned after a document is deleted from vector database"

    filename: str
    chunks_deleted: int
    message: str


# QUERYING


class QueryRequest(BaseModel):
    "What user sends when asking a question/query"

    query: str = Field(..., min_length=1, max_length=2000)
    top_k_chunks: Optional[int] = Field(
        default=5, ge=1, le=20
    )  # greater than or equal to 1 and less than or equal to 20


class SourceChunk(BaseModel):
    "chunk retrived from Chroma DataBase"

    text: str  # The text chunk that retrived from Chroma DB
    file_name: str  # Nmae of the file the text is retrived from
    chunk_index: int  # Index of the chunk
    similarity_score: float  # Cosine similariy score


class QueryResponse(BaseModel):
    "What FastAPI returns after generating an answer through LLM"

    query: str  # The original query
    response: str  # The response LLM generated
    sources: list[SourceChunk]  # The souce chunks LLM used to generate the answer


class DbInfo(BaseModel):
    "What is stored in Chroma DB"

    total_chunks: int
    documents: list[str]
