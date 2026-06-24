import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from app.models.schemas import DbInfo, DeleteResponse, IngestResponse, SummaryResponse
from app.services import embedder, parser, vectorstore
from app.services.llm import generate_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    File upload, validation, processing and storing the
    doument to a vector database.

    When the user uploads the file, it is:

    * It is validated aganist the supported file extensions
    * saved to upload directory
    * Parsed into raw text
    * Split into chunks
    * Stored in vector database


    Args:
        file (UploadFile): The uploaded document file

    Raises:
        HTTPException: When the file is unsupported, text
                        extraction fails or when document
                        processing encounters an error.

    Returns:
        IngestResponse: Metadata about the stored documents
                        which includes file name, number of
                        chunks created and ingestion status
                        message.
    """

    if not file.filename:
        raise HTTPException(
            status_code=422, detail="No file was attached. Please upload a file."
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type:{extension}. "
                f"Allowed extensions: {' '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    safe_name = Path(file.filename).name
    file_path = (Path(UPLOAD_DIR) / safe_name).resolve()
    if not str(file_path).startswith(str(Path(UPLOAD_DIR).resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingested = False
    try:

        raw_text = parser.extract_text(str(file_path))

        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not extract any text from the document."
                    " The file may be empty or scanned"
                ),
            )

        chunks = parser.chunk_text(raw_text)

        embeddings = embedder.embed_text(chunks)

        chunks_created = vectorstore.add_chunks(
            chunks=chunks, embeddings=embeddings, filename=safe_name
        )
        ingested = True
    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to process document '%s'", safe_name)
        raise HTTPException(
            status_code=500, detail="Internal error processing the document."
        )
    finally:
        if not ingested:
            file_path.unlink(missing_ok=True)

    return IngestResponse(
        filename=safe_name,
        chunks_created=chunks_created,
        message=(f"'{safe_name}' successfully ingested into the vector store."),
    )


@router.get("/documents", response_model=DbInfo)
async def list_documents():
    """
    Retrieve information about the stored documents in
        Chroma Database

    Returns:
        DbInfo: ChromaDB information, which contains the chunk count
                and stored document names.
    """
    stats = vectorstore.get_stats()
    return DbInfo(**stats)


@router.get("/documents/{filename}/summary", response_model=SummaryResponse)
def summarize_document(filename: str):
    """
    Generate a summary of a previously uploaded document on demand.

    The original file is re-read from the upload directory and parsed
    back into text, which is then summarized by the LLM.

    Args:
        filename (str): Name of the document to summarize.

    Raises:
        HTTPException: if the filename is invalid, the document is not
            found, or summary generation fails.

    Returns:
        SummaryResponse: The filename and the generated summary.
    """
    safe_name = Path(filename).name
    file_path = (Path(UPLOAD_DIR) / safe_name).resolve()
    if not str(file_path).startswith(str(Path(UPLOAD_DIR).resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{safe_name}' not found.",
        )

    try:
        raw_text = parser.extract_text(str(file_path))

        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the document.",
            )

        summary = generate_summary(raw_text)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to summarize document '%s'", safe_name)
        raise HTTPException(
            status_code=500, detail="Internal error summarizing the document."
        )

    return SummaryResponse(filename=safe_name, summary=summary)


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    """
    Delete a docunment and its chunks from ChromaDB

    Args:
        filename (str): Name of the document to delete

    Raises:
        HTTPException: if the document does not exist in ChromaDB

    Returns:
        DeleteResponse: Information about the deleted document,
        which includes the filename, number of chunks removed and
        the delete status message.
    """
    chunks_deleted = vectorstore.delete_by_file_name(filename)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in the vector store",
        )

    return DeleteResponse(
        filename=filename,
        chunks_deleted=chunks_deleted,
        message=f"'{filename}' is deleted successfully",
    )
