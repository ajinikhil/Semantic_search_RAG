import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from app.models.schemas import IngestResponse  # DeleteResponse
from app.services import embedder, parser, vectorstore

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}."
                "Allowed extensions: {' '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    file_path = Path(UPLOAD_DIR) / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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
                chunks=chunks, embeddings=embeddings, filename=file.filename
            )

        except Exception as e:

            raise HTTPException(
                status_code=500, detail=f"Failed to process document: {str(e)}"
            )

        return IngestResponse(
            filename=file.filename,
            chunks_created=chunks_created,
            message=(f"'{file.filename}' successfully ingested into the vector store."),
        )
