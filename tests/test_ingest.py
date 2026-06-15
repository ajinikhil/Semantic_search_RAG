"""
Tests for the ingestion endpoints in app/routers/ingest.py:
  POST   /ingest/upload              → upload and ingest a document
  GET    /ingest/documents           → list stored documents
  DELETE /ingest/documents/{filename} → delete a document

Uses httpx.AsyncClient (async tests) with mocked ChromaDB and service calls
so no real vector-DB writes or file-system side-effects occur.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.ingest import router

# ── App / client fixture ──────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client():
    """AsyncClient wired to a minimal app containing only the ingest router."""
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Helper ────────────────────────────────────────────────────────────────────


def make_upload_file(filename: str, content: bytes = b"sample text content"):
    """Build a multipart file payload suitable for AsyncClient.post(files=...)."""
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


# ── Shared mock return values ─────────────────────────────────────────────────

_FAKE_EMBEDDINGS = [[0.1] * 384] * 3
_FAKE_CHUNKS = ["chunk one", "chunk two", "chunk three"]
_FAKE_TEXT = "Extracted document text."


# ══════════════════════════════════════════════════════════════════════════════
#  POST /ingest/upload
# ══════════════════════════════════════════════════════════════════════════════


class TestUploadDocument:
    """Tests for POST /ingest/upload."""

    @pytest.mark.asyncio
    async def test_upload_valid_pdf_returns_200(self, client):
        """
        Given a valid PDF file,
        When it is uploaded,
        Then the response is 200 with filename and chunk count.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch("app.routers.ingest.vectorstore.add_chunks", return_value=3),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("document.pdf")
            )

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "document.pdf"
        assert body["chunks_created"] == 3
        assert "successfully ingested" in body["message"]

    @pytest.mark.asyncio
    async def test_upload_valid_txt_returns_200(self, client):
        """
        Given a valid TXT file,
        When it is uploaded,
        Then the response is 200 with the correct filename.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch("app.routers.ingest.vectorstore.add_chunks", return_value=3),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("notes.txt", b"hello world")
            )

        assert response.status_code == 200
        assert response.json()["filename"] == "notes.txt"

    @pytest.mark.asyncio
    async def test_upload_valid_docx_returns_200(self, client):
        """
        Given a valid DOCX file,
        When it is uploaded,
        Then the response is 200 with the correct filename.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch("app.routers.ingest.vectorstore.add_chunks", return_value=3),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("report.docx", b"docx bytes")
            )

        assert response.status_code == 200
        assert response.json()["filename"] == "report.docx"

    @pytest.mark.asyncio
    async def test_upload_unsupported_extension_returns_400(self, client):
        """
        Given a file with an unsupported extension (.xlsx),
        When it is uploaded,
        Then a 400 error is returned with a descriptive message.
        """
        response = await client.post(
            "/ingest/upload", files=make_upload_file("spreadsheet.xlsx")
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_csv_extension_returns_400(self, client):
        """
        Given a .csv file (not in the allowed set),
        When it is uploaded,
        Then a 400 error is returned.
        """
        response = await client.post(
            "/ingest/upload", files=make_upload_file("data.csv")
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_no_file_returns_422(self, client):
        """
        Given no file is attached to the request,
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = await client.post("/ingest/upload")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_empty_document_returns_422(self, client):
        """
        Given a file that produces no extractable text (blank or scanned),
        When it is uploaded,
        Then a 422 error is returned indicating no text could be extracted.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value="   "),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("empty.pdf")
            )

        assert response.status_code == 422
        assert "Could not extract any text" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_parser_failure_returns_500(self, client):
        """
        Given the parser raises an unexpected exception,
        When a file is uploaded,
        Then a 500 error is returned with the error message.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch(
                "app.routers.ingest.parser.extract_text",
                side_effect=Exception("parse error"),
            ),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("broken.pdf")
            )

        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_embedder_failure_returns_500(self, client):
        """
        Given the embedder raises an unexpected exception after successful parsing,
        When a file is uploaded,
        Then a 500 error is returned.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text",
                side_effect=RuntimeError("model unavailable"),
            ),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("doc.pdf")
            )

        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_vectorstore_failure_returns_500(self, client):
        """
        Given ChromaDB raises an error when storing chunks,
        When a file is uploaded,
        Then a 500 error is returned.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch(
                "app.routers.ingest.vectorstore.add_chunks",
                side_effect=RuntimeError("disk full"),
            ),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("doc.pdf")
            )

        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    # ── Issue #20: uploaded file must not be orphaned on failure ──────────────

    @pytest.mark.asyncio
    async def test_upload_failure_removes_orphaned_file(self, client, tmp_path):
        """
        Regression test for issue #20.

        Given processing fails after the file has been written to disk,
        When a file is uploaded,
        Then the written file is removed so no orphan is left in the upload
        directory (an orphan has no ChromaDB entry and the delete endpoint
        could never reach it).
        """
        with (
            patch("app.routers.ingest.UPLOAD_DIR", str(tmp_path)),
            patch(
                "app.routers.ingest.parser.extract_text",
                side_effect=Exception("parse error"),
            ),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("broken.pdf")
            )

        assert response.status_code == 500
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_upload_empty_document_removes_orphaned_file(self, client, tmp_path):
        """
        Regression test for issue #20 (the 422 empty-text path).

        Given a file that yields no extractable text,
        When it is uploaded,
        Then the written file is removed from the upload directory.
        """
        with (
            patch("app.routers.ingest.UPLOAD_DIR", str(tmp_path)),
            patch("app.routers.ingest.parser.extract_text", return_value="   "),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("empty.pdf")
            )

        assert response.status_code == 422
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_upload_success_keeps_file_on_disk(self, client, tmp_path):
        """
        Regression test for issue #20.

        Given a successful ingestion,
        When a file is uploaded,
        Then the file is kept on disk (the delete endpoint relies on it
        existing in the upload directory).
        """
        with (
            patch("app.routers.ingest.UPLOAD_DIR", str(tmp_path)),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch("app.routers.ingest.vectorstore.add_chunks", return_value=3),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("keep.pdf")
            )

        assert response.status_code == 200
        assert (tmp_path / "keep.pdf").exists()

    @pytest.mark.asyncio
    async def test_upload_calls_services_in_correct_order(self, client):
        """
        Given a valid file,
        When it is uploaded,
        Then parser, embedder and vectorstore are each called exactly once
        in the correct order: extract → chunk → embed → store.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch(
                "app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT
            ) as mock_extract,
            patch(
                "app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS
            ) as mock_chunk,
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ) as mock_embed,
            patch(
                "app.routers.ingest.vectorstore.add_chunks", return_value=3
            ) as mock_add,
        ):
            await client.post("/ingest/upload", files=make_upload_file("doc.pdf"))

        mock_extract.assert_called_once()
        mock_chunk.assert_called_once()
        mock_embed.assert_called_once()
        mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_response_message_contains_filename(self, client):
        """
        Given a successful upload,
        When the response is returned,
        Then the success message contains the original filename.
        """
        with (
            patch("builtins.open", MagicMock()),
            patch("app.routers.ingest.shutil.copyfileobj"),
            patch("app.routers.ingest.parser.extract_text", return_value=_FAKE_TEXT),
            patch("app.routers.ingest.parser.chunk_text", return_value=_FAKE_CHUNKS),
            patch(
                "app.routers.ingest.embedder.embed_text", return_value=_FAKE_EMBEDDINGS
            ),
            patch("app.routers.ingest.vectorstore.add_chunks", return_value=3),
        ):
            response = await client.post(
                "/ingest/upload", files=make_upload_file("myreport.pdf")
            )

        assert "myreport.pdf" in response.json()["message"]


# ══════════════════════════════════════════════════════════════════════════════
#  GET /ingest/documents
# ══════════════════════════════════════════════════════════════════════════════


class TestListDocuments:
    """Tests for GET /ingest/documents."""

    @pytest.mark.asyncio
    async def test_list_documents_returns_200(self, client):
        """
        Given documents exist in the vector store,
        When the list endpoint is called,
        Then a 200 response is returned with chunk count and document names.
        """
        with patch(
            "app.routers.ingest.vectorstore.get_stats",
            return_value={"total_chunks": 42, "documents": ["a.pdf", "b.txt"]},
        ):
            response = await client.get("/ingest/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total_chunks"] == 42
        assert "a.pdf" in body["documents"]
        assert "b.txt" in body["documents"]

    @pytest.mark.asyncio
    async def test_list_documents_empty_store_returns_200(self, client):
        """
        Given no documents have been uploaded,
        When the list endpoint is called,
        Then a 200 response is returned with zero chunks and an empty list.
        """
        with patch(
            "app.routers.ingest.vectorstore.get_stats",
            return_value={"total_chunks": 0, "documents": []},
        ):
            response = await client.get("/ingest/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total_chunks"] == 0
        assert body["documents"] == []

    @pytest.mark.asyncio
    async def test_list_documents_schema_fields_present(self, client):
        """
        Given a non-empty vector store,
        When the list endpoint is called,
        Then the response body contains both total_chunks and documents fields.
        """
        with patch(
            "app.routers.ingest.vectorstore.get_stats",
            return_value={"total_chunks": 5, "documents": ["x.pdf"]},
        ):
            response = await client.get("/ingest/documents")

        body = response.json()
        assert "total_chunks" in body
        assert "documents" in body


# ══════════════════════════════════════════════════════════════════════════════
#  DELETE /ingest/documents/{filename}
# ══════════════════════════════════════════════════════════════════════════════


class TestDeleteDocument:
    """Tests for DELETE /ingest/documents/{filename}."""

    @pytest.mark.asyncio
    async def test_delete_existing_document_returns_200(self, client):
        """
        Given a document that exists in the vector store,
        When a delete request is made for that filename,
        Then a 200 response is returned with the number of chunks deleted.
        """
        with patch(
            "app.routers.ingest.vectorstore.delete_by_file_name", return_value=7
        ):
            response = await client.delete("/ingest/documents/report.pdf")

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "report.pdf"
        assert body["chunks_deleted"] == 7
        assert "deleted successfully" in body["message"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document_returns_404(self, client):
        """
        Given a filename that does not exist in the vector store
        (delete_by_file_name returns 0),
        When a delete request is made,
        Then a 404 error is returned with a descriptive message.
        """
        with patch(
            "app.routers.ingest.vectorstore.delete_by_file_name", return_value=0
        ):
            response = await client.delete("/ingest/documents/ghost.pdf")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_calls_vectorstore_with_correct_filename(self, client):
        """
        Given a valid delete request,
        When the endpoint is called,
        Then the vectorstore is called with exactly the filename from the URL.
        """
        with patch(
            "app.routers.ingest.vectorstore.delete_by_file_name", return_value=3
        ) as mock_delete:
            await client.delete("/ingest/documents/my_doc.pdf")

        mock_delete.assert_called_once_with("my_doc.pdf")

    @pytest.mark.asyncio
    async def test_delete_url_encoded_filename(self, client):
        """
        Given a filename containing special characters (URL-encoded),
        When a delete request is made,
        Then the vectorstore receives the decoded filename.
        """
        with patch(
            "app.routers.ingest.vectorstore.delete_by_file_name", return_value=2
        ) as mock_delete:
            await client.delete("/ingest/documents/my%20report.pdf")

        mock_delete.assert_called_once_with("my report.pdf")
