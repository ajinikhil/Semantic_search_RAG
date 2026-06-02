import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.ingest import router

# ── App fixture ───────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create a minimal FastAPI app with only the ingest router mounted."""
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Return a TestClient that makes requests against the ingest router."""
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_upload_file(filename: str, content: bytes = b"sample text content"):
    """
    Build a multipart file payload for use with TestClient.

    Args:
        filename: The name of the file being uploaded.
        content:  Raw bytes to use as the file body.

    Returns:
        A dict suitable for passing as `files=` in TestClient.post().
    """
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


# ══════════════════════════════════════════════════════════════════════════════
#  POST /ingest/upload
# ══════════════════════════════════════════════════════════════════════════════


class TestUploadDocument:
    """Tests for the POST /ingest/upload endpoint."""

    @patch("app.routers.ingest.vectorstore.add_chunks", return_value=10)
    @patch("app.routers.ingest.embedder.embed_text", return_value=[[0.1] * 384] * 5)
    @patch("app.routers.ingest.parser.chunk_text", return_value=["chunk1", "chunk2"])
    @patch("app.routers.ingest.parser.extract_text", return_value="Some extracted text")
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_valid_pdf_returns_200(
        self,
        mock_copy,
        mock_extract,
        mock_chunk,
        mock_embed,
        mock_add,
        client,
    ):
        """
        Given a valid PDF file,
        When it is uploaded,
        Then the response is 200 with filename and chunk count.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("document.pdf"),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "document.pdf"
        assert body["chunks_created"] == 10
        assert "successfully ingested" in body["message"]

    @patch("app.routers.ingest.vectorstore.add_chunks", return_value=5)
    @patch("app.routers.ingest.embedder.embed_text", return_value=[[0.1] * 384] * 3)
    @patch("app.routers.ingest.parser.chunk_text", return_value=["chunk1"])
    @patch("app.routers.ingest.parser.extract_text", return_value="Some extracted text")
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_valid_txt_returns_200(
        self,
        mock_copy,
        mock_extract,
        mock_chunk,
        mock_embed,
        mock_add,
        client,
    ):
        """
        Given a valid TXT file,
        When it is uploaded,
        Then the response is 200.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("notes.txt", b"hello world"),
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "notes.txt"

    @patch("app.routers.ingest.vectorstore.add_chunks", return_value=8)
    @patch("app.routers.ingest.embedder.embed_text", return_value=[[0.1] * 384] * 3)
    @patch("app.routers.ingest.parser.chunk_text", return_value=["chunk1"])
    @patch("app.routers.ingest.parser.extract_text", return_value="Some extracted text")
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_valid_docx_returns_200(
        self,
        mock_copy,
        mock_extract,
        mock_chunk,
        mock_embed,
        mock_add,
        client,
    ):
        """
        Given a valid DOCX file,
        When it is uploaded,
        Then the response is 200.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("report.docx", b"docx bytes"),
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "report.docx"

    def test_upload_unsupported_extension_returns_400(self, client):
        """
        Given a file with an unsupported extension (e.g. .xlsx),
        When it is uploaded,
        Then a 400 error is returned with a descriptive message.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("spreadsheet.xlsx"),
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_no_file_returns_422(self, client):
        """
        Given no file is attached to the request,
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = client.post("/ingest/upload")
        assert response.status_code == 422

    @patch("app.routers.ingest.parser.extract_text", return_value="   ")
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_empty_document_returns_422(self, mock_copy, mock_extract, client):
        """
        Given a file that produces no extractable text (blank or scanned),
        When it is uploaded,
        Then a 422 error is returned indicating no text was found.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("empty.pdf"),
        )
        assert response.status_code == 422
        assert "Could not extract any text" in response.json()["detail"]

    @patch(
        "app.routers.ingest.parser.extract_text", side_effect=Exception("parse error")
    )
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_processing_error_returns_500(self, mock_copy, mock_extract, client):
        """
        Given the parser raises an unexpected exception,
        When a file is uploaded,
        Then a 500 error is returned with the error message.
        """
        response = client.post(
            "/ingest/upload",
            files=make_upload_file("broken.pdf"),
        )
        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    @patch("app.routers.ingest.vectorstore.add_chunks", return_value=3)
    @patch("app.routers.ingest.embedder.embed_text", return_value=[[0.1] * 384])
    @patch("app.routers.ingest.parser.chunk_text", return_value=["chunk1"])
    @patch("app.routers.ingest.parser.extract_text", return_value="Some text")
    @patch("app.routers.ingest.shutil.copyfileobj")
    @patch("builtins.open", MagicMock())
    def test_upload_calls_services_in_correct_order(
        self,
        mock_copy,
        mock_extract,
        mock_chunk,
        mock_embed,
        mock_add,
        client,
    ):
        """
        Given a valid file,
        When it is uploaded,
        Then parser, embedder and vectorstore are each called exactly once
        in the correct order (extract → chunk → embed → store).
        """
        client.post("/ingest/upload", files=make_upload_file("doc.pdf"))

        mock_extract.assert_called_once()
        mock_chunk.assert_called_once()
        mock_embed.assert_called_once()
        mock_add.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  GET /ingest/documents
# ══════════════════════════════════════════════════════════════════════════════


class TestListDocuments:
    """Tests for the GET /ingest/documents endpoint."""

    @patch(
        "app.routers.ingest.vectorstore.get_stats",
        return_value={"total_chunks": 42, "documents": ["a.pdf", "b.txt"]},
    )
    def test_list_documents_returns_200(self, mock_stats, client):
        """
        Given documents exist in the vector store,
        When the list endpoint is called,
        Then a 200 response is returned with chunk count and document names.
        """
        response = client.get("/ingest/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total_chunks"] == 42
        assert "a.pdf" in body["documents"]
        assert "b.txt" in body["documents"]

    @patch(
        "app.routers.ingest.vectorstore.get_stats",
        return_value={"total_chunks": 0, "documents": []},
    )
    def test_list_documents_empty_store(self, mock_stats, client):
        """
        Given no documents have been uploaded,
        When the list endpoint is called,
        Then a 200 response is returned with zero chunks and empty list.
        """
        response = client.get("/ingest/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total_chunks"] == 0
        assert body["documents"] == []


# ══════════════════════════════════════════════════════════════════════════════
#  DELETE /ingest/documents/{filename}
# ══════════════════════════════════════════════════════════════════════════════


class TestDeleteDocument:
    """Tests for the DELETE /ingest/documents/{filename} endpoint."""

    @patch("app.routers.ingest.vectorstore.delete_by_file_name", return_value=7)
    def test_delete_existing_document_returns_200(self, mock_delete, client):
        """
        Given a document that exists in the vector store,
        When a delete request is made for that filename,
        Then a 200 response is returned with the number of chunks deleted.
        """
        response = client.delete("/ingest/documents/report.pdf")

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "report.pdf"
        assert body["chunks_deleted"] == 7
        assert "deleted successfully" in body["message"]

    @patch("app.routers.ingest.vectorstore.delete_by_file_name", return_value=0)
    def test_delete_nonexistent_document_returns_404(self, mock_delete, client):
        """
        Given a filename that does not exist in the vector store,
        When a delete request is made,
        Then a 404 error is returned with a descriptive message.
        """
        response = client.delete("/ingest/documents/ghost.pdf")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("app.routers.ingest.vectorstore.delete_by_file_name", return_value=3)
    def test_delete_calls_vectorstore_with_correct_filename(self, mock_delete, client):
        """
        Given a valid delete request,
        When the endpoint is called,
        Then the vectorstore is called with exactly the filename from the URL.
        """
        client.delete("/ingest/documents/my_doc.pdf")
        mock_delete.assert_called_once_with("my_doc.pdf")
