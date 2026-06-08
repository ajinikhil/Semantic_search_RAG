import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level patch: intercept chromadb.PersistentClient before the service
# module is imported so the global ``collection`` is a MagicMock.
# ---------------------------------------------------------------------------

_mock_collection = MagicMock()
_mock_client = MagicMock()
_mock_client.get_or_create_collection.return_value = _mock_collection

with patch("chromadb.PersistentClient", return_value=_mock_client):
    import app.services.vectorstore as chroma_service

    # Inject the mock collection directly so every call in tests hits our mock
    chroma_service.collection = _mock_collection


def _reset_mock():
    """Reset the shared collection mock between tests.

    ``reset_mock()`` alone does NOT clear ``side_effect`` in Python 3.8+,
    so leftover side effects from error-path tests would bleed into the next
    test.  We clear them explicitly on every child mock we use.
    """
    _mock_collection.reset_mock(side_effect=True, return_value=True)
    # Belt-and-suspenders: explicitly nil out side_effect on each child
    for attr in ("add", "query", "get", "delete", "count"):
        child = getattr(_mock_collection, attr)
        child.side_effect = None
        child.return_value = MagicMock()


# ===========================================================================
# add_chunks
# ===========================================================================


class TestAddChunks:
    """Tests for :func:`chroma_service.add_chunks`."""

    def setup_method(self):
        _reset_mock()

    def test_returns_chunk_count_on_success(self):
        """
        ``add_chunks`` should return the number of chunks inserted when
        ``collection.add`` succeeds without raising.
        """
        chunks = ["chunk one", "chunk two", "chunk three"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        filename = "report.pdf"

        result = chroma_service.add_chunks(chunks, embeddings, filename)

        assert result == 3

    def test_calls_collection_add_with_correct_args(self):
        """
        ``collection.add`` must be called exactly once and receive the right
        documents, embeddings, and metadatas.  UUIDs are verified by count
        only (non-deterministic values).
        """
        chunks = ["a", "b"]
        embeddings = [[1.0], [2.0]]
        filename = "notes.txt"

        chroma_service.add_chunks(chunks, embeddings, filename)

        _mock_collection.add.assert_called_once()
        kwargs = _mock_collection.add.call_args.kwargs

        assert kwargs["documents"] == chunks
        assert kwargs["embeddings"] == embeddings
        assert len(kwargs["ids"]) == 2
        assert kwargs["metadatas"] == [
            {"filename": "notes.txt", "chunk_index": 0},
            {"filename": "notes.txt", "chunk_index": 1},
        ]

    def test_raises_runtime_error_when_collection_add_fails(self):
        """
        If ``collection.add`` raises any exception, ``add_chunks`` must wrap it
        in a ``RuntimeError`` with a descriptive message.
        """
        _mock_collection.add.side_effect = Exception("disk full")

        with pytest.raises(
            RuntimeError, match="Error while adding documents to ChromaDB"
        ):
            chroma_service.add_chunks(["x"], [[0.0]], "file.pdf")

    def test_generates_unique_ids_per_chunk(self):
        """Each chunk must receive a distinct UUID string."""
        chunks = ["p", "q", "r"]
        chroma_service.add_chunks(chunks, [[0.0]] * 3, "doc.pdf")

        ids = _mock_collection.add.call_args.kwargs["ids"]
        assert len(ids) == len(set(ids)), "IDs must be unique"


# ===========================================================================
# search
# ===========================================================================


class TestSearch:
    """Tests for :func:`chroma_service.search`."""

    def setup_method(self):
        _reset_mock()

    def test_returns_query_results(self):
        """
        ``search`` should return whatever ``collection.query`` returns, passing
        the query vector wrapped in a list.
        """
        fake_results = {
            "documents": [["chunk text"]],
            "metadatas": [[{"filename": "a.pdf", "chunk_index": 0}]],
            "distances": [[0.12]],
        }
        _mock_collection.query.return_value = fake_results

        result = chroma_service.search([0.1, 0.2, 0.3])

        assert result is fake_results

    def test_passes_query_vector_as_list(self):
        """
        ChromaDB expects ``query_embeddings`` to be a list of vectors.
        ``search`` must wrap the single vector correctly.
        """
        vector = [0.5, 0.6]
        chroma_service.search(vector)

        call_kwargs = _mock_collection.query.call_args.kwargs
        assert call_kwargs["query_embeddings"] == [vector]

    def test_includes_correct_fields(self):
        """``search`` must request documents, metadatas, and distances."""
        chroma_service.search([0.0])

        call_kwargs = _mock_collection.query.call_args.kwargs
        assert set(call_kwargs["include"]) == {"documents", "metadatas", "distances"}

    def test_raises_runtime_error_on_query_failure(self):
        """
        If ``collection.query`` raises, ``search`` must re-raise as
        ``RuntimeError`` with context.
        """
        _mock_collection.query.side_effect = Exception("connection lost")

        with pytest.raises(RuntimeError, match="Error while searching from ChromaDB"):
            chroma_service.search([0.0])


# ===========================================================================
# delete_by_file_name
# ===========================================================================


class TestDeleteByFileName:
    """Tests for :func:`chroma_service.delete_by_file_name`."""

    def setup_method(self):
        _reset_mock()

    def test_deletes_chunks_and_file_when_exists(self):
        """
        When chunks are found and the file exists on disk, both
        ``collection.delete`` and ``os.remove`` must be called.
        """
        _mock_collection.get.return_value = {"ids": ["id1", "id2"]}

        with patch("os.path.exists", return_value=True), patch(
            "os.remove"
        ) as mock_remove:
            result = chroma_service.delete_by_file_name("report.pdf")

        _mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])
        mock_remove.assert_called_once()
        assert result == 2

    def test_returns_number_of_deleted_chunks(self):
        """Return value must equal the number of chunk IDs deleted."""
        _mock_collection.get.return_value = {"ids": ["a", "b", "c"]}

        with patch("os.path.exists", return_value=False):
            result = chroma_service.delete_by_file_name("data.csv")

        assert result == 3

    def test_skips_os_remove_when_file_not_on_disk(self):
        """
        If the physical file is not present (already cleaned up elsewhere),
        ``os.remove`` must NOT be called — no error should be raised.
        """
        _mock_collection.get.return_value = {"ids": ["id1"]}

        with patch("os.path.exists", return_value=False), patch(
            "os.remove"
        ) as mock_remove:
            chroma_service.delete_by_file_name("ghost.pdf")

        mock_remove.assert_not_called()

    def test_returns_zero_when_no_chunks_found(self):
        """
        If ``collection.get`` returns an empty ID list, ``delete_by_file_name``
        must return 0 so the router can raise a 404.
        """
        _mock_collection.get.return_value = {"ids": []}

        result = chroma_service.delete_by_file_name("missing.pdf")

        assert result == 0

    def test_raises_runtime_error_on_collection_error(self):
        """
        An unexpected exception from ``collection.get`` must be wrapped in a
        ``RuntimeError``.
        """
        _mock_collection.get.side_effect = Exception("timeout")

        with pytest.raises(RuntimeError, match="Error deleting the file"):
            chroma_service.delete_by_file_name("any.pdf")

    def test_constructs_correct_file_path(self):
        """
        The path passed to ``os.remove`` must be the join of ``UPLOAD_DIR``
        and the filename.
        """
        _mock_collection.get.return_value = {"ids": ["x"]}
        expected_path = os.path.join(chroma_service.UPLOAD_DIR, "invoice.pdf")

        with patch("os.path.exists", return_value=True), patch(
            "os.remove"
        ) as mock_remove:
            chroma_service.delete_by_file_name("invoice.pdf")

        mock_remove.assert_called_once_with(expected_path)


# ===========================================================================
# get_stats
# ===========================================================================


class TestGetStats:
    """Tests for :func:`chroma_service.get_stats`."""

    def setup_method(self):
        _reset_mock()

    def test_returns_zero_stats_when_collection_empty(self):
        """
        When ``collection.count()`` returns 0, ``get_stats`` must return
        ``{"total_chunks": 0, "documents": []}`` without calling
        ``collection.get``.
        """
        _mock_collection.count.return_value = 0

        result = chroma_service.get_stats()

        assert result == {"total_chunks": 0, "documents": []}
        _mock_collection.get.assert_not_called()

    def test_returns_correct_total_and_unique_filenames(self):
        """
        ``get_stats`` should return the total chunk count and a sorted,
        deduplicated list of filenames from the metadata.
        """
        _mock_collection.count.return_value = 5
        _mock_collection.get.return_value = {
            "metadatas": [
                {"filename": "b.pdf", "chunk_index": 0},
                {"filename": "a.pdf", "chunk_index": 0},
                {"filename": "a.pdf", "chunk_index": 1},
                {"filename": "c.pdf", "chunk_index": 0},
                {"filename": "b.pdf", "chunk_index": 1},
            ]
        }

        result = chroma_service.get_stats()

        assert result["total_chunks"] == 5
        assert result["documents"] == ["a.pdf", "b.pdf", "c.pdf"]  # sorted

    def test_deduplicates_filenames(self):
        """
        The ``documents`` list must contain each filename only once even when
        many chunks share the same filename.
        """
        _mock_collection.count.return_value = 3
        _mock_collection.get.return_value = {
            "metadatas": [
                {"filename": "x.pdf", "chunk_index": 0},
                {"filename": "x.pdf", "chunk_index": 1},
                {"filename": "x.pdf", "chunk_index": 2},
            ]
        }

        result = chroma_service.get_stats()

        assert result["documents"] == ["x.pdf"]

    def test_raises_runtime_error_on_failure(self):
        """
        Any exception from ChromaDB calls must be wrapped in a ``RuntimeError``
        with a message referencing the stats retrieval.
        """
        _mock_collection.count.side_effect = Exception("network error")

        with pytest.raises(
            RuntimeError, match="Error while retreving stats from ChromaDB"
        ):
            chroma_service.get_stats()
