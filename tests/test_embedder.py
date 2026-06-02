from unittest.mock import patch

import numpy as np
import pytest

from app.services.embedder import embed_query, embed_text


@pytest.fixture
def mock_model():
    """Mock the SentenceTransformer model to avoid loading it during tests."""
    with patch("app.services.embedder.model") as mock:
        yield mock


@pytest.fixture
def sample_text():
    """A valid text chunk as would be extracted from a document."""
    return "FastAPI is a modern web framework for building APIs."


@pytest.fixture
def sample_query():
    """A valid user query as would be submitted to the search endpoint."""
    return "What is FastAPI?"


@pytest.fixture
def mock_embedding():
    """A fake normalized embedding vector of dimension 384 (all-MiniLM-L6-v2)."""
    return np.array([0.1, 0.2, 0.3] * 128, dtype=np.float32)  # 384 dims


class TestEmbedText:
    """Tests for the embed_text function."""

    def test_returns_list(self, mock_model, sample_text, mock_embedding):
        """embed_text should always return a Python list."""
        mock_model.encode.return_value = mock_embedding
        result = embed_text(sample_text)
        assert isinstance(result, list)

    def test_returns_floats(self, mock_model, sample_text, mock_embedding):
        """Each element in the returned list should be a float."""
        mock_model.encode.return_value = mock_embedding
        result = embed_text(sample_text)
        assert all(isinstance(v, float) for v in result)

    def test_correct_dimensions(self, mock_model, sample_text, mock_embedding):
        """Returned embedding should have 384 dimensions (all-MiniLM-L6-v2)."""
        mock_model.encode.return_value = mock_embedding
        result = embed_text(sample_text)
        assert len(result) == 384

    def test_calls_model_once(self, mock_model, sample_text, mock_embedding):
        """embed_text should call the model exactly once per invocation."""
        mock_model.encode.return_value = mock_embedding
        embed_text(sample_text)
        mock_model.encode.assert_called_once()

    def test_normalize_embeddings_true(self, mock_model, sample_text, mock_embedding):
        """embed_text should always pass normalize_embeddings=True to the model."""
        mock_model.encode.return_value = mock_embedding
        embed_text(sample_text)
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is True

    def test_raises_value_error_on_empty_string(self):
        """embed_text should raise ValueError when given an empty string."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            embed_text("")

    def test_raises_value_error_on_whitespace(self):
        """embed_text should raise ValueError when given only whitespace."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            embed_text("   ")

    def test_raises_runtime_error_on_model_failure(self, mock_model, sample_text):
        """embed_text should raise RuntimeError if the model throws an exception."""
        mock_model.encode.side_effect = Exception("Model crashed")
        with pytest.raises(RuntimeError, match="Error encoding text"):
            embed_text(sample_text)


class TestEmbedQuery:
    """Tests for the embed_query function."""

    def test_returns_list(self, mock_model, sample_query, mock_embedding):
        """embed_query should always return a Python list."""
        mock_model.encode.return_value = np.array([mock_embedding])
        result = embed_query(sample_query)
        assert isinstance(result, list)

    def test_returns_floats(self, mock_model, sample_query, mock_embedding):
        """Each element in the returned list should be a float."""
        mock_model.encode.return_value = np.array([mock_embedding])
        result = embed_query(sample_query)
        assert all(isinstance(v, float) for v in result)

    def test_correct_dimensions(self, mock_model, sample_query, mock_embedding):
        """Returned query embedding should have 384 dimensions."""
        mock_model.encode.return_value = np.array([mock_embedding])
        result = embed_query(sample_query)
        assert len(result) == 384

    def test_calls_model_once(self, mock_model, sample_query, mock_embedding):
        """embed_query should call the model exactly once per invocation."""
        mock_model.encode.return_value = np.array([mock_embedding])
        embed_query(sample_query)
        mock_model.encode.assert_called_once()

    def test_normalize_embeddings_true(self, mock_model, sample_query, mock_embedding):
        """embed_query should always pass normalize_embeddings=True to the model."""
        mock_model.encode.return_value = np.array([mock_embedding])
        embed_query(sample_query)
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is True

    def test_query_wrapped_in_list(self, mock_model, sample_query, mock_embedding):
        """embed_query should pass the query as a list
        to the model, not a raw string."""
        mock_model.encode.return_value = np.array([mock_embedding])
        embed_query(sample_query)
        args, _ = mock_model.encode.call_args
        assert isinstance(args[0], list)
        assert args[0][0] == sample_query

    def test_raises_runtime_error_on_model_failure(self, mock_model, sample_query):
        """embed_query should raise RuntimeError if the model throws an exception."""
        mock_model.encode.side_effect = Exception("Model crashed")
        with pytest.raises(RuntimeError, match="Error encoding query"):
            embed_query(sample_query)

    def test_empty_query_does_not_raise_value_error(self, mock_model, mock_embedding):
        """
        embed_query does not validate empty strings (unlike embed_text).
        This test documents that behaviour explicitly.
        """
        mock_model.encode.return_value = np.array([mock_embedding])
        result = embed_query("")
        assert isinstance(result, list)
