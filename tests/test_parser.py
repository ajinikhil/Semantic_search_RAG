import pytest
from unittest.mock import patch, MagicMock

from app.services.parser import extract_text, chunk_text
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── extract_text tests ────────────────────────────────────────────────────────

class TestExtractText:

    def test_extract_text_returns_string(self):
        """extract_text should return a string when xtxt succeeds"""
        with patch("app.services.parser.xtxt", return_value="sample extracted text"):
            result = extract_text("dummy.pdf")
        assert isinstance(result, str)

    def test_extract_text_returns_correct_content(self):
        """extract_text should return exactly what xtxt returns"""
        expected = "Hello this is the document content."
        with patch("app.services.parser.xtxt", return_value=expected):
            result = extract_text("dummy.pdf")
        assert result == expected

    def test_extract_text_raises_runtime_error_on_failure(self):
        """extract_text should raise RuntimeError if xtxt fails"""
        with patch("app.services.parser.xtxt", side_effect=Exception("file not found")):
            with pytest.raises(RuntimeError) as exc_info:
                extract_text("bad_file.pdf")
        assert "error extracting text" in str(exc_info.value)

    def test_extract_text_error_message_contains_original_error(self):
        """RuntimeError message should include the original exception message"""
        with patch("app.services.parser.xtxt", side_effect=Exception("permission denied")):
            with pytest.raises(RuntimeError) as exc_info:
                extract_text("locked.pdf")
        assert "permission denied" in str(exc_info.value)

    def test_extract_text_called_with_correct_argument(self):
        """extract_text should pass the document argument to xtxt"""
        with patch("app.services.parser.xtxt", return_value="text") as mock_xtxt:
            extract_text("myfile.pdf")
        mock_xtxt.assert_called_once_with("myfile.pdf")


# ── chunk_text tests ──────────────────────────────────────────────────────────

class TestChunkText:

    def test_chunk_text_returns_list(self):
        """chunk_text should return a list"""
        text = "word " * 300  # enough text to produce chunks
        result = chunk_text(text)
        assert isinstance(result, list)

    def test_chunk_text_returns_non_empty_list(self):
        """chunk_text should return at least one chunk for non-empty input"""
        text = "word " * 300
        result = chunk_text(text)
        assert len(result) > 0

    def test_chunk_text_each_chunk_is_string(self):
        """every item in the returned list should be a string"""
        text = "word " * 300
        result = chunk_text(text)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_chunk_text_respects_chunk_size(self):
        """no chunk should exceed CHUNK_SIZE characters"""
        text = "word " * 500
        result = chunk_text(text)
        for chunk in result:
            assert len(chunk) <= CHUNK_SIZE, (
                f"Chunk of size {len(chunk)} exceeds CHUNK_SIZE {CHUNK_SIZE}"
            )

    def test_chunk_text_empty_string_returns_empty_list(self):
        """empty string input should return an empty list"""
        result = chunk_text("")
        assert result == []

    def test_chunk_text_short_text_returns_single_chunk(self):
        """text shorter than CHUNK_SIZE should return exactly one chunk"""
        text = "This is a very short text."
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_long_text_produces_multiple_chunks(self):
        """text much longer than CHUNK_SIZE should produce multiple chunks"""
        text = "word " * 1000
        result = chunk_text(text)
        assert len(result) > 1

    def test_chunk_text_overlap_means_chunks_share_content(self):
        """consecutive chunks should share some content due to overlap"""
        if CHUNK_OVERLAP == 0:
            pytest.skip("CHUNK_OVERLAP is 0, overlap test not applicable")
        text = "word " * 500
        result = chunk_text(text)
        if len(result) > 1:
            # Last words of chunk[0] should appear at start of chunk[1]
            end_of_first = result[0][-CHUNK_OVERLAP:]
            assert end_of_first in result[1]

    def test_chunk_text_raises_runtime_error_on_failure(self):
        """chunk_text should raise RuntimeError if splitter fails"""
        with patch(
            "app.services.parser.RecursiveCharacterTextSplitter",
            side_effect=Exception("splitter crashed")
        ):
            with pytest.raises(RuntimeError) as exc_info:
                chunk_text("some text")
        assert "error chunking text" in str(exc_info.value)

    def test_chunk_text_error_contains_original_message(self):
        """RuntimeError should include the original exception message"""
        with patch(
            "app.services.parser.RecursiveCharacterTextSplitter",
            side_effect=Exception("out of memory")
        ):
            with pytest.raises(RuntimeError) as exc_info:
                chunk_text("some text")
        assert "out of memory" in str(exc_info.value)