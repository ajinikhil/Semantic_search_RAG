from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------
# `xtxt` is imported lazily inside the function body (`from pyxtxt import xtxt`),
# so it is never bound as an attribute on app.services.parser.
# The correct target is the *source* module: "pyxtxt.xtxt".
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_returns_extracted_text(self):
        """Happy path: pyxtxt returns a string."""
        mock_document = MagicMock()
        expected_text = "Hello, world!"

        with patch("pyxtxt.xtxt", return_value=expected_text):
            from app.services.parser import extract_text

            result = extract_text(mock_document)

        assert result == expected_text

    def test_passes_document_to_xtxt(self):
        """Ensures the document argument is forwarded to pyxtxt unchanged."""
        mock_document = MagicMock()

        with patch("pyxtxt.xtxt") as mock_xtxt:
            from app.services.parser import extract_text

            extract_text(mock_document)

        mock_xtxt.assert_called_once_with(mock_document)

    def test_raises_runtime_error_when_xtxt_fails(self):
        """Any exception from pyxtxt is wrapped in a RuntimeError."""
        mock_document = MagicMock()

        with patch("pyxtxt.xtxt", side_effect=Exception("parse failure")):
            from app.services.parser import extract_text

            with pytest.raises(
                RuntimeError, match="error extracting text: parse failure"
            ):
                extract_text(mock_document)

    def test_runtime_error_wraps_original_message(self):
        """RuntimeError message includes the original exception detail."""
        original_message = "unsupported file format"

        with patch("pyxtxt.xtxt", side_effect=ValueError(original_message)):
            from app.services.parser import extract_text

            with pytest.raises(RuntimeError) as exc_info:
                extract_text(MagicMock())

        assert original_message in str(exc_info.value)

    def test_returns_empty_string_for_empty_document(self):
        """pyxtxt returning an empty string is passed through as-is."""
        with patch("pyxtxt.xtxt", return_value=""):
            from app.services.parser import extract_text

            result = extract_text(MagicMock())

        assert result == ""

    def test_multipage_text_returned_intact(self):
        """Multiline / multipage text is not altered."""
        multipage = "Page 1 content\n\nPage 2 content\n\nPage 3 content"

        with patch("pyxtxt.xtxt", return_value=multipage):
            from app.services.parser import extract_text

            result = extract_text(MagicMock())

        assert result == multipage


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    """Inject deterministic config values for every test in this module."""
    monkeypatch.setattr("app.services.parser.CHUNK_SIZE", CHUNK_SIZE)
    monkeypatch.setattr("app.services.parser.CHUNK_OVERLAP", CHUNK_OVERLAP)


class TestChunkText:
    def test_short_text_returned_as_single_chunk(self):
        """Text shorter than CHUNK_SIZE comes back as one element."""
        from app.services.parser import chunk_text

        short_text = "A" * (CHUNK_SIZE - 1)
        chunks = chunk_text(short_text)

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_long_text_split_into_multiple_chunks(self):
        """Text much longer than CHUNK_SIZE is split into multiple chunks."""
        from app.services.parser import chunk_text

        long_text = "word " * 200  # well over CHUNK_SIZE
        chunks = chunk_text(long_text)

        assert len(chunks) > 1

    def test_all_chunks_respect_chunk_size(self):
        """No individual chunk exceeds CHUNK_SIZE characters."""
        from app.services.parser import chunk_text

        long_text = "x" * 500
        chunks = chunk_text(long_text)

        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE

    def test_chunks_cover_full_content(self):
        """
        With overlap the splitter may repeat characters across boundaries,
        but the union of all chunks must contain every word of the original.
        Checks that no content is silently dropped.
        """
        from app.services.parser import chunk_text

        words = [f"word{i}" for i in range(200)]
        original = " ".join(words)
        chunks = chunk_text(original)

        reconstructed = "".join(chunks)
        for word in words:
            assert word in reconstructed

    def test_returns_list(self):
        """Return type is always a list."""
        from app.services.parser import chunk_text

        result = chunk_text("Some text.")
        assert isinstance(result, list)

    def test_empty_string_returns_empty_list(self):
        """Empty input produces no chunks."""
        from app.services.parser import chunk_text

        result = chunk_text("")
        assert result == []

    def test_raises_runtime_error_on_splitter_failure(self):
        """Exceptions from RecursiveCharacterTextSplitter
        are wrapped in RuntimeError."""
        with patch(
            "app.services.parser.RecursiveCharacterTextSplitter"
        ) as MockSplitter:
            MockSplitter.return_value.split_text.side_effect = Exception(
                "splitter boom"
            )

            from app.services.parser import chunk_text

            with pytest.raises(
                RuntimeError, match="error chunking text: splitter boom"
            ):
                chunk_text("Some text that triggers the failure.")

    def test_splitter_initialised_with_config_values(self):
        """RecursiveCharacterTextSplitter receives
        the configured chunk_size and overlap."""
        with patch(
            "app.services.parser.RecursiveCharacterTextSplitter"
        ) as MockSplitter:
            MockSplitter.return_value.split_text.return_value = ["chunk"]

            from app.services.parser import chunk_text

            chunk_text("any text")

        MockSplitter.assert_called_once_with(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def test_chunk_overlap_creates_shared_content(self):
        """
        With overlap > 0 adjacent chunks should share a substring at their
        boundary (assuming the text is long enough to force a split).
        """
        from app.services.parser import chunk_text

        sentence = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(sentence)

        if len(chunks) > 1:
            tail = chunks[0][-CHUNK_OVERLAP:]
            head = chunks[1][:CHUNK_OVERLAP]
            assert any(c in head for c in tail)
