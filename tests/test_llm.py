from unittest.mock import patch

import pytest

import app.services.llm as llm
from app.models.schemas import SourceChunk
from app.services.llm import build_prompt, generate_answer, get_client

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks():
    return [
        SourceChunk(
            text="FastAPI is a modern web framework.",
            file_name="doc1.pdf",
            chunk_index=0,
            similarity_score=0.95,
        ),
        SourceChunk(
            text="ChromaDB is a vector database.",
            file_name="doc2.pdf",
            chunk_index=1,
            similarity_score=0.89,
        ),
    ]


@pytest.fixture
def single_chunk():
    return [
        SourceChunk(
            text="Python is a programming language.",
            file_name="doc1.pdf",
            chunk_index=0,
            similarity_score=0.91,
        ),
    ]


@pytest.fixture
def mock_client():
    """
    Patch get_client() so no real Gemini API call is made.

    get_client() is called inside generate_answer() — patching it
    means mock_client.return_value is the client instance used
    inside the function.
    """
    with patch("app.services.llm.get_client") as mock:
        yield mock


# ══════════════════════════════════════════════════════════════════════════════
#  get_client
# ══════════════════════════════════════════════════════════════════════════════


class TestGetClient:
    """Tests for the lazy module-level Gemini client singleton (issue #17)."""

    def test_client_is_cached_and_built_once(self):
        """get_client should construct genai.Client once and reuse it."""
        llm._client = None  # reset the module-level singleton
        with patch("app.services.llm.genai.Client") as mock_ctor:
            first = get_client()
            second = get_client()

        assert first is second
        mock_ctor.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  build_prompt
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_prompt_returns_string(self, sample_chunks):
        """build_prompt should always return a string."""
        prompt = build_prompt("What is FastAPI?", sample_chunks)
        assert isinstance(prompt, str)

    def test_prompt_contains_question(self, sample_chunks):
        """build_prompt should include the user question in the prompt."""
        prompt = build_prompt("What is FastAPI?", sample_chunks)
        assert "What is FastAPI?" in prompt

    def test_prompt_contains_chunk_text(self, sample_chunks):
        """build_prompt should include the text of every chunk."""
        prompt = build_prompt("What is FastAPI?", sample_chunks)
        assert "FastAPI is a modern web framework." in prompt
        assert "ChromaDB is a vector database." in prompt

    def test_prompt_contains_source_filenames(self, sample_chunks):
        """build_prompt should include the source filename of every chunk."""
        prompt = build_prompt("What is FastAPI?", sample_chunks)
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt

    def test_prompt_contains_chunk_index(self, sample_chunks):
        """build_prompt should include the chunk index of every chunk."""
        prompt = build_prompt("What is FastAPI?", sample_chunks)
        assert "chunk 0" in prompt
        assert "chunk 1" in prompt

    def test_prompt_single_chunk(self, single_chunk):
        """build_prompt should work correctly with a single chunk."""
        prompt = build_prompt("What is Python?", single_chunk)
        assert "Python is a programming language." in prompt

    def test_prompt_empty_chunks(self):
        """build_prompt should still return a string when chunks is empty."""
        prompt = build_prompt("What is FastAPI?", [])
        assert isinstance(prompt, str)

    def test_prompt_empty_question(self, sample_chunks):
        """build_prompt should still return a string when question is empty."""
        prompt = build_prompt("", sample_chunks)
        assert isinstance(prompt, str)

    def test_prompt_raises_runtime_error_on_invalid_chunks(self):
        """build_prompt should raise RuntimeError if chunks are malformed."""
        with pytest.raises(RuntimeError):
            build_prompt("What is FastAPI?", ["invalid", "chunks"])


# ══════════════════════════════════════════════════════════════════════════════
#  generate_answer
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateAnswer:
    """Tests for the generate_answer function."""

    def test_generate_answer_returns_answer_and_used_sources(
        self, mock_client, sample_chunks
    ):
        """generate_answer should return an (answer, used_sources) tuple."""
        mock_client.return_value.models.generate_content.return_value.text = (
            '{"answer": "FastAPI is a web framework.", "used_sources": [1]}'
        )
        answer, used = generate_answer("What is FastAPI?", sample_chunks)
        assert answer == "FastAPI is a web framework."
        assert used == [1]

    def test_generate_answer_returns_llm_response(self, mock_client, sample_chunks):
        """generate_answer should return exactly the answer the LLM responds with."""
        mock_client.return_value.models.generate_content.return_value.text = (
            '{"answer": "FastAPI is a web framework.", "used_sources": [1, 2]}'
        )
        answer, used = generate_answer("What is FastAPI?", sample_chunks)
        assert answer == "FastAPI is a web framework."
        assert used == [1, 2]

    def test_generate_answer_calls_api_once(self, mock_client, sample_chunks):
        """generate_answer should call the Gemini API exactly once."""
        mock_client.return_value.models.generate_content.return_value.text = (
            '{"answer": "Some answer.", "used_sources": []}'
        )
        generate_answer("What is FastAPI?", sample_chunks)
        mock_client.return_value.models.generate_content.assert_called_once()

    def test_generate_answer_raises_runtime_error_on_api_failure(
        self, mock_client, sample_chunks
    ):
        """generate_answer should raise RuntimeError if the API call fails."""
        mock_client.return_value.models.generate_content.side_effect = Exception(
            "API unavailable"
        )
        with pytest.raises(RuntimeError):
            generate_answer("What is FastAPI?", sample_chunks)

    def test_generate_answer_empty_chunks(self, mock_client):
        """generate_answer should report no used sources when there are none."""
        mock_client.return_value.models.generate_content.return_value.text = (
            '{"answer": "I don\'t know.", "used_sources": []}'
        )
        answer, used = generate_answer("What is FastAPI?", [])
        assert isinstance(answer, str)
        assert used == []
