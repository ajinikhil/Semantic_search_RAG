from unittest.mock import patch

import pytest

from app.models.schemas import SourceChunk
from app.services.llm import build_prompt, generate_answer


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


def test_prompt_contains_question(sample_chunks):
    prompt = build_prompt("What is FastAPI?", sample_chunks)
    assert "What is FastAPI?" in prompt


def test_prompt_contains_chunk_text(sample_chunks):
    prompt = build_prompt("What is FastAPI?", sample_chunks)
    assert "FastAPI is a modern web framework." in prompt
    assert "ChromaDB is a vector database." in prompt


def test_prompt_contains_source_filenames(sample_chunks):
    prompt = build_prompt("What is FastAPI?", sample_chunks)
    assert "doc1.pdf" in prompt
    assert "doc2.pdf" in prompt


def test_prompt_contains_chunk_index(sample_chunks):
    prompt = build_prompt("What is FastAPI?", sample_chunks)
    assert "chunk 0" in prompt
    assert "chunk 1" in prompt


def test_prompt_returns_string(sample_chunks):
    prompt = build_prompt("What is FastAPI?", sample_chunks)
    assert isinstance(prompt, str)


def test_prompt_single_chunk(single_chunk):
    prompt = build_prompt("What is Python?", single_chunk)
    assert "Python is a programming language." in prompt


def test_prompt_empty_chunks():
    prompt = build_prompt("What is FastAPI?", [])
    assert isinstance(prompt, str)


def test_prompt_empty_question(sample_chunks):
    prompt = build_prompt("", sample_chunks)
    assert isinstance(prompt, str)


def test_prompt_raises_runtime_error_on_invalid_chunks():
    with pytest.raises(RuntimeError):
        build_prompt("What is FastAPI?", ["invalid", "chunks"])


@patch("app.services.llm.client")
def test_generate_answer_returns_string(mock_client, sample_chunks):
    mock_client.models.generate_content.return_value.text = (
        "FastAPI is a web framework."
    )
    answer = generate_answer("What is FastAPI?", sample_chunks)
    assert isinstance(answer, str)


@patch("app.services.llm.client")
def test_generate_answer_returns_llm_response(mock_client, sample_chunks):
    mock_client.models.generate_content.return_value.text = (
        "FastAPI is a web framework."
    )
    answer = generate_answer("What is FastAPI?", sample_chunks)
    assert answer == "FastAPI is a web framework."


@patch("app.services.llm.client")
def test_generate_answer_calls_api_once(mock_client, sample_chunks):
    mock_client.models.generate_content.return_value.text = "Some answer."
    generate_answer("What is FastAPI?", sample_chunks)
    mock_client.models.generate_content.assert_called_once()


@patch("app.services.llm.client")
def test_generate_answer_raises_runtime_error_on_api_failure(
    mock_client, sample_chunks
):
    mock_client.models.generate_content.side_effect = Exception("API unavailable")
    with pytest.raises(RuntimeError):
        generate_answer("What is FastAPI?", sample_chunks)


@patch("app.services.llm.client")
def test_generate_answer_empty_chunks(mock_client):
    mock_client.models.generate_content.return_value.text = "I don't know."
    answer = generate_answer("What is FastAPI?", [])
    assert isinstance(answer, str)
