import pytest

from app.models.schemas import SourceChunk
from app.services.llm import build_prompt


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
