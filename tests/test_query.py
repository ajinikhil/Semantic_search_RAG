"""
Tests for the query endpoint in app/routers/query.py:
  POST /query → answer a question using RAG

Uses httpx.AsyncClient (async tests) with mocked embed_query, ChromaDB search,
and Gemini generate_answer so no real API keys or models are needed.
"""

from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.query import router

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client():
    """AsyncClient wired to a minimal app containing only the query router."""
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Shared mock data ──────────────────────────────────────────────────────────

_VECTOR = [0.1] * 384

_SEARCH_RESULTS = {
    "documents": [["Chunk one text.", "Chunk two text."]],
    "metadatas": [
        [
            {"filename": "doc_a.pdf", "chunk_index": 0},
            {"filename": "doc_b.pdf", "chunk_index": 3},
        ]
    ],
    "distances": [[0.1, 0.4]],
}

_ANSWER = "Based on the documents, the answer is X."

_EMPTY_SEARCH = {"documents": [[]], "metadatas": [[]], "distances": [[]]}


# ══════════════════════════════════════════════════════════════════════════════
#  POST /query — happy paths
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryEndpointHappyPath:
    """Tests for successful POST /query calls."""

    @pytest.mark.asyncio
    async def test_valid_query_returns_200(self, client):
        """
        Given a valid question,
        When the query endpoint is called,
        Then a 200 response is returned with an answer and sources.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "What is machine learning?"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == _ANSWER
        assert body["query"] == "What is machine learning?"
        assert len(body["sources"]) == 2

    @pytest.mark.asyncio
    async def test_response_contains_correct_source_fields(self, client):
        """
        Given a valid query,
        When the endpoint returns sources,
        Then each source contains text, file_name, chunk_index and similarity_score.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "What is deep learning?"}
            )

        first = response.json()["sources"][0]
        assert first["text"] == "Chunk one text."
        assert first["file_name"] == "doc_a.pdf"
        assert first["chunk_index"] == 0
        assert "similarity_score" in first

    @pytest.mark.asyncio
    async def test_similarity_score_calculated_correctly(self, client):
        """
        Given distances [0.1, 0.4] from ChromaDB,
        When the endpoint processes the result,
        Then similarity scores are 1 - distance/2:
          0.1 → 0.95, 0.4 → 0.8.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "Tell me about neural nets."}
            )

        sources = response.json()["sources"]
        assert sources[0]["similarity_score"] == round(1 - 0.1 / 2, 4)
        assert sources[1]["similarity_score"] == round(1 - 0.4 / 2, 4)

    @pytest.mark.asyncio
    async def test_query_is_echoed_in_response(self, client):
        """
        Given a query string,
        When the endpoint responds,
        Then the original query is included verbatim in the response body.
        """
        question = "How does ChromaDB store vectors?"
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post("/query", json={"query": question})

        assert response.json()["query"] == question

    @pytest.mark.asyncio
    async def test_services_called_in_correct_order(self, client):
        """
        Given a valid query,
        When the endpoint is called,
        Then embed_query, search, and generate_answer are each called exactly
        once in the correct order.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR) as mock_embed,
            patch(
                "app.routers.query.search", return_value=_SEARCH_RESULTS
            ) as mock_search,
            patch(
                "app.routers.query.generate_answer", return_value=_ANSWER
            ) as mock_gen,
        ):
            await client.post("/query", json={"query": "What is RAG?"})

        mock_embed.assert_called_once_with("What is RAG?")
        mock_search.assert_called_once_with(_VECTOR)
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_receives_correct_arguments(self, client):
        """
        Given a valid query,
        When the endpoint calls generate_answer,
        Then it passes the original query string and the list of SourceChunk
        objects as arguments.
        """
        question = "What is a transformer model?"
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch(
                "app.routers.query.generate_answer", return_value=_ANSWER
            ) as mock_gen,
        ):
            await client.post("/query", json={"query": question})

        call_args = mock_gen.call_args
        assert call_args[0][0] == question
        assert isinstance(call_args[0][1], list)
        assert len(call_args[0][1]) == 2

    @pytest.mark.asyncio
    async def test_no_matching_chunks_returns_empty_sources(self, client):
        """
        Given ChromaDB returns no matching documents,
        When the endpoint is called,
        Then the response has an empty sources list and still returns 200.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_EMPTY_SEARCH),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "Completely unrelated topic."}
            )

        assert response.status_code == 200
        assert response.json()["sources"] == []

    @pytest.mark.asyncio
    async def test_query_at_max_length_returns_200(self, client):
        """
        Given a query string at the maximum allowed length (2000 chars),
        When the endpoint is called,
        Then the request is accepted and returns 200.
        """
        long_query = "a" * 2000
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post("/query", json={"query": long_query})

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_top_k_chunks_default_is_accepted(self, client):
        """
        Given a request body with no top_k_chunks field,
        When the endpoint is called,
        Then the default of 5 is used and the request succeeds.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post("/query", json={"query": "What is Python?"})

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_top_k_chunks_within_bounds_accepted(self, client):
        """
        Given top_k_chunks is set to a valid value (e.g. 10),
        When the endpoint is called,
        Then the request is accepted.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "What is Python?", "top_k_chunks": 10}
            )

        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  POST /query — validation errors (422)
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryValidation:
    """Tests for request body validation on POST /query."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_422(self, client):
        """
        Given an empty query string (violates min_length=1),
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = await client.post("/query", json={"query": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_query_field_returns_422(self, client):
        """
        Given a request body with no query field,
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = await client.post("/query", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_exceeding_max_length_returns_422(self, client):
        """
        Given a query string exceeding the maximum allowed length (2001 chars),
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        too_long = "a" * 2001
        response = await client.post("/query", json={"query": too_long})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_chunks_zero_returns_422(self, client):
        """
        Given top_k_chunks=0 (violates ge=1),
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = await client.post(
            "/query", json={"query": "test", "top_k_chunks": 0}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_chunks_above_max_returns_422(self, client):
        """
        Given top_k_chunks=21 (violates le=20),
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = await client.post(
            "/query", json={"query": "test", "top_k_chunks": 21}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_chunks_boundary_min_accepted(self, client):
        """
        Given top_k_chunks=1 (the minimum valid value),
        When the endpoint is called,
        Then the request is accepted (200).
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "test", "top_k_chunks": 1}
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_top_k_chunks_boundary_max_accepted(self, client):
        """
        Given top_k_chunks=20 (the maximum valid value),
        When the endpoint is called,
        Then the request is accepted (200).
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch("app.routers.query.generate_answer", return_value=_ANSWER),
        ):
            response = await client.post(
                "/query", json={"query": "test", "top_k_chunks": 20}
            )

        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  POST /query — service failures (500)
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryServiceFailures:
    """Tests covering 500 responses when downstream services raise exceptions."""

    @pytest.mark.asyncio
    async def test_embedder_failure_returns_500(self, client):
        """
        Given the embedder raises an unexpected exception,
        When the query endpoint is called,
        Then a 500 Internal Server Error is returned.
        """
        with patch(
            "app.routers.query.embed_query", side_effect=Exception("model not loaded")
        ):
            response = await client.post("/query", json={"query": "What is attention?"})

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_search_failure_returns_500(self, client):
        """
        Given ChromaDB raises an exception during the similarity search,
        When the query endpoint is called,
        Then a 500 Internal Server Error is returned.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch(
                "app.routers.query.search", side_effect=RuntimeError("connection lost")
            ),
        ):
            response = await client.post("/query", json={"query": "What is RAG?"})

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_llm_failure_returns_500(self, client):
        """
        Given the LLM raises an unexpected exception,
        When the query endpoint is called,
        Then a 500 Internal Server Error is returned.
        """
        with (
            patch("app.routers.query.embed_query", return_value=_VECTOR),
            patch("app.routers.query.search", return_value=_SEARCH_RESULTS),
            patch(
                "app.routers.query.generate_answer",
                side_effect=Exception("LLM unavailable"),
            ),
        ):
            response = await client.post(
                "/query", json={"query": "What is transfer learning?"}
            )

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_500_response_contains_error_detail(self, client):
        """
        Given the embedder raises an exception with a specific message,
        When the query endpoint is called,
        Then the 500 response body contains the error detail.
        """
        with patch(
            "app.routers.query.embed_query", side_effect=Exception("out of memory")
        ):
            response = await client.post("/query", json={"query": "test question"})

        assert response.status_code == 500
        assert response.json().get("detail") is not None
