from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.query import router

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create a minimal FastAPI app with only the query router mounted."""
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Return a TestClient for the query router."""
    return TestClient(app)


# ── Shared mock data ──────────────────────────────────────────────────────────

MOCK_QUERY_VECTOR = [0.1] * 384

MOCK_SEARCH_RESULTS = {
    "documents": [["Chunk one text.", "Chunk two text."]],
    "metadatas": [
        [
            {"filename": "doc_a.pdf", "chunk_index": 0},
            {"filename": "doc_b.pdf", "chunk_index": 3},
        ]
    ],
    "distances": [[0.1, 0.4]],
}

MOCK_ANSWER = "Based on the documents, the answer is X."


# ══════════════════════════════════════════════════════════════════════════════
#  POST /query
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryEndpoint:
    """Tests for the POST /query endpoint."""

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_valid_query_returns_200(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a valid question,
        When the query endpoint is called,
        Then a 200 response is returned with an answer and sources.
        """
        response = client.post("/query", json={"query": "What is machine learning?"})

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == MOCK_ANSWER
        assert body["query"] == "What is machine learning?"
        assert len(body["sources"]) == 2

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_response_contains_correct_source_fields(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a valid query,
        When the endpoint returns sources,
        Then each source contains text, file_name, chunk_index
        and similarity_score.
        """
        response = client.post("/query", json={"query": "What is deep learning?"})

        sources = response.json()["sources"]
        first = sources[0]

        assert first["text"] == "Chunk one text."
        assert first["file_name"] == "doc_a.pdf"
        assert first["chunk_index"] == 0
        assert "similarity_score" in first

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_similarity_score_is_calculated_correctly(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a distance of 0.1 returned by ChromaDB,
        When the endpoint processes the result,
        Then the similarity score is correctly converted to 1 - distance/2.
        Distance 0.1 → score 0.95, Distance 0.4 → score 0.8.
        """
        response = client.post("/query", json={"query": "Tell me about neural nets."})

        sources = response.json()["sources"]
        assert sources[0]["similarity_score"] == round(1 - 0.1 / 2, 4)  # 0.95
        assert sources[1]["similarity_score"] == round(1 - 0.4 / 2, 4)  # 0.8

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_services_called_in_correct_order(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a valid query,
        When the endpoint is called,
        Then embed_query, search, and generate_answer are each
        called exactly once in the correct order.
        """
        client.post("/query", json={"query": "What is RAG?"})

        mock_embed.assert_called_once_with("What is RAG?")
        mock_search.assert_called_once_with(MOCK_QUERY_VECTOR)
        mock_generate.assert_called_once()

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_query_is_echoed_in_response(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a query string,
        When the endpoint responds,
        Then the original query is included in the response body.
        """
        question = "How does ChromaDB store vectors?"
        response = client.post("/query", json={"query": question})

        assert response.json()["query"] == question

    def test_empty_query_returns_422(self, client):
        """
        Given an empty query string,
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = client.post("/query", json={"query": ""})

        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        """
        Given a request body with no query field,
        When the endpoint is called,
        Then a 422 Unprocessable Entity is returned.
        """
        response = client.post("/query", json={})

        assert response.status_code == 422

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch(
        "app.routers.query.search",
        return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_no_matching_chunks_returns_empty_sources(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a query with no matching documents in ChromaDB,
        When the endpoint is called,
        Then the response has an empty sources list.
        """
        response = client.post("/query", json={"query": "Completely unrelated topic."})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    @patch("app.routers.query.embed_query", side_effect=Exception("model not loaded"))
    def test_embedder_failure_returns_500(self, mock_embed, client):
        """
        Given the embedder raises an unexpected exception,
        When the query endpoint is called,
        Then a 500 Internal Server Error is returned.
        """
        response = client.post("/query", json={"query": "What is attention?"})

        assert response.status_code == 500

    @patch(
        "app.routers.query.generate_answer", side_effect=Exception("LLM unavailable")
    )
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_llm_failure_returns_500(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given the LLM raises an unexpected exception,
        When the query endpoint is called,
        Then a 500 Internal Server Error is returned.
        """
        response = client.post("/query", json={"query": "What is transfer learning?"})

        assert response.status_code == 500

    @patch("app.routers.query.generate_answer", return_value=MOCK_ANSWER)
    @patch("app.routers.query.search", return_value=MOCK_SEARCH_RESULTS)
    @patch("app.routers.query.embed_query", return_value=MOCK_QUERY_VECTOR)
    def test_generate_answer_receives_correct_arguments(
        self, mock_embed, mock_search, mock_generate, client
    ):
        """
        Given a valid query,
        When the endpoint calls generate_answer,
        Then it passes the original query string and the
        list of SourceChunk objects as arguments.
        """
        question = "What is a transformer model?"
        client.post("/query", json={"query": question})

        call_args = mock_generate.call_args
        assert call_args[0][0] == question  # first arg: query string
        assert isinstance(call_args[0][1], list)  # second arg: list of sources
        assert len(call_args[0][1]) == 2  # matches MOCK_SEARCH_RESULTS
