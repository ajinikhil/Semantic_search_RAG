"""
Tests for the root-level FastAPI endpoints defined in app/main.py:
  GET /       → {"status": "running"}
  GET /health → {"health": "ok"}

Uses httpx.AsyncClient (via the shared ``app_client`` fixture in conftest.py).
"""

import pytest

# ══════════════════════════════════════════════════════════════════════════════
#  GET /
# ══════════════════════════════════════════════════════════════════════════════


class TestRootEndpoint:
    """Tests for the GET / endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, app_client):
        """Root endpoint should respond with HTTP 200."""
        response = await app_client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_root_returns_status_running(self, app_client):
        """Root endpoint body should be exactly {"status": "running"}."""
        response = await app_client.get("/")
        assert response.json() == {"status": "running"}

    @pytest.mark.asyncio
    async def test_root_content_type_is_json(self, app_client):
        """Root endpoint should return JSON content."""
        response = await app_client.get("/")
        assert "application/json" in response.headers["content-type"]


# ══════════════════════════════════════════════════════════════════════════════
#  GET /health
# ══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, app_client):
        """Health endpoint should respond with HTTP 200."""
        response = await app_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app_client):
        """Health endpoint body should be exactly {"health": "ok"}."""
        response = await app_client.get("/health")
        assert response.json() == {"health": "ok"}

    @pytest.mark.asyncio
    async def test_health_content_type_is_json(self, app_client):
        """Health endpoint should return JSON content."""
        response = await app_client.get("/health")
        assert "application/json" in response.headers["content-type"]
