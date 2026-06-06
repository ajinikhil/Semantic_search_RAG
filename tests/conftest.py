"""
Shared pytest fixtures for the Semantic Search RAG test suite.

Async fixtures use ``pytest_asyncio.fixture`` (strict mode — the default).
Every async test must carry ``@pytest.mark.asyncio``.
"""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def app_client():
    """AsyncClient wired to the full FastAPI application (both routers)."""
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
