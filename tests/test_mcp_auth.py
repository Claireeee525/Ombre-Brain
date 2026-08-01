from unittest.mock import AsyncMock

import pytest

from server import McpBearerAuthMiddleware


async def _call(path="/mcp", method="POST", headers=None):
    inner = AsyncMock()
    middleware = McpBearerAuthMiddleware(inner, "private-token")
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    encoded = [
        (str(key).encode("latin-1"), str(value).encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    await middleware(
        {"type": "http", "path": path, "method": method, "headers": encoded},
        receive,
        send,
    )
    return inner, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer wrong"}])
async def test_remote_mcp_rejects_missing_or_wrong_token(headers):
    inner, sent = await _call(headers=headers)

    inner.assert_not_awaited()
    assert sent[0]["status"] == 401
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]


@pytest.mark.asyncio
async def test_remote_mcp_accepts_correct_bearer_token():
    inner, sent = await _call(headers={"authorization": "Bearer private-token"})

    inner.assert_awaited_once()
    assert sent == []


@pytest.mark.asyncio
async def test_health_and_preflight_remain_public():
    health, _ = await _call(path="/health", method="GET")
    options, _ = await _call(path="/mcp", method="OPTIONS")

    health.assert_awaited_once()
    options.assert_awaited_once()
