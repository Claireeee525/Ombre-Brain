import json

import httpx
import pytest

import server


def _payload(response):
    return json.loads(response.text)


@pytest.mark.asyncio
async def test_remote_mcp_requests_do_not_depend_on_process_session_ids():
    app = server.mcp.streamable_http_app()
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {"accept": "application/json, text/event-stream"}
            initialized = await client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "stateless-regression", "version": "1"},
                },
            })
            listed = await client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            })

    assert initialized.status_code == 200
    assert "mcp-session-id" not in initialized.headers
    assert listed.status_code == 200
    tool_names = {tool["name"] for tool in _payload(listed)["result"]["tools"]}
    assert {"breath", "somatic_read", "hold", "memory_review"} <= tool_names
