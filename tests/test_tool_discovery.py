"""Regression tests for official MCP tool discovery metadata."""

import json
from types import SimpleNamespace

import pytest

import server


EXPECTED_TOOL_ALIASES = {
    "breath": ("记忆检索", "recall", "memory"),
    "hold": ("保存", "remember", "memory"),
    "curate": ("记忆整理", "organize", "memory"),
    "memory_review": ("候选记忆审核", "review", "memory"),
    "grow": ("日记归档", "archive", "memory"),
    "trace": ("修改", "edit", "memory"),
    "somatic_read": ("身体状态读取", "read", "state"),
    "somatic_feel": ("情绪事件", "feel", "emotion"),
    "somatic_digest": ("情绪消化", "digest", "emotion"),
    "somatic_integrate": ("情绪余波合并", "integrate", "emotion"),
    "constellation": ("记忆星图", "graph", "memory"),
    "herbier": ("记忆藏页", "catalogue", "memory"),
    "pulse": ("系统状态", "status", "memories"),
    "dream": ("做梦", "reflect", "memory"),
}


@pytest.mark.asyncio
async def test_every_official_tool_description_starts_with_its_name_and_aliases():
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}

    assert set(tools) == set(EXPECTED_TOOL_ALIASES)
    for name, aliases in EXPECTED_TOOL_ALIASES.items():
        description = tools[name].description
        assert description.lower().startswith(name.lower())
        for alias in aliases:
            assert alias.lower() in description.lower()
    herbier_schema = tools["herbier"].inputSchema
    assert "offset" in herbier_schema["properties"]
    assert herbier_schema["properties"]["offset"]["default"] == 0
    assert herbier_schema["properties"]["limit"]["default"] == 100


@pytest.mark.asyncio
async def test_public_health_exposes_the_deployed_version(monkeypatch):
    monkeypatch.setattr(
        server.bucket_mgr,
        "get_stats",
        lambda: _async_value({"permanent_count": 2, "dynamic_count": 3}),
    )

    response = await server.health_check(SimpleNamespace())
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["version"] == server.OMBRE_VERSION
    assert payload["buckets"] == 5


async def _async_value(value):
    return value
