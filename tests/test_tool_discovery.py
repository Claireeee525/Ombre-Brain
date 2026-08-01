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
    "memory_stance": ("记忆表态", "annotate", "memory"),
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
    assert herbier_schema["properties"]["include_rejected"]["default"] is False
    breath_schema = tools["breath"].inputSchema
    assert breath_schema["properties"]["response_format"]["default"] == "text"


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


def test_herbier_full_catalogue_lenses_use_bucket_metadata():
    assert server._herbier_memory_kind({"metadata": {"pinned": True}}) == "lasting"
    assert server._herbier_memory_kind({"metadata": {"domain": ["梦境"]}}) == "dream"
    assert server._herbier_memory_kind({"metadata": {"tags": ["身体状态"]}}) == "state"
    assert server._herbier_memory_kind({"metadata": {"domain": ["回忆"]}}) == "event"


@pytest.mark.asyncio
async def test_breath_packet_preserves_provenance_and_render_rules(monkeypatch):
    monkeypatch.setattr(server.dehydrator, "dehydrate", lambda content, meta: _async_value("相关记忆摘要"))
    bucket = {
        "id": "bucket-1",
        "content": "Claire 当时说过的原话。",
        "metadata": {
            "name": "那次谈话",
            "evidence_speakers": ["Claire"],
            "signed_by": ["珂洛"],
            "curated_by": "Calder",
            "source_session_id": "session-1",
            "source_message_ids": ["message-1"],
            "valid_from": "2026-07-27T12:00:00+08:00",
        },
    }

    direct = await server._breath_packet_item(bucket, "direct")
    related = await server._breath_packet_item(bucket, "related")

    assert direct == {
        "bucket_id": "bucket-1",
        "title": "那次谈话",
        "summary": "Claire 当时说过的原话。",
        "source_actor": "Claire",
        "recorded_by": "Calder",
        "source_ref": "message:message-1",
        "conversation_id": "session-1",
        "event_date": "2026-07-27T12:00:00+08:00",
        "match_kind": "direct",
        "render_kind": "original",
        "why_recalled": "关键词直接命中",
    }
    assert related["summary"] == "相关记忆摘要"
    assert related["render_kind"] == "summary"
    assert related["match_kind"] == "related"


async def _async_value(value):
    return value
