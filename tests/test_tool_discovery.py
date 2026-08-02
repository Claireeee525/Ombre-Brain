"""Regression tests for official MCP tool discovery metadata."""

import json
from types import SimpleNamespace

import pytest

import server


EXPECTED_TOOL_ALIASES = {
    "breath": ("记忆检索", "recall", "memory"),
    "source_read": ("原文证据", "exact", "source"),
    "embedding_queue": ("向量重试", "embedding", "queue"),
    "hold": ("保存", "remember", "memory"),
    "curate": ("记忆整理", "organize", "memory"),
    "memory_review": ("候选记忆审核", "review", "memory"),
    "review_queue": ("待审候选", "review", "candidate"),
    "memory_stance": ("记忆表态", "annotate", "memory"),
    "grow": ("日记归档", "archive", "memory"),
    "trace": ("修改", "edit", "memory"),
    "somatic_read": ("身体状态读取", "read", "state"),
    "somatic_feel": ("情绪事件", "feel", "emotion"),
    "somatic_digest": ("情绪消化", "digest", "emotion"),
    "somatic_integrate": ("情绪余波合并", "integrate", "emotion"),
    "constellation": ("记忆星图", "graph", "memory"),
    "herbier": ("记忆藏页", "catalogue", "memory"),
    "inventory": ("只读盘点", "audit", "inventory"),
    "dupes": ("重复审核组", "duplicate", "review"),
    "pulse": ("系统状态", "status", "memories"),
    "dream": ("做梦", "reflect", "memory"),
    "handoff": ("短期状态", "handoff", "short-term"),
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
async def test_inventory_is_read_only_and_can_return_id_only_review_payload(monkeypatch):
    monkeypatch.setattr(
        server,
        "build_inventory",
        lambda buckets_dir, include_archive=True: {
            "read_only": True,
            "records": [{"id": "a1"}],
            "raw_transcript_records": [{"id": "a1"}],
            "source_unknown_records": [{"id": "a2"}],
            "low_confidence_records": [],
            "protected_records": [{"id": "a3"}],
        },
    )

    payload = json.loads(await server.inventory(include_records=False))

    assert payload == {
        "read_only": True,
        "records": ["a1"],
        "raw_transcript_records": ["a1"],
        "source_unknown_records": ["a2"],
        "low_confidence_records": [],
        "protected_records": ["a3"],
    }


@pytest.mark.asyncio
async def test_inventory_http_routes_are_authenticated_and_read_only(monkeypatch):
    from starlette.datastructures import QueryParams

    class Request:
        query_params = QueryParams("records=0&archive=0")
        cookies = {}
        headers = {}

    monkeypatch.setattr(server, "_require_auth", lambda request: None)
    monkeypatch.setattr(
        server,
        "build_inventory",
        lambda buckets_dir, include_archive=True: {
            "read_only": True,
            "records": [{"id": "a1"}],
            "raw_transcript_records": [],
            "source_unknown_records": [],
            "low_confidence_records": [],
            "protected_records": [],
            "duplicate_content_groups": [],
            "same_name_review_groups": [],
            "review_policy": {"physical_delete_performed": False},
        },
    )

    inventory_response = await server.api_inventory(Request())
    assert inventory_response.status_code == 200
    assert json.loads(inventory_response.body)["records"] == ["a1"]

    dupes_request = Request()
    dupes_request.query_params = QueryParams("limit=2")
    dupes_response = await server.api_dupes(dupes_request)
    payload = json.loads(dupes_response.body)
    assert dupes_response.status_code == 200
    assert payload["read_only"] is True
    assert payload["review_policy"]["physical_delete_performed"] is False


@pytest.mark.asyncio
async def test_backup_route_returns_verified_receipt(monkeypatch, tmp_path):
    class Request:
        cookies = {}
        headers = {}

        async def json(self):
            return {"include_archive": False, "label": "regression"}

    monkeypatch.setattr(server, "_require_auth", lambda request: None)
    monkeypatch.setattr(server, "create_backup", lambda *args, **kwargs: {
        "ok": True,
        "archive": str(tmp_path / "backup.tar.gz"),
        "file_count": 2,
    })
    monkeypatch.setattr(server, "verify_backup", lambda *args, **kwargs: {
        "ok": True,
        "restore_tested": True,
    })

    response = await server.api_backup(Request())
    payload = json.loads(response.body)
    assert response.status_code == 201
    assert payload["verification"]["restore_tested"] is True
    assert payload["ok"] is True


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
