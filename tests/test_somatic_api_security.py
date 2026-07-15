# -*- coding: utf-8 -*-
"""Security boundary tests for the full and redacted somatic HTTP views."""
import json
from types import SimpleNamespace

import pytest

import server
import somatic_state as S


def _request(headers=None, cookies=None):
    return SimpleNamespace(headers=headers or {}, cookies=cookies or {})


@pytest.mark.asyncio
async def test_full_somatic_api_requires_dashboard_session(monkeypatch):
    def must_not_read():
        raise AssertionError("unauthenticated request reached private state")

    monkeypatch.setattr(server.somatic_state, "read_state", must_not_read)
    response = await server.somatic_api(_request())

    assert response.status_code == 401


def test_home_summary_auth_accepts_only_configured_constant_time_token(monkeypatch):
    monkeypatch.setattr(server, "OMBRE_HOME_READ_TOKEN", "home-private-token")

    assert server._require_home_read_auth(
        _request(headers={"authorization": "Bearer home-private-token"})
    ) is None
    assert server._require_home_read_auth(
        _request(headers={"x-ombre-home-token": "home-private-token"})
    ) is None
    assert server._require_home_read_auth(
        _request(headers={"authorization": "Bearer wrong"})
    ).status_code == 401
    assert server._require_home_read_auth(_request()).status_code == 401


@pytest.mark.asyncio
async def test_home_summary_route_returns_only_redacted_shape(monkeypatch):
    monkeypatch.setattr(server, "OMBRE_HOME_READ_TOKEN", "home-private-token")
    state = S.fresh_state(1_700_000_000_000)
    state["events"] = [{"label": "PRIVATE EVENT"}]
    state["thoughts"] = [{
        "id": "private",
        "text": "PRIVATE THOUGHT",
        "drive": "attachment",
        "kind": "flit",
        "strength": 50,
        "peakStrength": 50,
        "fedCount": 0,
    }]
    monkeypatch.setattr(server.somatic_state, "read_state", lambda: state)
    monkeypatch.setattr(server.somatic_state, "live", lambda stored: (stored, False))

    response = await server.somatic_summary_api(
        _request(headers={"authorization": "Bearer home-private-token"})
    )
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "PRIVATE" not in body
    summary = json.loads(body)["summary"]
    assert not ({"events", "thoughts", "echoes"} & set(summary))


@pytest.mark.asyncio
async def test_official_somatic_read_tool_keeps_full_mcp_path_working(monkeypatch):
    state = S.fresh_state(1_700_000_000_000)
    monkeypatch.setattr(server.somatic_state, "read_state", lambda: state)
    monkeypatch.setattr(server.somatic_state, "live", lambda stored: (stored, False))

    block = await server.somatic_read()

    assert block.startswith("[Kelo Somatic Field]")
    assert block.endswith("[/Kelo Somatic Field]")
    assert "当前倾向" in block


@pytest.mark.asyncio
async def test_legacy_echo_recovery_is_not_a_remote_mcp_tool():
    names = {tool.name for tool in await server.mcp.list_tools()}

    assert "somatic_recover_echoes" not in names
    assert {"somatic_read", "somatic_feel", "somatic_digest"} <= names


@pytest.mark.asyncio
async def test_somatic_feel_never_persists_freeform_note(monkeypatch):
    state = S.fresh_state(1_700_000_000_000)
    captured = {}
    monkeypatch.setattr(server.somatic_state, "read_state", lambda: state)
    monkeypatch.setattr(server.somatic_state, "write_state", lambda value: captured.setdefault("state", value))

    await server.somatic_feel("affection", "PRIVATE NOTE: a long raw Claude sentence")
    encoded = json.dumps(captured["state"], ensure_ascii=False)

    assert "PRIVATE NOTE" not in encoded
    assert "long raw Claude sentence" not in encoded
    assert captured["state"]["thoughts"][0]["text"] == "我想把她给我的亲昵留在心里"


@pytest.mark.asyncio
async def test_unlabelled_digest_warns_and_only_refreshes_contact(monkeypatch):
    state = S.fresh_state(1_700_000_000_000)
    captured = {}
    monkeypatch.setattr(server.somatic_state, "read_state", lambda: state)
    monkeypatch.setattr(server.somatic_state, "write_state", lambda value: captured.setdefault("state", value))

    result = await server.somatic_digest("我回复她：我爱你，好想你。")

    assert "未检测到" in result
    assert "没有写入事件或 Thought Pool" in result
    assert captured["state"]["events"] == []
    assert captured["state"]["thoughts"] == []
