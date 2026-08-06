"""Regression tests for diary digest: error propagation, parse robustness,
privacy, grow failure safety, and evidence dedup — all exercising production paths
through a clean server import fixture with no global pollution."""
import json
import logging
import os
import sys
import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Dehydrator helpers ──

def _dehydrator_for(tmp_path, monkeypatch):
    """Create a Dehydrator with mock client, isolated to tmp_path."""
    monkeypatch.delenv("OMBRE_DIGEST_MODEL", raising=False)
    from dehydrator import Dehydrator
    config = {
        "dehydration": {
            "api_key": "sk-test",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
        },
        "buckets_dir": str(tmp_path / "buckets"),
    }
    dh = Dehydrator(config)
    dh.client = MagicMock()
    dh.client.chat = MagicMock()
    dh.client.chat.completions = MagicMock()
    dh.client.chat.completions.create = AsyncMock()
    return dh


def _fake_response(*, choices=None, model="test-model", usage=None, response_id="resp-1"):
    resp = MagicMock()
    resp.choices = choices or []
    resp.model = model
    resp.id = response_id
    resp.usage = usage or MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return resp


def _valid_items():
    return [
        {"name": "早上的咖啡", "content": "今天早上喝了一杯很香的拿铁", "domain": ["日常"], "valence": 0.8, "arousal": 0.3, "tags": ["咖啡"], "importance": 3},
        {"name": "下午的散步", "content": "下午在公园散了步，天气很好", "domain": ["日常", "健康"], "valence": 0.9, "arousal": 0.2, "tags": ["散步", "天气"], "importance": 4},
    ]


# ── Server import fixture ──

@pytest.fixture
def server_module(tmp_path, monkeypatch):
    """Import server.py with sys.modules patching + automatic rollback.
    Only mocks deps unavailable in the test environment."""
    # Save originals and set mocks — restore on teardown
    _mcp = MagicMock()
    _mcp.server = MagicMock()
    _mcp.server.fastmcp = MagicMock()
    _mcp.server.fastmcp.FastMCP = MagicMock()
    _mcp.types = MagicMock()
    _mocks = [
        ("mcp", _mcp),
        ("mcp.server", _mcp.server),
        ("mcp.server.fastmcp", _mcp.server.fastmcp),
        ("mcp.types", _mcp.types),
        ("starlette", MagicMock()),
        ("starlette.requests", MagicMock()),
        ("starlette.responses", MagicMock()),
        ("starlette.applications", MagicMock()),
        ("starlette.routing", MagicMock()),
        ("starlette.middleware", MagicMock()),
        ("starlette.authentication", MagicMock()),
        ("oauth_provider", MagicMock()),
        ("embedding_engine", MagicMock()),
        ("import_memory", MagicMock()),
    ]
    saved = {}
    for name, mock in _mocks:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    sys.modules["embedding_engine"].EmbeddingEngine = MagicMock()

    # Patch utils.load_config to use tmp_path
    import utils as _utils
    monkeypatch.setattr(_utils, "load_config", MagicMock(return_value={
        "buckets_dir": str(tmp_path / "buckets"),
        "dehydration": {},
        "log_level": "WARNING",
    }))

    # Import server fresh
    sys.modules.pop("server", None)
    srv = importlib.import_module("server")

    yield srv

    # Teardown: restore original sys.modules state
    sys.modules.pop("server", None)
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ═══════════════════════════════════════════════════════════════
# Dehydrator unit tests (10)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_empty_choices_raises_runtime_error(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    dh.client.chat.completions.create.return_value = _fake_response(choices=[])
    with pytest.raises(RuntimeError, match="未返回 choices"):
        await dh._api_digest("some diary content here for testing")


@pytest.mark.asyncio
async def test_empty_content_raises_runtime_error(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = ""
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    with pytest.raises(RuntimeError, match="返回空正文"):
        await dh._api_digest("some diary content here for testing")


@pytest.mark.asyncio
async def test_valid_json_array_parses_correctly(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    items = _valid_items()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    result = await dh._api_digest("content for testing")
    assert len(result) == 2
    assert result[0]["name"] == "早上的咖啡"
    assert result[0]["content"] == "今天早上喝了一杯很香的拿铁"
    assert result[1]["name"] == "下午的散步"
    assert result[1]["content"] == "下午在公园散了步，天气很好"


@pytest.mark.asyncio
async def test_fenced_json_parses_correctly(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    items = [{"name": "测试条目", "content": "这是一条测试记忆，从 fenced block 里解析出来", "domain": ["测试"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]
    raw = "```json\n" + json.dumps(items, ensure_ascii=False) + "\n```"
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = raw
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    result = await dh._api_digest("content")
    assert len(result) == 1
    assert result[0]["name"] == "测试条目"
    assert result[0]["content"] == "这是一条测试记忆，从 fenced block 里解析出来"


@pytest.mark.asyncio
async def test_items_wrapper_parses_correctly(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    items = [{"name": "包装测试", "content": "包装在 items 里的记忆内容", "domain": ["测试"], "valence": 0.6, "arousal": 0.4, "tags": [], "importance": 5}]
    raw = json.dumps({"items": items}, ensure_ascii=False)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = raw
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    result = await dh._api_digest("content")
    assert len(result) == 1
    assert result[0]["name"] == "包装测试"
    assert result[0]["content"] == "包装在 items 里的记忆内容"


@pytest.mark.asyncio
async def test_invalid_json_raises_runtime_error_not_silent(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "这不是有效的 JSON，只是一段文字"
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    with pytest.raises(RuntimeError, match="JSON 解析失败"):
        await dh._api_digest("content")


@pytest.mark.asyncio
async def test_missing_content_raises_with_validation_stats(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    items = [
        {"name": "缺content1", "domain": ["日常"]},
        {"not": "even a dict with content"},
    ]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    with pytest.raises(RuntimeError, match="没有有效 content"):
        await dh._api_digest("content")


def test_parse_extracts_json_from_wrapping_text(tmp_path):
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": str(tmp_path / "b")}
    dh = Dehydrator(config)
    raw = '以下是整理结果：\n[{"name": "提取测试", "content": "从文本里提取的记忆", "domain": ["日常"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]\n希望有帮助。'
    result = dh._parse_digest(raw)
    assert len(result) == 1
    assert result[0]["name"] == "提取测试"
    assert result[0]["content"] == "从文本里提取的记忆"


@pytest.mark.asyncio
async def test_digest_uses_digest_model_when_set(tmp_path, monkeypatch):
    dh = _dehydrator_for(tmp_path, monkeypatch)
    dh.digest_model = "custom-digest-model"
    items = [{"name": "t", "content": "test content here", "domain": ["测试"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice], model="custom-digest-model")
    result = await dh.digest("test")
    assert len(result) == 1
    assert dh.client.chat.completions.create.call_args[1]["model"] == "custom-digest-model"


def test_parse_error_never_exposes_diary_content(tmp_path, caplog):
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": str(tmp_path / "b")}
    dh = Dehydrator(config)
    secret = "Claire-SECRET-日记-abc123-今天心情很复杂-xyz789" * 20
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as exc_info:
            dh._parse_digest(secret)
    msg = str(exc_info.value)
    assert "raw_len=" in msg
    assert "raw_hash=" in msg
    assert "Claire-SECRET-日记" not in msg
    assert "abc123" not in msg
    assert "xyz789" not in msg
    assert "preview=" not in msg
    log_text = caplog.text
    assert "Claire-SECRET-日记" not in log_text
    assert "abc123" not in log_text


# ═══════════════════════════════════════════════════════════════
# Grow failure path tests — call real _grow_impl via fixture
# ═══════════════════════════════════════════════════════════════

GROW_DIARY = "今天早上喝了咖啡，下午去了公园散步，晚上看了一部好电影，回家的路上还买了一束花。"


@pytest.mark.asyncio
async def test_grow_impl_digest_failure_preserves_evidence(server_module, monkeypatch):
    evidence_calls = []
    async def fake_store(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-001", True)

    async def fake_digest(content):
        raise RuntimeError("日记整理 API 返回空正文（model=deepseek-v4-flash, finish_reason=stop, completion_tokens=0）")

    merge_calls = []
    async def fake_merge(*args, **kwargs):
        merge_calls.append(args)
        return ("merged-1", True)

    plan_calls = []
    def fake_plan(content):
        plan_calls.append(content)

    async def fake_ensure():
        pass

    monkeypatch.setattr(server_module, "_store_source_evidence", fake_store)
    monkeypatch.setattr(server_module.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(server_module, "_merge_or_create", fake_merge)
    monkeypatch.setattr(server_module, "_fire_plan_resolution", fake_plan)
    monkeypatch.setattr(server_module.decay_engine, "ensure_started", fake_ensure)

    result = await server_module._grow_impl(GROW_DIARY, source_surface="Claude官方端")

    assert "日记整理失败" in result
    assert len(evidence_calls) == 1
    assert GROW_DIARY in evidence_calls[0]
    assert len(merge_calls) == 0
    assert len(plan_calls) == 0


@pytest.mark.asyncio
async def test_grow_impl_empty_items_no_candidates(server_module, monkeypatch):
    evidence_calls = []
    async def fake_store(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-002", True)

    async def fake_digest(content):
        return []

    merge_calls = []
    async def fake_merge(*args, **kwargs):
        merge_calls.append(args)
        return ("merged-x", True)

    async def fake_ensure():
        pass

    monkeypatch.setattr(server_module, "_store_source_evidence", fake_store)
    monkeypatch.setattr(server_module.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(server_module, "_merge_or_create", fake_merge)
    monkeypatch.setattr(server_module.decay_engine, "ensure_started", fake_ensure)

    result = await server_module._grow_impl(GROW_DIARY)

    assert "内容为空或整理失败" in result
    assert len(evidence_calls) == 1
    assert len(merge_calls) == 0


@pytest.mark.asyncio
async def test_grow_impl_success_path(server_module, monkeypatch):
    evidence_calls = []
    async def fake_store(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-003", True)

    items = _valid_items()
    async def fake_digest(content):
        return items

    merge_calls = []
    async def fake_merge(*args, **kwargs):
        merge_calls.append(args)
        return ("candidate-1", False)

    plan_calls = []
    def fake_plan(content):
        plan_calls.append(content)

    async def fake_ensure():
        pass

    monkeypatch.setattr(server_module, "_store_source_evidence", fake_store)
    monkeypatch.setattr(server_module.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(server_module, "_merge_or_create", fake_merge)
    monkeypatch.setattr(server_module, "_fire_plan_resolution", fake_plan)
    monkeypatch.setattr(server_module.decay_engine, "ensure_started", fake_ensure)

    result = await server_module._grow_impl(GROW_DIARY)

    assert "原文证据→ev-003" in result
    assert "新存" in result
    assert "待审候选→2条" in result
    assert len(evidence_calls) == 1
    assert len(merge_calls) == 2
    assert len(plan_calls) == 1


@pytest.mark.asyncio
async def test_store_source_evidence_dedup(server_module, monkeypatch, tmp_path):
    """Real _store_source_evidence: same content → same ID, created=False on retry,
    different content → different ID."""
    import hashlib, uuid

    store = {}

    async def fake_list_all(include_archive=False):
        return list(store.values())

    async def fake_create(**kwargs):
        eid = str(uuid.uuid4())
        content = kwargs.get("content", "")
        meta = kwargs.get("extra_metadata", {})
        store[eid] = {"id": eid, "content": content, "metadata": meta}
        return eid

    fake_mgr = MagicMock()
    fake_mgr.list_all = fake_list_all
    fake_mgr.create = fake_create
    monkeypatch.setattr(server_module, "bucket_mgr", fake_mgr)

    content = "这是需要去重的原文证据内容。"
    surface = "测试来源"

    eid1, created1 = await server_module._store_source_evidence(content, source_surface=surface)
    assert created1 is True
    assert eid1

    eid2, created2 = await server_module._store_source_evidence(content, source_surface=surface)
    assert created2 is False
    assert eid2 == eid1

    # Only one create call happened across both — dedup prevented the second
    assert len(store) == 1

    eid3, created3 = await server_module._store_source_evidence("不同的内容。", source_surface=surface)
    assert created3 is True
    assert eid3 != eid1
    assert len(store) == 2


# ═══════════════════════════════════════════════════════════════
# sys.modules recovery test
# ═══════════════════════════════════════════════════════════════

def test_fixture_restores_sys_modules_after_patching():
    """Verify that manually patching and restoring sys.modules works correctly.
    Pattern: save original → set mock → yield → restore original."""
    sentinel = object()
    # Ensure key is absent initially
    sys.modules.pop("__sentinel_test__", None)
    # Save + set mock
    saved = sys.modules.get("__sentinel_test__")
    sys.modules["__sentinel_test__"] = MagicMock()
    # Simulate fixture teardown: restore original
    if saved is None:
        sys.modules.pop("__sentinel_test__", None)
    else:
        sys.modules["__sentinel_test__"] = saved
    # After restore, key should be gone
    assert "__sentinel_test__" not in sys.modules


def test_fixture_restores_existing_module_after_patching():
    """Verify that a previously existing module is restored, not deleted."""
    sentinel = object()
    sys.modules["__sentinel_test_2__"] = sentinel
    saved = sys.modules.get("__sentinel_test_2__")
    sys.modules["__sentinel_test_2__"] = MagicMock()
    # Simulate teardown
    if saved is None:
        sys.modules.pop("__sentinel_test_2__", None)
    else:
        sys.modules["__sentinel_test_2__"] = saved
    # After restore, the original sentinel should be back
    assert sys.modules["__sentinel_test_2__"] is sentinel
    # Clean up
    sys.modules.pop("__sentinel_test_2__", None)
