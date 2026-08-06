"""Regression tests for diary digest: error propagation, parse robustness,
privacy, grow failure safety, and evidence dedup — all exercising production paths."""
import json
import logging
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock heavy server dependencies so `import server` works in tests ──
_mcp = MagicMock()
_mcp.server = MagicMock()
_mcp.server.fastmcp = MagicMock()
_mcp.server.fastmcp.FastMCP = MagicMock()
_mcp.types = MagicMock()
sys.modules["mcp"] = _mcp
sys.modules["mcp.server"] = _mcp.server
sys.modules["mcp.server.fastmcp"] = _mcp.server.fastmcp
sys.modules["mcp.types"] = _mcp.types

_NEED_MOCK = [
    "decay_engine", "embedding_engine",
    "family_engine", "nudge_engine", "curator", "memory_layers", "inventory",
    "import_memory", "write_memory", "backup", "recall_cooldown",
    "reclassify_api", "reclassify_domains", "oauth_provider",
    "starlette", "starlette.requests", "starlette.responses",
    "starlette.applications", "starlette.routing", "starlette.middleware",
    "starlette.authentication",
    "yaml", "yaml.cyaml", "frontmatter", "bucket_manager",
]
for _mod in _NEED_MOCK:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["bucket_manager"].BucketManager = MagicMock()

# utils needs real return values
_utils_mock = MagicMock()
_utils_mock.load_config = MagicMock(return_value={"buckets_dir": "/tmp/test-buckets", "dehydration": {}})
_utils_mock.setup_logging = MagicMock()
_utils_mock.strip_wikilinks = MagicMock(return_value="")
_utils_mock.count_tokens_approx = MagicMock(return_value=100)
_utils_mock.now_iso = MagicMock(return_value="2026-08-06T12:00:00Z")
sys.modules["utils"] = _utils_mock

# Fallback for any other missing module
import builtins
_orig_import = builtins.__import__
def _safe_import(name, *a, **kw):
    try: return _orig_import(name, *a, **kw)
    except ModuleNotFoundError:
        if name not in sys.modules: sys.modules[name] = MagicMock()
        return sys.modules[name]
builtins.__import__ = _safe_import


# ── Helpers ──

def _make_dehydrator(tmp_path):
    """Create a Dehydrator with a mock OpenAI client, using isolated tmp_path."""
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


# ═══════════════════════════════════════════════════════════════
# Dehydrator unit tests (10 tests) — all use tmp_path
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_empty_choices_raises_runtime_error(tmp_path):
    dh = _make_dehydrator(tmp_path)
    dh.client.chat.completions.create.return_value = _fake_response(choices=[])
    with pytest.raises(RuntimeError, match="未返回 choices"):
        await dh._api_digest("some diary content here for testing")


@pytest.mark.asyncio
async def test_empty_content_raises_runtime_error(tmp_path):
    dh = _make_dehydrator(tmp_path)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = ""
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    with pytest.raises(RuntimeError, match="返回空正文"):
        await dh._api_digest("some diary content here for testing")


@pytest.mark.asyncio
async def test_valid_json_array_parses_correctly(tmp_path):
    dh = _make_dehydrator(tmp_path)
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
async def test_fenced_json_parses_correctly(tmp_path):
    dh = _make_dehydrator(tmp_path)
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
async def test_items_wrapper_parses_correctly(tmp_path):
    dh = _make_dehydrator(tmp_path)
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
async def test_invalid_json_raises_runtime_error_not_silent(tmp_path):
    dh = _make_dehydrator(tmp_path)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "这不是有效的 JSON，只是一段文字"
    dh.client.chat.completions.create.return_value = _fake_response(choices=[choice])
    with pytest.raises(RuntimeError, match="JSON 解析失败"):
        await dh._api_digest("content")


@pytest.mark.asyncio
async def test_missing_content_raises_with_validation_stats(tmp_path):
    dh = _make_dehydrator(tmp_path)
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
    monkeypatch.delenv("OMBRE_DIGEST_MODEL", raising=False)
    dh = _make_dehydrator(tmp_path)
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
    """Error msg and caplog must never contain the secret diary string or raw preview."""
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": str(tmp_path / "b")}
    dh = Dehydrator(config)
    secret = "Claire-SECRET-日记-abc123-今天心情很复杂-xyz789" * 20

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as exc_info:
            dh._parse_digest(secret)

    msg = str(exc_info.value)
    # Must contain safe metadata
    assert "raw_len=" in msg
    assert "raw_hash=" in msg
    # Must NOT contain the secret
    assert "Claire-SECRET-日记" not in msg
    assert "abc123" not in msg
    assert "xyz789" not in msg
    # Must NOT contain a raw preview
    assert "preview=" not in msg
    # caplog must also be clean
    log_text = caplog.text
    assert "Claire-SECRET-日记" not in log_text
    assert "abc123" not in log_text


# ═══════════════════════════════════════════════════════════════
# Real _grow_impl failure path tests — exercises production code
# ═══════════════════════════════════════════════════════════════

GROW_DIARY = "今天早上喝了咖啡，下午去了公园散步，晚上看了一部好电影，回家的路上还买了一束花。"


@pytest.mark.asyncio
async def test_grow_impl_digest_failure_preserves_evidence(monkeypatch):
    """Real _grow_impl: digest RuntimeError → evidence saved, no merge, no plan resolution."""
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

    import server as srv
    monkeypatch.setattr(srv, "_store_source_evidence", fake_store)
    monkeypatch.setattr(srv.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(srv, "_merge_or_create", fake_merge)
    monkeypatch.setattr(srv, "_fire_plan_resolution", fake_plan)
    monkeypatch.setattr(srv.decay_engine, "ensure_started", fake_ensure)

    result = await srv._grow_impl(GROW_DIARY, source_surface="Claude官方端")

    assert "日记整理失败" in result
    assert len(evidence_calls) == 1
    assert GROW_DIARY in evidence_calls[0]
    assert len(merge_calls) == 0
    assert len(plan_calls) == 0


@pytest.mark.asyncio
async def test_grow_impl_empty_items_no_candidates(monkeypatch):
    """Real _grow_impl: digest returns [] → evidence saved, no candidates created."""
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

    import server as srv
    monkeypatch.setattr(srv, "_store_source_evidence", fake_store)
    monkeypatch.setattr(srv.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(srv, "_merge_or_create", fake_merge)
    monkeypatch.setattr(srv.decay_engine, "ensure_started", fake_ensure)

    result = await srv._grow_impl(GROW_DIARY)

    assert "内容为空或整理失败" in result
    assert len(evidence_calls) == 1
    assert len(merge_calls) == 0


@pytest.mark.asyncio
async def test_grow_impl_success_path(monkeypatch):
    """Real _grow_impl: valid items → merge called, plan resolution fires."""
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

    import server as srv
    monkeypatch.setattr(srv, "_store_source_evidence", fake_store)
    monkeypatch.setattr(srv.dehydrator, "digest", fake_digest)
    monkeypatch.setattr(srv, "_merge_or_create", fake_merge)
    monkeypatch.setattr(srv, "_fire_plan_resolution", fake_plan)
    monkeypatch.setattr(srv.decay_engine, "ensure_started", fake_ensure)

    result = await srv._grow_impl(GROW_DIARY)

    assert "原文证据→ev-003" in result
    assert "新存" in result
    assert "待审候选→2条" in result
    assert len(evidence_calls) == 1
    assert len(merge_calls) == 2  # one per item
    assert len(plan_calls) == 1


# ═══════════════════════════════════════════════════════════════
# Real _store_source_evidence dedup test
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_store_source_evidence_dedup(monkeypatch, tmp_path):
    """Verify _store_source_evidence dedup contract via in-memory simulation.
    Since bucket_manager requires yaml C extension (incompatible with Python 3.9
    on arm64), we test the dedup logic against an in-memory store that mimics
    the real BucketManager.list_all + create behavior."""
    import hashlib, uuid
    import server as srv

    # In-memory evidence store
    store = {}
    seen_hashes = set()

    async def fake_list_all(include_archive=False):
        return list(store.values())

    async def fake_create(**kwargs):
        eid = str(uuid.uuid4())
        store[eid] = {"id": eid, "content": kwargs.get("content", ""), "metadata": kwargs.get("extra_metadata", {})}
        return eid

    # Wire the real _store_source_evidence to use our in-memory store
    # by monkeypatching bucket_mgr
    fake_mgr = MagicMock()
    fake_mgr.list_all = fake_list_all
    fake_mgr.create = fake_create
    monkeypatch.setattr(srv, "bucket_mgr", fake_mgr)

    content = "这是需要去重的原文证据内容。"
    surface = "测试来源"

    # First call — should create
    eid1, created1 = await srv._store_source_evidence(content, source_surface=surface)
    assert created1 is True
    assert eid1

    # Second call — same content and surface → same ID, created=False
    eid2, created2 = await srv._store_source_evidence(content, source_surface=surface)
    assert created2 is False
    assert eid2 == eid1

    # Different content → different ID
    eid3, created3 = await srv._store_source_evidence("不同的内容。", source_surface=surface)
    assert created3 is True
    assert eid3 != eid1
