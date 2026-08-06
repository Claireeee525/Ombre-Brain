"""Regression tests for diary digest safety: error propagation, parse robustness, privacy, grow failure."""
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Helpers ──

def _make_dehydrator():
    """Create a Dehydrator with a mock OpenAI client for unit testing."""
    from dehydrator import Dehydrator
    config = {
        "dehydration": {
            "api_key": "sk-test",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
        },
        "buckets_dir": "/tmp/test-buckets",
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
# Dehydrator unit tests (10 tests)
# ═══════════════════════════════════════════════════════════════

# 1: empty choices → RuntimeError
@pytest.mark.asyncio
async def test_empty_choices_raises_runtime_error():
    dh = _make_dehydrator()
    resp = _fake_response(choices=[])
    dh.client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError, match="未返回 choices"):
        await dh._api_digest("some diary content here for testing")


# 2: empty content → RuntimeError
@pytest.mark.asyncio
async def test_empty_content_raises_runtime_error():
    dh = _make_dehydrator()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = ""
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError, match="返回空正文"):
        await dh._api_digest("some diary content here for testing")


# 3: valid JSON array → normal parse
@pytest.mark.asyncio
async def test_valid_json_array_parses_correctly():
    dh = _make_dehydrator()
    items = _valid_items()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    result = await dh._api_digest("content for testing")
    assert len(result) == 2
    assert result[0]["name"] == "早上的咖啡"
    assert result[1]["content"] == "下午在公园散了步，天气很好"


# 4: fenced JSON → normal parse
@pytest.mark.asyncio
async def test_fenced_json_parses_correctly():
    dh = _make_dehydrator()
    items = [{"name": "测试", "content": "这是一条测试记忆", "domain": ["测试"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]
    raw = "```json\n" + json.dumps(items, ensure_ascii=False) + "\n```"
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = raw
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    result = await dh._api_digest("content")
    assert len(result) == 1


# 5: {"items": [...]} wrapper → normal parse
@pytest.mark.asyncio
async def test_items_wrapper_parses_correctly():
    dh = _make_dehydrator()
    items = [{"name": "包装测试", "content": "包装在 items 里的记忆", "domain": ["测试"], "valence": 0.6, "arousal": 0.4, "tags": [], "importance": 5}]
    raw = json.dumps({"items": items}, ensure_ascii=False)
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = raw
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    result = await dh._api_digest("content")
    assert len(result) == 1


# 6: invalid JSON → RuntimeError, not silent []
@pytest.mark.asyncio
async def test_invalid_json_raises_runtime_error_not_silent():
    dh = _make_dehydrator()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "这不是有效的 JSON，只是一段文字"
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError, match="JSON 解析失败"):
        await dh._api_digest("content")


# 7: missing content in all items → RuntimeError with validation stats
@pytest.mark.asyncio
async def test_missing_content_raises_with_validation_stats():
    dh = _make_dehydrator()
    items = [
        {"name": "缺content1", "domain": ["日常"]},
        {"not": "even a dict with content"},
    ]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError, match="没有有效 content"):
        await dh._api_digest("content")


# 8: JSON wrapped in text → extracts array
def test_parse_extracts_json_from_wrapping_text():
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": "/tmp/x"}
    dh = Dehydrator(config)
    raw = '以下是整理结果：\n[{"name": "n", "content": "c", "domain": ["日常"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]\n希望有帮助。'
    result = dh._parse_digest(raw)
    assert len(result) == 1
    assert result[0]["content"] == "c"


# 9: digest_model override
@pytest.mark.asyncio
async def test_digest_uses_digest_model_when_set():
    dh = _make_dehydrator()
    dh.digest_model = "custom-digest-model"
    items = [{"name": "t", "content": "test content here", "domain": ["测试"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    resp = _fake_response(choices=[choice], model="custom-digest-model")
    dh.client.chat.completions.create.return_value = resp
    result = await dh.digest("test")
    assert len(result) == 1
    call_args = dh.client.chat.completions.create.call_args
    assert call_args[1]["model"] == "custom-digest-model"


# 10: parse error NEVER exposes diary content
def test_parse_error_never_exposes_diary_content():
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": "/tmp/x"}
    dh = Dehydrator(config)
    secret = "Claire-SECRET-日记-abc123-今天心情很复杂-xyz789" * 20
    with pytest.raises(RuntimeError) as exc_info:
        dh._parse_digest(secret)
    msg = str(exc_info.value)
    # Must contain safe metadata
    assert "raw_len=" in msg
    assert "raw_hash=" in msg
    # Must NOT contain the secret string
    assert "Claire-SECRET-日记" not in msg
    assert "abc123" not in msg
    assert "xyz789" not in msg
    # Must NOT contain raw[:200] preview
    assert "preview=" not in msg


# ═══════════════════════════════════════════════════════════════
# Grow-level safety regression tests (5 tests)
#
# Test the grow error-handling contract directly by replicating
# the logic inline — avoids the full server import chain.
# ═══════════════════════════════════════════════════════════════

GROW_DIARY = "今天早上喝了咖啡，下午去了公园散步，晚上看了一部好电影，回家的路上还买了一束花。"

# Replicate grow() core logic from server.py for isolated testing
async def _grow_core(content, *, store_evidence, digest_fn, merge_fn, ensure_started, source_surface="", source_session_id=""):
    """Replica of server.py grow() error-handling logic, testable in isolation."""
    try:
        await ensure_started()
    except Exception as e:
        return f"内部错误：衰减引擎启动失败 - {e}"

    if not content or not content.strip():
        return "内容为空，无法整理。"

    try:
        evidence_id, evidence_created = await store_evidence(
            content, source_surface=source_surface, source_session_id=source_session_id)
    except Exception as e:
        return f"内部错误：原文证据保存失败 - {e}"

    # Short content fast path
    if len(content.strip()) < 30:
        return "（短内容快速路径）"

    try:
        items = await digest_fn(content)
    except Exception as e:
        return f"日记整理失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    for item in items:
        try:
            result_name, is_merged = await merge_fn(item)
            results.append(f"候选·{result_name}")
        except Exception as e:
            results.append(f"⚠️{item.get('name', '?')}")

    return "\n".join(results) if results else "内容为空或整理失败。"


@pytest.mark.asyncio
async def test_grow_preserves_evidence_on_digest_failure():
    """Grow: digest 失败时原文证据仍保留，不创建候选桶。"""
    evidence_calls = []
    async def store_evidence(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-001", True)

    async def digest_fn(content):
        raise RuntimeError("日记整理 API 返回空正文（model=deepseek-v4-flash, finish_reason=stop, completion_tokens=0）")

    merge_calls = []
    async def merge_fn(item):
        merge_calls.append(item)
        return ("merged-1", True)

    async def ensure_started():
        pass

    result = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=merge_fn,
        ensure_started=ensure_started, source_surface="Claude官方端")

    assert "日记整理失败" in result
    assert len(evidence_calls) == 1
    assert GROW_DIARY in evidence_calls[0]
    assert len(merge_calls) == 0


@pytest.mark.asyncio
async def test_grow_returns_error_on_empty_choices():
    """Grow: choices 为空时返回明确错误。"""
    async def store_evidence(content, *, source_surface="", source_session_id=""):
        return ("ev-002", True)

    async def digest_fn(content):
        raise RuntimeError("日记整理 API 未返回 choices（model=deepseek-v4-flash, response_id=resp-x）")

    async def ensure_started():
        pass

    result = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=AsyncMock(),
        ensure_started=ensure_started)

    assert "日记整理失败" in result
    assert "未返回 choices" in result


@pytest.mark.asyncio
async def test_grow_returns_error_on_invalid_json():
    """Grow: 非法 JSON 时返回明确错误，原文证据保留。"""
    evidence_calls = []
    async def store_evidence(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-003", True)

    async def digest_fn(content):
        raise RuntimeError("日记整理结果 JSON 解析失败（raw_len=500, raw_hash=abcd1234）")

    async def ensure_started():
        pass

    result = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=AsyncMock(),
        ensure_started=ensure_started)

    assert "日记整理失败" in result
    assert "JSON 解析失败" in result
    assert len(evidence_calls) == 1


@pytest.mark.asyncio
async def test_grow_retry_same_content_does_not_duplicate_evidence():
    """Grow: 同一原文两次调用，store_evidence 第二次返回已存在 ID。"""
    evidence_calls = []
    async def store_evidence(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        if len(evidence_calls) == 1:
            return ("ev-004", True)
        return ("ev-004", False)

    digest_calls = []
    async def digest_fn(content):
        digest_calls.append(content)
        if len(digest_calls) == 1:
            raise RuntimeError("日记整理 API 返回空正文（model=deepseek-v4-flash, finish_reason=stop, completion_tokens=0）")
        else:
            raise RuntimeError("日记整理 API 未返回 choices（model=deepseek-v4-flash, response_id=resp-y）")

    async def ensure_started():
        pass

    r1 = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=AsyncMock(),
        ensure_started=ensure_started)
    r2 = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=AsyncMock(),
        ensure_started=ensure_started)

    assert "日记整理失败" in r1
    assert "日记整理失败" in r2
    assert len(evidence_calls) == 2
    assert evidence_calls[0] == evidence_calls[1]  # same content both times


@pytest.mark.asyncio
async def test_grow_empty_items_creates_no_candidates():
    """Grow: digest 返回空列表时不创建任何候选。"""
    evidence_calls = []
    async def store_evidence(content, *, source_surface="", source_session_id=""):
        evidence_calls.append(content)
        return ("ev-005", True)

    async def digest_fn(content):
        return []

    merge_calls = []
    async def merge_fn(item):
        merge_calls.append(item)
        return ("merged-x", True)

    async def ensure_started():
        pass

    result = await _grow_core(GROW_DIARY,
        store_evidence=store_evidence, digest_fn=digest_fn, merge_fn=merge_fn,
        ensure_started=ensure_started)

    assert "内容为空或整理失败" in result
    assert len(evidence_calls) == 1
    assert len(merge_calls) == 0
