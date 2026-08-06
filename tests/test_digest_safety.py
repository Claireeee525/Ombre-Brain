"""Regression tests for diary digest safety: error propagation, parse robustness, privacy."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ──

def _make_dehydrator(digest_model=None):
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
    if digest_model:
        dh.digest_model = digest_model
    # Replace the real client with a mock
    dh.client = MagicMock()
    dh.client.chat = MagicMock()
    dh.client.chat.completions = MagicMock()
    dh.client.chat.completions.create = AsyncMock()
    return dh


def _fake_response(*, choices=None, model="test-model", usage=None, response_id="resp-1"):
    """Build a mock OpenAI response object."""
    resp = MagicMock()
    resp.choices = choices or []
    resp.model = model
    resp.id = response_id
    resp.usage = usage or MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return resp


# ── Test 1: empty choices → RuntimeError ──

@pytest.mark.asyncio
async def test_empty_choices_raises_runtime_error():
    dh = _make_dehydrator()
    resp = _fake_response(choices=[])
    dh.client.chat.completions.create.return_value = resp

    with pytest.raises(RuntimeError, match="未返回 choices"):
        await dh._api_digest("some diary content here for testing")


# ── Test 2: empty content → RuntimeError ──

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


# ── Test 3: valid JSON array → normal parse ──

@pytest.mark.asyncio
async def test_valid_json_array_parses_correctly():
    dh = _make_dehydrator()
    items = [
        {"name": "早上的咖啡", "content": "今天早上喝了一杯很香的拿铁", "domain": ["日常"], "valence": 0.8, "arousal": 0.3, "tags": ["咖啡"], "importance": 3},
        {"name": "下午的散步", "content": "下午在公园散了步，天气很好", "domain": ["日常", "健康"], "valence": 0.9, "arousal": 0.2, "tags": ["散步", "天气"], "importance": 4},
    ]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    resp = _fake_response(choices=[choice])
    dh.client.chat.completions.create.return_value = resp

    result = await dh._api_digest("content for testing")
    assert len(result) == 2
    assert result[0]["name"] == "早上的咖啡"
    assert result[1]["content"] == "下午在公园散了步，天气很好"


# ── Test 4: fenced JSON → normal parse ──

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
    assert result[0]["content"] == "这是一条测试记忆"


# ── Test 5: {"items": [...]} wrapper → normal parse ──

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
    assert result[0]["content"] == "包装在 items 里的记忆"


# ── Test 6: invalid JSON → RuntimeError, not silent [] ──

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


# ── Test 7: missing content in all items → RuntimeError with stats ──

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


# ── Test 8: _parse_digest handles text wrapped around JSON array ──

def test_parse_extracts_json_from_wrapping_text():
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": "/tmp/x"}
    dh = Dehydrator(config)
    raw = "以下是整理结果：\n[{\"name\": \"n\", \"content\": \"c\", \"domain\": [\"日常\"], \"valence\": 0.5, \"arousal\": 0.3, \"tags\": [], \"importance\": 5}]\n希望有帮助。"
    result = dh._parse_digest(raw)
    assert len(result) == 1
    assert result[0]["content"] == "c"


# ── Test 9: digest() with digest_model override ──

@pytest.mark.asyncio
async def test_digest_uses_digest_model_when_set():
    dh = _make_dehydrator(digest_model="deepseek-v4-pro")
    items = [{"name": "t", "content": "test content here", "domain": ["测试"], "valence": 0.5, "arousal": 0.3, "tags": [], "importance": 5}]
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = json.dumps(items, ensure_ascii=False)
    resp = _fake_response(choices=[choice], model="deepseek-v4-pro")
    dh.client.chat.completions.create.return_value = resp

    result = await dh.digest("test")
    assert len(result) == 1
    # Verify it used the digest_model, not the default
    call_args = dh.client.chat.completions.create.call_args
    assert call_args[1]["model"] == "deepseek-v4-pro"


# ── Test 10: parse error contains safe metadata, preview capped at 200 chars ──

def test_parse_error_contains_safe_metadata():
    from dehydrator import Dehydrator
    config = {"dehydration": {"api_key": "sk-test", "model": "x", "base_url": "x"}, "buckets_dir": "/tmp/x"}
    dh = Dehydrator(config)
    long_content = "x" * 500
    with pytest.raises(RuntimeError) as exc_info:
        dh._parse_digest(long_content)
    msg = str(exc_info.value)
    # Must contain safe metadata
    assert "raw_len=500" in msg
    assert "raw_hash=" in msg
    # Preview must be capped — full 500 chars not in message
    assert "x" * 250 not in msg
    # But 200-char preview is allowed
    assert "x" * 200 in msg
