from unittest.mock import AsyncMock

import pytest

import server
import utils
from bucket_manager import BucketManager


def _bucket(bucket_id, content, *, name="记忆", importance=5, bucket_type="dynamic"):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "name": name,
            "tags": [],
            "domain": ["回忆"],
            "importance": importance,
            "valence": 0.5,
            "arousal": 0.3,
            "type": bucket_type,
            "created": "2026-08-01T00:00:00",
            "last_active": "2026-08-01T00:00:00",
            "activation_count": 0,
            "memory_status": "confirmed",
        },
    }


def test_surfacing_builtin_defaults_are_two_dynamic_and_three_pinned(tmp_path, monkeypatch):
    monkeypatch.delenv("OMBRE_SURFACING_MAX_DYNAMIC_PER_CALL", raising=False)
    monkeypatch.delenv("OMBRE_SURFACING_MAX_PINNED_PER_CALL", raising=False)
    cfg = utils.load_config(str(tmp_path / "missing.yaml"))
    assert cfg["surfacing"]["max_dynamic_per_call"] == 2
    assert cfg["surfacing"]["max_pinned_per_call"] == 3


def test_surfacing_env_overrides_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_SURFACING_MAX_DYNAMIC_PER_CALL", "4")
    monkeypatch.setenv("OMBRE_SURFACING_MAX_PINNED_PER_CALL", "5")
    cfg = utils.load_config(str(tmp_path / "missing.yaml"))
    assert cfg["surfacing"]["max_dynamic_per_call"] == 4
    assert cfg["surfacing"]["max_pinned_per_call"] == 5


@pytest.mark.asyncio
async def test_search_requires_real_evidence_before_soft_ranking(test_config):
    manager = BucketManager(test_config)
    manager.list_all = AsyncMock(return_value=[
        _bucket("match", "我们在阳台种下了一盆冰茉莉。", importance=2),
        _bucket("noise", "刚完成一次很重要的新版本部署。", importance=10),
    ])

    results = await manager.search("冰茉莉", limit=10)

    assert [item["id"] for item in results] == ["match"]


@pytest.mark.asyncio
async def test_short_query_must_be_literal_not_fuzzy_noise(test_config):
    manager = BucketManager(test_config)
    manager.list_all = AsyncMock(return_value=[
        _bucket("match", "猫趴在窗边晒太阳。"),
        _bucket("noise", "今天讨论了长期版本规划。", importance=10),
    ])

    results = await manager.search("猫", limit=10)

    assert [item["id"] for item in results] == ["match"]


@pytest.mark.asyncio
async def test_semantic_threshold_admits_only_clear_related_result(test_config):
    class FakeEmbedding:
        enabled = True

        async def search_similar(self, query, top_k=50):
            return [("weak", 0.71), ("strong", 0.73)]

    manager = BucketManager(test_config, embedding_engine=FakeEmbedding())
    manager.list_all = AsyncMock(return_value=[
        _bucket("weak", "完全不同的占位文字 A。"),
        _bucket("strong", "完全不同的占位文字 B。"),
    ])

    results = await manager.search("海边约定", limit=10)

    assert [item["id"] for item in results] == ["strong"]
    assert results[0]["vector_match"] is True


class _BreathManager:
    def __init__(self, results):
        self.results = results
        self.touch = AsyncMock()
        self.list_all = AsyncMock(side_effect=AssertionError("query recall must not drift into list_all"))

    async def search(self, *args, **kwargs):
        return self.results


@pytest.mark.asyncio
async def test_query_recall_is_read_only_and_never_self_reinforces(monkeypatch):
    manager = _BreathManager([_bucket("match", "冰茉莉开花了。")])
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    monkeypatch.setattr(server.dehydrator, "dehydrate", AsyncMock(side_effect=AssertionError("query recall must not call LLM dehydration")))
    monkeypatch.setattr(server.family_engine, "families_for", lambda bucket_ids: {})
    monkeypatch.setattr(server, "_fire_webhook", AsyncMock())

    result = await server.breath(query="冰茉莉")

    # 检索逐字返回原文，不经 LLM 摘要
    assert "冰茉莉开花了。" in result
    manager.touch.assert_not_awaited()
    manager.list_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_keeps_digested_and_pinned_memories(monkeypatch):
    manager = _BreathManager([
        _bucket("digested", "那次体检的结论。", name="体检"),
        _bucket("pinned", "我们之间的一条核心约定。", name="约定"),
    ])
    manager.results[0]["metadata"]["digested"] = True
    manager.results[1]["metadata"]["pinned"] = True
    manager.results[1]["metadata"]["protected"] = True
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    monkeypatch.setattr(server.family_engine, "families_for", lambda bucket_ids: {})
    monkeypatch.setattr(server, "_fire_webhook", AsyncMock())

    result = await server.breath(query="体检 约定")

    assert "那次体检的结论。" in result
    assert "我们之间的一条核心约定。" in result
    assert "已消化，仍可检索" in result


@pytest.mark.asyncio
async def test_empty_query_match_does_not_append_random_old_memory(monkeypatch):
    manager = _BreathManager([])
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    monkeypatch.setattr(server, "_fire_webhook", AsyncMock())

    result = await server.breath(query="不存在的专名")

    assert result == "未找到相关记忆。"
    manager.list_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_uses_duplicate_meaning_not_recall_rank(monkeypatch):
    unrelated = _bucket("fresh-important", "刚完成一次很重要的新版本部署。", importance=10)

    class MergeManager:
        def __init__(self):
            self.create = AsyncMock(return_value="new-memory")
            self.update = AsyncMock()

        async def list_all(self, include_archive=False):
            return [unrelated]

    manager = MergeManager()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.embedding_engine, "generate_and_store", AsyncMock(return_value=True))

    result, merged = await server._merge_or_create(
        "我们在阳台种下了一盆冰茉莉。", [], 5, ["回忆"], 0.6, 0.3, "冰茉莉"
    )

    assert (result, merged) == ("new-memory", False)
    manager.update.assert_not_awaited()
    manager.create.assert_awaited_once()
