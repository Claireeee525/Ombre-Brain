"""Tests for plan (promise/todo) lifecycle and dream integration."""

import pytest

import server
from bucket_manager import BucketManager


class FakeEmbedding:
    async def generate_and_store(self, *args, **kwargs):
        return True


async def _done():
    return None


def _plan_bucket(bucket_id, content, **meta):
    m = {
        "type": "plan",
        "status": "active",
        "name": content[:20],
        "created": "2026-08-04T10:00:00",
        "last_active": "2026-08-04T10:00:00",
        "memory_status": "confirmed",
        "domain": ["plan"],
        "weight": 0.5,
        "change_log": [],
    }
    m.update(meta)
    return {"id": bucket_id, "content": content, "metadata": m}


@pytest.mark.asyncio
async def test_plan_create_writes_plan_bucket_and_dedups_exact_active(test_config, monkeypatch):
    manager = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())

    first = await server.plan("周末带她去看海")
    assert first.startswith("📋plan→")
    bucket_id = first.split("→")[1].split(" ")[0]
    bucket = await manager.get(bucket_id)
    assert bucket["metadata"]["type"] == "plan"
    assert bucket["metadata"]["status"] == "active"
    assert bucket["metadata"]["weight"] == 0.5

    second = await server.plan("周末带她去看海")
    assert bucket_id in second
    plans = [b for b in await manager.list_all(include_archive=True) if b["metadata"].get("type") == "plan"]
    assert len(plans) == 1


@pytest.mark.asyncio
async def test_plan_list_filters_status(test_config, monkeypatch):
    manager = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())
    await server.plan("买回家的牛奶")
    active_id = (await server.plan("修好她的台灯")).split("→")[1].split(" ")[0]
    await server.trace(active_id, status="resolved")

    active_list = await server.plan_list(status="active")
    assert "买回家的牛奶" in active_list
    assert "修好她的台灯" not in active_list
    all_list = await server.plan_list(status="all")
    assert "修好她的台灯" in all_list


@pytest.mark.asyncio
async def test_trace_resolves_plan_and_rejects_status_on_non_plan(test_config, monkeypatch):
    manager = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    plan_id = await manager.create(
        "把体检报告发给妈妈",
        bucket_type="plan",
        extra_metadata={"status": "active", "change_log": []},
    )
    await server.trace(plan_id, status="resolved")
    bucket = await manager.get(plan_id)
    assert bucket["metadata"]["status"] == "resolved"
    assert bucket["metadata"]["resolved"] is True

    normal_id = await manager.create("普通记忆")
    reply = await server.trace(normal_id, status="resolved")
    assert "只能用于 plan 桶" in reply


@pytest.mark.asyncio
async def test_auto_resolution_closes_plan_when_event_matches(test_config, monkeypatch):
    manager = BucketManager(test_config)
    plan_id = await manager.create(
        "周末带她去看海",
        bucket_type="plan",
        extra_metadata={"status": "active", "change_log": []},
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)

    class FakeJudge:
        async def judge_plan_resolution(self, plan_content, event_content):
            return {"resolved": True, "reason": "她真的去了海边"}

    monkeypatch.setattr(server, "dehydrator", FakeJudge())
    await server._check_plan_resolution("我们周六真的去了海边，浪很大")
    bucket = await manager.get(plan_id)
    assert bucket["metadata"]["status"] == "resolved"
    assert bucket["metadata"]["resolved"] is True


@pytest.mark.asyncio
async def test_plan_never_surfaces_in_breath_search(test_config, monkeypatch):
    manager = BucketManager(test_config)

    class SearchManager:
        def __init__(self, results):
            self.results = results

        async def search(self, *args, **kwargs):
            return self.results

        async def list_all(self, **kwargs):
            raise AssertionError("query recall must not drift into list_all")

    plan_bucket = _plan_bucket("plan-1", "周末带她去看海")
    normal_bucket = {
        "id": "mem-1",
        "content": "我们去看了海。",
        "metadata": {"type": "dynamic", "name": "看海", "created": "2026-08-04T10:00:00", "memory_status": "confirmed", "domain": ["回忆"]},
    }
    monkeypatch.setattr(server, "bucket_mgr", SearchManager([plan_bucket, normal_bucket]))
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())
    monkeypatch.setattr(server.family_engine, "families_for", lambda bucket_ids: {})
    monkeypatch.setattr(server, "_fire_webhook", lambda *args, **kwargs: _done())

    result = await server.breath(query="看海")
    assert "我们去看了海。" in result
    assert "周末带她去看海" not in result


@pytest.mark.asyncio
async def test_dream_lists_active_plans_and_skips_digested_memories(test_config, monkeypatch):
    manager = BucketManager(test_config)
    for bid, content, meta in [
        ("normal-1", "今天一起看了场电影。", {"type": "dynamic", "name": "看电影", "created": "2026-08-04T10:00:00", "last_active": "2026-08-04T10:00:00", "memory_status": "confirmed", "domain": ["回忆"]}),
        ("digested-1", "这条已经消化了。", {"type": "dynamic", "name": "消化", "digested": True, "created": "2026-08-04T09:00:00", "last_active": "2026-08-04T09:00:00", "memory_status": "confirmed", "domain": ["回忆"]}),
        ("old-1", "很久以前的事。", {"type": "dynamic", "name": "旧事", "created": "2026-07-01T10:00:00", "last_active": "2026-07-01T10:00:00", "memory_status": "confirmed", "domain": ["回忆"]}),
    ]:
        created_id = await manager.create(content, bucket_type=meta["type"], extra_metadata={k: v for k, v in meta.items() if k != "type"})
        if meta.get("last_active", "").startswith("2026-07-01"):
            import frontmatter as _fm
            from pathlib import Path
            old_bucket = await manager.get(created_id)
            post = _fm.load(Path(old_bucket["path"]))
            post["last_active"] = meta["last_active"]
            post["created"] = meta["created"]
            _fm.dump(post, Path(old_bucket["path"]))
    await manager.create(
        "把体检报告发给妈妈",
        bucket_type="plan",
        extra_metadata={"status": "active", "change_log": [], "created": "2026-08-04T10:00:00", "last_active": "2026-08-04T10:00:00"},
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())

    class FakeEmbeddingOff:
        enabled = False

    monkeypatch.setattr(server, "embedding_engine", FakeEmbeddingOff())
    monkeypatch.setattr(server, "_fire_webhook", lambda *args, **kwargs: _done())

    result = await server.dream(window_hours=48)
    assert "今天一起看了场电影。" in result
    assert "这条已经消化了。" not in result
    assert "很久以前的事。" not in result
    assert "=== Active Plans ===" in result
    assert "把体检报告发给妈妈" in result
