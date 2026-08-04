"""Tests for anchor/release coordinate system."""

import pytest

import server
from bucket_manager import BucketManager


async def _done():
    return None


def _bucket(bucket_id, content, **meta):
    m = {
        "type": "dynamic",
        "name": content[:20],
        "created": "2026-08-04T10:00:00",
        "last_active": "2026-08-04T10:00:00",
        "memory_status": "confirmed",
        "domain": ["回忆"],
        "importance": 5,
        "valence": 0.5,
        "arousal": 0.3,
    }
    m.update(meta)
    return {"id": bucket_id, "content": content, "metadata": m}


@pytest.mark.asyncio
async def test_anchor_set_release_and_limit(test_config, monkeypatch):
    manager = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    first_id = await manager.create("我们的关系基调。")
    second_id = await manager.create("另一条普通记忆。")

    reply = await server.anchor(first_id)
    assert "已锚定" in reply
    assert "1/24" in reply
    bucket = await manager.get(first_id)
    assert bucket["metadata"]["anchor"] is True

    noop = await server.anchor(first_id)
    assert "已经是 anchor" in noop

    released = await server.release(first_id)
    assert "已解除锚定" in released
    bucket = await manager.get(first_id)
    assert bucket["metadata"].get("anchor") is False

    released_again = await server.release(second_id)
    assert "本来就不是 anchor" in released_again


@pytest.mark.asyncio
async def test_anchor_hard_limit_is_24(test_config, monkeypatch):
    manager = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    ids = []
    for i in range(24):
        ids.append(await manager.create(f"坐标第{i+1}条。"))
    for bucket_id in ids:
        await server.anchor(bucket_id)
    extra_id = await manager.create("第25条。")
    reply = await server.anchor(extra_id)
    assert "已满" in reply
    bucket = await manager.get(extra_id)
    assert not bucket["metadata"].get("anchor", False)


@pytest.mark.asyncio
async def test_anchor_does_not_surface_but_is_searchable(test_config, monkeypatch):
    manager = BucketManager(test_config)

    class SurfacingManager:
        def __init__(self, buckets):
            self.buckets = buckets

        async def list_all(self, include_archive=False):
            return self.buckets

        async def search(self, *args, **kwargs):
            return [b for b in self.buckets if b["id"] == "anchor-1"]

    buckets = [
        _bucket("anchor-1", "我们的坐标系。", anchor=True, importance=9),
        _bucket("normal-1", "一件待办的小事。", importance=8),
    ]
    monkeypatch.setattr(server, "bucket_mgr", SurfacingManager(buckets))
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())
    monkeypatch.setattr(server, "_get_recall_cooldown", lambda: None)
    monkeypatch.setattr(server.dehydrator, "dehydrate", lambda content, meta=None: _async_str(content))
    monkeypatch.setattr(server.family_engine, "families_for", lambda bucket_ids: {})
    monkeypatch.setattr(server, "_fire_webhook", lambda *args, **kwargs: _done())

    surfaced = await server.breath()
    assert "我们的坐标系。" not in surfaced
    assert "一件待办的小事。" in surfaced

    searched = await server.breath(query="坐标系")
    assert "我们的坐标系。" in searched


async def _async_str(value):
    return value
