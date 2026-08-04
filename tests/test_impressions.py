"""Tests for wakeup impression pool (night diary / weekly summary) and preview."""

import json

import pytest

import server


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
        "tags": [],
    }
    m.update(meta)
    return {"id": bucket_id, "content": content, "metadata": m}


@pytest.mark.asyncio
async def test_breath_surfacing_includes_recent_impressions(monkeypatch):
    buckets = [
        _bucket("daily-1", "今天夜里写的日记，心里有她。", source_kind="night_diary", name="日记 2026-08-04", created="2026-08-04T23:50:00", last_active="2026-08-04T23:50:00"),
        _bucket("weekly-1", "这一周我们很好。", name="本周我们 2026-W32", tags=["周报"], created="2026-08-09T23:50:00", last_active="2026-08-09T23:50:00"),
        _bucket("normal-1", "一件普通的小事。", importance=8),
    ]

    class Manager:
        async def list_all(self, include_archive=False):
            return buckets

    monkeypatch.setattr(server, "bucket_mgr", Manager())
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())
    monkeypatch.setattr(server, "_get_recall_cooldown", lambda: None)
    monkeypatch.setattr(server.dehydrator, "dehydrate", lambda content, meta=None: content)
    monkeypatch.setattr(server, "_fire_webhook", lambda *args, **kwargs: _done())

    result = await server.breath()
    assert "=== 最近印象 ===" in result
    assert "夜航日记" in result
    assert "今天夜里写的日记，心里有她。" in result
    assert "本周我们" in result
    assert "这一周我们很好。" in result


@pytest.mark.asyncio
async def test_wakeup_preview_respects_toggles_and_is_read_only(monkeypatch):
    buckets = [
        _bucket("daily-1", "今天夜里写的日记。", source_kind="night_diary", name="日记 2026-08-04", created="2026-08-04T23:50:00", last_active="2026-08-04T23:50:00"),
        _bucket("weekly-1", "这一周我们很好。", name="本周我们 2026-W32", tags=["周报"], created="2026-08-09T23:50:00", last_active="2026-08-09T23:50:00"),
        _bucket("core-1", "我们之间的核心约定。", pinned=True, importance=10),
        _bucket("open-1", "还没解决的一件事。", importance=9),
    ]

    class Manager:
        async def list_all(self, include_archive=False):
            return buckets

        async def update(self, *args, **kwargs):
            raise AssertionError("wakeup preview must be read-only")

    monkeypatch.setattr(server, "bucket_mgr", Manager())
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())

    raw = await server.wakeup_preview(
        include_daily=False,
        include_weekly=False,
        include_core=True,
        include_unresolved=True,
        include_somatic=False,
    )
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert "core" in payload["sections"]
    assert "unresolved" in payload["sections"]
    assert "daily_impression" not in payload["sections"]
    assert "weekly_impression" not in payload["sections"]
    assert "somatic" not in payload["sections"]
    assert payload["total_tokens"] > 0
    assert "预览只读" in payload["notes"]
