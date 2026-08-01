import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

import server
from bucket_manager import BucketManager
from memory_layers import (
    classify_memory_layer,
    memory_recallable,
    normalize_layer_metadata,
)


def _bucket(bucket_id, content, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {"name": bucket_id, "domain": ["测试"], **metadata},
    }


def test_legacy_buckets_are_derived_without_rewriting_frontmatter():
    raw = _bucket(
        "raw",
        "时间：2026/8/1 10:00\nClaire：记下这句。\n珂洛：好。",
        source_kind="legacy",
    )
    candidate = _bucket("candidate", "可能的偏好", memory_status="candidate")
    active = _bucket("active", "已经确认的约定", memory_status="confirmed")
    short_term = _bucket(
        "short",
        "当前项目线头",
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    assert classify_memory_layer(raw["metadata"], raw["content"]) == "evidence"
    assert classify_memory_layer(candidate["metadata"], candidate["content"]) == "candidate"
    assert classify_memory_layer(active["metadata"], active["content"]) == "active"
    assert normalize_layer_metadata(short_term["metadata"], short_term["content"])["memory_layer"] == "short_term"
    assert "memory_layer" not in raw["metadata"]


@pytest.mark.parametrize(
    ("layer", "mode", "expected"),
    [
        ("evidence", "normal", False),
        ("evidence", "exact", True),
        ("candidate", "normal", False),
        ("candidate", "review", True),
        ("active", "normal", True),
        ("short_term", "normal", False),
        ("short_term", "handoff", True),
        ("feel", "normal", False),
        ("feel", "accompany", True),
        ("archive", "normal", False),
        ("archive", "archive", True),
    ],
)
def test_layer_recall_policy_is_explicit(layer, mode, expected):
    metadata = {"memory_layer": layer}
    if layer == "short_term":
        metadata["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert memory_recallable(metadata, "内容", mode=mode) is expected


@pytest.mark.asyncio
async def test_bucket_search_applies_layer_gate_before_ranking(test_config):
    manager = BucketManager(test_config)
    manager.list_all = AsyncMock(return_value=[
        _bucket("raw", "时间：2026/8/1\nClaire：猫。\n珂洛：记下。"),
        _bucket("candidate", "猫的偏好", memory_status="candidate"),
        _bucket("active", "猫喜欢趴在窗边", memory_status="confirmed"),
    ])

    normal = await manager.search("猫", limit=10)
    exact = await manager.search("猫", limit=10, recall_mode="exact")
    review = await manager.search("猫", limit=10, recall_mode="review")

    assert [item["id"] for item in normal] == ["active"]
    assert {item["id"] for item in exact} == {"raw", "active"}
    assert [item["id"] for item in review] == ["candidate"]


@pytest.mark.asyncio
async def test_reject_and_restore_switch_layer_without_deleting_evidence(monkeypatch):
    class Manager:
        def __init__(self):
            self.bucket = _bucket("memory-1", "可恢复正文", memory_status="confirmed")

        async def get(self, bucket_id):
            return self.bucket if bucket_id == "memory-1" else None

        async def update(self, bucket_id, **updates):
            self.bucket["metadata"].update(updates)
            return True

    manager = Manager()
    monkeypatch.setattr(server, "bucket_mgr", manager)

    rejected = json.loads(await server.memory_review("memory-1", "reject"))
    assert rejected["ok"] is True
    assert manager.bucket["metadata"]["memory_layer"] == "archive"
    assert manager.bucket["content"] == "可恢复正文"
    assert not server._curator_recallable(manager.bucket["metadata"], content=manager.bucket["content"])

    restored = json.loads(await server.memory_review("memory-1", "restore"))
    assert restored["ok"] is True
    assert manager.bucket["metadata"]["memory_layer"] == "active"
    assert server._curator_recallable(manager.bucket["metadata"], content=manager.bucket["content"])
