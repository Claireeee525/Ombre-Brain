import json

import pytest

import server


class FakeBucketManager:
    def __init__(self, status="confirmed", *, pinned=False, protected=False):
        self.bucket = {
            "id": "memory-1",
            "content": "保留原文证据。",
            "metadata": {
                "memory_status": status,
                "pinned": pinned,
                "protected": protected,
            },
        }

    async def get(self, bucket_id):
        return self.bucket if bucket_id == self.bucket["id"] else None

    async def update(self, bucket_id, **updates):
        if bucket_id != self.bucket["id"]:
            return False
        self.bucket["metadata"].update(updates)
        return True


@pytest.mark.asyncio
async def test_confirmed_memory_can_be_rejected_without_deleting_evidence(monkeypatch):
    manager = FakeBucketManager("confirmed")
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_review(
        "memory-1",
        "reject",
        actor="Claire via Kelo Home",
        reason="user_deleted_in_home",
        request_id="audit-123",
    ))

    assert result["ok"] is True
    assert result["memory_status"] == "rejected"
    assert result["reviewed_by"] == "Claire via Kelo Home"
    assert result["review_reason"] == "user_deleted_in_home"
    assert result["review_request_id"] == "audit-123"
    assert manager.bucket["content"] == "保留原文证据。"
    assert manager.bucket["metadata"]["status_before_reject"] == "confirmed"
    assert manager.bucket["metadata"]["resolved"] is True
    assert server._curator_recallable(manager.bucket["metadata"]) is False


@pytest.mark.asyncio
async def test_rejected_memory_can_be_restored_to_its_previous_status(monkeypatch):
    manager = FakeBucketManager("rejected")
    manager.bucket["metadata"]["status_before_reject"] = "confirmed"
    manager.bucket["metadata"]["resolved"] = True
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_review("memory-1", "restore"))

    assert result["ok"] is True
    assert result["memory_status"] == "confirmed"
    assert manager.bucket["metadata"]["resolved"] is False
    assert server._curator_recallable(manager.bucket["metadata"]) is True


@pytest.mark.asyncio
async def test_restore_is_idempotent_for_an_already_active_memory(monkeypatch):
    manager = FakeBucketManager("confirmed")
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_review("memory-1", "restore"))

    assert result == {
        "ok": True,
        "bucket_id": "memory-1",
        "memory_status": "confirmed",
        "duplicate": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["pinned", "protected"])
async def test_core_memory_cannot_be_removed_accidentally(monkeypatch, guard):
    manager = FakeBucketManager("confirmed", **{guard: True})
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_review("memory-1", "reject"))

    assert result["ok"] is False
    assert "核心记忆" in result["error"]
    assert manager.bucket["metadata"]["memory_status"] == "confirmed"


@pytest.mark.asyncio
async def test_herbier_can_include_rejected_pages_only_for_recovery(monkeypatch):
    class CatalogueManager:
        async def list_all(self, include_archive=False):
            return [
                {
                    "id": "active",
                    "content": "仍在使用。",
                    "metadata": {"name": "有效页", "memory_status": "confirmed", "created": "2026-08-01"},
                },
                {
                    "id": "removed",
                    "content": "保留底稿。",
                    "metadata": {"name": "已移除页", "memory_status": "rejected", "created": "2026-07-31"},
                },
            ]

    monkeypatch.setattr(server, "bucket_mgr", CatalogueManager())

    active = json.loads(await server.herbier(limit=20))
    recovery = json.loads(await server.herbier(limit=20, include_rejected=True))

    assert [page["id"] for page in active["pages"]] == ["active"]
    assert {page["id"] for page in recovery["pages"]} == {"active", "removed"}
