import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from archivist import MemoryArchivist


class FakeBucketManager:
    def __init__(self, buckets):
        self.buckets = {bucket["id"]: deepcopy(bucket) for bucket in buckets}

    async def list_all(self, include_archive=False):
        return [deepcopy(bucket) for bucket in self.buckets.values()]

    async def get(self, bucket_id):
        bucket = self.buckets.get(bucket_id)
        return deepcopy(bucket) if bucket else None

    async def update(self, bucket_id, **updates):
        bucket = self.buckets[bucket_id]
        bucket.setdefault("metadata", {}).update(updates)
        return True


class FakeCompletions:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {"decisions": self.decisions}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))],
            usage=SimpleNamespace(prompt_tokens=240, completion_tokens=60),
        )


class FakeDehydrator:
    def __init__(self, decisions):
        self.completions = FakeCompletions(decisions)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))


class SequencedCompletions:
    def __init__(self, decisions_by_call):
        self.decisions_by_call = list(decisions_by_call)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        decisions = self.decisions_by_call.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"decisions": decisions}, ensure_ascii=False)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def bucket(bucket_id, content, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": metadata.pop("name", bucket_id),
            "memory_status": "confirmed",
            "memory_layer": "active",
            "recall_policy": "normal",
            "importance": 5,
            "type": "dynamic",
            **metadata,
        },
    }


async def wait_terminal(runner, job_id):
    for _ in range(100):
        current = runner.get(job_id)
        if current["status"] not in {"running"}:
            return current
        await asyncio.sleep(0.01)
    raise AssertionError("archivist job did not finish")


@pytest.mark.asyncio
async def test_archivist_archives_exact_duplicates_and_keeps_hard_guards(tmp_path):
    buckets = [
        bucket("keeper", "完全一样的普通记录", created="2026-01-02"),
        bucket("copy", "完全一样的普通记录", created="2026-01-01"),
        bucket("protected", "模型不能动我", protected=True),
        bucket("important", "这是一个重要承诺", importance=9),
        bucket("raw", "时间：2026/7/15 09:05:50\nClaire：你好\n珂洛：我在"),
        bucket("normal", "Claire 最近开始学习陶艺。"),
    ]
    manager = FakeBucketManager(buckets)
    dehydrator = FakeDehydrator([
        {"id": "normal", "action": "keep", "confidence": 0.97, "reason": "稳定偏好"},
    ])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(bucket_id, decision, **kwargs):
        meta = manager.buckets[bucket_id]["metadata"]
        if decision == "reject":
            meta.update(memory_status="rejected", memory_layer="archive", recall_policy="hidden")
        elif decision == "restore":
            meta.update(memory_status="confirmed", memory_layer="active", recall_policy="normal")
        return json.dumps({"ok": True, "bucket_id": bucket_id})

    started = await runner.start(review_handler)
    finished = await wait_terminal(runner, started["id"])

    assert finished["status"] == "completed"
    assert finished["archived"] == 1
    assert finished["evidence_only"] == 1
    assert manager.buckets["copy"]["metadata"]["memory_status"] == "rejected"
    assert manager.buckets["raw"]["metadata"]["memory_layer"] == "evidence"
    assert manager.buckets["protected"]["metadata"]["memory_status"] == "confirmed"
    assert manager.buckets["important"]["metadata"]["memory_status"] == "confirmed"
    assert {item["id"] for item in dehydrator.completions.decisions} == {"normal"}
    assert (tmp_path / "archivist" / "audit.jsonl").is_file()


@pytest.mark.asyncio
async def test_archivist_never_uses_same_name_as_an_archive_reason(tmp_path):
    manager = FakeBucketManager([
        bucket("first", "第一件不同的事", name="同名记忆"),
        bucket("second", "第二件不同的事", name="同名记忆"),
    ])
    dehydrator = FakeDehydrator([])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(**kwargs):
        raise AssertionError("same-name records must not be archived")

    started = await runner.start(review_handler)
    finished = await wait_terminal(runner, started["id"])

    assert finished["review"] == 2
    assert finished["archived"] == 0
    assert dehydrator.completions.calls == []


@pytest.mark.asyncio
async def test_archivist_whole_job_restore_replays_only_changed_records(tmp_path):
    manager = FakeBucketManager([
        bucket("keeper", "完全一样"),
        bucket("copy", "完全一样"),
        bucket("raw", "时间：2026/7/15 09:05:50\nClaire：你好\n珂洛：我在"),
    ])
    dehydrator = FakeDehydrator([])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(bucket_id, decision, **kwargs):
        meta = manager.buckets[bucket_id]["metadata"]
        if decision == "reject":
            meta.update(memory_status="rejected", memory_layer="archive", recall_policy="hidden")
        elif decision == "restore":
            meta.update(memory_status="confirmed", memory_layer="active", recall_policy="normal")
        return json.dumps({"ok": True, "bucket_id": bucket_id})

    started = await runner.start(review_handler)
    await wait_terminal(runner, started["id"])
    restored = await runner.restore(started["id"], review_handler)

    assert restored["status"] == "restored"
    assert restored["restored"] == 2
    assert manager.buckets["copy"]["metadata"]["memory_status"] == "confirmed"
    assert manager.buckets["raw"]["metadata"]["memory_layer"] == "active"


@pytest.mark.asyncio
async def test_archivist_uses_pro_only_to_recheck_flash_boundaries(tmp_path):
    manager = FakeBucketManager([bucket("uncertain", "一次没有事实内容的寒暄占位。")])
    completions = SequencedCompletions([
        [{"id": "uncertain", "action": "review", "confidence": 0.6, "reason": "需要复核"}],
        [{"id": "uncertain", "action": "archive", "confidence": 0.97, "reason": "明确无效占位"}],
    ])
    dehydrator = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(bucket_id, decision, **kwargs):
        assert decision == "reject"
        manager.buckets[bucket_id]["metadata"].update(memory_status="rejected", memory_layer="archive", recall_policy="hidden")
        return json.dumps({"ok": True, "bucket_id": bucket_id})

    started = await runner.start(review_handler)
    finished = await wait_terminal(runner, started["id"])

    assert finished["archived"] == 1
    assert [call["model"] for call in completions.calls] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert finished["usage"]["review_requests"] == 1


def test_worker_save_cannot_overwrite_a_concurrent_pause_request(tmp_path):
    manager = FakeBucketManager([])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, FakeDehydrator([]))
    job = {
        "id": "pause-race",
        "status": "running",
        "pause_requested": False,
        "pause_reason": "",
    }
    runner._save_job(job)

    paused = runner._load_job(job["id"])
    paused["pause_requested"] = True
    paused["pause_reason"] = "用户暂停"
    runner._save_job(paused)

    stale_worker_copy = {**job, "processed": 1}
    runner._save_job(stale_worker_copy)

    saved = runner._load_job(job["id"])
    assert saved["pause_requested"] is True
    assert saved["pause_reason"] == "用户暂停"
