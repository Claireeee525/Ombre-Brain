import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import frontmatter

from archivist import MemoryArchivist
from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine


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

    async def create(self, content, name=None, extra_metadata=None, **kwargs):
        bucket_id = f"canonical-{len(self.buckets) + 1}"
        self.buckets[bucket_id] = bucket(
            bucket_id,
            content,
            name=name or bucket_id,
            tags=kwargs.get("tags") or [],
            domain=kwargs.get("domain") or ["未分类"],
            importance=kwargs.get("importance") or 5,
            **(extra_metadata or {}),
        )
        return bucket_id


class LegacyLocatorBucketManager(FakeBucketManager):
    async def get(self, bucket_id):
        if bucket_id == "legacy-file-name":
            return deepcopy(self.buckets.get("frontmatter-id"))
        if bucket_id == "frontmatter-id":
            return None
        return await super().get(bucket_id)


class SnapshotOnlyBucketManager(FakeBucketManager):
    async def get(self, bucket_id):
        return None


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


class FakeSimilarityEngine:
    def __init__(self, groups):
        self.groups = groups
        self.generated = []

    def find_related_groups(self, bucket_ids, **kwargs):
        allowed = set(bucket_ids)
        return [group for group in self.groups if set(group["ids"]) <= allowed]

    async def generate_and_store(self, bucket_id, content):
        self.generated.append((bucket_id, content))
        return True


class ConsolidationCompletions:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        user_prompt = kwargs["messages"][-1]["content"]
        if "长期记忆整合员" in user_prompt:
            payload = {"groups": [self.proposal]}
        else:
            payload = {"decisions": []}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))],
            usage=SimpleNamespace(prompt_tokens=180, completion_tokens=80),
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


@pytest.mark.asyncio
async def test_archivist_uses_filename_locator_for_legacy_frontmatter_ids(tmp_path):
    legacy = bucket("frontmatter-id", "一次无事实内容的寒暄占位。")
    legacy["path"] = "/vault/legacy-file-name.md"
    manager = LegacyLocatorBucketManager([legacy])
    dehydrator = FakeDehydrator([
        {"id": "frontmatter-id", "action": "archive", "confidence": 0.98, "reason": "明确无效占位"},
    ])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)
    calls = []

    async def review_handler(bucket_id, decision, **kwargs):
        calls.append((bucket_id, decision))
        meta = manager.buckets["frontmatter-id"]["metadata"]
        if decision == "reject":
            meta.update(memory_status="rejected", memory_layer="archive", recall_policy="hidden")
        elif decision == "restore":
            meta.update(memory_status="confirmed", memory_layer="active", recall_policy="normal")
        return json.dumps({"ok": True})

    started = await runner.start(review_handler)
    finished = await wait_terminal(runner, started["id"])
    assert finished["status"] == "completed"
    assert finished["failed"] == 0
    assert calls == [("legacy-file-name", "reject")]

    restored = await runner.restore(started["id"], review_handler)
    assert restored["status"] == "restored"
    assert calls[-1] == ("legacy-file-name", "restore")


@pytest.mark.asyncio
async def test_bucket_manager_resolves_legacy_frontmatter_id_not_in_filename(tmp_path):
    buckets_dir = tmp_path / "buckets"
    legacy_path = buckets_dir / "dynamic" / "旧导入" / "人机恋焦虑_legacy-internal-id.md"
    legacy_path.parent.mkdir(parents=True)
    (legacy_path.parent / f"._{legacy_path.name}").write_bytes(b"\x00\x05\x16\x07appledouble")
    legacy_path.write_text(
        frontmatter.dumps(frontmatter.Post("旧记录正文", id="legacy-internal-id", importance=5)),
        encoding="utf-8",
    )
    manager = BucketManager({"buckets_dir": str(buckets_dir), "matching": {}})

    loaded = await manager.get("legacy-internal-id")
    assert loaded["content"] == "旧记录正文"
    assert await manager.update("legacy-internal-id", memory_layer="evidence") is True
    assert (await manager.get("legacy-internal-id"))["metadata"]["memory_layer"] == "evidence"


@pytest.mark.asyncio
async def test_read_only_keep_uses_catalogue_snapshot_without_reopening_file(tmp_path):
    manager = SnapshotOnlyBucketManager([bucket("snapshot-only", "应当保留的普通事实")])
    dehydrator = FakeDehydrator([
        {"id": "snapshot-only", "action": "keep", "confidence": 0.99, "reason": "稳定事实"},
    ])
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(**kwargs):
        raise AssertionError("read-only keep must not mutate")

    started = await runner.start(review_handler, dry_run=True)
    finished = await wait_terminal(runner, started["id"])
    assert finished["status"] == "completed"
    assert finished["processed"] == 1
    assert finished["failed"] == 0


@pytest.mark.asyncio
async def test_archivist_consolidates_cross_surface_preferences_and_restores_sources(tmp_path):
    manager = FakeBucketManager([
        bucket(
            "kelo-pref", "Claire 喜欢亲密回应写得自然直接，不要客服腔。",
            name="亲密回应偏好", source_surface="Kelo Home", signed_by=["Kelo"], participants=["Claire", "Kelo"],
        ),
        bucket(
            "official-pref", "亲密场景里要有真实情绪和动作细节，避免模板化措辞。",
            name="亲密写作要求", source_surface="Claude 官方端", signed_by=["Calder"], participants=["Claire", "Calder"],
        ),
    ])
    manager.embedding_engine = FakeSimilarityEngine([
        {"ids": ["kelo-pref", "official-pref"], "similarity": 0.91},
    ])
    completions = ConsolidationCompletions({
        "group_id": "placeholder",
        "decision": "merge",
        "title": "亲密回应与写作偏好",
        "content": "Claire 希望亲密回应自然直接，有真实情绪与动作细节，避免客服腔和模板化措辞。",
        "topic": "亲密偏好",
        "confidence": 0.96,
        "reason": "两条是同一项稳定偏好的互补表述",
    })

    async def create_with_group(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        match = __import__("re").search(r'"group_id":"([a-f0-9]+)"', prompt)
        completions.proposal["group_id"] = match.group(1)
        return await ConsolidationCompletions.create(completions, **kwargs)

    completions.create = create_with_group
    dehydrator = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    runner = MemoryArchivist({"buckets_dir": str(tmp_path), "archivist": {}}, manager, dehydrator)

    async def review_handler(bucket_id, decision, **kwargs):
        meta = manager.buckets[bucket_id]["metadata"]
        if decision in {"reject", "supersede"}:
            meta.update(memory_status="rejected", memory_layer="archive", recall_policy="hidden")
        elif decision == "restore":
            meta.update(memory_status="confirmed", memory_layer="active", recall_policy="normal")
        return json.dumps({"ok": True, "bucket_id": bucket_id})

    started = await runner.start(review_handler)
    finished = await wait_terminal(runner, started["id"])

    assert finished["merged_groups"] == 1
    assert finished["merged_sources"] == 2
    merge = next(item for item in finished["actions"] if item["action"] == "merge")
    assert merge["canonical_content"].startswith("Claire 希望亲密回应")
    assert {item["surface"] for item in merge["sources"]} == {"Kelo Home", "Claude 官方端"}
    canonical = manager.buckets[merge["canonical_id"]]
    assert canonical["metadata"]["consolidated_from"] == ["kelo-pref", "official-pref"]
    assert canonical["metadata"]["source_surfaces"] == ["Kelo Home", "Claude 官方端"]
    assert manager.buckets["kelo-pref"]["metadata"]["memory_status"] == "rejected"

    restored = await runner.restore(started["id"], review_handler)
    assert restored["status"] == "restored"
    assert manager.buckets["kelo-pref"]["metadata"]["memory_status"] == "confirmed"
    assert manager.buckets["official-pref"]["metadata"]["memory_status"] == "confirmed"
    assert manager.buckets[merge["canonical_id"]]["metadata"]["memory_status"] == "rejected"


def test_local_embedding_candidates_are_grouped_without_api_calls(tmp_path):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "dehydration": {},
        "embedding": {"enabled": False},
    })
    engine._store_embedding("kelo", [1.0, 0.0, 0.0])
    engine._store_embedding("official", [0.99, 0.1, 0.0])
    engine._store_embedding("unrelated", [0.0, 1.0, 0.0])

    groups = engine.find_related_groups(
        ["kelo", "official", "unrelated"], threshold=0.9,
    )

    assert len(groups) == 1
    assert set(groups[0]["ids"]) == {"kelo", "official"}
    assert groups[0]["similarity"] > 0.99
