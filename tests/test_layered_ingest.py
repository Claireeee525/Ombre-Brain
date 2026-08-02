import json
import sqlite3

import pytest


@pytest.mark.asyncio
async def test_grow_persists_raw_evidence_and_candidate_without_merging(test_config, monkeypatch):
    import server
    from bucket_manager import BucketManager

    manager = BucketManager(test_config)

    class FakeDehydrator:
        api_available = False

        async def analyze(self, content):
            return {"domain": ["相处"], "valence": 0.7, "arousal": 0.4, "tags": ["测试"], "suggested_name": "一次对话"}

        async def digest(self, content):
            return [{
                "content": "Claire 确认了这个安排。",
                "name": "安排确认",
                "tags": ["约定"],
                "importance": 6,
                "domain": ["约定"],
                "valence": 0.8,
                "arousal": 0.3,
            }]

    class FakeEmbedding:
        async def generate_and_store(self, *args, **kwargs):
            return True

    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server.decay_engine, "ensure_started", lambda: _done())
    monkeypatch.setattr(server, "config", test_config)

    raw = "时间：2026/08/02 20:00\nClaire：我们把这个安排定下来了。\n珂洛：我记住了。"
    result = await server.grow(raw, source_surface="Claude 官方端", source_session_id="session-test")
    buckets = await manager.list_all(include_archive=True)
    layers = {bucket["metadata"].get("memory_layer") for bucket in buckets}

    assert "原文证据→" in result
    assert "待审候选→" in result
    assert layers == {"evidence", "candidate"}
    evidence = next(bucket for bucket in buckets if bucket["metadata"].get("memory_layer") == "evidence")
    candidate = next(bucket for bucket in buckets if bucket["metadata"].get("memory_layer") == "candidate")
    assert evidence["metadata"]["source_kind"] == "original_evidence"
    assert evidence["metadata"]["evidence_ranges"][0]["speaker"] == "Claire"
    assert candidate["metadata"]["source_evidence_id"] == evidence["id"]
    assert candidate["metadata"]["memory_status"] == "candidate"


@pytest.mark.asyncio
async def test_source_read_resolves_candidate_to_exact_evidence(test_config, monkeypatch):
    import server
    from bucket_manager import BucketManager

    manager = BucketManager(test_config)
    evidence_id = await manager.create(
        "时间：2026/08/02\nClaire：这句原话应该能被查到。",
        domain=["原文证据"],
        extra_metadata={
            "memory_layer": "evidence",
            "recall_policy": "exact_only",
            "source_kind": "original_evidence",
            "evidence_digest": "digest-1",
            "evidence_ranges": [{"message_id": "line:2", "speaker": "Claire", "start": 2, "end": 2}],
        },
    )
    candidate_id = await manager.create(
        "这条摘要不能冒充原话。",
        domain=["测试"],
        extra_metadata={
            "memory_layer": "candidate",
            "memory_status": "candidate",
            "source_evidence_id": evidence_id,
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)

    payload = json.loads(await server.source_read(bucket_id=candidate_id, message_id="line:2"))
    assert payload["ok"] is True
    assert payload["source_bucket_id"] == evidence_id
    assert payload["content"] == "Claire：这句原话应该能被查到。"
    assert payload["recall_policy"] == "exact_only"


@pytest.mark.asyncio
async def test_chinese_phrase_and_bigram_evidence_reaches_memory(test_config):
    from bucket_manager import BucketManager

    manager = BucketManager(test_config)
    await manager.create("小家记忆库的原文证据测试", name="小家证据", domain=["原文证据"])
    matches = await manager.search("记忆库", limit=5)
    assert matches
    assert matches[0]["metadata"]["name"] == "小家证据"


@pytest.mark.asyncio
async def test_embedding_failure_is_persisted_and_later_retried(test_config, monkeypatch):
    from embedding_engine import EmbeddingEngine

    config = dict(test_config)
    config["embedding"] = {
        "enabled": True,
        "api_key": "test-key",
        "base_url": "https://example.invalid",
        "model": "test-model",
    }
    engine = EmbeddingEngine(config)

    async def empty_embedding(_content):
        return []

    monkeypatch.setattr(engine, "_generate_embedding", empty_embedding)
    assert await engine.generate_and_store("bucket-queue", "待向量化的记忆") is False
    assert engine.queue_status()["pending"] == 1

    conn = sqlite3.connect(engine.db_path)
    conn.execute("UPDATE embedding_jobs SET next_retry_at = ? WHERE bucket_id = ?", ("1970-01-01T00:00:00+00:00", "bucket-queue"))
    conn.commit()
    conn.close()

    async def valid_embedding(_content):
        return [1.0, 0.0]

    monkeypatch.setattr(engine, "_generate_embedding", valid_embedding)
    result = await engine.retry_pending()
    assert result["succeeded"] == 1
    assert engine.queue_status()["pending"] == 0
    assert await engine.get_embedding("bucket-queue") == [1.0, 0.0]


async def _done():
    return None
