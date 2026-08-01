import json
import sqlite3

import frontmatter

from inventory import build_inventory


def _write_bucket(root, layer, name, *, bucket_id, content, **metadata):
    directory = root / layer / "测试"
    directory.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(content, id=bucket_id, name=name, **metadata)
    path = directory / f"{name}_{bucket_id}.md"
    path.write_text(frontmatter.dumps(post), encoding="utf-8")


def test_inventory_reports_sources_status_raw_transcripts_and_review_groups(tmp_path):
    buckets = tmp_path / "buckets"
    for layer in ("permanent", "dynamic", "feel", "archive"):
        (buckets / layer).mkdir(parents=True)

    _write_bucket(
        buckets,
        "dynamic",
        "共同约定",
        bucket_id="a1",
        content="周末一起整理照片。",
        memory_status="candidate",
        source_kind="memory_secretary",
        source_session_id="session-1",
        confidence=0.8,
    )
    _write_bucket(
        buckets,
        "dynamic",
        "共同约定-副本",
        bucket_id="a2",
        content="周末一起整理照片。",
        memory_status="confirmed",
        source_kind="kelo_home",
        confidence=0.4,
    )
    _write_bucket(
        buckets,
        "archive",
        "旧对话",
        bucket_id="a3",
        content="时间：2026/7/15 09:05:50\nClaire：先记下。\n珂洛：好。",
        memory_status="rejected",
    )
    _write_bucket(
        buckets,
        "permanent",
        "共同约定",
        bucket_id="a4",
        content="内容不同，但同名不等于重复。",
        source_kind="manual",
        pinned=True,
    )

    report = build_inventory(buckets)

    assert report["read_only"] is True
    assert report["counts"]["valid_records"] == 4
    assert report["counts"]["status"] == {"candidate": 1, "confirmed": 2, "rejected": 1}
    assert report["counts"]["raw_transcripts"] == 1
    assert report["counts"]["source_unknown"] == 1
    assert report["counts"]["low_confidence"] == 1
    assert report["counts"]["protected_or_pinned"] == 1
    assert report["counts"]["duplicate_content_groups"] == 1
    assert report["counts"]["same_name_review_groups"] == 1
    assert report["review_policy"]["same_name_is_not_duplicate"] is True
    assert report["review_policy"]["physical_delete_performed"] is False
    assert report["review_policy"]["automatic_merge_performed"] is False


def test_inventory_keeps_malformed_files_and_index_anomalies_visible(tmp_path):
    buckets = tmp_path / "buckets"
    (buckets / "dynamic").mkdir(parents=True)
    (buckets / "permanent").mkdir(parents=True)
    (buckets / "feel").mkdir(parents=True)
    (buckets / "archive").mkdir(parents=True)
    _write_bucket(
        buckets,
        "dynamic",
        "有效",
        bucket_id="known",
        content="正文",
        source_kind="manual",
    )
    (buckets / "dynamic" / "坏.md").write_text("---\n: bad: yaml\n---\n正文", encoding="utf-8")

    db_path = buckets / "embeddings.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE embeddings (bucket_id TEXT PRIMARY KEY, embedding TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            ("missing", json.dumps([0.1, 0.2]), "2026-08-01"),
        )
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            ("known", "not-json", "2026-08-01"),
        )
        connection.execute(
            "CREATE TABLE families (id TEXT PRIMARY KEY, member_ids TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO families VALUES (?, ?)",
            ("family-1", json.dumps(["known", "missing"])),
        )

    report = build_inventory(buckets)

    assert report["counts"]["malformed_files"] == 1
    assert report["index_anomalies"]["orphan_embedding_ids"] == ["missing"]
    assert report["index_anomalies"]["malformed_embedding_ids"] == ["known"]
    assert report["index_anomalies"]["orphan_family_members"] == [
        {"family_id": "family-1", "member_id": "missing"}
    ]


def test_inventory_can_exclude_archive_without_mutating_files(tmp_path):
    buckets = tmp_path / "buckets"
    for layer in ("permanent", "dynamic", "feel", "archive"):
        (buckets / layer).mkdir(parents=True)
    _write_bucket(buckets, "dynamic", "当前", bucket_id="live", content="现在", source_kind="manual")
    _write_bucket(buckets, "archive", "旧", bucket_id="old", content="旧内容", source_kind="manual")

    before = (buckets / "archive" / "测试" / "旧_old.md").read_bytes()
    report = build_inventory(buckets, include_archive=False)
    after = (buckets / "archive" / "测试" / "旧_old.md").read_bytes()

    assert report["counts"]["valid_records"] == 1
    assert after == before
