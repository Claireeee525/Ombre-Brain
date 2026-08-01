import json

from backup import create_backup, verify_backup


def _write_bucket(root, layer, name, content):
    path = root / layer / "测试" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_backup_verifies_hashes_and_restore_without_touching_source(tmp_path):
    buckets = tmp_path / "buckets"
    _write_bucket(buckets, "dynamic", "当前", "---\nid: current\n---\n正文")
    _write_bucket(buckets, "archive", "旧", "---\nid: old\n---\n旧正文")
    (buckets / "embeddings.db").write_bytes(b"index")

    receipt = create_backup(buckets, tmp_path / "backups", label="test")
    verification = verify_backup(receipt["archive"], restore_test=True)

    assert receipt["read_only_source"] is True
    assert receipt["file_count"] == 3
    assert verification["ok"] is True
    assert verification["restore_tested"] is True
    assert verification["file_count"] == 3
    assert json.loads((tmp_path / "backups" / receipt["manifest"].split("/")[-1]).read_text())[
        "include_archive"
    ] is True


def test_backup_can_exclude_archive_and_detect_tampering(tmp_path):
    buckets = tmp_path / "buckets"
    _write_bucket(buckets, "dynamic", "当前", "正文")
    _write_bucket(buckets, "archive", "旧", "旧正文")

    receipt = create_backup(buckets, tmp_path / "backups", include_archive=False)
    verification = verify_backup(receipt["archive"])
    assert verification["ok"] is True
    assert verification["file_count"] == 1

    archive_path = tmp_path / "tampered.tar.gz"
    archive_path.write_bytes(b"not-a-valid-tar")
    tampered = verify_backup(archive_path)
    assert tampered["ok"] is False
