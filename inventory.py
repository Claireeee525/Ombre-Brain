"""Read-only memory inventory for Ombre Brain.

The inventory deliberately reads Markdown and derived SQLite indexes without
changing either one.  It is meant to produce a reviewable snapshot before any
cleanup or status change is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import frontmatter

from memory_layers import normalize_layer_metadata


STORAGE_DIRS = ("permanent", "dynamic", "feel", "archive")
SOURCE_FIELDS = (
    "source_kind",
    "source_surface",
    "source_session_id",
    "source_message_ids",
    "source_fingerprint",
)
RAW_TRANSCRIPT_HEAD_RE = re.compile(r"^\s*时间\s*[：:]\s*\d{4}\s*[/年-]", re.M)
RAW_TRANSCRIPT_SPEAKER_RE = re.compile(
    r"(^|\n)\s*(Claire|珂洛|爸爸|Kael|Calder|用户)\s*[：:]",
    re.M,
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def _content_hash(content: str) -> str:
    normalized = content.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_transcript_reason(content: str) -> str:
    if RAW_TRANSCRIPT_HEAD_RE.search(content):
        return "timestamped_transcript_head"
    speakers = RAW_TRANSCRIPT_SPEAKER_RE.findall(content)
    if len(speakers) >= 2:
        return "multiple_speaker_lines"
    return ""


def _source_known(metadata: dict[str, Any]) -> bool:
    for field in SOURCE_FIELDS:
        value = metadata.get(field)
        if isinstance(value, (list, tuple, set)):
            if any(str(item).strip() for item in value):
                return True
        elif str(value or "").strip() and str(value).strip().lower() not in {
            "legacy",
            "manual_or_legacy",
            "unknown",
        }:
            return True
    return False


def _confidence_flags(metadata: dict[str, Any]) -> tuple[float | None, bool, bool]:
    raw = metadata.get("confidence")
    if raw in (None, ""):
        return None, False, True
    try:
        value = max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return None, False, True
    return value, value < 0.5, False


def _safe_preview(content: str, limit: int = 180) -> str:
    return _normalized_text(content)[:limit]


def _record_from_post(path: Path, root: Path, post: Any) -> dict[str, Any]:
    metadata = dict(post.metadata)
    content = str(post.content or "")
    layer_metadata = normalize_layer_metadata(metadata, content)
    bucket_id = str(metadata.get("id") or path.stem)
    status = str(metadata.get("memory_status") or "confirmed")
    confidence, low_confidence, missing_confidence = _confidence_flags(metadata)
    raw_reason = _raw_transcript_reason(content)
    source_known = _source_known(metadata)
    return {
        "id": bucket_id,
        "name": str(metadata.get("name") or bucket_id),
        "path": str(path.relative_to(root)),
        "storage_layer": path.relative_to(root).parts[0] if path.relative_to(root).parts else "",
        "bytes": path.stat().st_size,
        "file_sha256": _file_hash(path),
        "content_sha256": _content_hash(content),
        "content_preview": _safe_preview(content),
        "created": metadata.get("created", ""),
        "last_active": metadata.get("last_active", ""),
        "memory_status": status,
        "memory_layer": layer_metadata["memory_layer"],
        "recall_policy": layer_metadata["recall_policy"],
        "expired": layer_metadata["expired"],
        "expires_at": metadata.get("expires_at", ""),
        "source_kind": str(metadata.get("source_kind") or "legacy"),
        "source_surface": str(metadata.get("source_surface") or ""),
        "source_known": source_known,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "missing_confidence": missing_confidence,
        "raw_transcript": bool(raw_reason),
        "raw_transcript_reason": raw_reason,
        "pinned": bool(metadata.get("pinned")),
        "protected": bool(metadata.get("protected")),
        "resolved": bool(metadata.get("resolved")),
        "tags": list(metadata.get("tags") or [])[:12],
        "domain": list(metadata.get("domain") or [])[:4],
    }


def _scan_buckets(buckets_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for storage_dir in STORAGE_DIRS:
        directory = buckets_dir / storage_dir
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.md")):
            try:
                post = frontmatter.load(str(path))
                records.append(_record_from_post(path, buckets_dir, post))
            except Exception as exc:
                malformed.append({
                    "path": str(path.relative_to(buckets_dir)),
                    "file_sha256": _file_hash(path),
                    "bytes": path.stat().st_size,
                    "error": str(exc)[:320],
                })
    return records, malformed


def _read_sqlite_indexes(db_path: Path, bucket_ids: set[str]) -> dict[str, Any]:
    empty = {
        "path": str(db_path),
        "present": db_path.exists(),
        "read_only": True,
        "embedding_count": 0,
        "orphan_embedding_ids": [],
        "malformed_embedding_ids": [],
        "family_count": 0,
        "orphan_family_members": [],
        "duplicate_family_memberships": [],
        "error": "",
    }
    if not db_path.exists():
        return empty

    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "embeddings" in tables:
                rows = connection.execute(
                    "SELECT bucket_id, embedding FROM embeddings"
                ).fetchall()
                empty["embedding_count"] = len(rows)
                for bucket_id, raw_embedding in rows:
                    bucket_id = str(bucket_id)
                    if bucket_id not in bucket_ids:
                        empty["orphan_embedding_ids"].append(bucket_id)
                    try:
                        embedding = json.loads(raw_embedding)
                        if not isinstance(embedding, list) or not embedding:
                            raise ValueError("embedding is not a non-empty list")
                        if not all(isinstance(value, (int, float)) for value in embedding):
                            raise ValueError("embedding contains non-numeric values")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        empty["malformed_embedding_ids"].append(bucket_id)

            if "families" in tables:
                rows = connection.execute(
                    "SELECT id, member_ids FROM families"
                ).fetchall()
                empty["family_count"] = len(rows)
                memberships: defaultdict[str, list[str]] = defaultdict(list)
                for family_id, raw_member_ids in rows:
                    try:
                        member_ids = json.loads(raw_member_ids)
                        if not isinstance(member_ids, list):
                            raise ValueError("member_ids is not a list")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        empty["orphan_family_members"].append({
                            "family_id": str(family_id),
                            "member_id": "<malformed_member_ids>",
                        })
                        continue
                    for member_id in member_ids:
                        member_id = str(member_id)
                        memberships[member_id].append(str(family_id))
                        if member_id not in bucket_ids:
                            empty["orphan_family_members"].append({
                                "family_id": str(family_id),
                                "member_id": member_id,
                            })
                empty["duplicate_family_memberships"] = [
                    {"member_id": member_id, "family_ids": family_ids}
                    for member_id, family_ids in memberships.items()
                    if len(family_ids) > 1
                ]
    except sqlite3.Error as exc:
        empty["error"] = str(exc)[:320]
    return empty


def _groups(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = str(record.get(key) or "").strip()
        if value:
            grouped[value].append(record)
    return [
        {
            "key": value,
            "count": len(items),
            "ids": [item["id"] for item in items],
            "paths": [item["path"] for item in items],
        }
        for value, items in sorted(grouped.items())
        if len(items) > 1
    ]


def build_inventory(buckets_dir: str | Path, *, include_archive: bool = True) -> dict[str, Any]:
    """Build a JSON-serializable, read-only inventory report."""
    root = Path(buckets_dir).expanduser().resolve()
    scan_root = root if include_archive else root
    records, malformed = _scan_buckets(scan_root)
    if not include_archive:
        records = [record for record in records if record["storage_layer"] != "archive"]

    status_counts = Counter(record["memory_status"] for record in records)
    layer_counts = Counter(record["memory_layer"] for record in records)
    source_counts = Counter(record["source_kind"] for record in records)
    id_groups = _groups(records, "id")
    content_groups = _groups(records, "content_sha256")
    name_groups = _groups(
        [{**record, "name": _normalized_text(record["name"]).casefold()} for record in records],
        "name",
    )
    bucket_ids = {record["id"] for record in records}
    index_report = _read_sqlite_indexes(root / "embeddings.db", bucket_ids)

    raw_records = [record for record in records if record["raw_transcript"]]
    unknown_records = [record for record in records if not record["source_known"]]
    low_confidence_records = [record for record in records if record["low_confidence"]]
    protected_records = [record for record in records if record["pinned"] or record["protected"]]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "buckets_dir": str(root),
        "counts": {
            "total_files": len(records) + len(malformed),
            "valid_records": len(records),
            "malformed_files": len(malformed),
            "archived_records": sum(record["storage_layer"] == "archive" for record in records),
            "status": dict(sorted(status_counts.items())),
            "memory_layer": dict(sorted(layer_counts.items())),
            "source_kind": dict(sorted(source_counts.items())),
            "raw_transcripts": len(raw_records),
            "source_unknown": len(unknown_records),
            "low_confidence": len(low_confidence_records),
            "missing_confidence": sum(record["missing_confidence"] for record in records),
            "protected_or_pinned": len(protected_records),
            "rejected": status_counts.get("rejected", 0),
            "duplicate_content_groups": len(content_groups),
            "duplicate_content_records": sum(group["count"] for group in content_groups),
            "same_name_review_groups": len(name_groups),
            "same_name_review_records": sum(group["count"] for group in name_groups),
            "duplicate_id_groups": len(id_groups),
        },
        "records": records,
        "raw_transcript_records": raw_records,
        "source_unknown_records": unknown_records,
        "low_confidence_records": low_confidence_records,
        "protected_records": protected_records,
        "malformed_files": malformed,
        "duplicate_content_groups": content_groups,
        "same_name_review_groups": name_groups,
        "duplicate_id_groups": id_groups,
        "index_anomalies": index_report,
        "review_policy": {
            "same_name_is_not_duplicate": True,
            "physical_delete_performed": False,
            "automatic_merge_performed": False,
            "next_action": "review IDs before any status change",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Ombre memory inventory")
    parser.add_argument("--buckets-dir", required=True, help="Ombre buckets directory")
    parser.add_argument(
        "--exclude-archive",
        action="store_true",
        help="Exclude archive/ from the report (default includes it)",
    )
    args = parser.parse_args(argv)
    report = build_inventory(args.buckets_dir, include_archive=not args.exclude_archive)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
