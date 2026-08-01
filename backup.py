"""Create and verify recoverable Ombre Brain backups.

Backups are explicit archives of the Markdown source buckets and SQLite
indexes.  Verification checks the archive's manifest, hashes every member,
and optionally extracts into a temporary directory to prove restoration
without touching the live buckets directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


STORAGE_DIRS = ("permanent", "dynamic", "feel", "archive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root: Path, *, include_archive: bool) -> Iterable[Path]:
    layers = STORAGE_DIRS if include_archive else STORAGE_DIRS[:-1]
    for layer in layers:
        directory = root / layer
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.md")):
            if path.is_file() and not path.is_symlink():
                yield path
    # Indexes are derived data but are included so a restore can be checked
    # against the exact source state that produced them.
    for path in sorted(root.glob("*.db")):
        if path.is_file() and not path.is_symlink():
            yield path


def _manifest(root: Path, files: list[Path], *, include_archive: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backup_type": "ombre_brain_source_and_indexes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_buckets_dir": str(root),
        "include_archive": include_archive,
        "files": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }


def create_backup(
    buckets_dir: str | Path,
    output_dir: str | Path,
    *,
    include_archive: bool = True,
    label: str = "",
) -> dict[str, Any]:
    """Create a tar.gz plus sidecar manifest and return its receipt."""
    root = Path(buckets_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = list(_source_files(root, include_archive=include_archive))
    manifest = _manifest(root, files, include_archive=include_archive)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{label.strip()}" if label.strip() else ""
    archive_path = destination / f"ombre-backup-{stamp}{suffix}.tar.gz"
    manifest_path = destination / f"ombre-backup-{stamp}{suffix}.manifest.json"
    counter = 2
    while archive_path.exists() or manifest_path.exists():
        archive_path = destination / f"ombre-backup-{stamp}{suffix}-{counter}.tar.gz"
        manifest_path = destination / f"ombre-backup-{stamp}{suffix}-{counter}.manifest.json"
        counter += 1

    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        import io

        archive.addfile(info, io.BytesIO(manifest_bytes))
        for path in files:
            relative = str(path.relative_to(root)).replace("\\", "/")
            archive.add(path, arcname=f"data/{relative}", recursive=False)

    manifest_path.write_bytes(manifest_bytes)
    return {
        "ok": True,
        "read_only_source": True,
        "archive": str(archive_path),
        "manifest": str(manifest_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
        "file_count": len(files),
        "include_archive": include_archive,
        "generated_at": manifest["generated_at"],
    }


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def verify_backup(archive_path: str | Path, *, restore_test: bool = True) -> dict[str, Any]:
    """Verify archive contents and optionally extract into a temporary folder."""
    path = Path(archive_path).expanduser().resolve()
    result: dict[str, Any] = {
        "ok": False,
        "archive": str(path),
        "archive_bytes": path.stat().st_size if path.exists() else 0,
        "archive_sha256": _sha256(path) if path.exists() else "",
        "restore_tested": False,
        "file_count": 0,
        "errors": [],
    }
    if not path.exists():
        result["errors"].append("archive_not_found")
        return result

    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            unsafe = [member.name for member in members if not _safe_member_name(member.name)]
            if unsafe:
                result["errors"].append({"unsafe_members": unsafe[:10]})
                return result
            manifest_member = next((member for member in members if member.name == "manifest.json"), None)
            if manifest_member is None:
                result["errors"].append("manifest_missing")
                return result
            manifest = json.load(archive.extractfile(manifest_member))
            expected = {f"data/{item['path']}": item for item in manifest.get("files", [])}
            actual = {
                member.name: member
                for member in members
                if member.name.startswith("data/") and member.isfile()
            }
            if set(expected) != set(actual):
                result["errors"].append({
                    "member_set_mismatch": {
                        "missing": sorted(set(expected) - set(actual))[:10],
                        "unexpected": sorted(set(actual) - set(expected))[:10],
                    }
                })
                return result

            for name, entry in expected.items():
                member = actual[name]
                if member.size != entry.get("bytes"):
                    result["errors"].append({"size_mismatch": name})
                    continue
                extracted = archive.extractfile(member)
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != entry.get("sha256"):
                    result["errors"].append({"sha256_mismatch": name})

            if result["errors"]:
                return result

            if restore_test:
                with tempfile.TemporaryDirectory(prefix="ombre-backup-verify-") as temp_dir:
                    archive.extractall(temp_dir, members=[manifest_member, *actual.values()], filter="data")
                    for name, entry in expected.items():
                        restored = Path(temp_dir) / name
                        if not restored.is_file() or restored.stat().st_size != entry.get("bytes"):
                            result["errors"].append({"restore_mismatch": name})
                            continue
                        if _sha256(restored) != entry.get("sha256"):
                            result["errors"].append({"restore_sha256_mismatch": name})
                result["restore_tested"] = True

            result["file_count"] = len(expected)
            result["manifest"] = manifest
            result["ok"] = not result["errors"]
            return result
    except (OSError, tarfile.TarError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result["errors"].append(str(exc)[:320])
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and verify Ombre Brain backups")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--buckets-dir", required=True)
    create_parser.add_argument("--output-dir", required=True)
    create_parser.add_argument("--exclude-archive", action="store_true")
    create_parser.add_argument("--label", default="")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive")
    verify_parser.add_argument("--no-restore-test", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "create":
        receipt = create_backup(
            args.buckets_dir,
            args.output_dir,
            include_archive=not args.exclude_archive,
            label=args.label,
        )
        receipt["verification"] = verify_backup(receipt["archive"], restore_test=True)
    else:
        receipt = verify_backup(args.archive, restore_test=not args.no_restore_test)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt.get("ok") and receipt.get("verification", {"ok": True}).get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
