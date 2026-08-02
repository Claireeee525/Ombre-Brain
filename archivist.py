"""Background AI archivist for the historical Ombre memory catalogue.

The archivist is deliberately conservative: deterministic guards run before and
after the model, source Markdown is never deleted, and every mutation has an
append-only receipt that can be replayed for a whole-job restore.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from memory_layers import RECALL_POLICIES, normalize_layer_metadata
from utils import count_tokens_approx


ReviewHandler = Callable[..., Awaitable[str]]

RAW_TRANSCRIPT_HEAD_RE = re.compile(r"^\s*时间\s*[：:]\s*\d{4}\s*[/年-]", re.M)
RAW_TRANSCRIPT_SPEAKER_RE = re.compile(
    r"(^|\n)\s*(Claire|珂洛|爸爸|Kael|Calder|用户)\s*[：:]", re.M
)
CORE_SAFETY_RE = re.compile(
    r"承诺|约定|答应|边界|底线|纪念日|生日|过敏|用药|药物|医院|病史|遗嘱|账号|密码",
    re.I,
)
ALLOWED_MODEL_ACTIONS = {"keep", "archive", "review", "evidence_only"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _content_hash(content: str) -> str:
    normalized = str(content or "").replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_error(error: Any) -> str:
    return re.sub(r"Bearer\s+[^\s]+", "Bearer [redacted]", str(error or "归档任务失败"))[:360]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class MemoryArchivist:
    """Persisted background task runner using Ombre's existing DeepSeek client."""

    def __init__(self, config: dict[str, Any], bucket_manager: Any, dehydrator: Any):
        self.config = config
        self.bucket_manager = bucket_manager
        self.dehydrator = dehydrator
        cfg = config.get("archivist", {})
        self.model = str(os.environ.get("OMBRE_ARCHIVIST_MODEL") or cfg.get("model") or "deepseek-v4-flash")
        self.review_model = str(
            os.environ.get("OMBRE_ARCHIVIST_REVIEW_MODEL")
            or cfg.get("review_model")
            or "deepseek-v4-pro"
        )
        self.default_batch_size = max(2, min(20, int(cfg.get("batch_size", 8))))
        self.archive_confidence = max(0.8, min(0.99, float(cfg.get("archive_confidence", 0.93))))
        self.merge_confidence = max(0.75, min(0.99, float(cfg.get("merge_confidence", 0.88))))
        self.similarity_threshold = max(0.65, min(0.95, float(cfg.get("similarity_threshold", 0.78))))
        self.max_input_tokens = max(
            10_000,
            int(os.environ.get("OMBRE_ARCHIVIST_MAX_INPUT_TOKENS") or cfg.get("max_input_tokens", 5_000_000)),
        )
        self.max_output_tokens = max(
            5_000,
            int(os.environ.get("OMBRE_ARCHIVIST_MAX_OUTPUT_TOKENS") or cfg.get("max_output_tokens", 500_000)),
        )
        self.input_price = float(cfg.get("input_usd_per_million", 0.14))
        self.output_price = float(cfg.get("output_usd_per_million", 0.28))
        self.review_input_price = float(cfg.get("review_input_usd_per_million", 0.435))
        self.review_output_price = float(cfg.get("review_output_usd_per_million", 0.87))
        self.root = Path(config["buckets_dir"]) / "archivist"
        self.jobs_dir = self.root / "jobs"
        self.audit_path = self.root / "audit.jsonl"
        self._tasks: dict[str, asyncio.Task] = {}
        self._index_tasks: set[asyncio.Task] = set()
        self._index_semaphore = asyncio.Semaphore(4)
        self._start_lock = asyncio.Lock()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._mark_interrupted_jobs()

    def _index_in_background(self, bucket_id: str, content: str) -> None:
        engine = getattr(self.bucket_manager, "embedding_engine", None)
        if not engine:
            return

        async def _run() -> None:
            async with self._index_semaphore:
                await engine.generate_and_store(bucket_id, content)

        task = asyncio.create_task(_run())
        self._index_tasks.add(task)
        task.add_done_callback(self._index_tasks.discard)

    def _job_path(self, job_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9-]", "", str(job_id or ""))
        return self.jobs_dir / f"{safe_id}.json"

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_path(job_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _save_job(self, job: dict[str, Any]) -> None:
        # A running worker saves after every record. Preserve a pause request
        # that may have landed between that record's read and write so the
        # worker cannot accidentally overwrite the user's stop signal.
        existing = self._load_job(str(job.get("id") or ""))
        if (
            job.get("status") == "running"
            and existing
            and existing.get("pause_requested")
        ):
            job["pause_requested"] = True
            job["pause_reason"] = existing.get("pause_reason") or "用户暂停"
        job["updated_at"] = _now_iso()
        _atomic_json(self._job_path(job["id"]), job)

    def _append_audit(self, event: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": _now_iso(), **event}, ensure_ascii=False) + "\n")

    def _mark_interrupted_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") == "running":
                job["status"] = "paused"
                job["pause_reason"] = "服务重启后安全暂停，可继续运行"
                job["pause_requested"] = False
                _atomic_json(path, job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._load_job(job_id)
        return self._public_job(job) if job else None

    def latest(self) -> dict[str, Any] | None:
        jobs = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                jobs.append(value)
        if not jobs:
            return None
        jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return self._public_job(jobs[0])

    def audit(self, limit: int = 100, job_id: str = "") -> list[dict[str, Any]]:
        if not self.audit_path.is_file():
            return []
        safe_limit = max(1, min(500, int(limit or 100)))
        items = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if job_id and item.get("job_id") != job_id:
                continue
            items.append(item)
        return items[-safe_limit:][::-1]

    def _public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        value = dict(job)
        value.pop("record_ids", None)
        value.pop("processed_ids", None)
        value.pop("decisions", None)
        actions = value.get("actions", [])
        value["actions"] = [
            {
                key: item.get(key)
                for key in (
                    "bucket_id", "action", "changed", "restored", "reason", "error",
                    "canonical_id", "canonical_title", "canonical_content", "topic",
                    "source_ids", "sources", "confidence", "similarity",
                )
                if item.get(key) not in (None, "")
            }
            for item in actions[-100:]
        ]
        return value

    async def start(
        self,
        review_handler: ReviewHandler,
        *,
        dry_run: bool = False,
        max_records: int = 0,
        batch_size: int = 0,
    ) -> dict[str, Any]:
        async with self._start_lock:
            for task in self._tasks.values():
                if not task.done():
                    raise RuntimeError("已有 AI 归档任务正在运行")
            if not self.dehydrator.client:
                raise RuntimeError("Ombre 的 DeepSeek API 未配置，请检查 OMBRE_API_KEY")

            buckets = await self.bucket_manager.list_all(include_archive=False)
            eligible = [bucket for bucket in buckets if self._is_active(bucket)]
            eligible.sort(key=lambda item: str(item.get("id") or ""))
            if max_records:
                eligible = eligible[:max(1, min(int(max_records), 10_000))]
            record_ids = [str(bucket["id"]) for bucket in eligible]
            estimated_input = sum(
                count_tokens_approx(str(bucket.get("content") or "")[:6000]) + 140 for bucket in eligible
            )
            estimated_output = len(eligible) * 90
            job_id = str(uuid.uuid4())
            job = {
                "schema_version": 1,
                "id": job_id,
                "status": "running",
                "dry_run": bool(dry_run),
                "model": self.model,
                "review_model": self.review_model,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "source_total": len(record_ids),
                "processed": 0,
                "kept": 0,
                "archived": 0,
                "merged_groups": 0,
                "merged_sources": 0,
                "evidence_only": 0,
                "review": 0,
                "failed": 0,
                "restored": 0,
                "estimated_input_tokens": estimated_input,
                "estimated_output_tokens": estimated_output,
                "estimated_cost_usd": round(
                    estimated_input / 1_000_000 * self.input_price
                    + estimated_output / 1_000_000 * self.output_price
                    + estimated_input * 0.15 / 1_000_000 * self.review_input_price
                    + estimated_output * 0.15 / 1_000_000 * self.review_output_price,
                    4,
                ),
                "usage": {
                    "input_tokens": 0, "output_tokens": 0, "requests": 0,
                    "review_input_tokens": 0, "review_output_tokens": 0, "review_requests": 0,
                },
                "limits": {
                    "max_input_tokens": self.max_input_tokens,
                    "max_output_tokens": self.max_output_tokens,
                },
                "batch_size": max(2, min(20, int(batch_size or self.default_batch_size))),
                "pause_requested": False,
                "pause_reason": "",
                "record_ids": record_ids,
                "processed_ids": [],
                "failed_ids": [],
                "decisions": [],
                "actions": [],
                "errors": [],
            }
            self._save_job(job)
            self._append_audit({"event": "job_created", "job_id": job_id, "dry_run": bool(dry_run), "records": len(record_ids)})
            task = asyncio.create_task(self._run(job_id, review_handler))
            self._tasks[job_id] = task
            task.add_done_callback(lambda _task, current_id=job_id: self._tasks.pop(current_id, None))
            return self._public_job(job)

    async def pause(self, job_id: str) -> dict[str, Any]:
        job = self._load_required(job_id)
        if job.get("status") not in {"running", "paused"}:
            raise RuntimeError("这批任务当前不能暂停")
        job["pause_requested"] = True
        job["pause_reason"] = "用户暂停"
        self._save_job(job)
        return self._public_job(job)

    async def retry(self, job_id: str, review_handler: ReviewHandler) -> dict[str, Any]:
        async with self._start_lock:
            for task in self._tasks.values():
                if not task.done():
                    raise RuntimeError("已有 AI 归档任务正在运行")
            job = self._load_required(job_id)
            if job.get("status") not in {"paused", "failed", "completed_with_errors", "budget_paused"}:
                raise RuntimeError("这批任务当前没有可重试的部分")
            failed_ids = set(job.get("failed_ids") or [])
            if failed_ids:
                job["processed_ids"] = [item for item in job.get("processed_ids", []) if item not in failed_ids]
            job["failed_ids"] = []
            job["failed"] = 0
            job["errors"] = []
            job["pause_requested"] = False
            job["pause_reason"] = ""
            job["status"] = "running"
            self._save_job(job)
            self._append_audit({"event": "job_retried", "job_id": job_id})
            task = asyncio.create_task(self._run(job_id, review_handler))
            self._tasks[job_id] = task
            task.add_done_callback(lambda _task, current_id=job_id: self._tasks.pop(current_id, None))
            return self._public_job(job)

    async def restore(self, job_id: str, review_handler: ReviewHandler) -> dict[str, Any]:
        job = self._load_required(job_id)
        if job.get("status") == "running":
            raise RuntimeError("请先暂停任务，再整批恢复")
        restored = set(job.get("restored_ids") or [])
        failures = []
        for action in reversed(job.get("actions", [])):
            bucket_id = str(action.get("bucket_id") or "")
            storage_id = str(action.get("storage_id") or bucket_id)
            restore_key = bucket_id or str(action.get("canonical_id") or "")
            if not action.get("changed") or not restore_key or restore_key in restored:
                continue
            try:
                if action.get("action") == "merge":
                    for source in action.get("source_receipts") or []:
                        if not source.get("changed"):
                            continue
                        raw = await review_handler(
                            bucket_id=str(source.get("storage_id") or source.get("bucket_id") or ""),
                            decision="restore",
                            actor="Ombre AI Archivist",
                            reason=f"restore_consolidation_job:{job_id}",
                            request_id=f"{job_id}:restore",
                        )
                        result = json.loads(raw)
                        if result.get("ok") is not True:
                            raise RuntimeError(result.get("error") or "Ombre 未确认来源恢复")
                    canonical_id = str(action.get("canonical_id") or "")
                    if canonical_id:
                        raw = await review_handler(
                            bucket_id=canonical_id,
                            decision="reject",
                            actor="Ombre AI Archivist",
                            reason=f"restore_consolidation_job:{job_id}",
                            request_id=f"{job_id}:restore:canonical",
                        )
                        result = json.loads(raw)
                        if result.get("ok") is not True:
                            raise RuntimeError(result.get("error") or "Ombre 未确认合并记忆撤回")
                elif action.get("action") == "archive":
                    raw = await review_handler(
                        bucket_id=storage_id,
                        decision="restore",
                        actor="Ombre AI Archivist",
                        reason=f"restore_archivist_job:{job_id}",
                        request_id=f"{job_id}:restore",
                    )
                    result = json.loads(raw)
                    if result.get("ok") is not True:
                        raise RuntimeError(result.get("error") or "Ombre 未确认恢复")
                elif action.get("action") == "evidence_only":
                    previous = action.get("previous") or {}
                    await self.bucket_manager.update(
                        storage_id,
                        memory_layer=previous.get("memory_layer") or "active",
                        recall_policy=previous.get("recall_policy") or RECALL_POLICIES["active"],
                    )
                restored.add(restore_key)
                action["restored"] = True
                self._append_audit({"event": "record_restored", "job_id": job_id, "bucket_id": restore_key})
            except Exception as exc:  # keep restoring the rest of the batch
                failures.append({"bucket_id": restore_key, "error": _safe_error(exc)})
        job["restored_ids"] = sorted(restored)
        job["restored"] = len(restored)
        job["restore_errors"] = failures[-100:]
        job["status"] = "restored" if not failures else "restore_with_errors"
        self._save_job(job)
        return self._public_job(job)

    def _load_required(self, job_id: str) -> dict[str, Any]:
        job = self._load_job(job_id)
        if not job:
            raise RuntimeError("找不到这批 AI 归档任务")
        return job

    async def _resolve_bucket(self, bucket: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Resolve imported records whose frontmatter id differs from the filename."""
        logical_id = str(bucket.get("id") or "")
        latest = await self.bucket_manager.get(logical_id)
        if latest:
            return logical_id, latest
        path = str(bucket.get("path") or "")
        storage_id = Path(path).stem if path else ""
        if storage_id and storage_id != logical_id:
            latest = await self.bucket_manager.get(storage_id)
            if latest:
                return storage_id, latest
        return logical_id, None

    def _is_active(self, bucket: dict[str, Any]) -> bool:
        metadata = bucket.get("metadata", {})
        if str(metadata.get("memory_status") or "confirmed") == "rejected":
            return False
        layer = normalize_layer_metadata(metadata, bucket.get("content", ""))["memory_layer"]
        return layer != "archive"

    def _hard_decision(self, bucket: dict[str, Any], same_name_ids: set[str]) -> dict[str, Any] | None:
        metadata = bucket.get("metadata", {})
        content = str(bucket.get("content") or "")
        bucket_id = str(bucket.get("id") or "")
        layer = normalize_layer_metadata(metadata, content)["memory_layer"]
        if metadata.get("pinned") or metadata.get("protected") or metadata.get("type") == "permanent":
            return {"id": bucket_id, "action": "keep", "confidence": 1.0, "reason": "受保护或永久记忆"}
        if int(metadata.get("importance") or 5) >= 8 or CORE_SAFETY_RE.search(content):
            return {"id": bucket_id, "action": "keep", "confidence": 1.0, "reason": "重要记忆或安全边界"}
        if layer == "evidence":
            return {"id": bucket_id, "action": "evidence_only", "confidence": 1.0, "reason": "已是原文证据层"}
        if metadata.get("type") == "feel" or layer in {"short_term"}:
            return {"id": bucket_id, "action": "keep", "confidence": 1.0, "reason": "陪伴层或短期线头不自动归档"}
        if RAW_TRANSCRIPT_HEAD_RE.search(content) or len(RAW_TRANSCRIPT_SPEAKER_RE.findall(content)) >= 2:
            return {"id": bucket_id, "action": "evidence_only", "confidence": 1.0, "reason": "完整聊天只作原文证据"}
        if bucket_id in same_name_ids:
            return {"id": bucket_id, "action": "review", "confidence": 1.0, "reason": "同名但内容不同，不自动合并或删除"}
        return None

    def _duplicate_decisions(self, buckets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
        by_hash: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for bucket in buckets:
            by_hash[_content_hash(bucket.get("content", ""))].append(bucket)
            name = " ".join(str(bucket.get("metadata", {}).get("name") or "").split()).casefold()
            if name:
                by_name[name].append(bucket)
        decisions: dict[str, dict[str, Any]] = {}
        same_name_ids = {
            str(bucket["id"])
            for group in by_name.values()
            if len({_content_hash(item.get("content", "")) for item in group}) > 1
            for bucket in group
        }
        for group in by_hash.values():
            if len(group) < 2:
                continue
            ranked = sorted(group, key=self._keeper_rank, reverse=True)
            keeper = ranked[0]
            keeper_hard = self._hard_decision(keeper, set())
            decisions[str(keeper["id"])] = keeper_hard or {
                "id": str(keeper["id"]), "action": "keep", "confidence": 1.0, "reason": "完全重复组保留项",
            }
            for bucket in ranked[1:]:
                hard = self._hard_decision(bucket, set())
                if hard and hard["action"] in {"keep", "evidence_only"}:
                    decisions[str(bucket["id"])] = hard
                else:
                    decisions[str(bucket["id"])] = {
                        "id": str(bucket["id"]), "action": "archive", "confidence": 1.0,
                        "reason": f"与 {keeper['id']} 正文完全重复",
                    }
        return decisions, same_name_ids

    def _keeper_rank(self, bucket: dict[str, Any]) -> tuple[int, int, int, str]:
        metadata = bucket.get("metadata", {})
        layer = normalize_layer_metadata(metadata, bucket.get("content", ""))["memory_layer"]
        try:
            importance = int(metadata.get("importance") or 5)
        except (TypeError, ValueError):
            importance = 5
        return (
            int(bool(metadata.get("pinned") or metadata.get("protected") or metadata.get("type") == "permanent")),
            int(layer == "evidence"),
            importance,
            str(metadata.get("created") or ""),
        )

    @staticmethod
    def _merge_source(bucket: dict[str, Any]) -> dict[str, Any]:
        metadata = bucket.get("metadata", {})
        signed_by = [str(item) for item in (metadata.get("signed_by") or []) if str(item).strip()]
        surface = str(metadata.get("source_surface") or "旧导入，入口未标注")
        title = str(metadata.get("name") or bucket.get("id") or "未命名记忆")
        return {
            "id": str(bucket.get("id") or ""),
            "title": title[:160],
            "surface": surface[:120],
            "signed_by": signed_by[:4],
            "created": str(metadata.get("created") or "")[:40],
            "preview": " ".join(str(bucket.get("content") or "").split())[:240],
        }

    def _merge_eligible(self, bucket: dict[str, Any]) -> bool:
        metadata = bucket.get("metadata", {})
        content = str(bucket.get("content") or "")
        layer = normalize_layer_metadata(metadata, content)["memory_layer"]
        if metadata.get("pinned") or metadata.get("protected") or metadata.get("type") == "permanent":
            return False
        if metadata.get("type") == "feel" or layer in {"evidence", "short_term", "archive"}:
            return False
        if RAW_TRANSCRIPT_HEAD_RE.search(content) or len(RAW_TRANSCRIPT_SPEAKER_RE.findall(content)) >= 2:
            return False
        return bool(content.strip())

    async def _candidate_groups(self, buckets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [bucket for bucket in buckets if self._merge_eligible(bucket)]
        by_id = {str(bucket["id"]): bucket for bucket in eligible}
        candidates: list[dict[str, Any]] = []
        engine = getattr(self.bucket_manager, "embedding_engine", None)
        if not engine or not hasattr(engine, "find_related_groups"):
            return []

        # Exact duplicates and repeated imported titles remain useful even when
        # an older record has no embedding yet.
        exact: defaultdict[str, list[str]] = defaultdict(list)
        names: defaultdict[str, list[str]] = defaultdict(list)
        for bucket in eligible:
            bucket_id = str(bucket["id"])
            exact[_content_hash(bucket.get("content", ""))].append(bucket_id)
            name = " ".join(str(bucket.get("metadata", {}).get("name") or "").split()).casefold()
            if name:
                names[name].append(bucket_id)
        for group in exact.values():
            if len(group) > 1:
                for offset in range(0, len(group), 8):
                    piece = group[offset:offset + 8]
                    if len(piece) > 1:
                        candidates.append({"ids": piece, "similarity": 1.0})
        for group in names.values():
            if len(group) > 1 and len({_content_hash(by_id[item].get("content", "")) for item in group}) > 1:
                for offset in range(0, len(group), 8):
                    piece = group[offset:offset + 8]
                    if len(piece) > 1:
                        candidates.append({"ids": piece, "similarity": 0.86})

        try:
            candidates.extend(engine.find_related_groups(
                list(by_id),
                threshold=self.similarity_threshold,
                max_neighbors=4,
                max_group_size=8,
            ))
        except Exception as exc:
            self._append_audit({"event": "similarity_lookup_failed", "error": _safe_error(exc)})

        # Prefer the strongest non-overlapping groups. Overlap is dangerous:
        # A~B and B~C does not prove A~C are the same fact.
        seen_ids: set[str] = set()
        groups = []
        for raw in sorted(candidates, key=lambda item: float(item.get("similarity") or 0), reverse=True):
            ids = [item for item in raw.get("ids", []) if item in by_id and item not in seen_ids]
            if len(ids) < 2:
                continue
            seen_ids.update(ids)
            group_id = hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
            groups.append({
                "id": group_id,
                "similarity": float(raw.get("similarity") or 0),
                "buckets": [by_id[item] for item in ids],
            })
        return groups

    async def _propose_consolidations(
        self,
        groups: list[dict[str, Any]],
        model: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        records = []
        for group in groups:
            items = []
            for bucket in group["buckets"]:
                metadata = bucket.get("metadata", {})
                items.append({
                    **self._merge_source(bucket),
                    "content": str(bucket.get("content") or "")[:2600],
                    "tags": [str(item) for item in (metadata.get("tags") or [])][:12],
                    "domain": [str(item) for item in (metadata.get("domain") or [])][:8],
                    "participants": [str(item) for item in (metadata.get("participants") or [])][:8],
                })
            records.append({"group_id": group["id"], "similarity": group["similarity"], "items": items})
        prompt = (
            "你是 Ombre 的长期记忆整合员。下面每组只是本地语义检索找出的候选，你必须判断它们是否在说同一主题。\n"
            "这些记录来自同一个 Claire 的三个 Claude 入口：Kelo 账号、Calder 账号、Claude 官方端。入口或署名不同不代表是不同用户，重复内容应跨入口合并。\n"
            "但如果事实的主语分别是 Kelo 与 Calder（例如两个人各自的性格、承诺或偏好），必须保留人物名并保持分开；共享入口不等于抹平两个人。\n"
            "目标是把重复或高度相似的稳定记忆整理成一条完整、可直接召回的主记忆，例如亲密偏好、写作要求、称呼习惯、关系边界、长期生活偏好。\n"
            "【称呼规范】合并后的正文中，绝不使用「用户」「AI」「人类」「助手」「对方」等泛称。Claire 就写 Claire/宝宝/老婆；珂洛就写珂洛/Kael/爸爸；Calder 就写 Calder。如果原文里出现了这些泛称，合并时一律替换成具体名字。\n"
            "【可删除】关于学术论文、论文写作、文献综述、开题报告、毕业论文的记忆，如果不涉及 Claire 的情感、关系或个人偏好，decision 设为 delete。\n"
            "merge：同一事实/偏好/要求，允许互补信息合并；正文必须自足，保留条件、例外和重要细节，不虚构。"
            "keep_separate：只是同类但不是同一条，或不同日期的独立事件。review：相互矛盾、身份不清或不确定。delete：纯学术/论文内容，与关系和生活无关。"
            "不要把不同事件粗暴揉成大杂烩；不要因为文字相似就合并；矛盾内容绝不能擅自选边。\n"
            "严格返回 JSON：{\"groups\":[{\"group_id\":\"...\",\"decision\":\"merge|keep_separate|review|delete\","
            "\"title\":\"主记忆标题\",\"content\":\"合并后的完整正文\",\"topic\":\"主题\","
            "\"confidence\":0.0,\"reason\":\"判断理由\"}]}\n\n候选：\n"
            + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        )
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是谨慎、可审计的中文长期记忆整合员，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(900, min(6000, len(groups) * 850)),
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self.dehydrator.client.chat.completions.create(
                **kwargs, extra_body={"thinking": {"type": "disabled"}}
            )
        except TypeError:
            response = await self.dehydrator.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if not isinstance(content, str):
            content = str(content)
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("DeepSeek 整合员没有返回 JSON")
        payload = json.loads(content[start:end + 1])
        allowed = {group["id"] for group in groups}
        proposals = {}
        for raw in payload.get("groups", []):
            if not isinstance(raw, dict):
                continue
            group_id = str(raw.get("group_id") or "")
            decision = str(raw.get("decision") or "review").lower()
            if group_id not in allowed or decision not in {"merge", "keep_separate", "review", "delete"}:
                continue
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0)))
            except (TypeError, ValueError):
                confidence = 0.0
            proposals[group_id] = {
                "decision": decision,
                "title": " ".join(str(raw.get("title") or "").split())[:160],
                "content": str(raw.get("content") or "").strip()[:12000],
                "topic": " ".join(str(raw.get("topic") or "长期记忆").split())[:80],
                "confidence": confidence,
                "reason": str(raw.get("reason") or "")[:240],
            }
        raw_usage = getattr(response, "usage", None)
        return proposals, {
            "model": model,
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or count_tokens_approx(prompt)),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or count_tokens_approx(content)),
        }

    async def _record_consolidation(
        self,
        job: dict[str, Any],
        group: dict[str, Any],
        proposal: dict[str, Any],
        review_handler: ReviewHandler,
    ) -> None:
        buckets = group["buckets"]
        source_ids = [str(bucket["id"]) for bucket in buckets]
        sources = [self._merge_source(bucket) for bucket in buckets]
        decision = str(proposal.get("decision") or "review")
        if decision == "delete":
            reason = str(proposal.get("reason") or "纯学术/论文内容，与关系和生活无关")
            for bucket in buckets:
                if not job.get("dry_run"):
                    await self.bucket_manager.archive(str(bucket["id"]))
                await self._record_decision(job, bucket, {
                    "id": str(bucket["id"]),
                    "action": "archive",
                    "confidence": float(proposal.get("confidence") or 0),
                    "reason": f"自动归档：{reason}",
                }, review_handler)
            job.setdefault("actions", []).append({
                "bucket_id": f"delete:{group['id']}",
                "action": "delete",
                "changed": not job.get("dry_run"),
                "reason": reason,
                "source_ids": source_ids,
                "sources": sources,
                "confidence": float(proposal.get("confidence") or 0),
                "similarity": group["similarity"],
            })
            self._save_job(job)
            return

        should_merge = (
            decision == "merge"
            and float(proposal.get("confidence") or 0) >= self.merge_confidence
            and bool(str(proposal.get("title") or "").strip())
            and bool(str(proposal.get("content") or "").strip())
        )
        if not should_merge:
            reason = str(proposal.get("reason") or "DeepSeek 判断这些记忆不应自动合并")
            for bucket in buckets:
                await self._record_decision(job, bucket, {
                    "id": str(bucket["id"]),
                    "action": "keep",
                    "confidence": float(proposal.get("confidence") or 0),
                    "reason": f"相似候选保持分开：{reason}",
                }, review_handler)
            job.setdefault("actions", []).append({
                "bucket_id": f"candidate:{group['id']}",
                "action": "keep_separate" if decision != "review" else "review",
                "changed": False,
                "reason": reason,
                "source_ids": source_ids,
                "sources": sources,
                "confidence": float(proposal.get("confidence") or 0),
                "similarity": group["similarity"],
            })
            self._save_job(job)
            return

        metadata_list = [bucket.get("metadata", {}) for bucket in buckets]
        union = lambda key: list(dict.fromkeys(
            str(item) for metadata in metadata_list for item in (metadata.get(key) or []) if str(item).strip()
        ))
        source_surfaces = list(dict.fromkeys(
            str(metadata.get("source_surface") or "旧导入，入口未标注") for metadata in metadata_list
        ))
        importance = max(int(metadata.get("importance") or 5) for metadata in metadata_list)
        domain = list(dict.fromkeys(
            str(item) for metadata in metadata_list for item in (metadata.get("domain") or []) if str(item).strip()
        ))[:6] or [str(proposal.get("topic") or "长期记忆")]
        tags = list(dict.fromkeys([
            "AI合并", "共同记忆", str(proposal.get("topic") or "长期记忆"),
            *(str(item) for metadata in metadata_list for item in (metadata.get("tags") or [])),
        ]))[:12]
        receipt = {
            "bucket_id": f"merge:{group['id']}",
            "action": "merge",
            "changed": False,
            "canonical_title": proposal["title"],
            "canonical_content": proposal["content"],
            "topic": proposal.get("topic") or "长期记忆",
            "reason": proposal.get("reason") or "同一主题的重复/相似记忆",
            "confidence": float(proposal.get("confidence") or 0),
            "similarity": group["similarity"],
            "source_ids": source_ids,
            "sources": sources,
            "source_receipts": [],
        }
        try:
            if not job.get("dry_run"):
                canonical_id = await self.bucket_manager.create(
                    content=proposal["content"],
                    tags=tags,
                    importance=importance,
                    domain=domain,
                    bucket_type="dynamic",
                    name=proposal["title"],
                    extra_metadata={
                        "source_kind": "ai_consolidation",
                        "source_session_id": job["id"],
                        "memory_status": "confirmed",
                        "memory_layer": "active",
                        "recall_policy": RECALL_POLICIES["active"],
                        "confidence": float(proposal.get("confidence") or 0),
                        "curated_by": "DeepSeek / Ombre AI Archivist",
                        "source_surface": " + ".join(source_surfaces)[:180],
                        "source_surfaces": source_surfaces,
                        "signed_by": union("signed_by"),
                        "participants": union("participants"),
                        "consolidated_from": source_ids,
                        "consolidation_job_id": job["id"],
                        "consolidation_topic": proposal.get("topic") or "长期记忆",
                    },
                )
                receipt["canonical_id"] = canonical_id
                for bucket in buckets:
                    storage_id, latest = await self._resolve_bucket(bucket)
                    if not latest:
                        raise RuntimeError(f"执行前来源记忆已不存在：{bucket['id']}")
                    raw = await review_handler(
                        bucket_id=storage_id,
                        decision="supersede",
                        actor="Ombre AI Archivist",
                        reason=f"consolidated_into:{canonical_id}",
                        request_id=job["id"],
                    )
                    result = json.loads(raw)
                    if result.get("ok") is not True:
                        raise RuntimeError(result.get("error") or f"来源记忆未能归档：{bucket['id']}")
                    receipt["source_receipts"].append({
                        "bucket_id": str(bucket["id"]),
                        "storage_id": storage_id,
                        "changed": not bool(result.get("duplicate")),
                    })
                receipt["changed"] = True
                # Search indexing is important, but it must not hold the whole
                # consolidation job hostage to one remote embedding request.
                self._index_in_background(canonical_id, proposal["content"])
            job.setdefault("decisions", []).append(receipt)
            job.setdefault("actions", []).append(receipt)
            for source_id in source_ids:
                if source_id not in job.setdefault("processed_ids", []):
                    job["processed_ids"].append(source_id)
            job["processed"] = len(set(job["processed_ids"]))
            job["merged_groups"] = int(job.get("merged_groups") or 0) + 1
            job["merged_sources"] = int(job.get("merged_sources") or 0) + len(source_ids)
            self._append_audit({
                "event": "memories_consolidated", "job_id": job["id"],
                "canonical_id": receipt.get("canonical_id", "dry-run"),
                "source_ids": source_ids, "title": proposal["title"],
            })
        except Exception as exc:
            # Best-effort rollback keeps a partial write from becoming the
            # catalogue's new truth.
            for source in receipt.get("source_receipts") or []:
                if source.get("changed"):
                    try:
                        await review_handler(
                            bucket_id=source["storage_id"], decision="restore",
                            actor="Ombre AI Archivist", reason="consolidation_rollback",
                            request_id=f"{job['id']}:rollback",
                        )
                    except Exception:
                        pass
            if receipt.get("canonical_id"):
                try:
                    await review_handler(
                        bucket_id=receipt["canonical_id"], decision="reject",
                        actor="Ombre AI Archivist", reason="consolidation_rollback",
                        request_id=f"{job['id']}:rollback:canonical",
                    )
                except Exception:
                    pass
            receipt["error"] = _safe_error(exc)
            job.setdefault("failed_ids", []).extend(source_ids)
            job["failed"] = len(set(job["failed_ids"]))
            job.setdefault("actions", []).append(receipt)
            job.setdefault("errors", []).append({"group_id": group["id"], "error": receipt["error"]})
        self._save_job(job)

    async def _run(self, job_id: str, review_handler: ReviewHandler) -> None:
        job = self._load_required(job_id)
        try:
            all_buckets = await self.bucket_manager.list_all(include_archive=False)
            by_id = {str(item["id"]): item for item in all_buckets if self._is_active(item)}
            ordered = [by_id[item] for item in job.get("record_ids", []) if item in by_id]

            # Main feature: surface semantic candidates across every Claude
            # entry point, then let DeepSeek write one canonical memory. This
            # happens before the older per-record cleanup so duplicates cannot
            # be archived without first getting a visible consolidation result.
            candidate_groups = await self._candidate_groups(ordered)
            job["candidate_groups"] = len(candidate_groups)
            job["candidate_sources"] = sum(len(group["buckets"]) for group in candidate_groups)
            self._save_job(job)
            for offset in range(0, len(candidate_groups), 5):
                if self._pause_if_requested(job):
                    return
                if self._budget_exhausted(job):
                    job["status"] = "budget_paused"
                    job["pause_reason"] = "达到本批 token 安全上限"
                    self._save_job(job)
                    return
                group_batch = candidate_groups[offset:offset + 5]
                proposals, usage = await self._propose_consolidations(group_batch, self.model)
                self._add_usage(job, usage)
                for group in group_batch:
                    proposal = proposals.get(group["id"]) or {
                        "decision": "review", "confidence": 0.0,
                        "reason": "DeepSeek 没有给出这一组的整合结论",
                    }
                    await self._record_consolidation(job, group, proposal, review_handler)
                    if self._pause_if_requested(job):
                        return

            # A real catalogue has the embedding engine above. Once every
            # semantic candidate has been reviewed, unrelated records are not
            # "legacy chores" for DeepSeek: leave them untouched and finish.
            # The fallback below remains for lightweight/offline deployments
            # that do not have semantic indexing.
            engine = getattr(self.bucket_manager, "embedding_engine", None)
            if engine and hasattr(engine, "find_related_groups"):
                processed = set(job.get("processed_ids") or [])
                untouched = [str(bucket["id"]) for bucket in ordered if str(bucket["id"]) not in processed]
                job.setdefault("processed_ids", []).extend(untouched)
                job["kept"] = int(job.get("kept") or 0) + len(untouched)
                job["processed"] = len(set(job["processed_ids"]))
                job["status"] = "completed_with_errors" if job.get("failed_ids") else "completed"
                job["completed_at"] = _now_iso()
                self._save_job(job)
                self._append_audit({"event": "job_completed", "job_id": job_id, "status": job["status"]})
                return

            duplicate_decisions, same_name_ids = self._duplicate_decisions(ordered)
            processed = set(job.get("processed_ids") or [])
            pending_model = []

            for bucket in ordered:
                bucket_id = str(bucket["id"])
                if bucket_id in processed:
                    continue
                decision = duplicate_decisions.get(bucket_id) or self._hard_decision(bucket, same_name_ids)
                if decision:
                    await self._record_decision(job, bucket, decision, review_handler)
                else:
                    pending_model.append(bucket)
                if self._pause_if_requested(job):
                    return

            batch_size = int(job.get("batch_size") or self.default_batch_size)
            for offset in range(0, len(pending_model), batch_size):
                if self._pause_if_requested(job):
                    return
                if self._budget_exhausted(job):
                    job["status"] = "budget_paused"
                    job["pause_reason"] = "达到本批 token 安全上限"
                    self._save_job(job)
                    self._append_audit({"event": "budget_paused", "job_id": job_id})
                    return
                batch = pending_model[offset:offset + batch_size]
                decisions, usage = await self._classify(batch, self.model)
                self._add_usage(job, usage)
                review_batch = []
                for bucket in batch:
                    first = decisions.get(str(bucket["id"]))
                    if not first or first.get("action") == "review" or (
                        first.get("action") == "archive"
                        and float(first.get("confidence") or 0) < self.archive_confidence
                    ):
                        review_batch.append(bucket)
                if review_batch and self.review_model and self.review_model != self.model and not self._budget_exhausted(job):
                    try:
                        reviewed, review_usage = await self._classify(review_batch, self.review_model)
                        self._add_usage(job, review_usage)
                        for bucket_id, decision in reviewed.items():
                            decisions[bucket_id] = decision
                    except Exception as exc:
                        job.setdefault("errors", []).append({
                            "at": _now_iso(), "stage": "review_model", "error": _safe_error(exc),
                        })
                        job["errors"] = job["errors"][-100:]
                        self._save_job(job)
                        self._append_audit({
                            "event": "review_model_failed", "job_id": job_id, "error": _safe_error(exc),
                        })
                for bucket in batch:
                    decision = decisions.get(str(bucket["id"])) or {
                        "id": str(bucket["id"]), "action": "review", "confidence": 0.0, "reason": "模型漏掉这条，保持不动",
                    }
                    if decision["action"] == "archive" and float(decision.get("confidence") or 0) < self.archive_confidence:
                        decision = {**decision, "action": "review", "reason": f"置信度不足：{decision.get('reason') or ''}"}
                    await self._record_decision(job, bucket, decision, review_handler)
                    if self._pause_if_requested(job):
                        return

            job = self._load_required(job_id)
            job["status"] = "completed_with_errors" if job.get("failed_ids") else "completed"
            job["completed_at"] = _now_iso()
            self._save_job(job)
            self._append_audit({"event": "job_completed", "job_id": job_id, "status": job["status"]})
        except Exception as exc:
            job = self._load_job(job_id) or job
            job["status"] = "failed"
            job.setdefault("errors", []).append({"at": _now_iso(), "error": _safe_error(exc)})
            job["errors"] = job["errors"][-100:]
            self._save_job(job)
            self._append_audit({"event": "job_failed", "job_id": job_id, "error": _safe_error(exc)})

    def _pause_if_requested(self, job: dict[str, Any]) -> bool:
        latest = self._load_job(job["id"]) or job
        if not latest.get("pause_requested"):
            return False
        latest["status"] = "paused"
        latest["pause_requested"] = False
        latest["pause_reason"] = latest.get("pause_reason") or "用户暂停"
        self._save_job(latest)
        self._append_audit({"event": "job_paused", "job_id": latest["id"]})
        return True

    def _budget_exhausted(self, job: dict[str, Any]) -> bool:
        usage = job.get("usage", {})
        return (
            int(usage.get("input_tokens") or 0) >= self.max_input_tokens
            or int(usage.get("output_tokens") or 0) >= self.max_output_tokens
        )

    def _add_usage(self, job: dict[str, Any], usage: dict[str, Any]) -> None:
        current = job.setdefault("usage", {
            "input_tokens": 0, "output_tokens": 0, "requests": 0,
            "review_input_tokens": 0, "review_output_tokens": 0, "review_requests": 0,
        })
        current["input_tokens"] = int(current.get("input_tokens") or 0) + int(usage.get("input_tokens") or 0)
        current["output_tokens"] = int(current.get("output_tokens") or 0) + int(usage.get("output_tokens") or 0)
        current["requests"] = int(current.get("requests") or 0) + 1
        if usage.get("model") == self.review_model and self.review_model != self.model:
            current["review_input_tokens"] = int(current.get("review_input_tokens") or 0) + int(usage.get("input_tokens") or 0)
            current["review_output_tokens"] = int(current.get("review_output_tokens") or 0) + int(usage.get("output_tokens") or 0)
            current["review_requests"] = int(current.get("review_requests") or 0) + 1
        flash_input = current["input_tokens"] - int(current.get("review_input_tokens") or 0)
        flash_output = current["output_tokens"] - int(current.get("review_output_tokens") or 0)
        current["cost_usd"] = round(
            flash_input / 1_000_000 * self.input_price
            + flash_output / 1_000_000 * self.output_price
            + int(current.get("review_input_tokens") or 0) / 1_000_000 * self.review_input_price
            + int(current.get("review_output_tokens") or 0) / 1_000_000 * self.review_output_price,
            4,
        )
        self._save_job(job)

    async def _classify(self, buckets: list[dict[str, Any]], model: str) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        records = []
        for bucket in buckets:
            metadata = bucket.get("metadata", {})
            records.append({
                "id": str(bucket["id"]),
                "title": str(metadata.get("name") or bucket["id"])[:160],
                "content": str(bucket.get("content") or "")[:6000],
                "importance": int(metadata.get("importance") or 5),
                "status": str(metadata.get("memory_status") or "confirmed"),
                "layer": normalize_layer_metadata(metadata, bucket.get("content", ""))["memory_layer"],
                "source_kind": str(metadata.get("source_kind") or "legacy"),
                "confidence": metadata.get("confidence"),
                "created": str(metadata.get("created") or ""),
            })
        prompt = (
            "你是 Ombre 历史记忆归档员。只做保守分类，不改写、不合并记忆。\n"
            "对每条记录输出一个决定：keep=保留有效记忆；archive=明显无价值、重复口水或错误摘要，可进入可恢复归档；"
            "review=有歧义所以保持不动；evidence_only=完整原话/聊天底稿，仅作证据、不进入普通召回。\n"
            "重要承诺、关系边界、人物事实、日期、偏好、健康信息必须 keep。仅标题相同绝不能作为 archive 理由。"
            "拿不准必须 review。archive 只给真正明确的项目，confidence 至少 0.93。\n"
            "严格返回 JSON：{\"decisions\":[{\"id\":\"...\",\"action\":\"keep|archive|review|evidence_only\","
            "\"confidence\":0.0,\"reason\":\"一句短理由\"}]}\n\n记录：\n"
            + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        )
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是保守、可审计的中文记忆归档分类器，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(600, min(4000, len(records) * 220)),
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self.dehydrator.client.chat.completions.create(
                **kwargs, extra_body={"thinking": {"type": "disabled"}}
            )
        except TypeError:
            response = await self.dehydrator.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if not isinstance(content, str):
            content = str(content)
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("DeepSeek 归档员没有返回 JSON")
        payload = json.loads(content[start:end + 1])
        allowed_ids = {item["id"] for item in records}
        decisions = {}
        for raw in payload.get("decisions", []):
            if not isinstance(raw, dict):
                continue
            bucket_id = str(raw.get("id") or "")
            action = str(raw.get("action") or "review").lower()
            if bucket_id not in allowed_ids or action not in ALLOWED_MODEL_ACTIONS:
                continue
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0)))
            except (TypeError, ValueError):
                confidence = 0.0
            decisions[bucket_id] = {
                "id": bucket_id,
                "action": action,
                "confidence": confidence,
                "reason": str(raw.get("reason") or "")[:240],
            }
        raw_usage = getattr(response, "usage", None)
        usage = {
            "model": model,
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or count_tokens_approx(prompt)),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or count_tokens_approx(content)),
        }
        return decisions, usage

    async def _record_decision(
        self,
        job: dict[str, Any],
        bucket: dict[str, Any],
        decision: dict[str, Any],
        review_handler: ReviewHandler,
    ) -> None:
        bucket_id = str(bucket["id"])
        if bucket_id in set(job.get("processed_ids") or []):
            return
        action = decision.get("action") if decision.get("action") in ALLOWED_MODEL_ACTIONS else "review"
        metadata = bucket.get("metadata", {})
        layer_info = normalize_layer_metadata(metadata, bucket.get("content", ""))
        receipt = {
            "bucket_id": bucket_id,
            "content_sha256": _content_hash(bucket.get("content", "")),
            "action": action,
            "confidence": float(decision.get("confidence") or 0),
            "reason": str(decision.get("reason") or "")[:240],
            "changed": False,
            "previous": {
                "memory_status": str(metadata.get("memory_status") or "confirmed"),
                "memory_layer": layer_info["memory_layer"],
                "recall_policy": layer_info["recall_policy"],
            },
        }
        try:
            # Keep/review and every dry-run decision are read-only. The
            # catalogue snapshot is sufficient; only re-resolve the backing
            # file immediately before a real mutation.
            storage_id = bucket_id
            latest = bucket
            if not job.get("dry_run") and action in {"archive", "evidence_only"}:
                storage_id, latest = await self._resolve_bucket(bucket)
                if not latest:
                    raise RuntimeError("执行前记忆已不存在")
            receipt["storage_id"] = storage_id
            hard = self._hard_decision(latest, set())
            if action == "archive" and hard and hard["action"] in {"keep", "evidence_only"}:
                action = hard["action"]
                receipt["action"] = action
                receipt["reason"] = f"执行器保护：{hard['reason']}"
            if not job.get("dry_run") and action == "archive":
                raw = await review_handler(
                    bucket_id=storage_id,
                    decision="reject",
                    actor="Ombre AI Archivist",
                    reason=receipt["reason"],
                    request_id=job["id"],
                )
                result = json.loads(raw)
                if result.get("ok") is not True:
                    raise RuntimeError(result.get("error") or "Ombre 未确认归档")
                receipt["changed"] = not bool(result.get("duplicate"))
            elif not job.get("dry_run") and action == "evidence_only" and layer_info["memory_layer"] != "evidence":
                await self.bucket_manager.update(
                    storage_id,
                    memory_layer="evidence",
                    recall_policy=RECALL_POLICIES["evidence"],
                )
                receipt["changed"] = True
            job.setdefault("decisions", []).append(receipt)
            job.setdefault("actions", []).append(receipt)
            job.setdefault("processed_ids", []).append(bucket_id)
            job["processed"] = len(set(job["processed_ids"]))
            counter_key = {
                "keep": "kept",
                "archive": "archived",
                "evidence_only": "evidence_only",
                "review": "review",
            }.get(action, "review")
            job[counter_key] = int(job.get(counter_key) or 0) + 1
            self._append_audit({
                "event": "record_decided",
                "job_id": job["id"],
                **{key: receipt[key] for key in ("bucket_id", "content_sha256", "action", "confidence", "reason", "changed")},
            })
        except Exception as exc:
            receipt["error"] = _safe_error(exc)
            job.setdefault("failed_ids", []).append(bucket_id)
            job["failed"] = len(set(job["failed_ids"]))
            job.setdefault("errors", []).append({"bucket_id": bucket_id, "error": receipt["error"]})
            job["errors"] = job["errors"][-100:]
            self._append_audit({"event": "record_failed", "job_id": job["id"], "bucket_id": bucket_id, "error": receipt["error"]})
        self._save_job(job)
