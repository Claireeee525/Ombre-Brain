# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in SQLite, and provides cosine similarity search.
# 通过 Gemini API（OpenAI 兼容）生成 embedding，
# 存储在 SQLite 中，提供余弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被谁依赖：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging
from datetime import datetime, timedelta, timezone

from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存储 + 余弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = (embed_cfg.get("api_key") or dehy_cfg.get("api_key") or "").strip()
        self.base_url = (
            (embed_cfg.get("base_url") or "").strip()
            or (dehy_cfg.get("base_url") or "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        # --- Initialize SQLite ---
        self._init_db()

        # --- Optional hooks (set by server.py): fired after store / delete ---
        # --- 可选回调（server.py 注入）：向量入库后 / 删除后触发（记忆家族用）---
        self.on_stored = None    # async fn(bucket_id, embedding, content)
        self.on_deleted = None   # sync fn(bucket_id)

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                bucket_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                next_retry_at TEXT NOT NULL,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        为内容生成 embedding 并存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                self._enqueue_retry(bucket_id, content, "empty embedding response")
                return False
            self._store_embedding(bucket_id, embedding)
            self._clear_retry(bucket_id)
            if self.on_stored:
                try:
                    await self.on_stored(bucket_id, embedding, content)
                except Exception as hook_err:
                    logger.warning(f"on_stored hook failed for {bucket_id}: {hook_err}")
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            self._enqueue_retry(bucket_id, content, str(e))
            return False

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _enqueue_retry(self, bucket_id: str, content: str, error: str):
        """Persist a failed job so transient provider failures do not lose indexing."""
        if not bucket_id or not content or not content.strip():
            return
        from utils import now_iso
        content = content[:20000]
        content_hash = __import__("hashlib").sha256(content.encode("utf-8")).hexdigest()[:32]
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT attempts FROM embedding_jobs WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        attempts = int(row[0] if row else 0)
        delay = min(3600, 30 * (2 ** min(attempts, 6)))
        now = self._now()
        next_retry = now + timedelta(seconds=delay)
        conn.execute(
            """INSERT INTO embedding_jobs
               (bucket_id, content, content_hash, attempts, status, next_retry_at, last_error, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
               ON CONFLICT(bucket_id) DO UPDATE SET
                 content=excluded.content,
                 content_hash=excluded.content_hash,
                 attempts=excluded.attempts,
                 status='pending',
                 next_retry_at=excluded.next_retry_at,
                 last_error=excluded.last_error,
                 updated_at=excluded.updated_at""",
            (bucket_id, content, content_hash, attempts + 1, next_retry.isoformat(), str(error)[:500], now.isoformat(), now.isoformat()),
        )
        conn.commit()
        conn.close()

    def _clear_retry(self, bucket_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embedding_jobs WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    def queue_status(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(attempts), 0) FROM embedding_jobs WHERE status = 'pending'"
        ).fetchone()
        conn.close()
        return {
            "enabled": bool(self.enabled),
            "pending": int(row[0] or 0),
            "attempts": int(row[1] or 0),
        }

    async def retry_pending(self, limit: int = 20) -> dict:
        """Retry due jobs and keep failed rows for the next backoff window."""
        if not self.enabled:
            return {**self.queue_status(), "retried": 0, "succeeded": 0, "failed": 0}
        now = self._now().isoformat()
        limit = max(1, min(int(limit or 20), 100))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT bucket_id, content FROM embedding_jobs
               WHERE status = 'pending' AND next_retry_at <= ?
               ORDER BY next_retry_at ASC LIMIT ?""",
            (now, limit),
        ).fetchall()
        conn.close()
        succeeded = 0
        failed = 0
        for bucket_id, content in rows:
            try:
                embedding = await self._generate_embedding(content)
                if not embedding:
                    raise RuntimeError("empty embedding response")
                self._store_embedding(bucket_id, embedding)
                self._clear_retry(bucket_id)
                if self.on_stored:
                    try:
                        await self.on_stored(bucket_id, embedding, content)
                    except Exception as hook_err:
                        logger.warning(f"on_stored hook failed for retry {bucket_id}: {hook_err}")
                succeeded += 1
            except Exception as exc:
                self._enqueue_retry(bucket_id, content, str(exc))
                failed += 1
        return {**self.queue_status(), "retried": len(rows), "succeeded": succeeded, "failed": failed}

    async def _generate_embedding(self, text: str) -> list[float]:
        """Call API to generate embedding vector."""
        # Truncate to avoid token limits
        truncated = text[:2000]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            (bucket_id, json.dumps(embedding), now_iso()),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()
        if self.on_deleted:
            try:
                self.on_deleted(bucket_id)
            except Exception as hook_err:
                logger.warning(f"on_deleted hook failed for {bucket_id}: {hook_err}")

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    def find_related_groups(
        self,
        bucket_ids: list[str],
        *,
        threshold: float = 0.78,
        max_neighbors: int = 4,
        max_group_size: int = 8,
    ) -> list[dict]:
        """Return local semantic candidate groups without making API calls.

        The archivist uses these as *candidates* only. DeepSeek still decides
        whether the records describe the same memory before anything changes.
        Keeping this lookup local makes a full catalogue scan cheap and lets
        records imported by different Claude surfaces meet each other.
        """
        wanted = {str(item) for item in bucket_ids if str(item).strip()}
        if len(wanted) < 2:
            return []
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT bucket_id, embedding FROM embeddings",
        ).fetchall()
        conn.close()
        ids = []
        vectors = []
        expected_size = None
        for bucket_id, raw in rows:
            if str(bucket_id) not in wanted:
                continue
            try:
                vector = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(vector, list) or not vector:
                continue
            if expected_size is None:
                expected_size = len(vector)
            if len(vector) != expected_size:
                continue
            ids.append(str(bucket_id))
            vectors.append(vector)
        if len(ids) < 2:
            return []

        try:
            import numpy as np

            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
            similarities = matrix @ matrix.T
        except (ImportError, ValueError):
            # Production includes numpy. A small pure-Python fallback keeps
            # the method usable in lightweight test/repair environments.
            similarities = [
                [self._cosine_similarity(left, right) for right in vectors]
                for left in vectors
            ]

        pairs = []
        for left_index, left_id in enumerate(ids):
            row = similarities[left_index]
            candidates = sorted(
                (
                    (float(row[right_index]), right_index)
                    for right_index in range(left_index + 1, len(ids))
                    if float(row[right_index]) >= threshold
                ),
                reverse=True,
            )[:max(1, int(max_neighbors))]
            pairs.extend((score, left_index, right_index) for score, right_index in candidates)
        pairs.sort(reverse=True)

        groups: list[dict] = []
        assigned: dict[int, int] = {}
        for score, left_index, right_index in pairs:
            left_group = assigned.get(left_index)
            right_group = assigned.get(right_index)
            if left_group is None and right_group is None:
                group_index = len(groups)
                groups.append({"ids": [ids[left_index], ids[right_index]], "scores": [score]})
                assigned[left_index] = group_index
                assigned[right_index] = group_index
                continue
            if left_group is not None and right_group is None:
                group_index, new_index = left_group, right_index
            elif right_group is not None and left_group is None:
                group_index, new_index = right_group, left_index
            else:
                # Do not join two existing clusters: transitive similarity can
                # otherwise collapse a whole category into one giant memory.
                continue
            group = groups[group_index]
            if len(group["ids"]) >= max(2, int(max_group_size)):
                continue
            member_indexes = [ids.index(item) for item in group["ids"]]
            member_scores = [float(similarities[new_index][member]) for member in member_indexes]
            if min(member_scores, default=0.0) < threshold:
                continue
            group["ids"].append(ids[new_index])
            group["scores"].extend(member_scores)
            assigned[new_index] = group_index

        return [
            {
                "ids": group["ids"],
                "similarity": round(sum(group["scores"]) / max(1, len(group["scores"])), 4),
            }
            for group in groups
            if len(group["ids"]) >= 2
        ]

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        搜索与查询文本相似的桶。返回 (bucket_id, 相似度分数) 列表。
        """
        if not self.enabled:
            return []

        try:
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # Load all embeddings from SQLite
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT bucket_id, embedding FROM embeddings").fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate cosine similarity
        results = []
        for bucket_id, emb_json in rows:
            try:
                stored_embedding = json.loads(emb_json)
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                results.append((bucket_id, sim))
            except (json.JSONDecodeError, Exception):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
