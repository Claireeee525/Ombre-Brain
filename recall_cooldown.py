"""Recall cooldown table (SQLite).

教程《记忆召回全链路》第 6 节的 ID 级冷却：
同一条记忆被动浮现过之后，60 轮内不再浮现；
落 SQLite 而不是内存，重启不丢。

这里以「每次无参数 breath 浮现调用」为一轮。pinned/钉选核心不参与冷却
（它们是"每次都要在场"的宪法级内容，由预算上限单独控制）。
"""

import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_cooldown (
    bucket_id   TEXT PRIMARY KEY,
    round       INTEGER NOT NULL,
    surfaced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_rounds (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""


class RecallCooldown:
    def __init__(self, db_path: str, window: int = 60):
        self.db_path = str(db_path)
        self.window = max(1, int(window or 60))
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _current_round(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM recall_rounds WHERE key='round'"
        ).fetchone()
        return int(row[0]) if row else 0

    def next_round(self) -> int:
        """推进一轮并返回新轮次号。"""
        nxt = self._current_round() + 1
        self._conn.execute(
            "INSERT INTO recall_rounds(key, value) VALUES('round', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (nxt,),
        )
        self._conn.commit()
        return nxt

    def cooling_ids(self, round_no: int = None) -> set:
        """返回当前仍在冷却期内的 bucket_id 集合。"""
        current = self._current_round() if round_no is None else int(round_no)
        cutoff = current - (self.window - 1)
        rows = self._conn.execute(
            "SELECT bucket_id FROM recall_cooldown WHERE round >= ?",
            (cutoff,),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def mark(self, bucket_ids, round_no: int = None) -> None:
        """把刚浮现过的桶记入冷却表。"""
        current = self._current_round() if round_no is None else int(round_no)
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for bucket_id in bucket_ids:
            if not bucket_id:
                continue
            self._conn.execute(
                "INSERT INTO recall_cooldown(bucket_id, round, surfaced_at) "
                "VALUES(?, ?, ?) "
                "ON CONFLICT(bucket_id) DO UPDATE SET "
                "round=excluded.round, surfaced_at=excluded.surfaced_at",
                (str(bucket_id), current, now),
            )
        self._conn.commit()

    def prune(self, round_no: int = None) -> None:
        """清理已过冷却窗口的旧记录，防止表无限膨胀。"""
        current = self._current_round() if round_no is None else int(round_no)
        cutoff = current - (self.window - 1)
        self._conn.execute(
            "DELETE FROM recall_cooldown WHERE round < ?",
            (cutoff,),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
