"""Tests for the SQLite recall cooldown table（60 轮 ID 级冷却）。"""

from recall_cooldown import RecallCooldown


def test_round_counter_increments(tmp_path):
    cd = RecallCooldown(str(tmp_path / "recall_cooldown.sqlite3"), window=60)
    assert cd.next_round() == 1
    assert cd.next_round() == 2


def test_mark_and_cooling_within_window(tmp_path):
    cd = RecallCooldown(str(tmp_path / "recall_cooldown.sqlite3"), window=60)
    r1 = cd.next_round()
    cd.mark(["a1", "a2"], r1)
    r2 = cd.next_round()
    assert cd.cooling_ids(r2) == {"a1", "a2"}
    # 第 r1+60 轮时，r1 轮浮现的桶恰好离开 60 轮窗口
    assert cd.cooling_ids(r1 + 60) == set()


def test_re_mark_refreshes_round(tmp_path):
    cd = RecallCooldown(str(tmp_path / "recall_cooldown.sqlite3"), window=60)
    r1 = cd.next_round()
    cd.mark(["a1"], r1)
    r59 = r1 + 58
    cd.mark(["a1"], r59)
    # 刷新后窗口从 r59 起算，r1+60 仍在冷却
    assert cd.cooling_ids(r1 + 60) == {"a1"}
    assert cd.cooling_ids(r59 + 61) == set()


def test_prune_removes_old_rows(tmp_path):
    cd = RecallCooldown(str(tmp_path / "recall_cooldown.sqlite3"), window=60)
    r1 = cd.next_round()
    cd.mark(["old"], r1)
    cd.mark(["fresh"], r1 + 59)
    cd.prune(r1 + 60)
    assert cd.cooling_ids(r1 + 60) == {"fresh"}
