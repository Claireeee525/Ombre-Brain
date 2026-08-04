"""Tests for the soft date-window filter（"昨天/上周" → ±1 天日期过滤）。"""

from bucket_manager import _date_arg, _bucket_date_in_window, _shift_date


def test_date_arg_normalizes_iso_and_rejects_garbage():
    assert _date_arg("2026-07-30") == "2026-07-30"
    assert _date_arg("2026-07-30T12:00:00+08:00") == "2026-07-30"
    assert _date_arg("上周") == ""
    assert _date_arg("") == ""


def test_shift_date_adds_days():
    assert _shift_date("2026-07-30", -1) == "2026-07-29"
    assert _shift_date("2026-07-30", 1) == "2026-07-31"
    assert _shift_date("2026-08-01", -1) == "2026-07-31"


def test_bucket_date_in_window_is_soft_plus_minus_one_day():
    meta = {"valid_from": "2026-07-30T12:00:00+08:00"}
    assert _bucket_date_in_window(meta, "2026-07-30", "2026-07-30") is True
    assert _bucket_date_in_window(meta, "2026-07-29", "2026-07-29") is True
    assert _bucket_date_in_window(meta, "2026-07-31", "2026-07-31") is True
    assert _bucket_date_in_window(meta, "2026-07-28", "2026-07-28") is False
    assert _bucket_date_in_window(meta, "2026-08-01", "2026-08-01") is False


def test_created_fallback_when_no_valid_from():
    meta = {"created": "2026-07-02T08:00:00"}
    assert _bucket_date_in_window(meta, "2026-07-02", "2026-07-02") is True
    assert _bucket_date_in_window(meta, "2026-07-03", "2026-07-03") is True
    assert _bucket_date_in_window(meta, "2026-07-04", "2026-07-04") is False


def test_missing_date_only_included_without_window():
    assert _bucket_date_in_window({}, "", "") is True
    assert _bucket_date_in_window({}, "2026-07-30", "2026-07-30") is False
