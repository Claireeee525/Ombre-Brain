"""Canonical memory layers and recall gates.

The storage layout is intentionally kept backward compatible.  Older buckets
may not have ``memory_layer`` or ``recall_policy`` in their frontmatter; this
module derives both values at read time until an explicit review writes them.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


MEMORY_LAYERS = (
    "evidence",
    "candidate",
    "active",
    "short_term",
    "feel",
    "dream",
    "archive",
)

RECALL_POLICIES = {
    "evidence": "exact_only",
    "candidate": "review_only",
    "active": "normal",
    "short_term": "handoff_only",
    "feel": "accompany_only",
    "dream": "accompany_only",
    "archive": "hidden",
}

_VALID_MODES = {"normal", "exact", "evidence", "review", "handoff", "accompany", "archive"}
_RAW_SOURCE_KINDS = {
    "raw_transcript",
    "raw_chat",
    "conversation_export",
    "claude_export",
    "original_evidence",
}
_RAW_TRANSCRIPT_HEAD_RE = re.compile(r"^\s*时间\s*[：:]\s*\d{4}\s*[/年-]", re.M)
_RAW_TRANSCRIPT_SPEAKER_RE = re.compile(
    r"(^|\n)\s*(Claire|珂洛|爸爸|Kael|Calder|用户)\s*[：:]", re.M
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_text(item) for item in value if _text(item)]
    return []


def _is_raw_evidence(metadata: dict[str, Any], content: str) -> bool:
    source_kind = _text(metadata.get("source_kind")).lower()
    if source_kind in _RAW_SOURCE_KINDS:
        return True
    if _text(metadata.get("evidence_type")).lower() in {"raw", "transcript", "original"}:
        return True
    return bool(
        _RAW_TRANSCRIPT_HEAD_RE.search(content or "")
        or len(_RAW_TRANSCRIPT_SPEAKER_RE.findall(content or "")) >= 2
    )


def _parse_expiry(value: Any) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_expired(metadata: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Return whether an explicitly short-lived memory is past its expiry."""
    expiry = _parse_expiry(metadata.get("expires_at"))
    if expiry is None:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expiry <= current.astimezone(timezone.utc)


def classify_memory_layer(metadata: dict[str, Any] | None, content: str = "") -> str:
    """Derive one canonical layer without mutating the supplied metadata."""
    meta = metadata or {}
    status = _text(meta.get("memory_status") or "confirmed").lower()
    stored_type = _text(meta.get("type")).lower()
    explicit = _text(meta.get("memory_layer")).lower()

    # Rejected/archived material is always recoverable background, never live.
    if status in {"rejected", "archived"} or stored_type == "archived" or explicit == "archive":
        return "archive"
    if status == "candidate" or explicit == "candidate":
        return "candidate"
    if explicit in MEMORY_LAYERS:
        return explicit
    if _is_raw_evidence(meta, content):
        return "evidence"
    if stored_type == "feel":
        return "feel"

    words = " ".join(
        [
            *_list_text(meta.get("domain")),
            *_list_text(meta.get("tags")),
        ]
    )
    if re.search(r"梦境|梦|dream", words, re.I):
        return "dream"
    if _text(meta.get("expires_at")):
        return "short_term"
    return "active"


def normalize_layer_metadata(
    metadata: dict[str, Any] | None,
    content: str = "",
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return presentation metadata for a bucket, preserving original fields."""
    result = dict(metadata or {})
    layer = classify_memory_layer(result, content)
    result["memory_layer"] = layer
    result["recall_policy"] = RECALL_POLICIES[layer]
    result["expired"] = is_expired(result, now=now) if layer == "short_term" else False
    return result


def recall_mode(value: str = "normal") -> str:
    mode = _text(value).lower().replace("-", "_") or "normal"
    aliases = {
        "default": "normal",
        "query": "normal",
        "source": "evidence",
        "exact_only": "exact",
        "review_only": "review",
        "handoff_only": "handoff",
        "accompany_only": "accompany",
        "hidden": "archive",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in _VALID_MODES else "normal"


def memory_recallable(
    metadata: dict[str, Any] | None,
    content: str = "",
    *,
    mode: str = "normal",
    include_candidates: bool = False,
) -> bool:
    """Apply the layer gate before ranking, recency, emotion, or embeddings."""
    meta = metadata or {}
    layer = classify_memory_layer(meta, content)
    mode = recall_mode(mode)
    if include_candidates and mode == "normal":
        # The legacy flag is an explicit review request, never a silent recall.
        mode = "review"

    if layer == "archive":
        return mode == "archive"
    if layer == "candidate":
        return mode == "review"
    if layer == "evidence":
        return mode in {"exact", "evidence"}
    if layer == "active":
        return mode in {"normal", "exact", "evidence", "handoff"}
    if layer == "short_term":
        return mode == "handoff" and not is_expired(meta)
    if layer in {"feel", "dream"}:
        return mode == "accompany"
    return False


def layer_fields(layer: str, *, expires_at: str = "") -> dict[str, str]:
    """Build validated frontmatter fields for a new or reviewed memory."""
    normalized = _text(layer).lower().replace("-", "_")
    if normalized not in MEMORY_LAYERS:
        normalized = "active"
    if normalized == "short_term" and not _text(expires_at):
        raise ValueError("short_term memory requires expires_at")
    return {
        "memory_layer": normalized,
        "recall_policy": RECALL_POLICIES[normalized],
        **({"expires_at": _text(expires_at)[:64]} if _text(expires_at) else {}),
    }

