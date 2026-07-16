"""Pure validation helpers for the background memory curator."""

import hashlib
import json
from typing import Any

from rapidfuzz import fuzz

import somatic_engine as E


MEMORY_KINDS = {"lasting", "event", "state", "dream"}
MEMORY_STATUSES = {"confirmed", "candidate"}
MEMORY_OPERATIONS = {"add", "revision"}


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def memory_fingerprint(session_id: str, item: dict) -> str:
    evidence = sorted(str(value)[:100] for value in item.get("evidence_message_ids", []) if value)
    raw = json.dumps([
        session_id or "",
        item.get("operation") or "add",
        evidence,
        item.get("title") or "",
        item.get("content") or "",
        item.get("supersedes") or "",
    ], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def duplicate_similarity(item: dict, bucket: dict) -> float:
    """Compare memory meaning directly, without recall-time or importance boosts."""
    metadata = bucket.get("metadata") or {}
    content = _text(item.get("content"), 2400)
    existing_content = _text(bucket.get("content"), 2400)
    title = _text(item.get("title"), 160)
    existing_title = _text(metadata.get("name"), 160)
    content_score = float(fuzz.WRatio(content, existing_content)) if content and existing_content else 0.0
    title_score = float(fuzz.WRatio(title, existing_title)) if title and existing_title else 0.0
    new_tags = {_text(tag, 40).casefold() for tag in (item.get("tags") or []) if _text(tag, 40)}
    old_tags = {_text(tag, 40).casefold() for tag in (metadata.get("tags") or []) if _text(tag, 40)}
    overlap = len(new_tags & old_tags)
    tag_score = 100.0 * overlap / max(1, min(len(new_tags), len(old_tags))) if overlap else 0.0
    combined = max(
        content_score,
        content_score * 0.74 + title_score * 0.26,
        content_score * 0.78 + title_score * 0.12 + tag_score * 0.10,
    )
    if title_score >= 98 and len(title) >= 8 and (overlap or content_score >= 65):
        combined = max(combined, 88.0)
    return round(min(100.0, combined), 2)


def normalize_curate_payload(value: Any) -> dict:
    if isinstance(value, str):
        if len(value) > 80_000:
            raise ValueError("curator payload is too large")
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("curator payload must be an object")

    session_id = _text(value.get("session_id") or value.get("sessionId"), 180)
    batch_id = _text(value.get("batch_id") or value.get("batchId"), 100)
    batch_message_ids = [
        _text(item, 100) for item in (value.get("source_message_ids") or value.get("sourceMessageIds") or [])
        if _text(item, 100)
    ][:40]
    allowed_ids = set(batch_message_ids)
    memories = []
    for raw in (value.get("memories") or [])[:8]:
        if not isinstance(raw, dict):
            continue
        title = _text(raw.get("title"), 120)
        content = _text(raw.get("content"), 2000)
        if not title or not content:
            continue
        operation = raw.get("operation") if raw.get("operation") in MEMORY_OPERATIONS else "add"
        status = raw.get("status") if raw.get("status") in MEMORY_STATUSES else "candidate"
        evidence = [
            _text(item, 100) for item in (raw.get("evidence_message_ids") or raw.get("evidenceMessageIds") or [])
            if _text(item, 100) and (not allowed_ids or _text(item, 100) in allowed_ids)
        ][:12]
        confidence = _clamp(raw.get("confidence"), 0.0, 1.0, 0.7)
        if operation == "revision" or raw.get("inferred") or not evidence or confidence < 0.78:
            status = "candidate"
        item = {
            "title": title,
            "content": content,
            "kind": raw.get("kind") if raw.get("kind") in MEMORY_KINDS else "event",
            "operation": operation,
            "status": status,
            "confidence": round(confidence, 2),
            "importance": round(_clamp(raw.get("importance"), 1, 10, 5)),
            "tags": list(dict.fromkeys(
                _text(tag, 40) for tag in (raw.get("tags") or []) if _text(tag, 40)
            ))[:12],
            "domain": list(dict.fromkeys(
                _text(domain, 40) for domain in (raw.get("domain") or []) if _text(domain, 40)
            ))[:2],
            "valence": round(_clamp(raw.get("valence"), 0.0, 1.0, 0.5), 2),
            "arousal": round(_clamp(raw.get("arousal"), 0.0, 1.0, 0.3), 2),
            "evidence_message_ids": list(dict.fromkeys(evidence)),
            "valid_from": _text(raw.get("valid_from") or raw.get("validFrom"), 40),
            "valid_to": _text(raw.get("valid_to") or raw.get("validTo"), 40),
            "supersedes": _text(raw.get("supersedes") or raw.get("supersedes_bucket_id"), 100),
            "rationale": _text(raw.get("rationale"), 240),
        }
        item["source_fingerprint"] = _text(raw.get("source_fingerprint") or raw.get("sourceFingerprint"), 64) or memory_fingerprint(session_id, item)
        memories.append(item)

    return {
        "session_id": session_id,
        "batch_id": batch_id,
        "source_message_ids": batch_message_ids,
        "source_kind": _text(value.get("source_kind") or value.get("sourceKind") or "memory_secretary", 40),
        "memories": memories,
    }


def aggregate_somatic_signals(signals: Any) -> dict:
    """Turn several semantic signals into one conservative pulse table."""
    if not isinstance(signals, list):
        return {"pulses": {}, "dominant": "", "signals": []}
    normalized = []
    merged_weights = {}
    for raw in signals[:12]:
        if not isinstance(raw, dict):
            continue
        signal_type = _text(raw.get("type"), 40).lower()
        table = None
        if signal_type.startswith("mood:"):
            table = E.MOOD_PULSES.get(signal_type.split(":", 1)[1])
        else:
            table = E.EVENT_PULSES.get(signal_type)
        if not table:
            continue
        weight = _clamp(raw.get("weight"), 0.05, 1.0, 0.4)
        merged_weights[signal_type] = min(1.0, merged_weights.get(signal_type, 0.0) + weight)

    for signal_type, weight in sorted(merged_weights.items(), key=lambda item: item[1], reverse=True)[:5]:
        normalized.append({"type": signal_type, "weight": round(weight, 2)})

    net = {}
    satisfy = set()
    for signal in normalized:
        signal_type, weight = signal["type"], signal["weight"]
        table = E.MOOD_PULSES.get(signal_type.split(":", 1)[1]) if signal_type.startswith("mood:") else E.EVENT_PULSES.get(signal_type)
        for drive, amount in (table or {}).items():
            if drive == "_satisfy":
                if weight >= 0.65:
                    satisfy.update(amount or [])
                continue
            if drive in E.DRIVE_KEYS:
                # This is a residue after immediate per-turn reactions, so use
                # only sixty percent of the original pulse and cap the batch.
                net[drive] = max(-0.12, min(0.12, net.get(drive, 0.0) + float(amount) * weight * 0.6))
    pulses = {key: round(value, 4) for key, value in net.items() if abs(value) >= 0.005}
    if satisfy:
        pulses["_satisfy"] = sorted(satisfy)
    return {
        "pulses": pulses,
        "dominant": normalized[0]["type"] if normalized else "",
        "signals": normalized,
    }
