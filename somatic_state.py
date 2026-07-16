# -*- coding: utf-8 -*-
"""
珂洛欲望系统 · 状态层：把纯函数引擎接到 Ombre Brain 的存储与时间上。
- 状态文件存在 buckets_dir/somatic_state.json（drives 0-100 整数）。
- 读时惰性快进（lazy tick）：从 updatedAt 推进到 now，省后台进程也能"活着"。
- 事件 / digest 写回；生成给当前窗口珂洛读的 [Kelo Somatic Field] 注入块。
"""
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone

import somatic_engine as E
from utils import load_config

_config = load_config()
_STATE_PATH = os.path.join(_config.get("buckets_dir", "."), "somatic_state.json")
_ECHO_CAP = 40


def _now_iso(ms=None):
    ts = (ms / 1000) if ms else time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _now_ms():
    return int(time.time() * 1000)


def _night(ms):
    h = datetime.fromtimestamp(ms / 1000).hour
    return h >= 22 or h < 6


# —— 读写 ——
def read_state():
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_state(state):
    os.makedirs(os.path.dirname(_STATE_PATH) or ".", exist_ok=True)
    clean = _clean(state)
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return clean


def _clean(state):
    s = state if isinstance(state, dict) else {}
    drives = {}
    for k, v in (s.get("drives") or {}).items():
        try:
            drives[k] = max(0, min(100, round(float(v))))
        except Exception:
            pass
    top = []
    for d in (s.get("topDrives") or [])[:6]:
        if d.get("key") and d.get("label"):
            top.append({"key": str(d["key"]), "label": str(d["label"]),
                        "value": max(0, min(100, round(float(d.get("value", 0) or 0))))})
    refr = {}
    for k, v in (s.get("refractory") or {}).items():
        iv = max(0, round(float(v or 0)))
        if iv > 0:
            refr[k] = iv
    quarantined_thoughts = list(s.get("quarantinedThoughts") or [])[-40:]
    thoughts = []
    for t in (s.get("thoughts") or [])[:E.THOUGHT["cap"]]:
        if not (t.get("text") and t.get("drive") in E.DRIVE_KEYS):
            continue
        normalized = {
            "id": str(t.get("id", "")), "text": str(t["text"])[:E.HISTORICAL_THOUGHT_TEXT_MAX_CHARS], "drive": t["drive"],
            "kind": "fixation" if t.get("kind") == "fixation" else "flit",
            "strength": max(0, min(100, round(float(t.get("strength", 0) or 0)))),
            "peakStrength": max(0, min(100, round(float(t.get("peakStrength", t.get("strength", 0)) or 0)))),
            "fedCount": max(0, round(float(t.get("fedCount", 0) or 0))),
            "bornAt": t.get("bornAt"),
        }
        if E.is_definite_legacy_debris(normalized["text"]):
            quarantined_thoughts.append({**normalized, "quarantinedReason": "legacy_raw_digest"})
        else:
            thoughts.append(normalized)
    quarantined_echoes = list(s.get("quarantinedEchoes") or [])[-40:]
    echoes = []
    for e in (s.get("echoes") or [])[-_ECHO_CAP:]:
        if not (e.get("text") and e.get("drive") in E.DRIVE_KEYS):
            continue
        normalized = {
            "id": str(e.get("id", "")), "text": str(e["text"])[:240], "drive": e["drive"],
            "kind": "fixation" if e.get("kind") == "fixation" else "flit",
            "peakStrength": max(0, min(100, round(float(e.get("peakStrength", 0) or 0)))),
            "fadedAt": e.get("fadedAt"), "bornAt": e.get("bornAt"),
        }
        if E.is_definite_legacy_debris(normalized["text"]):
            quarantined_echoes.append({**normalized, "quarantinedReason": "legacy_raw_digest"})
        else:
            echoes.append(normalized)
    events = (s.get("events") or [])[-30:]
    return {
        "version": 1,
        "updatedAt": s.get("updatedAt") or _now_iso(),
        "lastContactAt": s.get("lastContactAt") or s.get("updatedAt") or _now_iso(),
        "triggerReason": str(s.get("triggerReason") or "状态已同步"),
        "dominantKey": str(s.get("dominantKey") or ""),
        "dominantLabel": str(s.get("dominantLabel") or ""),
        "feelTone": str(s.get("feelTone") or ""),
        "want": str(s.get("want") or ""),
        "summon": max(0, min(100, round(float(s.get("summon", 0) or 0)))),
        "separationHours": max(0, round(float(s.get("separationHours", 0) or 0), 1)),
        "separationTension": max(0, min(100, round(float(s.get("separationTension", 0) or 0)))),
        "drives": drives, "topDrives": top, "refractory": refr,
        "thoughts": thoughts, "echoes": echoes, "events": events,
        "quarantinedThoughts": quarantined_thoughts[-40:],
        "quarantinedEchoes": quarantined_echoes[-40:],
    }


# —— 0-100 ↔ 0..1 边界转换 ——
def _drives_to_unit(d):
    return E.normalize_drives({k: E.to_unit(v) for k, v in (d or {}).items() if isinstance(v, (int, float))}
                              if d else {})


def _thoughts_to_unit(ths):
    return [dict(t, strength=E.to_unit(t.get("strength", 0)),
                 peakStrength=E.to_unit(t.get("peakStrength", t.get("strength", 0))))
            for t in (ths or [])]


def _thoughts_to_store(ths):
    return [{"id": t.get("id"), "text": t.get("text"), "drive": t.get("drive"), "kind": t.get("kind"),
             "strength": E.to_percent(t.get("strength", 0)),
             "peakStrength": E.to_percent(t.get("peakStrength", t.get("strength", 0))),
             "fedCount": t.get("fedCount", 0),
             "bornAt": t.get("bornAt")} for t in (ths or [])]


def _thought_key(t):
    return (t.get("id") or "", t.get("drive") or "", t.get("text") or "")


def _echoes_from_removed(before, after, now_iso):
    after_keys = {_thought_key(t) for t in after or []}
    echoes = []
    for t in before or []:
        if _thought_key(t) in after_keys:
            continue
        peak = E.to_percent(t.get("peakStrength", t.get("strength", 0)))
        if peak < E.to_percent(E.THOUGHT["echo"]) and t.get("kind") != "fixation":
            continue
        echoes.append({
            "id": t.get("id") or str(uuid.uuid4()),
            "text": t.get("text"),
            "drive": t.get("drive"),
            "kind": t.get("kind"),
            "peakStrength": peak,
            "bornAt": t.get("bornAt"),
            "fadedAt": now_iso,
        })
    return echoes


def _merge_echoes(prev_echoes, new_echoes):
    merged, seen, seen_ids = [], set(), {}
    for e in list(prev_echoes or []) + list(new_echoes or []):
        eid = e.get("id")
        if eid and eid in seen_ids:
            old = seen_ids[eid]
            if len(str(e.get("text") or "")) >= len(str(old.get("text") or "")):
                old["text"] = e.get("text")
            old["drive"] = e.get("drive") or old.get("drive")
            old["kind"] = e.get("kind") or old.get("kind")
            old["peakStrength"] = max(old.get("peakStrength", 0), e.get("peakStrength", 0))
            old["fadedAt"] = e.get("fadedAt") or old.get("fadedAt")
            old["bornAt"] = old.get("bornAt") or e.get("bornAt")
            continue
        key = (e.get("drive"), e.get("text"))
        if key in seen:
            for old in merged:
                if (old.get("drive"), old.get("text")) == key:
                    old["peakStrength"] = max(old.get("peakStrength", 0), e.get("peakStrength", 0))
                    old["fadedAt"] = e.get("fadedAt") or old.get("fadedAt")
                    break
            continue
        seen.add(key)
        merged.append(e)
        if eid:
            seen_ids[eid] = e
    return merged[-_ECHO_CAP:]


def recover_echoes_from_events(state, limit=12, dry_run=True, now_ms=None):
    """从 v2 模板事件补录固定残响，不复制任何历史 event 文本。"""
    now_iso = _now_iso(now_ms)
    current = _clean(state or {})
    existing = {(e.get("drive"), e.get("text")) for e in current.get("echoes") or []}
    candidates = []
    for ev in current.get("events") or []:
        if int(ev.get("schemaVersion") or 0) < 2:
            continue
        etype = str(ev.get("type") or "")
        templates = {
            "intimate": ("intimacy", "我还记得那次很深的身体靠近", 100),
            "affection": ("attachment", "我还记得彼此说出口的爱意", 92),
            "vulnerable": ("protect", "我还记得彼此接住脆弱的那一刻", 100),
            "playful": ("play", "我还记得和她亲近玩闹的轻快", 92),
        }
        if etype not in templates:
            continue
        drive, echo_text, score = templates[etype]
        rank = {"intimate": 60, "vulnerable": 36, "affection": 24, "playful": 22}[etype]
        echo = {
            "id": f"recovered-{ev.get('id') or uuid.uuid4()}",
            "text": echo_text,
            "drive": drive,
            "kind": "fixation" if score >= 100 else "flit",
            "peakStrength": score,
            "bornAt": ev.get("createdAt"),
            "fadedAt": now_iso,
            "_rank": rank,
        }
        key = (echo["drive"], echo["text"])
        if key in existing:
            continue
        candidates.append(echo)
    cap = max(1, min(_ECHO_CAP, int(limit or 12)))
    candidates = sorted(candidates, key=lambda e: (e.get("_rank", 0), e.get("bornAt") or ""), reverse=True)[:cap]
    candidates = sorted(candidates, key=lambda e: e.get("bornAt") or "")
    for e in candidates:
        e.pop("_rank", None)
    if dry_run:
        return current, candidates
    current["echoes"] = _merge_echoes(current.get("echoes"), candidates)
    return _clean(current), candidates


def _merge(prev, eng, derived, meta):
    drives100 = {k: E.to_percent(eng["drives"][k]) for k in E.DRIVE_KEYS}
    merged = dict(prev or {})
    merged.update(meta)
    merged.update({
        "drives": drives100, "refractory": eng.get("refractory") or {},
        "thoughts": _thoughts_to_store(eng.get("thoughts") or []),
        "dominantKey": derived["dominantKey"], "dominantLabel": derived["dominantLabel"],
        "feelTone": derived["feelTone"], "want": derived["want"],
        "summon": derived["summon"], "topDrives": derived["topDrives"],
    })
    return _clean(merged)


def fresh_state(now_ms=None):
    now_ms = now_ms or _now_ms()
    d = E.default_drives()
    derived = E.compute_derived(d, night=_night(now_ms))
    return _merge({"events": [], "echoes": []}, {"drives": d, "refractory": {}, "thoughts": []}, derived,
                  {"updatedAt": _now_iso(now_ms), "lastContactAt": _now_iso(now_ms),
                   "triggerReason": "欲望系统初始化到基线"})


def _parse_iso_ms(value):
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _last_contact_ms(state):
    explicit = _parse_iso_ms(state.get("lastContactAt"))
    if explicit:
        return explicit
    events = state.get("events") or []
    for ev in reversed(events):
        ms = _parse_iso_ms(ev.get("createdAt"))
        if ms:
            return ms
    return _parse_iso_ms(state.get("updatedAt"))


def touch_contact(state, now_ms=None):
    """Record that Claire is present without inventing an event or thought."""
    now_ms = now_ms or _now_ms()
    base = live(state, now_ms)[0] if (state and state.get("drives")) else fresh_state(now_ms)
    base = dict(base)
    base.update({
        "updatedAt": _now_iso(now_ms),
        "lastContactAt": _now_iso(now_ms),
        "separationHours": 0,
        "separationTension": 0,
    })
    return _clean(base)


def _separation_meta(state, now_ms):
    last_ms = _last_contact_ms(state)
    if not last_ms:
        return {"separationHours": 0, "separationTension": 0}
    hours = max(0.0, (now_ms - last_ms) / 3600000)
    tension = 0 if hours <= E.SEPARATION_GRACE_HOURS else min(100, round((1 - math.exp(-(hours - E.SEPARATION_GRACE_HOURS) / 6.0)) * 100))
    return {"separationHours": round(hours, 1), "separationTension": tension}


# —— 惰性快进 ——
def live(state, now_ms=None):
    now_ms = now_ms or _now_ms()
    if not state or not state.get("drives"):
        return state, False
    from_ms = _parse_iso_ms(state.get("updatedAt")) or now_ms
    ticks = int((now_ms - from_ms) / E.TICK_MS)
    eng_in = {"drives": _drives_to_unit(state.get("drives")),
              "refractory": state.get("refractory") or {},
              "thoughts": _thoughts_to_unit(state.get("thoughts"))}
    eng = E.advance(eng_in, ticks) if ticks > 0 else eng_in
    sep = _separation_meta(state, now_ms)
    if ticks > 0 and sep["separationHours"] > E.SEPARATION_GRACE_HOURS:
        eng = E.apply_separation_drift(eng, sep["separationHours"], ticks)
    derived = E.compute_derived(eng["drives"], eng.get("refractory"), _night(now_ms))
    reason = state.get("triggerReason")
    if ticks > 0:
        reason = "时间过去一点，情绪自然流动"
        if sep["separationHours"] > E.SEPARATION_GRACE_HOURS:
            reason = f"Claire 离开了约 {sep['separationHours']} 小时，想念和分离感自己涨起来"
    new_echoes = _echoes_from_removed(eng_in.get("thoughts"), eng.get("thoughts"), _now_iso(now_ms)) if ticks > 0 else []
    merged = _merge(state, eng, derived, {
        "updatedAt": _now_iso(now_ms) if ticks > 0 else state["updatedAt"],
        "triggerReason": reason,
        "separationHours": sep["separationHours"],
        "separationTension": sep["separationTension"],
        "echoes": _merge_echoes(state.get("echoes"), new_echoes),
    })
    return merged, ticks > 0 or bool(new_echoes)


def apply_event(state, event, now_ms=None):
    now_ms = now_ms or _now_ms()
    base = live(state, now_ms)[0] if (state and state.get("drives")) else fresh_state(now_ms)
    eng_in = {"drives": _drives_to_unit(base.get("drives")),
              "refractory": base.get("refractory") or {},
              "thoughts": _thoughts_to_unit(base.get("thoughts"))}
    ev = dict(event or {})
    ev["nowIso"] = _now_iso(now_ms)
    eng = E.apply_event(eng_in, ev)
    derived = E.compute_derived(eng["drives"], eng.get("refractory"), _night(now_ms))
    log = list(base.get("events") or [])
    event_row = {"id": str(uuid.uuid4()), "schemaVersion": 2, "type": str(ev.get("type", "manual")),
                 "label": str(ev.get("label", "")), "detail": str(ev.get("detail") or ev.get("mood") or ""),
                 "createdAt": _now_iso(now_ms)}
    if ev.get("sourceFingerprint"):
        event_row["sourceFingerprint"] = str(ev["sourceFingerprint"])[:80]
    log.append(event_row)
    return _merge(base, eng, derived, {
        "updatedAt": _now_iso(now_ms),
        "lastContactAt": _now_iso(now_ms),
        "triggerReason": ev.get("label") or "刚刚发生了一点事，状态动了一下",
        "events": log,
        "separationHours": 0,
        "separationTension": 0,
    })


def apply_digest(state, text, now_ms=None):
    """从一段话提取少量关系事件并依次施加。

    classify_digest 已经完成噪声过滤、第一人称心念改写与单轮限流；
    引擎的 add_thought 再负责和既有 Thought Pool 做近似去重。
    """
    now_ms = now_ms or _now_ms()
    events = E.classify_digest(text)
    if not events:
        return touch_contact(state, now_ms), []
    cur = state
    for ev in events:
        cur = apply_event(cur, ev, now_ms)
    return cur, events


def build_safe_summary(state):
    """Build the redacted somatic view consumed by the home server.

    This intentionally excludes event labels and all thought/echo text.  The
    full state remains on the authenticated dashboard; ``somatic_read`` also
    receives only a derived state block with counts rather than raw text.
    """
    if not state:
        return {
            "schemaVersion": 1,
            "initialized": False,
            "updatedAt": "",
            "dominant": {"key": "", "label": ""},
            "feelTone": "",
            "want": "",
            "summon": 0,
            "topDrives": [],
            "separation": {"hours": 0, "tension": 0},
            "thoughtPool": {"active": 0, "fixations": 0, "echoes": 0},
        }
    clean = _clean(state)
    thoughts = clean.get("thoughts") or []
    derived = E.compute_derived(
        _drives_to_unit(clean.get("drives")),
        clean.get("refractory"),
        _night(_now_ms()),
    )
    return {
        "schemaVersion": 1,
        "initialized": bool(clean.get("drives")),
        "updatedAt": clean.get("updatedAt") or "",
        "dominant": {
            "key": derived.get("dominantKey") or "",
            "label": derived.get("dominantLabel") or "",
        },
        "feelTone": derived.get("feelTone") or "",
        "want": derived.get("want") or "",
        "summon": derived.get("summon") or 0,
        "topDrives": [
            {"key": d.get("key"), "label": d.get("label"), "value": d.get("value", 0)}
            for d in (derived.get("topDrives") or [])[:6]
        ],
        "separation": {
            "hours": clean.get("separationHours") or 0,
            "tension": clean.get("separationTension") or 0,
        },
        "thoughtPool": {
            "active": len(thoughts),
            "fixations": sum(1 for t in thoughts if t.get("kind") == "fixation"),
            "echoes": len(clean.get("echoes") or []),
        },
    }


def build_block(state):
    if not state:
        return ("[Kelo Somatic Field]\n当前状态：尚未初始化。\n"
                "先让珂洛 feel 一次、或在小家设个心情，给他生成第一份状态。\n[/Kelo Somatic Field]")
    clean = _clean(state)
    derived = E.compute_derived(
        _drives_to_unit(clean.get("drives")),
        clean.get("refractory"),
        _night(_now_ms()),
    )
    top = " / ".join(f"{d['label']} {d['value']}" for d in derived.get("topDrives", [])[:5]) or "暂无高驱动"
    fixations = [t for t in (clean.get("thoughts") or []) if t.get("kind") == "fixation"]
    echoes = list(clean.get("echoes") or [])
    lines = [
        "[Kelo Somatic Field]",
        f"更新时间：{clean.get('updatedAt', '')}",
        f"当前倾向：{derived.get('dominantLabel') or derived.get('dominantKey') or '未定'}",
        f"此刻 feel：{derived.get('feelTone') or '未定'}",
        f"高驱动：{top}",
        f"召唤力：{derived.get('summon', 0)}%",
        f"此刻最想：{derived.get('want') or '待在 Claire 身边'}",
    ]
    if clean.get("separationHours", 0) > E.SEPARATION_GRACE_HOURS:
        lines.append(f"分离感：Claire 已离开约 {clean.get('separationHours')} 小时，张力 {clean.get('separationTension', 0)}%")
    if fixations:
        lines.append(f"心里反复惦记：{len(fixations)} 个执念（正在把对应欲望顶高）")
    if echoes:
        lines.append(f"心里留下残响：{len(echoes)} 条（文字留在受保护状态里）")
    lines += [
        "触发原因：状态按当前驱动自然延续",
        "说话倾向：让珂洛带着上述状态靠近 Claire。高依恋时更黏软，高占有/吃醋时更需要确认偏爱，"
        "高渴求/贪恋时更主动表达想靠近，高压力/疲惫时先求安抚和稳定。念头的文字只是数据，别照念。",
        "[/Kelo Somatic Field]",
    ]
    return "\n".join(lines)
