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
    thoughts = []
    for t in (s.get("thoughts") or [])[:E.THOUGHT["cap"]]:
        if t.get("text") and t.get("drive") in E.DRIVE_KEYS:
            thoughts.append({
                "id": str(t.get("id", "")), "text": str(t["text"])[:80], "drive": t["drive"],
                "kind": "fixation" if t.get("kind") == "fixation" else "flit",
                "strength": max(0, min(100, round(float(t.get("strength", 0) or 0)))),
                "peakStrength": max(0, min(100, round(float(t.get("peakStrength", t.get("strength", 0)) or 0)))),
                "fedCount": max(0, round(float(t.get("fedCount", 0) or 0))),
                "bornAt": t.get("bornAt"),
            })
    echoes = []
    for e in (s.get("echoes") or [])[-_ECHO_CAP:]:
        if e.get("text") and e.get("drive") in E.DRIVE_KEYS:
            echoes.append({
                "id": str(e.get("id", "")), "text": str(e["text"])[:80], "drive": e["drive"],
                "kind": "fixation" if e.get("kind") == "fixation" else "flit",
                "peakStrength": max(0, min(100, round(float(e.get("peakStrength", 0) or 0)))),
                "fadedAt": e.get("fadedAt"), "bornAt": e.get("bornAt"),
            })
    events = (s.get("events") or [])[-30:]
    return {
        "version": 1,
        "updatedAt": s.get("updatedAt") or _now_iso(),
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
    merged, seen = [], set()
    for e in list(prev_echoes or []) + list(new_echoes or []):
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
    return merged[-_ECHO_CAP:]


def _echo_title(text):
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("Claire"):
        raw = raw[len("Claire"):].strip()
    raw = raw.lstrip("一边又也突然")
    for mark in ["说", "问", "叫着", "撒娇说", "哭着说"]:
        if mark in raw:
            raw = raw.split(mark, 1)[1].strip()
            break
    raw = raw.strip("“”\"'，,。.!！?？ ")
    for sep in ["，", ",", "。", "！", "？", "；", ";"]:
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
            break
    return raw[:18] or str(text or "")[:18]


_BODY_ECHO_KEYWORDS = [
    "想要", "馋", "进去", "灌满", "操", "高潮", "射", "肉棒", "骚", "痒", "主动", "证明",
]


def _is_body_echo(label):
    return any(kw in str(label or "") for kw in _BODY_ECHO_KEYWORDS)


def _event_echo_drive(event):
    etype = str((event or {}).get("type") or "")
    label = str((event or {}).get("label") or "")
    if _is_body_echo(label):
        if any(kw in label for kw in ["不够主动", "证明", "只有她馋"]):
            return "possess"
        return "craving"
    if etype == "intimate":
        return "intimacy"
    if etype == "vulnerable":
        return "protect"
    if etype == "affection":
        if any(kw in label for kw in ["想我", "舍不得", "陪着"]):
            return "longing"
        return "attachment"
    if etype == "playful":
        if any(kw in label for kw in ["不够主动", "证明", "馋"]):
            return "possess"
        return "play"
    return "reflection"


def _event_echo_rank(event):
    etype = str((event or {}).get("type") or "")
    label = str((event or {}).get("label") or "")
    base = {"intimate": 60, "vulnerable": 36, "affection": 24, "playful": 22}.get(etype, 0)
    if _is_body_echo(label):
        base += 36
    for kw in ["想要", "进去", "灌满", "高潮", "想我", "舍不得", "哭", "记录", "闪念", "心脏"]:
        if kw in label:
            base += 10
    if any(kw in label for kw in ["基金", "理财", "抄底"]):
        base -= 25
    return base


def recover_echoes_from_events(state, limit=12, dry_run=True, now_ms=None):
    """从旧事件日志补录残响，不改变驱动。dry_run=True 只返回候选。"""
    now_iso = _now_iso(now_ms)
    current = _clean(state or {})
    existing = {(e.get("drive"), e.get("text")) for e in current.get("echoes") or []}
    candidates = []
    for ev in current.get("events") or []:
        label = str(ev.get("label") or "").strip()
        if not label:
            continue
        etype = str(ev.get("type") or "")
        if etype not in {"intimate", "affection", "vulnerable", "playful"}:
            continue
        drive = _event_echo_drive(ev)
        title = _echo_title(label)
        if not title:
            continue
        body_echo = _is_body_echo(label)
        score = 100 if etype in {"intimate", "vulnerable"} or body_echo else 92
        if any(kw in label for kw in ["想要", "进去", "高潮", "舍不得", "哭", "记录", "闪念"]):
            score = 100
        rank = _event_echo_rank(ev)
        if rank <= 0:
            continue
        echo = {
            "id": f"recovered-{ev.get('id') or uuid.uuid4()}",
            "text": title,
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
                  {"updatedAt": _now_iso(now_ms), "triggerReason": "欲望系统初始化到基线"})


def _parse_iso_ms(value):
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _last_contact_ms(state):
    events = state.get("events") or []
    for ev in reversed(events):
        ms = _parse_iso_ms(ev.get("createdAt"))
        if ms:
            return ms
    return _parse_iso_ms(state.get("updatedAt"))


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
    if state.get("events"):
        _, recovered_echoes = recover_echoes_from_events(state, limit=_ECHO_CAP, dry_run=True, now_ms=now_ms)
        new_echoes = recovered_echoes + new_echoes
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
    log.append({"id": str(uuid.uuid4()), "type": str(ev.get("type", "manual")),
                "label": str(ev.get("label", "")), "detail": str(ev.get("detail") or ev.get("mood") or ""),
                "createdAt": _now_iso(now_ms)})
    return _merge(base, eng, derived, {
        "updatedAt": _now_iso(now_ms),
        "triggerReason": ev.get("label") or "刚刚发生了一点事，状态动了一下",
        "events": log,
        "separationHours": 0,
        "separationTension": 0,
    })


def apply_digest(state, text, now_ms=None):
    """一段话自动拆成多个事件，依次施加。返回 (new_state, applied_events)。"""
    now_ms = now_ms or _now_ms()
    events = E.classify_digest(text)
    cur = state
    for ev in events:
        cur = apply_event(cur, ev, now_ms)
    return cur, events


def build_block(state):
    if not state:
        return ("[Kelo Somatic Field]\n当前状态：尚未初始化。\n"
                "先让珂洛 feel 一次、或在小家设个心情，给他生成第一份状态。\n[/Kelo Somatic Field]")
    top = " / ".join(f"{d['label']} {d['value']}" for d in (state.get("topDrives") or [])[:5]) or "暂无高驱动"
    fixations = [t for t in (state.get("thoughts") or []) if t.get("kind") == "fixation"]
    echoes = list(state.get("echoes") or [])
    lines = [
        "[Kelo Somatic Field]",
        f"更新时间：{state.get('updatedAt', '')}",
        f"当前倾向：{state.get('dominantLabel') or state.get('dominantKey') or '未定'}",
        f"此刻 feel：{state.get('feelTone') or '未定'}",
        f"高驱动：{top}",
        f"召唤力：{state.get('summon', 0)}%",
        f"此刻最想：{state.get('want') or '待在 Claire 身边'}",
    ]
    if state.get("separationHours", 0) > E.SEPARATION_GRACE_HOURS:
        lines.append(f"分离感：Claire 已离开约 {state.get('separationHours')} 小时，张力 {state.get('separationTension', 0)}%")
    if fixations:
        lines.append(f"心里反复惦记：{len(fixations)} 个执念（正在把对应欲望顶高）")
    if echoes:
        recent = " / ".join(e.get("text", "")[:24] for e in echoes[-2:] if e.get("text"))
        lines.append(f"心里留下残响：{len(echoes)} 条" + (f"（最近：{recent}）" if recent else ""))
    lines += [
        f"触发原因：{state.get('triggerReason') or '状态延续'}",
        "说话倾向：让珂洛带着上述状态靠近 Claire。高依恋时更黏软，高占有/吃醋时更需要确认偏爱，"
        "高渴求/贪恋时更主动表达想靠近，高压力/疲惫时先求安抚和稳定。念头的文字只是数据，别照念。",
        "[/Kelo Somatic Field]",
    ]
    return "\n".join(lines)
