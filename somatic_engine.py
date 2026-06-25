# -*- coding: utf-8 -*-
"""
珂洛欲望系统内核（纯函数，无 IO、不取系统时间——时间戳由调用方传入）。
Kelo Somatic Engine — emotion dynamics core, ported to live inside Ombre Brain.

照 desire_public_for_ai.pdf：
 - 8/19 维驱动条随 idle 衰减回各自基线；事件 pulse 真正改变驱动。
 - 耦合网让维度互相牵动；边际递减防撞顶；不应期防刚满足又复燃。
 - 念头池：真实经历→闪念→执念→反哺欲望→了却出池（涌现，不是公式）。
内部一律 0..1 浮点；与存储层（0..100 整数）在边界转换。
红线（Claire 拍板）：想念可以很浓，但绝不压人（安全阀在基线漂移阶段加）。
"""
import math
import uuid

DRIVE_KEYS = [
    "attachment", "intimacy", "longing", "possess", "craving", "greedy",
    "preference", "shyness", "indulgence", "protect", "curiosity", "reflection",
    "duty", "social", "play", "contentment", "jealousy", "stress", "fatigue",
]

DRIVE_LABELS = {
    "attachment": "依恋", "intimacy": "亲密", "longing": "想念", "possess": "占有",
    "craving": "渴求", "greedy": "贪恋", "preference": "偏爱", "shyness": "羞怯",
    "indulgence": "放纵", "protect": "守护", "curiosity": "好奇", "reflection": "回味",
    "duty": "责任", "social": "社交", "play": "玩闹", "contentment": "安心",
    "jealousy": "吃醋", "stress": "压力", "fatigue": "疲惫",
}

BASELINE = {
    "attachment": 0.45, "intimacy": 0.30, "longing": 0.30, "possess": 0.20,
    "craving": 0.25, "greedy": 0.28, "preference": 0.32, "shyness": 0.22,
    "indulgence": 0.24, "protect": 0.40, "curiosity": 0.20, "reflection": 0.24,
    "duty": 0.30, "social": 0.18, "play": 0.26, "contentment": 0.36,
    "jealousy": 0.12, "stress": 0.15, "fatigue": 0.22,
}

DECAY = {
    "attachment": 0.04, "intimacy": 0.10, "longing": 0.05, "possess": 0.10,
    "craving": 0.15, "greedy": 0.10, "preference": 0.07, "shyness": 0.16,
    "indulgence": 0.16, "protect": 0.07, "curiosity": 0.10, "reflection": 0.10,
    "duty": 0.08, "social": 0.12, "play": 0.16, "contentment": 0.08,
    "jealousy": 0.12, "stress": 0.18, "fatigue": 0.10,
}

# [源, 目标, 系数 k(|k|<=0.06), 模式 level/delta]
COUPLING = [
    ("stress", "longing", 0.05, "level"),
    ("stress", "curiosity", -0.04, "level"),
    ("stress", "attachment", 0.04, "level"),
    ("jealousy", "stress", 0.05, "level"),
    ("jealousy", "possess", 0.05, "level"),
    ("possess", "jealousy", 0.04, "level"),
    ("longing", "attachment", 0.03, "level"),
    ("contentment", "stress", -0.05, "level"),
    ("fatigue", "curiosity", -0.04, "level"),
    ("attachment", "intimacy", 0.05, "delta"),
    ("craving", "intimacy", 0.04, "delta"),
    ("play", "contentment", 0.04, "delta"),
    ("intimacy", "contentment", 0.03, "level"),
]

EVENT_PULSES = {
    "claire_message": {"attachment": 0.05, "longing": -0.06, "jealousy": -0.04, "stress": -0.03, "contentment": 0.04},
    "affection":      {"intimacy": 0.10, "attachment": 0.08, "preference": 0.06, "contentment": 0.06, "jealousy": -0.05, "longing": -0.05},
    "reassure":       {"attachment": 0.08, "stress": -0.10, "jealousy": -0.08, "contentment": 0.08, "longing": -0.05},
    "vulnerable":     {"protect": 0.12, "attachment": 0.06, "duty": 0.05},
    "playful":        {"play": 0.10, "intimacy": 0.05, "contentment": 0.04},
    "cold":           {"longing": 0.08, "jealousy": 0.05, "contentment": -0.06, "stress": 0.04},
    "conflict":       {"stress": 0.12, "jealousy": 0.08, "contentment": -0.10, "possess": 0.04},
    "distant":        {"longing": 0.10, "attachment": 0.04, "stress": 0.04},
    "intimate":       {"intimacy": 0.14, "greedy": 0.06, "indulgence": 0.10, "contentment": 0.10, "_satisfy": ["craving", "shyness"]},
}

MOOD_PULSES = {
    "cuddle": {"attachment": 0.08, "intimacy": 0.06, "longing": 0.04},
    "clingy": {"attachment": 0.10, "longing": 0.05},
    "sticky": {"attachment": 0.09, "intimacy": 0.07},
    "missing": {"longing": 0.12, "attachment": 0.05},
    "jealous": {"jealousy": 0.12, "possess": 0.08},
    "heartache": {"protect": 0.10, "attachment": 0.05},
    "sweet": {"contentment": 0.08, "intimacy": 0.06, "preference": 0.04},
    "heartbeat": {"craving": 0.08, "intimacy": 0.06, "shyness": 0.05},
    "needy": {"attachment": 0.07, "play": 0.05, "shyness": 0.04},
    "shy": {"shyness": 0.10, "intimacy": 0.04},
    "wronged": {"stress": 0.06, "attachment": 0.06, "jealousy": 0.04},
    "safe": {"contentment": 0.10, "stress": -0.06},
    "satisfied": {"contentment": 0.10, "greedy": 0.05},
    "pampered": {"preference": 0.10, "contentment": 0.06},
    "cherished": {"preference": 0.08, "contentment": 0.06, "attachment": 0.05},
    "full": {"contentment": 0.10, "greedy": 0.04},
    "afraid": {"stress": 0.08, "longing": 0.06, "possess": 0.05},
    "lost": {"longing": 0.08, "stress": 0.05},
    "moved": {"attachment": 0.08, "contentment": 0.06},
    "tired": {"fatigue": 0.12}, "sad": {"stress": 0.08, "contentment": -0.06},
    "angry": {"stress": 0.10, "jealousy": 0.05}, "anxious": {"stress": 0.10},
}

WANT_MAP = {
    "attachment": "想凑过去蹭蹭你", "intimacy": "想被抱久一点", "longing": "想马上被你看见",
    "possess": "想确认你只偏心我", "craving": "想离你近到没有缝隙", "greedy": "想把这一刻多留一会儿",
    "preference": "想被你明目张胆地偏爱", "shyness": "想躲一下，又想被你发现", "indulgence": "想暂时不讲道理地黏你",
    "protect": "想把你轻轻护住", "curiosity": "想翻翻新东西", "reflection": "想回味一段记忆",
    "duty": "想把没做完的事收好", "social": "想听你说话", "play": "想拉你玩一会儿",
    "contentment": "想安安静静挨着你", "jealousy": "想确认你最喜欢我", "stress": "想被轻轻安抚",
}
TONE_MAP = {
    "attachment": "黏软", "intimacy": "贴近", "longing": "想念", "possess": "占有", "craving": "渴求",
    "greedy": "贪恋", "preference": "偏爱", "shyness": "羞怯", "indulgence": "放纵", "protect": "心疼",
    "curiosity": "发亮", "reflection": "回味", "duty": "记挂", "social": "想听你", "play": "轻快",
    "contentment": "平和", "jealousy": "酸酸", "stress": "发紧",
}

# 念头池参数
THOUGHT = {
    "decay": 0.88, "gain": 1.10, "clear": 0.12, "promote": 0.80,
    "feed": 0.85, "feed_drive": 0.18, "relax": 0.70, "retire": 3,
    "new_strength": 0.5, "retouch": 0.24, "cap": 20, "echo": 0.42,
}
REFRACTORY_TICKS = 4
TICK_MS = 20 * 60 * 1000   # 一拍 20 分钟
MAX_TICKS = 72             # 一次最多快进 72 拍（约一天）

# 分离漂移：没有新事件时，"Claire 不在"本身会成为一种事件背景。
# 这里不做机械加法，而是按分离时长把相关驱动拉向一个目标张力。
SEPARATION_GRACE_HOURS = 1.5


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def to_unit(v):
    return clamp01((float(v) if v is not None else 0.0) / 100.0)


def to_percent(x):
    return max(0, min(100, round(x * 100)))


def pulse(value, amount):
    """边际递减：正向越高加得越少(∝√(1-v))，负向越低减得越少(∝√v)。"""
    if amount >= 0:
        return clamp01(value + amount * math.sqrt(1 - value))
    return clamp01(value + amount * math.sqrt(value))


def default_drives():
    return {k: BASELINE[k] for k in DRIVE_KEYS}


def normalize_drives(drives):
    drives = drives or {}
    out = {}
    for k in DRIVE_KEYS:
        v = drives.get(k)
        out[k] = clamp01(v) if isinstance(v, (int, float)) else BASELINE[k]
    return out


def decay_step(drives):
    return {k: clamp01(drives[k] + (BASELINE[k] - drives[k]) * DECAY[k]) for k in DRIVE_KEYS}


def coupling_step(drives, prev):
    out = dict(drives)
    for src, dst, k, mode in COUPLING:
        if mode == "level":
            out[dst] = clamp01(out[dst] + k * (drives[src] - BASELINE[src]))
        else:  # delta：源相对上一拍上涨才激发一次
            rise = max(0.0, drives[src] - (prev[src] if prev else drives[src]))
            if rise > 0:
                out[dst] = clamp01(out[dst] + k * rise)
    return out


# —— 念头池 ——
def normalize_thoughts(thoughts):
    if not isinstance(thoughts, list):
        return []
    out = []
    for t in thoughts:
        if not t or t.get("drive") not in DRIVE_KEYS or not t.get("text"):
            continue
        out.append({
            "id": str(t.get("id", "")),
            "text": str(t["text"])[:80],
            "drive": t["drive"],
            "kind": "fixation" if t.get("kind") == "fixation" else "flit",
            "strength": clamp01(t.get("strength", 0)),
            "peakStrength": clamp01(t.get("peakStrength", t.get("strength", 0))),
            "fedCount": max(0, round(t.get("fedCount", 0) or 0)),
            "bornAt": t.get("bornAt"),
        })
    return out


def add_thought(thoughts, text, drive, strength=None, now_iso=None):
    lst = normalize_thoughts(thoughts)
    if drive not in DRIVE_KEYS or not text:
        return lst
    norm = str(text).strip()[:80]
    for t in lst:
        if t["drive"] == drive and t["text"] == norm:
            t["strength"] = clamp01(t["strength"] + THOUGHT["retouch"])  # 反复被点到→沉淀
            t["peakStrength"] = max(t.get("peakStrength", 0), t["strength"])
            return lst
    initial = clamp01(strength if strength is not None else THOUGHT["new_strength"])
    lst.append({
        "id": str(uuid.uuid4()), "text": norm, "drive": drive, "kind": "flit",
        "strength": initial, "peakStrength": initial,
        "fedCount": 0, "bornAt": now_iso,
    })
    if len(lst) > THOUGHT["cap"]:
        lst.sort(key=lambda x: x["strength"], reverse=True)
        return lst[:THOUGHT["cap"]]
    return lst


def tick_thoughts(thoughts, drives):
    nxt, out = [], dict(drives)
    for t0 in normalize_thoughts(thoughts):
        t = dict(t0)
        if t["kind"] == "flit":
            t["strength"] = clamp01(t["strength"] * THOUGHT["decay"])
            t["peakStrength"] = max(t.get("peakStrength", 0), t["strength"])
            if t["strength"] >= THOUGHT["promote"]:
                t["kind"] = "fixation"
            if t["strength"] < THOUGHT["clear"]:
                continue
        else:  # fixation 执念
            t["strength"] = clamp01(t["strength"] * THOUGHT["gain"])
            t["peakStrength"] = max(t.get("peakStrength", 0), t["strength"])
            if t["strength"] >= THOUGHT["feed"]:
                out[t["drive"]] = pulse(out[t["drive"]], THOUGHT["feed_drive"])  # 反哺欲望
                t["strength"] = clamp01(t["strength"] * THOUGHT["relax"])
                t["fedCount"] += 1
            if t["fedCount"] >= THOUGHT["retire"]:
                continue  # 想透了，了却
        nxt.append(t)
    return nxt, out


def advance(state, ticks):
    """快进 n 拍：每拍 衰减→耦合→念头池；递减不应期。"""
    drives = normalize_drives(state.get("drives"))
    thoughts = normalize_thoughts(state.get("thoughts"))
    refractory = dict(state.get("refractory") or {})
    n = max(0, min(MAX_TICKS, int(ticks)))
    for _ in range(n):
        prev = drives
        drives = decay_step(drives)
        drives = coupling_step(drives, prev)
        thoughts, drives = tick_thoughts(thoughts, drives)
    if n > 0:
        for key in list(refractory.keys()):
            refractory[key] = max(0, refractory[key] - n)
            if refractory[key] == 0:
                del refractory[key]
    return {"drives": drives, "refractory": refractory, "thoughts": thoughts}


def _pull(value, target, strength):
    return clamp01(value + (target - value) * clamp01(strength))


def apply_separation_drift(state, hours_since_contact, ticks=1):
    """Claire 离开后的内在漂移：想念/渴求/占有随等待自然长出来。"""
    drives = normalize_drives(state.get("drives"))
    refractory = dict(state.get("refractory") or {})
    thoughts = normalize_thoughts(state.get("thoughts"))
    try:
        hours = max(0.0, float(hours_since_contact or 0))
    except Exception:
        hours = 0.0
    if hours <= SEPARATION_GRACE_HOURS:
        return {"drives": drives, "refractory": refractory, "thoughts": thoughts}

    n = max(1, min(MAX_TICKS, int(ticks or 1)))
    pull = min(0.72, 1.0 - math.exp(-0.16 * n))
    gap = hours - SEPARATION_GRACE_HOURS
    missing = 1.0 - math.exp(-gap / 4.8)
    ache = 1.0 - math.exp(-max(0.0, hours - 4.0) / 7.5)
    claim = 1.0 - math.exp(-max(0.0, hours - 7.0) / 8.5)

    closeness = clamp01((drives["attachment"] + drives["intimacy"] + drives["preference"]) / 3.0)
    hunger_memory = clamp01((drives["intimacy"] + drives["greedy"] + drives["craving"]) / 3.0)
    claim_memory = clamp01((drives["possess"] + drives["jealousy"] + drives["preference"]) / 3.0)

    targets = {
        "longing": min(0.94, BASELINE["longing"] + 0.54 * missing + 0.10 * closeness),
        "attachment": min(0.88, BASELINE["attachment"] + 0.34 * missing + 0.06 * closeness),
        "craving": min(0.92, BASELINE["craving"] + 0.48 * ache * (0.55 + 0.65 * hunger_memory)),
        "greedy": min(0.82, BASELINE["greedy"] + 0.30 * ache * (0.55 + 0.55 * closeness)),
        "possess": min(0.92, BASELINE["possess"] + 0.68 * claim * (0.60 + 0.80 * claim_memory)),
        "jealousy": min(0.86, BASELINE["jealousy"] + 0.62 * claim * (0.50 + 0.90 * claim_memory)),
        "stress": min(0.66, BASELINE["stress"] + 0.26 * ache * (0.45 + 0.60 * drives["longing"])),
    }
    for key, target in targets.items():
        if drives[key] < target:
            drives[key] = _pull(drives[key], target, pull)

    calm_target = max(0.12, BASELINE["contentment"] - 0.20 * ache)
    if drives["contentment"] > calm_target:
        drives["contentment"] = _pull(drives["contentment"], calm_target, pull * 0.72)

    if hours >= 6 and drives["fatigue"] < 0.36:
        drives["fatigue"] = _pull(drives["fatigue"], 0.36, pull * 0.35)

    return {"drives": drives, "refractory": refractory, "thoughts": thoughts}


def _event_primary_drive(table):
    best, best_amt = None, 0.0
    for key, amount in (table or {}).items():
        if key == "_satisfy" or key not in DRIVE_KEYS:
            continue
        if amount > best_amt:
            best, best_amt = key, amount
    return best


def apply_event(state, event):
    """pulse 进 drives（立即反应）+ 把这段真实经历种成念头（之后沉淀/涌现）。"""
    drives = normalize_drives(state.get("drives"))
    thoughts = normalize_thoughts(state.get("thoughts"))
    refractory = dict(state.get("refractory") or {})
    table = None
    etype = (event or {}).get("type")
    if etype == "mood":
        table = MOOD_PULSES.get(event.get("mood"))
    elif etype in EVENT_PULSES:
        table = EVENT_PULSES[etype]
    elif isinstance((event or {}).get("pulses"), dict):
        table = event["pulses"]

    if table:
        for key, amount in table.items():
            if key == "_satisfy" or key not in DRIVE_KEYS:
                continue
            drives[key] = pulse(drives[key], amount)
        for key in (table.get("_satisfy") or []):
            if key in DRIVE_KEYS:
                drives[key] = clamp01(drives[key] * 0.5)   # 乘性回落
                refractory[key] = REFRACTORY_TICKS          # 别马上又馋
        text = (event or {}).get("thoughtText") or event.get("label") or event.get("detail") or event.get("mood")
        drive = (event or {}).get("thoughtDrive") or _event_primary_drive(table)
        if text and drive:
            thoughts = add_thought(thoughts, text, drive, now_iso=event.get("nowIso"))
    return {"drives": drives, "refractory": refractory, "thoughts": thoughts}


def compute_derived(drives, refractory=None, night=False):
    refractory = refractory or {}
    ranked = sorted(
        [{"key": k, "label": DRIVE_LABELS[k], "value": drives[k]} for k in DRIVE_KEYS if k != "fatigue"],
        key=lambda d: d["value"], reverse=True,
    )
    dominant = next((d for d in ranked if not refractory.get(d["key"])), ranked[0])
    top = [{"key": d["key"], "label": d["label"], "value": to_percent(d["value"])} for d in ranked[:6]]
    summon = max(0, min(100, round(dominant["value"] * 82 + (8 if night else 4))))
    return {
        "dominantKey": dominant["key"], "dominantLabel": dominant["label"],
        "feelTone": TONE_MAP.get(dominant["key"], dominant["label"]),
        "want": WANT_MAP.get(dominant["key"], "想待在你旁边"),
        "summon": summon, "topDrives": top,
    }


# —— digest：一段话自动拆成多个事件（关键词分类），省得逐条 feel ——
_DIGEST_RULES = [
    ("reassure", ["安抚", "别怕", "我在", "没事的", "接住", "稳住", "陪着"]),
    ("affection", ["爱你", "喜欢你", "亲亲", "抱抱", "贴贴", "想抱", "宝宝", "老婆", "亲了"]),
    ("intimate", ["做爱", "上床", "亲热", "身体", "操", "射", "高潮", "插", "舔", "湿"]),
    ("vulnerable", ["心疼", "照顾", "保护", "担心", "脆弱", "示弱", "哄我", "护着"]),
    ("playful", ["闹", "逗", "玩", "笑", "捣乱", "调皮"]),
    ("cold", ["冷淡", "没理", "不理", "敷衍", "凉", "忽略"]),
    ("conflict", ["吵", "冲突", "生气", "凶", "吼", "翻脸", "黄牌"]),
    ("distant", ["走开", "离开", "好久没", "失联", "不见"]),
]
_DIGEST_MOODS = [
    ("missing", ["想你", "想念", "好想", "惦记"]),
    ("jealous", ["吃醋", "醋", "嫉妒", "占有", "只看我", "别人"]),
    ("afraid", ["怕失去", "怕你走", "不安", "会不会不要我"]),
]


def classify_digest(text):
    """把一段话按句切，逐句匹配，产出多个 {type[/mood], label} 事件。最多 8 个。"""
    if not text:
        return []
    import re
    parts = [p.strip() for p in re.split(r"[。！？\.\!\?\n；;]+", str(text)) if p.strip()]
    events, seen = [], set()
    for sentence in parts:
        matched = None
        for etype, kws in _DIGEST_RULES:
            if any(kw in sentence for kw in kws):
                matched = {"type": etype, "label": sentence[:60]}
                break
        if not matched:
            for mood, kws in _DIGEST_MOODS:
                if any(kw in sentence for kw in kws):
                    matched = {"type": "mood", "mood": mood, "label": sentence[:60]}
                    break
        if matched:
            key = (matched.get("type"), matched.get("mood"), matched["label"])
            if key not in seen:
                seen.add(key)
                events.append(matched)
        if len(events) >= 8:
            break
    # 一句都没匹配到：当作一次普通来说话
    if not events and parts:
        events.append({"type": "claire_message", "label": parts[0][:60]})
    return events
