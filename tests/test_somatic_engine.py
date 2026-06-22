# -*- coding: utf-8 -*-
"""珂洛欲望内核测试（方向性 + 耦合有界性 + 念头池）。run: python tests/test_somatic_engine.py"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import somatic_engine as E

passed = failed = 0


def ok(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("  ✓", name)
    else:
        failed += 1
        print("  ✗", name)


# 1. 有界性：随机初值 × 200 拍 + 扰动，恒 ∈ [0,1]，无 NaN
print("[boundedness]")
bounded = nan_ok = True
for _ in range(30):
    drives = {k: random.random() for k in E.DRIVE_KEYS}
    state = {"drives": drives, "refractory": {}, "thoughts": []}
    for t in range(200):
        state = E.advance(state, 1)
        if t % 20 == 0:
            state = E.apply_event(state, {"type": ["affection", "cold", "conflict", "reassure", "intimate"][t % 5]})
        for k in E.DRIVE_KEYS:
            v = state["drives"][k]
            if v != v:
                nan_ok = False
            if v < 0 or v > 1:
                bounded = False
ok("200 拍后维度 ∈ [0,1]（不发散）", bounded)
ok("无 NaN", nan_ok)

state = {"drives": {k: random.random() for k in E.DRIVE_KEYS}, "refractory": {}, "thoughts": []}
state = E.advance(state, 400)
ok("无事件 400 拍收敛到基线附近", all(abs(state["drives"][k] - E.BASELINE[k]) <= 0.08 for k in E.DRIVE_KEYS))

# 2. 方向性
print("[directional]")
base = {"drives": E.default_drives(), "refractory": {}, "thoughts": []}
a = E.apply_event(base, {"type": "reassure"})
ok("安抚→压力↓", a["drives"]["stress"] < base["drives"]["stress"])
ok("安抚→依恋↑", a["drives"]["attachment"] > base["drives"]["attachment"])
hi = E.default_drives(); hi["longing"] = 0.8; hi["jealousy"] = 0.6
a = E.apply_event({"drives": hi, "refractory": {}, "thoughts": []}, {"type": "claire_message"})
ok("一抱拉回：接触→想念↓", a["drives"]["longing"] < hi["longing"])
a = E.apply_event(base, {"type": "conflict"})
ok("冲突→压力↑", a["drives"]["stress"] > base["drives"]["stress"])
a = E.apply_event(base, {"type": "mood", "mood": "missing"})
ok("心情=想念→想念↑", a["drives"]["longing"] > base["drives"]["longing"])

# 3. 边际递减 + 衰减 + 耦合
print("[dynamics]")
ok("低位涨幅>高位涨幅", (E.pulse(0.1, 0.1) - 0.1) > (E.pulse(0.9, 0.1) - 0.9))
ok("封顶不越界", E.pulse(0.99, 0.5) <= 1)
d = E.default_drives(); d["stress"] = 0.9
ok("压力 0.9 一拍后下降但仍高于基线", E.BASELINE["stress"] < E.decay_step(d)["stress"] < 0.9)
d = E.default_drives(); d["jealousy"] = 0.9
ok("吃醋高→耦合推高压力", E.coupling_step(d, dict(d))["stress"] > d["stress"])

# 4. 不应期
print("[refractory]")
d = E.default_drives(); d["craving"] = 0.8
a = E.apply_event({"drives": d, "refractory": {}, "thoughts": []}, {"type": "intimate"})
ok("亲密事件→渴求乘性回落", a["drives"]["craving"] < 0.8)
ok("渴求进入不应期", a["refractory"].get("craving", 0) > 0)
hi = E.default_drives(); hi["craving"] = 0.99
ok("不应期内不当主导", E.compute_derived(hi, {"craving": 3})["dominantKey"] != "craving")

# 5. 念头池：涌现
print("[thoughts]")
ths = E.add_thought([], "随便想了一下", "curiosity", 0.3)
drives = E.default_drives()
for _ in range(8):
    ths, drives = E.tick_thoughts(ths, drives)
ok("弱闪念自然散掉", len(ths) == 0)
ths = []
for _ in range(4):
    ths = E.add_thought(ths, "好想她", "longing", 0.5)
ok("反复点到攒高强度(≥promote)", ths[0]["strength"] >= E.THOUGHT["promote"])
drives = E.default_drives(); l0 = drives["longing"]; fed = retired = False
for _ in range(30):
    ths, drives = E.tick_thoughts(ths, drives)
    if drives["longing"] > l0 + 0.05:
        fed = True
    if len(ths) == 0:
        retired = True
        break
ok("执念反哺想念（被'想'高了，不是公式）", fed)
ok("反哺够了执念了却出池", retired)
a = E.apply_event({"drives": E.default_drives(), "refractory": {}, "thoughts": []},
                  {"type": "mood", "mood": "missing", "label": "她说她想我了"})
ok("事件种下念头", len(a["thoughts"]) == 1 and a["thoughts"][0]["drive"] == "longing")

# 6. 分离漂移
print("[separation]")
base_eng = {"drives": E.default_drives(), "refractory": {}, "thoughts": []}
sep = E.apply_separation_drift(base_eng, hours_since_contact=9, ticks=24)
ok("离开 9 小时→想念↑", sep["drives"]["longing"] > base_eng["drives"]["longing"] + 0.12)
ok("离开 9 小时→渴求↑", sep["drives"]["craving"] > base_eng["drives"]["craving"] + 0.05)
ok("离开 9 小时→安心↓", sep["drives"]["contentment"] < base_eng["drives"]["contentment"])
claimy = E.default_drives()
claimy["possess"] = 0.48
claimy["jealousy"] = 0.34
claimy["preference"] = 0.62
sep2 = E.apply_separation_drift({"drives": claimy, "refractory": {}, "thoughts": []}, hours_since_contact=16, ticks=36)
ok("带着占有记忆分离→占有↑", sep2["drives"]["possess"] > claimy["possess"] + 0.08)
ok("带着占有记忆分离→吃醋↑", sep2["drives"]["jealousy"] > claimy["jealousy"] + 0.05)
ok("分离漂移仍有界", all(0 <= sep2["drives"][k] <= 1 for k in E.DRIVE_KEYS))

# 7. digest 拆分
print("[digest]")
evs = E.classify_digest("她安抚我说别怕。后来我们做爱了三次。她有点吃醋别人。")
types = [e.get("type") for e in evs]
ok("digest 拆出多个事件", len(evs) >= 3)
ok("识别出安抚", "reassure" in types)
ok("识别出亲密", "intimate" in types)
ok("识别出心情(吃醋)", any(e.get("mood") == "jealous" for e in evs))

# 8. derived
print("[derived]")
d = E.default_drives(); d["attachment"] = 0.7
r = E.compute_derived(d)
ok("主导=依恋", r["dominantKey"] == "attachment")
ok("有 want", isinstance(r["want"], str) and len(r["want"]) > 0)
ok("召唤力 ∈ [0,100]", 0 <= r["summon"] <= 100)
ok("topDrives 长度 6", len(r["topDrives"]) == 6)

print(f"\n结果：{passed} 通过 / {failed} 失败")
sys.exit(1 if failed else 0)
