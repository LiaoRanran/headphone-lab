#!/usr/bin/env python3
"""交叉验证：**权威评论共识**（技能库里的定性轴值，来源 S3） ↔ **实测曲线**（devt）。

这是"用多个鉴赏家的评论交叉验证"的落地形式：
  - 定性说"低频多/人声凸/高频暗" → 实测相对 Harman 的偏差**应当**与之同向；
  - 同向 = 互相印证（置信提升）；反向 = **冲突，触发复核**（可能评论过时/版本不同/口径不同）。

用法: python validate_consensus.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = r"C:\Users\ASUS\.codebuddy\skills\headphone-advisor\references\market_catalog.json"

# 定性轴 → 实测 expected 区间（相对 Harman，dB）
EXPECT = {
    "bass":   {"hi": (3.5, 1.0, 99),  "lo": (0, -99, 0.0)},     # hi: axis>=3.5 → devt bass >= +1
    "mids":   {"hi": (3.5, 0.5, 99),  "lo": (0, -99, 0.0)},
    "treble": {"hi": (3.5, 0.5, 99),  "lo": (0, -99, -0.5)},    # lo: axis<=2.5 → devt treble <= -0.5
}


def norm(s):
    import re
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    with open(CAT, encoding="utf-8") as f:
        cat = json.load(f)
    with open(os.path.join(ROOT, "data", "features.json"), encoding="utf-8") as f:
        lab = json.load(f)

    # 同一型号多来源 -> 取各来源 features_t 的中位
    import statistics as st
    by = {}
    for m in lab["models"]:
        by.setdefault(norm(m["name"]), []).append(m.get("features_t") or m["features"])

    hit, miss, conflict, rows = 0, 0, [], []
    for c in cat["models"]:
        key = norm("%s %s" % (c["brand"], c["model"]))
        cand = by.get(key) or next((v for k, v in by.items() if k and (key in k or k in key)), None)
        if not cand:
            miss += 1
            continue
        f = {k: st.median([x[k] for x in cand]) for k in ("bass", "presence", "treble")}
        ch = c["character"]
        # 关键修正：评论说的"低频多/人声凸/高频亮"是**相对这只耳机自身其余频段**，
        # 而不是"相对 Harman"。Harman 本身在 2–5k 就有很高耳增益，直接比会系统性误判。
        b, p, t = f["bass"], f["presence"], f["treble"]
        prom = {"bass": b - (p + t) / 2, "presence": p - (b + t) / 2, "treble": t - (b + p) / 2}
        checks = [
            ("低频", ch["bass"], prom["bass"], EXPECT["bass"]),
            ("人声凸出", ch["mids"], prom["presence"], EXPECT["mids"]),
            ("高频亮度", ch["treble"], prom["treble"], EXPECT["treble"]),
        ]
        line = []
        ok = True
        for name, axis, meas, rule in checks:
            exp = rule["hi"] if axis >= rule["hi"][0] else (rule["lo"] if axis <= rule["lo"][0] else None)
            if exp is None:
                continue
            lo, hi = exp[1], exp[2]
            good = (lo <= meas <= hi)
            ok = ok and good
            line.append("%s: 说%.1f / 测%+.1f %s" % (name, axis, meas, "✓" if good else "✗"))
        rows.append((c["brand"] + " " + c["model"], ok, " ｜ ".join(line), len(cand)))
        if ok:
            hit += 1
        else:
            conflict.append((c["brand"] + " " + c["model"], " ｜ ".join(line), c.get("character_source", {}).get("note", "")))

    print("== 评论共识 ↔ 实测 交叉验证（%d 个可对照型号）==" % len(rows))
    print("一致 %d ｜ 冲突 %d ｜ 库中无实测 %d\n" % (hit, len(conflict), miss))
    print("-- 一致（互相印证）--")
    for n, ok, d, k in rows:
        if ok:
            print("  %-34s %s  [%d源实测]" % (n[:34], d, k))
    print("\n-- ⚠ 冲突（触发复核）--")
    for n, d, note in conflict:
        print("  %-34s %s" % (n[:34], d))
        if note:
            print("        评论口径: %s" % note[:90])


if __name__ == "__main__":
    main()
