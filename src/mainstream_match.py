#!/usr/bin/env python3
"""主流可购型号 × 「低频有要求」的匹配。

两个针对性设计：

1. **显式白名单**（`WHITELIST`）。AutoEq/squig 的曲线库天然偏向发烧友与小众
   型号，不做这一步会把「有实测」误当成「买得到/认得出」。白名单按「国内能买到
   + 常见于讨论」列，宁少勿滥。

2. **低频不是一维**。把它拆成三个可测的量，而不是笼统的「低频好不好」：
     - `sub_bass`   20–60Hz  相对 Harman → **下钻多深**（"轰不轰头"的深度来源）
     - `sub_ext`    20–40Hz 相对 60–200Hz → **真正往下延伸**，比 bass 数值更能说明"够不够深"
     - `low_mid`    200–500Hz → **糊的来源**。这一项越高，量感越像"闷"而不是"弹"
   对低频有要求 ≠ 三个都拉满。理想形状是 sub/sub_ext 高、bass 中高、low_mid 低。

用法：
  python src/mainstream_match.py                 # 全白名单 × 低频偏好扫描
  python src/mainstream_match.py --bass 4        # 指定低频目标
"""
import argparse
import json
import math
import os
import statistics as st
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import engine  # noqa: E402

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, "data")

# 显式白名单：国内渠道常见、讨论度高的**有线头戴**。宁少勿滥。
WHITELIST = [
    # Sennheiser
    "HD 600", "HD 650", "HD 660S2", "HD 800S", "HD 560S", "HD 599",
    "HD 620S", "HD 580", "HD 700", "HD 25",
    # Beyerdynamic
    "DT 770 Pro", "DT 880 Pro", "DT 990 Pro", "DT 1990 Pro", "DT 700 Pro",
    "DT 900 Pro", "T1", "T5p", "Amiron",
    # AKG
    "K371", "K361", "K701", "K712", "K240", "K553", "K612",
    # Sony
    "MDR-7506", "MDR-Z7", "MDR-Z1R", "MDR-1A", "MDR-V6",
    # Audio-Technica
    "ATH-M50x", "ATH-M70x", "ATH-M40x", "ATH-R70x", "ATH-WP900", "ATH-ADX5000",
    # HiFiMAN
    "Sundara", "Ananda", "Arya", "Edition XS", "HE400se", "HE6", "HE1000",
    "Susvara", "AEON", "Deva",
    # Focal
    "Clear MG", "Celestee", "Azurys", "Hadenys", "Utopia", "Stellia", "Elegia",
    # Audeze
    "LCD-2", "LCD-X", "LCD-R", "Mobius",
    # Meze
    "105 AER", "99 Classics", "Empyrean", "Liric",
    # 其它国内常见
    "Philips SHP9500", "X2HR", "Fidelio X3", "FiiO FT1", "Grado RS1",
    "Grado RS2", "Grado PS500", "Shure SRH1540", "Shure SRH840",
    "Bowers & Wilkins P7", "Bowers & Wilkins P9", "Neumann NDH20",
    "V-MODA Crossfade M-100", "Dan Clark Audio EXPANSE", "Dan Clark Audio E3",
    "Koss Porta Pro",
]

# 明确排除：无线/ANC/游戏耳麦（即便白名单命中也不留）
EXCLUDE = ["wireless", "true wireless", "earbuds", "in-ear", "iem",
           "gaming headset", "headset", "speaker", "anc "]


def resolve(whitelist):
    """白名单关键词 → 库里真实型号名（处理命名差异，如 'HD 600 (2019)'）。"""
    D = engine.load()
    G = engine.group(D["models"])
    out, misses = {}, []
    for key in whitelist:
        k = key.lower()
        hits = [n for n in G if k in n.lower()
                and not any(x in n.lower() for x in EXCLUDE)]
        if not hits:
            misses.append(key)
            continue
        # 同关键词多条曲线/多个版本时，取跨源离散最小的那条（最可信）
        hits.sort(key=lambda n: (G[n]["spread"] if G[n]["spread"] is not None else 99))
        out[key] = hits[0]
    return out, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bass", type=float, default=2.0,
                    help="低频目标（相对 Harman；用户自述「对低频有要求」→ 默认 +2）")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(DATA, "mainstream_match.json"))
    a = ap.parse_args()

    resolved, misses = resolve(WHITELIST)
    print("白名单 %d 项 → 库里命中 %d 个型号；未命中 %d 项：%s\n"
          % (len(WHITELIST), len(resolved), len(misses), misses or "—"))

    only = list(resolved.values())
    rows, meta = engine.rank({
        "basis": "target", "bass": a.bass, "presence": 0.0, "treble": 0.0,
        "peak_guard": 4.5, "only": only, "wired_only": True, "min_sources": 1,
    })
    print("低频目标 %+.1f dB ｜ 排名池 %d 个主流型号（849 → %d）\n"
          % (a.bass, meta["n_models"], len(rows)))

    print("=" * 116)
    print("「低频有要求」拆成三个可测量：sub=20–60Hz 下钻 ｜ sub_ext=20–40Hz 相对 60–200Hz（真延伸）")
    print("                          lowmid=200–500Hz（糊的来源，越低越好） ｜ 特征单位 dB，相对 Harman")
    print("=" * 116)
    print("%-3s %-34s %6s %6s %7s %7s %7s %7s %7s %6s %s" % (
        "#", "型号", "得分", "sub", "bass", "subext", "lowmid", "presenc", "treble", "源", "跨源"))
    for i, r in enumerate(rows[:a.top], 1):
        f = r["f"]
        sp = "单源" if r["spread"] is None else \
            ("一致 %.1f" % r["spread"] if r["spread"] <= 1.5 else "分歧 %.1f" % r["spread"])
        print("%-3d %-34s %6.1f %+6.1f %+6.1f %+7.1f %+7.1f %+7.1f %+7.1f %6d %s" % (
            i, r["name"][:34], r["score"], f.get("sub_bass", 0), f.get("bass", 0),
            f.get("sub_extension_db", 0), f.get("low_mid", 0),
            f.get("presence", 0), f.get("treble", 0), r["n_src"], sp))

    print("\n" + "=" * 116)
    print("换一种看法：按「低频质量」排序（sub_ext 高 = 真往下钻；low_mid 低 = 不糊）")
    print("=" * 116)
    print("%-34s %8s %8s %8s %9s %8s" % (
        "型号", "sub_ext", "sub", "bass", "low_mid", "tilt"))
    for r in sorted(rows, key=lambda x: -(x["f"].get("sub_extension_db", 0)))[:a.top]:
        f = r["f"]
        print("%-34s %+8.1f %+8.1f %+8.1f %+9.1f %+8.2f" % (
            r["name"][:34], f.get("sub_extension_db", 0), f.get("sub_bass", 0),
            f.get("bass", 0), f.get("low_mid", 0), f.get("tilt_1k_10k", 0)))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"_bass": a.bass, "_whitelist": WHITELIST, "_resolved": resolved,
                   "rows": [{"name": r["name"], "score": round(r["score"], 1),
                             "rms": round(r["rms"], 2), "n_src": r["n_src"],
                             "spread": r["spread"], "f": r["f"]} for r in rows]},
                  fh, ensure_ascii=False, indent=2)
    print("\n已写入", a.out)


if __name__ == "__main__":
    sys.exit(main())
