#!/usr/bin/env python3
"""预测「用这只耳机听你的曲库，各频段占比会变成多少」。

做法：
  1. 复用 music_trends.json 里已抽样的代表曲（保证抽样一致），但改用**与 lab
     特征层完全相同的 7 段定义**（fr_features.BANDS）——此前 music_trend 用的是
     trackscore 口径，两者频段边界不同（60–250 vs 60–200、2–6k vs 2–5k），
     不能直接拿来和耳机的 band 偏差相加。
  2. 耳机的 dB 偏差按物理规则作用到功率占比上：
         new_share ∝ old_share × 10^(dev/10)
     即耳机在某段 +3dB，该段功率贡献翻倍。再归一化回 100%。
  3. 输出「无耳机影响（音乐本来）」与「加上该耳机后」的对照。

用法：python src/predict_balance.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# 与 src/fr_features.py 的 BANDS 完全一致（Hz）
BANDS = [("sub_bass", 20, 60), ("bass", 60, 200), ("low_mid", 200, 500),
         ("mid", 500, 2000), ("presence", 2000, 5000), ("treble", 5000, 10000),
         ("air", 10000, 20000)]
BAND_CN = {"sub_bass": "超低 20-60", "bass": "低频 60-200", "low_mid": "中低 200-500",
           "mid": "中频 0.5-2k", "presence": "中高 2-5k", "treble": "高频 5-10k",
           "air": "空气 10-20k"}

NPERSEG = 16384
SEG = 60.0


def measure(path):
    """改用 music_trend.curve_of —— 此前本文件自己抄了一份测量代码，
    还是「先平均 L/R + 只取中段 60s」的旧版（audit.py 的 B1/B2 正是要抓这个）。
    共享实现，避免两边不一致。频段用 BANDS_LAB，与耳机 band 偏差同口径。"""
    import music_trend as mt
    _, _, bands, _ = mt.curve_of(path, bands=mt.BANDS_LAB)
    return bands


def main():
    with open(os.path.join(DATA, "music_trends.json"), encoding="utf-8") as f:
        trends = json.load(f)["folders"]

    # 汇总：每个目录取其已抽样曲目的均值，再按「曲目数」加权求全库均值
    folder_rows = {}
    for t in trends:
        vals = []
        for tr in t["tracks"]:
            if "bands_pct" not in tr:
                continue
            p = os.path.join(t["root"], tr["file"])
            if not os.path.isfile(p):
                continue
            vals.append(measure(p))
        if vals:
            folder_rows[t["name"]] = {k: float(np.mean([v[k] for v in vals]))
                                      for k, _, _ in BANDS}
            print("  已测 %-24s %d 曲" % (t["name"], len(vals)), flush=True)

    n_all = sum(1 for _ in folder_rows)
    lib = {k: float(np.mean([r[k] for r in folder_rows.values()])) for k, _, _ in BANDS}
    print("\n全库平均（lab 口径，%d 个目录）:" % n_all)
    for k, _, _ in BANDS:
        print("   %-14s %5.2f%%" % (BAND_CN[k], lib[k]))

    # ---- 候选耳机的 band 偏差（相对 Harman，来自 mainstream_match.json）----
    with open(os.path.join(DATA, "mainstream_match.json"), encoding="utf-8") as f:
        mm = json.load(f)
    rows = mm["rows"]

    def find(key):
        """按关键词在库里的全名里找（处理 'Focal Azurys' vs 'Azurys' 这类差异）。"""
        hits = [r for r in rows if key.lower() in r["name"].lower()]
        if not hits:
            return None
        # 同关键词多条时取跨源离散最小的
        hits.sort(key=lambda r: r["spread"] if r["spread"] is not None else 99)
        return hits[0]

    picks = [
        ("Focal Azurys", "Azurys"),
        ("AKG K371", "K371"),
        ("AKG K361", "K361"),
        ("Shure SRH1540", "SRH1540"),
        ("Sony MDR-7506", "MDR-7506"),
        ("Sennheiser HD 600", "HD 600"),
        ("Sennheiser HD 650", "HD 650"),
        ("Dan Clark Audio EXPANSE", "EXPANSE"),
        ("HIFIMAN Sundara", "Sundara"),
        ("Philips SHP9500", "SHP9500"),
    ]
    cands = {}
    for label, key in picks:
        r = find(key)
        if r:
            cands[label] = r["f"]
        else:
            print("  !! 未找到:", label)

    print("\n" + "=" * 104)
    print("预测：加上耳机后的各频段占比（%）。「原样」= 音乐本来（≈Harman 中性耳机的重放）")
    print("=" * 104)
    print("%-30s %8s %8s %8s %8s %8s %8s %8s" % (
        "耳机", "超低", "低频", "中低", "中频", "中高", "高频", "空气"))
    print("%-30s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f   ← 原样" % (
        "（无影响）", *[lib[k] for k, _, _ in BANDS]))

    out = {}
    for label, f in cands.items():
        raw = {k: lib[k] * 10 ** (f.get(k, 0) / 10.0) for k, _, _ in BANDS}
        s = sum(raw.values()) or 1.0
        pred = {k: raw[k] / s * 100 for k, _, _ in BANDS}
        out[label] = {"dev": {k: f.get(k, 0) for k, _, _ in BANDS}, "predicted": pred}
        print("%-30s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" % (
            label[:30], *[pred[k] for k, _, _ in BANDS]))

    # ---- 关键差值：低频合计与中频合计的变化 ----
    print("\n" + "=" * 104)
    print("关键差值（百分点变化）：低频合计(20-200Hz) / 中频(0.5-2k) / 中高(2-5k)")
    print("=" * 104)
    base_low = lib["sub_bass"] + lib["bass"]
    base_mid = lib["mid"]
    base_pres = lib["presence"]
    print("%-30s %10s %10s %10s" % ("耳机", "低频%", "中频%", "中高%"))
    print("%-30s %10.2f %10.2f %10.2f   ← 原样" % ("（无影响）", base_low, base_mid, base_pres))
    for label, d in out.items():
        p = d["predicted"]
        lo = p["sub_bass"] + p["bass"]
        print("%-30s %+10.2f %+10.2f %+10.2f" % (
            label[:30], lo - base_low, p["mid"] - base_mid, p["presence"] - base_pres))

    with open(os.path.join(DATA, "predicted_balance.json"), "w", encoding="utf-8") as f:
        json.dump({"_method":
                   "new_share ∝ old_share × 10^(dev/10)，dev = 相对 Harman 的 band 偏差（dB）",
                   "_band_defs": {k: [lo, hi] for k, lo, hi in BANDS},
                   "library_raw": lib, "per_folder": folder_rows,
                   "candidates": out}, f, ensure_ascii=False, indent=2)
    print("\n已写入 data/predicted_balance.json")


if __name__ == "__main__":
    sys.exit(main())
