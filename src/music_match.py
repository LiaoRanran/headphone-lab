#!/usr/bin/env python3
"""把「自测音乐走向」变成耳机候选集，并做**敏感性分析**。

方法论要点（为什么不是一个参数跑一次就完事）：
  音乐测量给出的是「素材里已经有什么」，而耳机给出的是「额外加什么」。
  两者叠加，风险主要在**同向叠加**（音乐本来低频多，耳机再抬低频 → 掩蔽中频）。
  但「该抵消多少」无法从测量单独推出——它取决于听音音量与个人偏好。
  所以这里**不假装有一个正确数值**，而是扫低频目标区间，
  看哪些型号在整段区间里始终靠前（稳健命中），哪些只是某个参数下的偶然产物。

用法：
  python src/music_match.py                 # 全部目录 × 低频目标扫描
  python src/music_match.py --folders "BA OST" --top 15
"""
import argparse
import json
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import engine  # noqa: E402

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, "data")
TRENDS = os.path.join(DATA, "music_trends.json")

# 低频目标扫描：0 = Harman 基准，负值 = 要求比 Harman 更少低频
BASS_GRID = [-4.0, -2.0, 0.0, 2.0]
TOP_N = 20


def load_folders(only=None):
    if not os.path.exists(TRENDS):
        raise SystemExit("缺少 data/music_trends.json —— 先跑 src/music_trend.py")
    with open(TRENDS, encoding="utf-8") as f:
        d = json.load(f)
    fs = d["folders"]
    if only:
        keys = [k.strip().lower() for k in only]
        fs = [x for x in fs if any(k in x["name"].lower() for k in keys)]
    return fs


def borda(rankings, top_n=TOP_N):
    """Borda 计数：把每次排名的名次折算成分数相加。

    比「平均得分」稳，因为不同参数下 RMS 的绝对值不可比，名次可比。
    """
    pts, hits, best = {}, {}, {}
    for rk in rankings:
        for pos, name in enumerate(rk[:top_n]):
            pts[name] = pts.get(name, 0) + (top_n - pos)
            hits[name] = hits.get(name, 0) + 1
            if pos < 3:
                best[name] = best.get(name, 0) + 1
    total = len(rankings)
    out = []
    for name, p in pts.items():
        out.append({"name": name, "borda": p, "hit": hits[name],
                    "hit_pct": round(hits[name] / total * 100, 0),
                    "top3": best.get(name, 0)})
    out.sort(key=lambda x: (-x["borda"], -x["hit"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--out", default=os.path.join(DATA, "music_match.json"))
    a = ap.parse_args()

    folders = load_folders(a.folders)
    if not folders:
        raise SystemExit("没有匹配的目录")

    per_folder, all_rankings, detail = {}, [], {}
    for f_ in folders:
        nm = f_["name"]
        print("\n" + "=" * 78)
        print("目录：%s  （低频占比 %.1f%% ｜ 中高 %.1f%% ｜ 倾斜 %.1f dB/oct）"
              % (nm, f_["low_share"], f_["presence_share"], f_["tilt_db_per_oct"]))
        print("=" * 78)
        rks = []
        for bass in BASS_GRID:
            rows, meta = engine.rank({
                "basis": "target", "bass": bass, "presence": 0.0, "treble": 0.0,
                "peak_guard": 4.5, "music_trend": nm, "wired_only": True,
                "min_sources": 1,
            })
            names = [r["name"] for r in rows]
            rks.append(names)
            all_rankings.append(names)
            detail.setdefault(nm, {})["bass%+.0f" % bass] = [
                {"name": r["name"], "score": round(r["score"], 1),
                 "rms": round(r["rms"], 2), "n_src": r["n_src"],
                 "spread": None if r["spread"] is None else round(r["spread"], 2),
                 "f": r["f"]}
                for r in rows[:a.top]
            ]
            top5 = " ｜ ".join("%s" % r["name"][:26] for r in rows[:5])
            print("  低频%+d: %s" % (bass, top5))

        cons = borda(rks, a.top)
        per_folder[nm] = cons[:a.top]
        print("  %s" % ("-" * 74))
        print("  【稳健集】低频目标整段区间的 Borda 共识（命中率 = 在 4 个设定里都进前 %d）" % a.top)
        for i, c in enumerate(cons[:10], 1):
            print("   %2d %-38s Borda %4d ｜ 命中 %.0f%% ｜ 前3 %d 次"
                  % (i, c["name"][:38], c["borda"], c["hit_pct"], c["top3"]))

    lib = borda(all_rankings, a.top)
    print("\n" + "=" * 78)
    print("全库共识（%d 个目录 × %d 个低频设定 = %d 次排名）"
          % (len(folders), len(BASS_GRID), len(all_rankings)))
    print("=" * 78)
    print("%-3s %-38s %6s %7s %6s  %s" % ("#", "型号", "Borda", "命中率", "前3",
                                          "相对 Harman 特征"))
    for i, c in enumerate(lib[:15], 1):
        ft = None
        for nm, blocks in detail.items():
            for b, arr in blocks.items():
                for r in arr:
                    if r["name"] == c["name"]:
                        ft = r["f"]
                        break
                if ft:
                    break
            if ft:
                break
        s = ""
        if ft:
            s = ("低频%+.1f 下潜%+.1f presence%+.1f 高频%+.1f 空气%+.1f 倾角%+.2f"
                 % (ft["bass"], ft["sub_extension_db"], ft["presence"],
                    ft["treble"], ft["air"], ft["tilt_1k_10k"]))
        print("%-3d %-38s %6d %6.0f%% %6d  %s"
              % (i, c["name"][:38], c["borda"], c["hit_pct"], c["top3"], s))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"_bass_grid": BASS_GRID, "_note":
                   "borda = 跨低频目标设定的名次共识；hit_pct = 进入前 %d 的比例" % a.top,
                   "per_folder": per_folder, "library": lib,
                   "detail": detail}, f, ensure_ascii=False, indent=2)
    print("\n已写入", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
