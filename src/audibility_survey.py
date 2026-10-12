#!/usr/bin/env python3
"""全库可闻性普查：文献阈值（1–4kHz ≥6 dB）到底有多少区分力？

这个普查回答一个关键问题：如果阈值把所有人都判成"听不出区别"，
那它就是没用的（阈值太高）。如果它把大多数人判成"越阈"，
说明阈值太低。只有当**两个方向都有**时，这个判据才有信息量。
"""
from __future__ import annotations

import os
import sys
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audibility  # noqa: E402
import engine  # noqa: E402
import filters  # noqa: E402


def main() -> int:
    D = engine.load()
    grid = D["grid"]
    G = engine.group(D["models"])
    wired = {k: v for k, v in G.items() if filters.is_wired_over_ear(k)}

    # 基准问题的核查记录（供后人省一次弯路）：
    # `curve_t` 的 `_t` 后缀表示「已相对来源目标曲线归一化」，所以拿它
    # 减 0 就等于「相对 Harman 的偏差」，**本来就是对的**。
    # 我曾怀疑 <200Hz 有 95.5% 越阈是基准设错造成的，于是改成从
    # rank() 里取真实 target 重算 —— 结果数字**一个不差**，因为
    # 无锚点时 rank() 的 target 就是全零（base=0 且 bass/presence/treble 全 0）。
    # 结论：不是基准错，是那个 3 dB 阈值本身没有区分力。见文末。
    cfg = {"basis": "target", "bass": 0.0, "presence": 0.0, "treble": 0.0,
           "peak_guard": 4.5, "wired_only": True, "min_sources": 1}
    rows, _meta = engine.rank(cfg)
    target = rows[0]["_target"]          # 无锚点时为全零，与 curve_t 口径一致

    print("普查范围：通过 wired_over_ear 的 %d 个型号" % len(wired))
    print("基准：来源目标曲线（curve_t 已归一化到目标，故 target 为全零）\n")

    vals, over, spans = [], [], []
    low_devs = []
    for k, v in wired.items():
        curve = v["curve_t"]
        a = audibility.analyse(curve, target, grid)
        if a.get("n"):
            vals.append(a["pref_band_worst_db"])
            spans.append(a["pref_band_pct_over"])
            low_devs.append(abs(a["low_shelf_worst_db"]))
            if a["pref_band_exceeds"]:
                over.append((k, a["pref_band_worst_db"], a["pref_band_worst_hz"]))

    vals.sort()
    n = len(vals)
    print("=== 1–4kHz 最差偏差的分布（相对目标）===")
    for q, lab in ((0.05, "5%"), (0.25, "25%"), (0.50, "中位"), (0.75, "75%"), (0.95, "95%")):
        print("  %-5s %+6.2f dB" % (lab, vals[int(q * (n - 1))]))
    print("  中心值 %.2f ｜ 最小 %.2f ｜ 最大 %.2f" % (st.mean(vals), vals[0], vals[-1]))
    print()

    print("=== 阈值区分力检验（阈值 %.1f dB）===" % audibility.PREFERENCE_JND_DB)
    print("  ≥阈值（判定为「可区分」）：%d / %d = %.1f%%"
          % (len(over), n, 100.0 * len(over) / n))
    print("  <阈值（判定为「双盲下难区分」）：%d / %d = %.1f%%"
          % (n - len(over), n, 100.0 * (n - len(over)) / n))
    print()
    if not over or len(over) == n:
        print("  !! 阈值把所有人都判到同一侧 → 该判据在当前库上没有区分力")
    else:
        print("  ✓ 两个方向都有 → 判据有区分力")
    print()

    print("=== 越阈的型号（按偏差排序，前 20）===")
    for k, d, f in sorted(over, key=lambda x: -abs(x[1]))[:20]:
        print("  %-40s %+6.2f dB @%6.0f Hz" % (k[:40], d, f or 0))
    print()

    # 低频谱：工程估计阈值——同样用目标当基准
    low_vals = sorted(low_devs)
    low_over = sum(1 for x in low_vals if x >= audibility.LOW_SHELF_JND_DB)
    print()
    print("=== <200Hz 低频搁架偏差（相对目标；阈值 %.1f dB 为工程估计，无文献硬依据）==="
          % audibility.LOW_SHELF_JND_DB)
    if low_vals:
        print("  5%% %.2f ｜ 中位 %.2f ｜ 95%% %.2f ｜ 最大 %.2f"
              % (low_vals[int(0.05 * (n - 1))], low_vals[n // 2],
                 low_vals[int(0.95 * (n - 1))], low_vals[-1]))
        print("  越阈 %d / %d = %.1f%%" % (low_over, n, 100.0 * low_over / n))
        frac = low_over / max(1, n)
        if frac > 0.9 or frac < 0.1:
            print("  !! 阈值把 %d%% 的型号判到同一侧 → **该估计阈值没有区分力**，"
                  % int(100 * (frac if frac > 0.9 else 1 - frac)))
            print("     因为它是我拍的不是文献给的。当作分级参考可以，不要当判据用。")
        else:
            print("  ✓ 两个方向都有 → 该估计阈值在当前库上有区分力")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
