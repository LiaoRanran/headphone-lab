#!/usr/bin/env python3
"""排名的统计严谨性：分的置信区间 + 相邻名次的可区分性。

为什么必须有这个：引擎输出「87.3 vs 86.6」却不说明这个差是否可信，
那只是**伪精度**。科研级的要求是给出区间，并明确指出哪些名次差异
在噪声范围内、不该被解读。

做法：**频率维 bootstrap**。把 121 个频点当样本、有放回重采样，
按同一公式重算得分，重复 n 次得到分的分布 → 百分位 CI。

它覆盖的是：**频谱形状的测量不确定性**（不同频点权重下的抖动）。
它**不**覆盖：单元个体差异、不同人工耳的传递函数差异、
同一型号不同批次/耳垫的变化 —— 那些需要多只样本的数据，本项目没有。
（这一点在输出里如实标注，不假装 CI 包含所有不确定性。）
"""
from __future__ import annotations

import math
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import engine  # noqa: E402


def _score(curve, target, grid, W, pg, idx):
    """给定频点子集 idx 时的得分（与 engine.rank 同公式，只是换了频点集）。"""
    sw = se = sp = 0.0
    for i in idx:
        w = W(grid[i])
        dd = curve[i] - target[i]
        sw += w
        se += w * dd * dd
        sp += w * max(0.0, dd - 2.0) ** 2
    if sw <= 0:
        return None
    rms = math.sqrt(se / sw)
    peak = math.sqrt(sp / sw)
    s = 100 - 7.0 * rms - 6.0 * peak
    pf = max((curve[i] for i in idx if 5000 <= grid[i] <= 10000), default=None)
    if pf is not None and pf > pg:
        s -= 4.0 * (pf - pg)
    return s


def score_with_ci(curve, target, grid, W, pg=4.5, n_boot=400, seed=0):
    """返回 (point, lo, hi, se)。lo/hi 为 2.5%/97.5% 百分位。"""
    import numpy as np
    n = len(grid)
    idx_all = list(range(n))
    point = _score(curve, target, grid, W, pg, idx_all)
    if point is None:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)          # 有放回重采样频点
        s = _score(curve, target, grid, W, pg, idx)
        if s is not None:
            samples.append(s)
    if len(samples) < n_boot * 0.5:
        return point, None, None, None
    a = np.array(samples)
    return (round(point, 2), round(float(np.percentile(a, 2.5)), 2),
            round(float(np.percentile(a, 97.5)), 2), round(float(a.std(ddof=1)), 2))


def annotate(rows, cfg, n_boot=400, seed=0, top=30):
    """给前 top 名加置信区间，并标注相邻名次是否可区分。

    可区分判据：两者 95% CI **不重叠**。重叠 → 名次差异在噪声内，
    不应被解读为"这款更好"。
    """
    D = engine.load()
    grid = D["grid"]
    n = len(grid)
    W = engine.base_weight()
    G = engine.group(D["models"])
    # 重建与 rank() 相同的目标曲线
    basis = cfg.get("basis", "target")
    KEY = "curve_t" if basis == "target" else "curve"
    chosen = [G[k][KEY] for k in G
              if any(p.lower() in k.lower() for p in cfg.get("like", []))]
    avoided = [G[k][KEY] for k in G
               if any(p.lower() in k.lower() for p in cfg.get("dislike", []))]
    import statistics as st
    base = [st.mean([c[i] for c in chosen]) for i in range(n)] if chosen else [0.0] * n
    avoid_dir = None
    if avoided:
        av = [st.mean([a[i] for a in avoided]) for i in range(n)]
        mu = sum(av) / n
        avoid_dir = [x - mu for x in av]
    lam = cfg.get("avoid_weight", 0.6)
    target = []
    for i, f in enumerate(grid):
        d = 0.0
        if f <= 200:
            if f < 40:
                d = cfg.get("bass", 0.0) * 0.4
            elif f < 60:
                d = cfg.get("bass", 0.0) * 0.7
            elif f <= 120:
                d = cfg.get("bass", 0.0) * 1.0
            else:
                d = cfg.get("bass", 0.0) * 0.7
        if 2000 <= f <= 5000:
            d += cfg.get("presence", 0.0)
        if 6000 <= f <= 12000:
            d += cfg.get("treble", 0.0)
        t = base[i] + d
        if avoid_dir:
            t -= lam * avoid_dir[i]
        target.append(t)

    out = []
    for rank, r in enumerate(rows[:top], 1):
        g = G.get(r["name"])
        if not g:
            continue
        lo, hi, se = None, None, None
        p, l, h, s = score_with_ci(g[KEY], target, grid, W,
                                   pg=cfg.get("peak_guard", 4.5),
                                   n_boot=n_boot, seed=seed + rank)
        lo, hi, se = l, h, s
        out.append({"rank": rank, "name": r["name"], "score": r["score"],
                    "ci_lo": lo, "ci_hi": hi, "se": se,
                    "n_src": r["n_src"], "spread": r["spread"]})
    # 相邻名次可区分性
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        ok = (a["ci_lo"] is not None and b["ci_hi"] is not None
              and a["ci_lo"] > b["ci_hi"])
        a["distinguishable_from_next"] = ok
    return out, target


def tiers(ann):
    """按 CI 重叠把型号聚成**档**。

    同一档内的名次差异在噪声范围内，**不应被解读为「这款更好」**。
    这是科研级输出与"排行榜"的本质区别：前者给档，后者给伪精度。
    """
    out: list[list[dict]] = []
    cur: list[dict] = []
    lo = hi = None
    for a in ann:
        if a.get("ci_lo") is None or a.get("ci_hi") is None:
            if cur:
                out.append(cur)
            cur, lo, hi = [a], None, None
            continue
        # 上一档没有有效 CI 时 lo/hi 是 None，**不能拿去做比较**（会 TypeError）——
        # 这种情况下无法判断重叠，只能开启新档。
        if not cur or lo is None:
            if cur:
                out.append(cur)
            cur, lo, hi = [a], a["ci_lo"], a["ci_hi"]
            continue
        # 与当前档的 CI 并集有重叠 → 同档
        if a["ci_hi"] >= lo and a["ci_lo"] <= hi:
            cur.append(a)
            lo, hi = min(lo, a["ci_lo"]), max(hi, a["ci_hi"])
        else:
            out.append(cur)
            cur, lo, hi = [a], a["ci_lo"], a["ci_hi"]
    if cur:
        out.append(cur)
    return out


def report(rows, cfg, top=15, n_boot=400):
    """打印带 CI 的排名表 + 分档结论。"""
    ann, _ = annotate(rows, cfg, n_boot=n_boot, top=top)
    print("\n" + "=" * 96)
    print("带置信区间的排名（频率维 bootstrap，%d 次）" % n_boot)
    print("CI 覆盖的是频谱形状的测量不确定性；**不含**单元个体差异/人工耳差异/批次差异")
    print("=" * 96)
    print("%-3s %-34s %7s %15s %7s  %s" % ("#", "型号", "得分", "95% CI", "SE", "与下一名"))
    for a in ann:
        ci = ("%.2f – %.2f" % (a["ci_lo"], a["ci_hi"])) if a["ci_lo"] is not None else "—"
        d = ("可区分" if a.get("distinguishable_from_next")
             else "**不可区分**") if a.get("distinguishable_from_next") is not None else "—"
        print("%-3d %-34s %7.2f %15s %7s  %s"
              % (a["rank"], a["name"][:34], a["score"], ci,
                 ("%.2f" % a["se"]) if a["se"] is not None else "—", d))

    ts = tiers(ann)
    print("\n" + "=" * 96)
    print("分档结论（同档内名次差异不可解读）")
    print("=" * 96)
    for i, t in enumerate(ts, 1):
        names = "、".join(x["name"][:24] for x in t)
        rng = ("%.1f–%.1f" % (min(x["ci_lo"] for x in t if x["ci_lo"] is not None),
                              max(x["ci_hi"] for x in t if x["ci_hi"] is not None))
               if any(x["ci_lo"] is not None for x in t) else "—")
        print("  第 %d 档（%d 款，CI 跨度 %s）：%s" % (i, len(t), rng, names))
    if len(ts) == 1:
        print("\n  **前 %d 名全部落在同一档** —— 这个排名没有区分度，"
              "按排行榜下单等于按噪声下单。" % len(ann))
    return ann


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bass", type=float, default=0.0)
    ap.add_argument("--presence", type=float, default=0.0)
    ap.add_argument("--treble", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    rows, meta = engine.rank({"basis": "target", "bass": a.bass,
                              "presence": a.presence, "treble": a.treble,
                              "peak_guard": 4.5, "wired_only": True,
                              "min_sources": 1})
    report(rows, {"basis": "target", "bass": a.bass, "presence": a.presence,
                  "treble": a.treble, "peak_guard": 4.5}, top=a.top, n_boot=a.boot)
