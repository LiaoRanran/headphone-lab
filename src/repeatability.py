#!/usr/bin/env python3
"""测量噪声基线：**同型号跨来源的离散度，就是这台"仪器"的可复现性上限。**

## 为什么这条判据要自己造

文献没给"空间可闻阈"（`spatial.py` 里已如实登记），但有一条判据**可以从数据里
自造出来**，而且比拍脑袋的阈值硬得多：

    同一个型号被不同人工耳 / 来源测过多次（本库 185 个型号有 ≥2 源）。
    这些曲线之间的差异 = 测量链路（人工耳形态、耦合、摆放、平滑）的
    **可复现性上限**。

于是得到一条**与文献无关、完全数据驱动**的判据：

    某型号在某频段的偏差，若**小于**该频段的跨来源典型离散度，
    那么量到的多半是测量架的差异，不是耳机本身的差异——
    此时不该对它下任何"听感"结论。

这是 ISO 5725（重复性与再现性）/ Gage R&R 的思路：**先量仪器的噪声，
再谈被测对象的差异**。它不告诉你"能不能听出来"，它告诉你
"**这个差异值不值得讨论**"——两层判据，各自回答各自的问题。

## 口径（每一条都可复算）

- 离散度用**极差（max−min）与标准差并行报告**：极差对坏点敏感（直观），
  标准差对坏点稳健（统计）。
- 每个型号的每条来源曲线先在该频段内取均值，再在**来源之间**取离散度
  ——而不是把所有来源的频点混在一起算（那会把频内起伏误当成来源差）。
- 汇总用**中位数**而不是均值：个别型号的来源差可能极大（耦合不良），
  中位数才代表"典型来源差"。
- 频段划分沿用 `spatial.BANDS`，**只对已核到的频段给结论**（S2/S3），
  未核频段照旧只登记。
"""
from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "data", "features.json")


def _band_mean(dev, grid, lo, hi) -> float | None:
    vals = [d for f, d in zip(grid, dev) if lo <= f <= hi]
    return sum(vals) / len(vals) if vals else None


def per_model_spread(models, grid, lo, hi) -> list[dict]:
    """逐型号计算某频段的跨来源离散度（只统计 ≥2 源的型号）。"""
    by_name: dict[str, list[float]] = {}
    for m in models:
        dev = m.get("dev")
        if not dev:
            continue
        mean = _band_mean(dev, grid, lo, hi)
        if mean is None:
            continue
        by_name.setdefault(m["name"], []).append(mean)
    out = []
    for name, means in by_name.items():
        if len(means) < 2:
            continue
        out.append({
            "name": name,
            "n_sources": len(means),
            "range_db": round(max(means) - min(means), 2),
            "std_db": round(statistics.pstdev(means), 2),
            "means": [round(v, 2) for v in means],
        })
    return out


def noise_floor(models, grid, lo, hi) -> dict:
    """某频段的测量噪声基线（跨来源离散度的中位数）。"""
    rows = per_model_spread(models, grid, lo, hi)
    if not rows:
        return {"n_models": 0}
    return {
        "n_models": len(rows),
        "median_range_db": round(statistics.median(r["range_db"] for r in rows), 2),
        "median_std_db": round(statistics.median(r["std_db"] for r in rows), 2),
        "p90_range_db": round(sorted(r["range_db"] for r in rows)[
            min(len(rows) - 1, int(0.9 * len(rows)))], 2),
    }


def exceeds(dev, grid, lo, hi, floor: dict) -> dict | None:
    """某条曲线在该频段的偏差是否**超出测量噪声**。

    返回 None 表示没有基线可比（该频段没有多源数据）——
    与 `spatial` 同一条规矩：**没测 ≠ 没差，不知道就说不知道**。
    """
    if not floor or floor.get("n_models", 0) < 3:
        return None
    mean = _band_mean(dev, grid, lo, hi)
    if mean is None:
        return None
    return {
        "band_mean_db": round(mean, 2),
        "floor_range_db": floor["median_range_db"],
        "exceeds": abs(mean) > floor["median_range_db"],
        "ratio": round(abs(mean) / max(floor["median_range_db"], 1e-9), 2),
    }


def report(models, grid, bands) -> str:
    """把"哪些频段的差异值得讨论"打成一张表。"""
    lines = ["频段 ｜ 多源型号数 ｜ 跨来源极差中位 ｜ 跨来源 σ 中位"]
    for key, lo, hi, *_ in bands:
        nf = noise_floor(models, grid, lo, hi)
        if nf.get("n_models"):
            lines.append("%-14s %6d  %8.2f dB  %8.2f dB"
                         % (key, nf["n_models"], nf["median_range_db"],
                            nf["median_std_db"]))
        else:
            lines.append("%-14s   （该频段没有 ≥2 源的型号）" % key)
    return "\n".join(lines)


def _selfcheck() -> None:
    import spatial

    with open(FEATURES, encoding="utf-8") as f:
        data = json.load(f)
    grid, models = data["grid"], data["models"]
    multi = sum(1 for m in models if m.get("dev")) and len({
        m["name"] for m in models if m.get("dev")})
    print("全库 %d 条曲线 / %d 个型号名" % (len(models), multi))

    print("\n== 各频段的测量噪声基线（跨来源离散度中位）==")
    print(report(models, grid, [(b.key, b.lo, b.hi) for b in spatial.BANDS]))

    print("\n== 用基线判一判：5–6 kHz（前后判别关键区）==")
    fb = next(b for b in spatial.BANDS if b.key == "frontback")
    floor = noise_floor(models, grid, fb.lo, fb.hi)
    print("基线：跨来源极差中位 %.2f dB（n=%d 型号）" % (floor["median_range_db"],
                                                   floor["n_models"]))
    over = under = 0
    worst = []
    for m in models:
        dev = m.get("dev")
        if not dev:
            continue
        r = exceeds(dev, grid, fb.lo, fb.hi, floor)
        if r is None:
            continue
        over += r["exceeds"]
        under += (not r["exceeds"])
        if r["exceeds"]:
            worst.append((r["ratio"], m["name"], r["band_mean_db"]))
    print("超出测量噪声：%d 款 ｜ 在噪声以内（差异多半是测量架）：%d 款" % (over, under))
    for ratio, name, mean in sorted(worst, reverse=True)[:6]:
        print("  %.1f× 噪声基线  %-40s %+.2f dB" % (ratio, name[:40], mean))
    print("\n提醒：'超出测量噪声'不等于'听得出来'——它只说明这个差异"
          "**值得讨论**；能不能听出来是另一层判据，本库没有依据。")


if __name__ == "__main__":
    _selfcheck()
