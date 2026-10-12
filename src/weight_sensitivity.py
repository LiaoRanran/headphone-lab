#!/usr/bin/env python3
"""感知权重的敏感性分析：手调的 base_weight() 到底影响排名多少？

为什么做这个而不是"换成 ISO 226 等响曲线"：
ISO 226:2003 的闭式公式与系数表**没能核到可引用的一手来源**
（标准正文是付费 PDF，抓到的镜像为二进制，PyPI/GitHub 均被拦）。
按方法论「能算的绝不猜 / 不编造标准」，这里**不凭记忆实现它**。

替代方案（信息量更大）：既然权重是手调的，就直接测它有多要紧——
跑若干种权重方案，看排名变化多少。变化小 → 手调权重不是瓶颈；
变化大 → 必须先把权重校准，否则排名不可信。

用法：python src/weight_sensitivity.py
"""
from __future__ import annotations

import json
import math
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import engine  # noqa: E402

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, "data")

# 七种权重方案。前两种是"极端"，其余代表常见取向。
def _iso226(phon=80.0, floor=0.05):
    import psycho
    return psycho.weight_fn(phon=phon, floor=floor)


SCHEMES = {
    "当前 base_weight（手调）": None,          # 用引擎默认
    "ISO 226 等响曲线 (80方)": _iso226(80.0, 0.05),
    "ISO 226 等响曲线 (40方)": _iso226(40.0, 0.05),
    "ISO 226 floor=0.15": _iso226(80.0, 0.15),
    "ISO 226 floor=0.30": _iso226(80.0, 0.30),
    "全平坦 w=1": lambda f: 1.0,
    "只信中频（1-4kHz=3，其余=1）": lambda f: 3.0 if 1000 <= f <= 4000 else 1.0,
    "偏重低频（<500Hz=3）": lambda f: 3.0 if f < 500 else 1.0,
    "偏重高频（>5kHz=3）": lambda f: 3.0 if f > 5000 else 1.0,
    "极端低频（<200Hz=10）": lambda f: 10.0 if f < 200 else 1.0,
}


def spearman(a: list[str], b: list[str]) -> float | None:
    """在两者交集上算 Spearman 秩相关（用名次差）。"""
    common = [x for x in a if x in b]
    if len(common) < 5:
        return None
    ra = {n: i for i, n in enumerate(a)}
    rb = {n: i for i, n in enumerate(b)}
    n = len(common)
    d2 = sum((ra[x] - rb[x]) ** 2 for x in common)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    D = engine.load()
    grid = D["grid"]
    base = engine.base_weight()

    # 固定目标与锚点，只变权重 —— 否则测的是两件事
    cfg = {"basis": "target", "bass": 0.0, "presence": 0.0, "treble": 0.0,
           "peak_guard": 4.5, "wired_only": True, "min_sources": 2}

    print("目标固定：Harman 基准 ｜ 低频/presence/高频 delta 全 0 ｜ 峰护栏 4.5")
    print("只改变重：感知权重 w(f)。min_sources=2（只留多来源型号，减少单源噪声）\n")

    tops: dict[str, list[str]] = {}
    for name, w in SCHEMES.items():
        rows, meta = engine.rank(cfg, w_override=w)
        tops[name] = [r["name"] for r in rows]
        print("=== %s ===" % name)
        for i, r in enumerate(rows[:8], 1):
            print("   %2d %-38s %6.1f" % (i, r["name"][:38], r["score"]))
        print()

    ref = tops["当前 base_weight（手调）"]
    print("=" * 78)
    print("与「当前手调权重」的排名一致性（Spearman 秩相关，1.0 = 完全相同）")
    print("=" * 78)
    print("%-30s %10s %12s" % ("权重方案", "Spearman", "前10重叠"))
    summary = []
    for name, t in tops.items():
        sp = spearman(ref, t)
        ov = len(set(ref[:10]) & set(t[:10]))
        summary.append({"scheme": name, "spearman": round(sp, 3) if sp else None,
                        "top10_overlap": ov})
        print("%-30s %10s %12d" % (name[:30],
                                   "%.3f" % sp if sp is not None else "—", ov))

    with open(os.path.join(DATA, "weight_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump({"_note": "只变感知权重，目标/锚点固定", "summary": summary,
                   "tops": {k: v[:30] for k, v in tops.items()}},
                  f, ensure_ascii=False, indent=2)
    print("\n已写入 data/weight_sensitivity.json")

    worst = min((s for s in summary if s["spearman"] is not None),
                key=lambda s: s["spearman"])
    print("\n结论：最激进的权重方案（%s）与当前手调权重的秩相关为 %.3f。"
          % (worst["scheme"], worst["spearman"]))
    if worst["spearman"] > 0.8:
        print("→ 排名对权重选择**不敏感**，手调权重不是当前瓶颈。")
    elif worst["spearman"] > 0.6:
        print("→ 中等敏感：权重取向会改变排序，但核心候选仍重叠。")
    else:
        print("→ **高度敏感**：先校准权重再谈排名，否则结论不可信。")


if __name__ == "__main__":
    sys.exit(main())
