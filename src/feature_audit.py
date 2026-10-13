#!/usr/bin/env python3
"""特征互审：**我们自己的特征之间有多冗余**。

## 为什么该审自己

README §4.5 里有一句断言：

> 「低频有要求」不是一维，拆成三个可测量：`sub_bass`(20–60Hz 下钻多深)、
> `sub_extension`(20–40Hz 相对 60–200Hz，真延伸)、`low_mid`(200–500Hz，糊的来源)。

这话要成立，三个特征就得**各自携带独立信息**。若两两高度相关，那"三维"只是
把同一份差异算了三遍——引擎按特征加权求和时，重复计分会让分数失真，
而且失真方向恰好偏向低频（那里权重本来最大）。

这不是审美问题，是可算的问题：**相关矩阵 + 方差膨胀因子（VIF）**。
VIF 更直接——它就是"重复计分"在加权求和里的后果：
VIF=1 表示与其他特征完全无关，VIF=10 常被当作"严重共线"的经验线。

## 本模块只做两件事

1. 在全部曲线上算 12 个特征的**相关矩阵**，把 |r| 高的成对列出来；
2. 算每个特征的 **VIF**，并按 VIF 排序——顺带把 README 那句断言摆到台面上：
   要么它被数据支持（三个低频特征两两不相关），要么它需要被改写。

**注意**：相关性高不等于该删。`sub_bass` 与 `bass` 相关是**物理必然**
（都在低频），删掉反而丢信息。这里给的是"**加权时别把它们当独立维度**"，
以及"输出里说'三个可测量'时要注明它们的相关程度"。
"""
from __future__ import annotations

import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "data", "features.json")

#: VIF 的经验线（统计惯例；不是本项目自己拍的，但也不是硬阈值，仅供分级）
VIF_WARN = 5.0
VIF_HIGH = 10.0

#: README §4.5 断言"低频要拆成三个可测量"时点名的三个特征。
#: 名字必须与 `fr_features.py` 产出的字段**逐字一致**——写错就会静默跳过，
#: 而"查不到"与"没问题"在报告里长得一样（本模块踩过，见 `report()`）。
LOW_BAND_FEATURES = ("sub_bass", "sub_extension_db", "low_mid")


def load(path: str = FEATURES):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [m["features"] for m in data["models"] if m.get("features")]
    names = sorted({k for r in rows for k in r})
    return names, rows


def _corr(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    den = math.sqrt(sxx * syy)
    return sxy / den if den else 0.0


def correlation(names, rows) -> dict[tuple[str, str], float]:
    cols = {k: [r.get(k) for r in rows] for k in names}
    cols = {k: [0.0 if v is None else float(v) for v in vs] for k, vs in cols.items()}
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = _corr(cols[a], cols[b])
    return out


def redundant(names, rows, threshold: float = 0.8) -> list[tuple[float, str, str]]:
    cor = correlation(names, rows)
    return sorted(((abs(r), a, b) for (a, b), r in cor.items() if abs(r) >= threshold),
                  reverse=True)


def vif(names, rows) -> dict[str, float]:
    """方差膨胀因子 VIF_j = 1 / (1 − R²_j)，R²_j 由"用**其它全部**特征线性回归
    第 j 个特征"得到。这里不解正规方程，改用**相关矩阵求逆的对角线**——
    两者等价，而 12×12 的求逆用高斯-约当写起来最省事、也不引第三方库。
    """
    n = len(names)
    idx = {k: i for i, k in enumerate(names)}
    cols = {}
    for k in names:
        vs = [r.get(k) for r in rows]
        cols[k] = [0.0 if v is None else float(v) for v in vs]
    # 相关矩阵（含对角线 1）
    R = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            R[i][j] = R[j][i] = _corr(cols[names[i]], cols[names[j]])
    # 求逆（高斯-约当，带部分主元）
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(R)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return {k: float("inf") for k in names}     # 完全共线
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return {k: M[idx[k]][n + idx[k]] for k in names}


def report(path: str = FEATURES, threshold: float = 0.8) -> str:
    names, rows = load(path)
    lines = ["样本 %d 条曲线 · %d 个特征" % (len(rows), len(names)), "",
             "== 高相关特征对（|r| ≥ %.2f）==" % threshold]
    pairs = redundant(names, rows, threshold)
    if not pairs:
        lines.append("  （没有）")
    for r, a, b in pairs:
        lines.append("  |r| = %.2f   %-18s ↔ %-18s  互相解释 %.0f%% 方差"
                     % (r, a, b, r * r * 100))
    lines += ["", "== 方差膨胀因子（VIF；越大越说明「被重复计分」）=="]
    vs = vif(names, rows)
    for k, v in sorted(vs.items(), key=lambda kv: -kv[1]):
        tag = ("严重共线" if v >= VIF_HIGH else
               "需留意" if v >= VIF_WARN else "")
        lines.append("  %-18s VIF = %6.2f  %s" % (k, v, tag))
    lines += ["", "== 低频三特征（README §4.5 的断言）=="]
    low = LOW_BAND_FEATURES
    missing = [k for k in low if k not in names]
    if missing:
        # **不许静默跳过**：早先这里把 `sub_extension_db` 写成 `sub_extension`，
        # 于是那一对直接不打印，报告看着"一切正常"——查不到东西和没有问题
        # 长得一模一样，这正是本项目最忌讳的那种失败。
        lines.append("  ⚠ 特征名不存在：%s（数据里是：%s）"
                     % (", ".join(missing), ", ".join(sorted(names))))
    cor = correlation(names, rows)
    for i, a in enumerate(low):
        for b in low[i + 1:]:
            if a in names and b in names:
                key = (a, b) if (a, b) in cor else (b, a)
                lines.append("  %-18s ↔ %-18s |r| = %.2f  （重叠 %.0f%% 方差）"
                             % (a, b, abs(cor[key]), cor[key] ** 2 * 100))
    return "\n".join(lines)


def _selfcheck() -> None:
    print(report())
    print("\n== 怎么读这张表（VIF 在本语境要小心）==")
    print("  这些「特征」不是独立测量，而是**同一条归一化曲线的分段积分**：")
    print("  曲线先做过 300–3k 均值归零，7 个频段特征只是同一条函数的切片，")
    print("  线性相关是**结构性**的（低频那几个 VIF 冲到几百就是这么来的）。")
    print("  所以 VIF 在这里不能按教科书那条「超过 10 就删变量」来读；")
    print("  它能回答的是：**加权求和时，同一份差异会不会被放大几遍**。")
    print("  真正要留意的，是那两个**不该相关却相关**的对：低频三兄弟里，")
    print("  `sub_bass` 与 `sub_extension_db` 的 |r| = 0.85 —— 见 README §4.5 的更正。")


if __name__ == "__main__":
    _selfcheck()


if __name__ == "__main__":
    _selfcheck()
