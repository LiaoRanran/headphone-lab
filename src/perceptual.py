#!/usr/bin/env python3
"""Bark 临界频带积分 —— 修掉引擎误差度量的一个数学缺陷。

### 缺陷是什么

引擎原来的误差度量是在 121 点 1/12 倍频程网格上求 `Σ w(f)·d(f)²`。
网格按恒定比例（5.9%）分布，而**人耳的频率分辨率是按临界频带走的**：

    Bark < 500Hz：带宽 ≈ 100 Hz（近似恒定）
    Bark > 500Hz：带宽 ≈ 0.2·f

1/12 倍频程在 500Hz 以下**远窄于** Bark 带（100 Hz 处 1/12 倍频带宽只有约 6 Hz），
所以按"网格点数"积分时，低频被过度代表、高频被代表不足。

感知权重 w(f) 部分补偿了这一点，但那是**另一件事**：w 说的是"哪里更敏感"，
不是"哪里该按什么带宽积分"。用一个参数同时干两件事，调参时必然互相牵扯。

### 修法

先按 Bark 带求**组内均值**，再对带求 RMS：

    rms_Bark = sqrt( Σ_bands w_b · mean_b(d)² / Σ_bands w_b )

于是每个临界频带对总误差的贡献只取决于它内部偏离多大，不再取决于它在 Hz 轴上占多宽。
`mean_b` 取组内均值而非总和，是关键的归一化——否则宽带（高频）又会自动获得更大权重。

### 关于响度模型：撤掉了，理由记在这里

本模块原打算同时实现 ISO 532-1（Zwicker）响度，把 dB 偏差换算成 sone/phon。
**做不出来，已删除**，原因是校不准：

    1kHz / 40 dB SPL 纯音 → 公式给出 31.1 phon（ISO 226 定义应为 40 phon）
    1kHz / 60 → 70 dB     → 响度比 6.51×（Stevens 定律期望约 4×）

第二条说明**不是纯尺度误差**——解出来指数项应约 0.16 而非公式里的 0.25，
即我对 `alpha` 或 `L_TQ` 的理解有误。在能核到 ISO 532-1 原文系数表之前，
宁可不提供响度换算，也不给一个差 9 phon 的数字。
按"能算的绝不猜"的规矩：** Bark 标度与积分可验证，响度常数不可验证，后者不做。**
"""
from __future__ import annotations

import math

#: Zwicker Bark 标度（Traunmüller 形式）
_BARK_A = 26.81
_BARK_B = 1960.0
_BARK_C = 0.53


def bark(f: float) -> float:
    """频率 (Hz) → Bark。与 critical_bw() 解析自洽。"""
    return _BARK_A * f / (_BARK_B + f) - _BARK_C


def bark_to_hz(z: float) -> float:
    """Bark → 频率 (Hz)。bark() 的解析逆。"""
    return _BARK_B * (z + _BARK_C) / (_BARK_A - _BARK_C - z)


def critical_bw(f: float) -> float:
    """临界带宽 (Hz) = 1 / (dz/df)，由 bark() 的导数导出。

    不另引一套独立公式，避免两个标度互相矛盾。
    校验点：1 kHz ≈ 167 Hz（文献常引 ~160 Hz）；100 Hz ≈ 81 Hz。
    """
    return (_BARK_B + f) ** 2 / (_BARK_A * _BARK_B)


def bark_edges(n_low: float = 20.0, n_high: float = 20000.0) -> list[float]:
    """把 [n_low, n_high] 按 Bark 等宽切分的边界频率（Hz）。

    返回 n_edges 个点，即 n_edges−1 个 Bark 带。等宽切在 Bark 域，
    换算到 Hz 域就是"每个带在人耳看来一样宽"。
    """
    lo, hi = bark(n_low), bark(n_high)
    n = max(1, int(round(hi - lo)))
    # 端点**不**做 bark_to_hz 往返：bark_to_hz(bark(20000)) 会因浮点回到
    # 20000.000000000062，断言端点相等会失败。直接取原值，语义也更直白。
    edges = [n_low]
    edges += [bark_to_hz(lo + (hi - lo) * i / n) for i in range(1, n)]
    edges.append(n_high)
    return edges


def band_index(grid, edges=None) -> list[int]:
    """频网格上每个点归到哪个 Bark 带（超出范围给 −1）。"""
    if edges is None:
        edges = bark_edges()
    out = []
    for f in grid:
        k = -1
        for i in range(len(edges) - 1):
            if edges[i] <= f < edges[i + 1]:
                k = i
                break
        out.append(k)
    return out


def bark_band_means(curve, target, grid, edges=None) -> dict[int, float]:
    """每个 Bark 带内的平均偏差（dB）。

    单列出来是因为引擎的分数里 RMS 项与峰护栏项**必须用同一个积分域**，
    否则等于把两套口径混在一个数里。两个项都从这里取带内均值。
    """
    idx = band_index(grid, edges)
    bands: dict[int, list[float]] = {}
    for i, k in enumerate(idx):
        if k >= 0:
            bands.setdefault(k, []).append(curve[i] - target[i])
    return {k: sum(v) / len(v) for k, v in bands.items()}


def bark_rms(curve, target, grid, weights=None, edges=None) -> float:
    """按 Bark 带积权的 RMS 偏差（dB）。

    weights: 可选的 {band_index: weight}（如 ISO 226 敏感度按带平均）。
             默认各带等权 —— 这正是与旧度量的区别所在。
    """
    idx = band_index(grid, edges)
    bands: dict[int, list[float]] = {}
    for i, k in enumerate(idx):
        if k >= 0:
            bands.setdefault(k, []).append(curve[i] - target[i])
    if not bands:
        return 0.0
    se = sw = 0.0
    for k, ds in bands.items():
        m = sum(ds) / len(ds)                      # 组内均值：宽带不会自动加权
        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        se += w * m * m
        sw += w
    return math.sqrt(se / sw) if sw > 0 else 0.0


def band_count(grid, edges=None) -> int:
    """网格点实际覆盖到的 Bark 带数（用于诊断：太少的带说明频段覆盖不足）。"""
    return len({k for k in band_index(grid, edges) if k >= 0})


def _selfcheck():
    # 1) Bark 标度对照文献
    print("Bark 校验：")
    for f, exp in ((100.0, 0.8), (1000.0, 8.5), (10000.0, 22.0)):
        print("  %6.0f Hz → Bark %5.2f（文献 ≈%s）｜临界带宽 %6.1f Hz"
              % (f, bark(f), exp, critical_bw(f)))
    # 2) 边界单调且覆盖
    e = bark_edges()
    print("  Bark 等宽切分：%d 个带，%.0f → %.0f Hz，单调=%s"
          % (len(e) - 1, e[0], e[-1], all(e[i] < e[i + 1] for i in range(len(e) - 1))))
    # 3) 旧度量的缺陷：100Hz 以下 1/12 倍频程比 Bark 窄多少
    for f in (50.0, 100.0, 1000.0, 8000.0):
        oct12 = f * (2 ** (1 / 12) - 1)
        print("  %6.0f Hz：1/12 倍频带宽 %6.1f Hz ｜ Bark 带宽 %6.1f Hz ｜ 比 %.2f"
              % (f, oct12, critical_bw(f), critical_bw(f) / oct12))


if __name__ == "__main__":
    _selfcheck()
