#!/usr/bin/env python3
"""ISO 226:2003 等响曲线 —— 感知权重的权威来源。

系数表与公式（可引用的一手来源）
--------------------------------
ISO 226:2003《Acoustics — Normal equal-loudness-level contours》：

    A_f = 4.47e-3 · (10^(0.025·L_N) − 1.15) + (0.4 · 10^((T_f + L_u)/10 − 9))^α_f
    L_F = (10/α_f) · log10(A_f) − L_u + 94

其中 L_N 是响度级（方，phon），T_f 是该频率的安静听阈，
α_f 与 L_u 是标准给出的、只与频率有关的系数表（下表）。

**为什么不用它直接当权重，而要加 floor**：按上式算出的"敏感度"在 20 Hz 只有
10^((40−99.9)/10) ≈ 1e-6。若直接拿它当误差权重，等于完全忽略超低频——
但音乐里少了超低频你**听得出来**（声音变薄）。纯音敏感度 ≠ 宽带谱差的显著性。
所以这里取 `max(敏感度, floor)`，floor 可调，默认 0.05（约 −13 dB 相对 1 kHz）。

自校验见 tests/test_psycho.py：
  * f = 1000 Hz 时必须给出 L_F = L_N（"方"的定义）——全 phon 档验证；
  * 40 方 @ 20 Hz ≈ 99.9 dB、@ 100 Hz ≈ 64.4 dB，与 ISO 226 公布值一致。
"""
from __future__ import annotations

import math

#: ISO 226:2003 首选频率（Hz）
F_TABLE = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
           500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
           6300, 8000, 10000, 12500]

#: α_f —— 响度感知指数
AF_TABLE = [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349,
            0.330, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250,
            0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254,
            0.271, 0.301]

#: L_u —— 以 1000 Hz 为基准的线性传输函数幅值相关量
LU_TABLE = [-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2,
            -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1,
            -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1]

#: T_f —— 安静听阈（dB SPL）
TF_TABLE = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
            14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3,
            -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3]

assert len(F_TABLE) == len(AF_TABLE) == len(LU_TABLE) == len(TF_TABLE), \
    "ISO 226 四张表必须等长"


def _interp(table: list[float], f: float) -> float:
    """在 ISO 226 的首选频率表上做对数频率线性插值。"""
    if f <= F_TABLE[0]:
        return table[0]
    if f >= F_TABLE[-1]:
        return table[-1]
    lo, hi = 0, len(F_TABLE) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if F_TABLE[mid] <= f:
            lo = mid
        else:
            hi = mid
    t = (math.log(f) - math.log(F_TABLE[lo])) / \
        (math.log(F_TABLE[hi]) - math.log(F_TABLE[lo]))
    return table[lo] + t * (table[hi] - table[lo])


def equal_loudness_spl(freq: float, phon: float) -> float:
    """给定响度级 L_N（方），返回频率 f 上听起来一样响所需的声压级（dB SPL）。

    这是 ISO 226:2003 §4 的闭式解，不是查表近似。
    """
    if not 0.0 <= phon <= 90.0:
        raise ValueError("phon 必须在 0–90（ISO 226 的有效范围）")
    af = _interp(AF_TABLE, freq)
    lu = _interp(LU_TABLE, freq)
    tf = _interp(TF_TABLE, freq)
    a_f = 4.47e-3 * (10 ** (0.025 * phon) - 1.15) + \
        (0.4 * 10 ** (((tf + lu) / 10.0) - 9.0)) ** af
    return (10.0 / af) * math.log10(a_f) - lu + 94.0


def equal_loudness_db(freqs, phon: float = 80.0) -> list[float]:
    """批量版。返回每个频率所需的 SPL。"""
    return [equal_loudness_spl(f, phon) for f in freqs]


def sensitivity(freqs, phon: float = 80.0, floor: float = 0.05) -> list[float]:
    """感知权重：1 kHz 处为 1，其余频率按等响曲线的相对敏感度缩放。

    构造（见模块 docstring 的理由）：
        w(f) = max(10^((L_F(1000) − L_F(f)) / 10), floor)

    phon 越大 → 曲线越平 → 低频权重越高（这就是"响度补偿"的量化形式）。
    """
    ref = equal_loudness_spl(1000.0, phon)
    out = []
    for f in freqs:
        lf = equal_loudness_spl(f, phon)
        w = 10 ** ((ref - lf) / 10.0)
        out.append(max(w, floor))
    return out


def weight_fn(phon: float = 80.0, floor: float = 0.05):
    """返回 engine.rank 可直接使用的 w(f) 调用式。"""
    ref = equal_loudness_spl(1000.0, phon)

    def w(f: float) -> float:
        return max(10 ** ((ref - equal_loudness_spl(f, phon)) / 10.0), floor)
    return w
