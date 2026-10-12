#!/usr/bin/env python3
"""频响偏差的专业化制图：多型号 · 基线 · Bark 带 · 参考线。

与 `fr_features.py`（算）和 `spatial.py`（判）配套的**看**。
图上每一样东西都注明口径与出处，不画没有依据的参考线。

用法：
    python src/fr_plot.py                       # 默认选 6 款（多源优先）
    python src/fr_plot.py --models "HD 600,HD 650" --out out/fr.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "data", "features.json")
OUT = os.path.join(ROOT, "out")

#: 默认对比名单：主流池里多源、讨论度高的几款（拿不到就退化为"取前 6"）
DEFAULT_MODELS = ("Sennheiser HD 600", "Sennheiser HD 650", "Sennheiser HD 560S",
                  "HiFiMan Sundara", "Beyerdynamic DT 1990 Pro", "Sony MDR-7506")

# ---- 参考线（口径与出处写在图例里，不画的就不标）--------------------------
#: Olive & Welti 2013（二手）：1–4 kHz 的 6 dB 是"双盲可区分偏好"的量级。
#: 只在这个频段内画，**不许平移到别的频段**——那是伪造依据（见 audibility.py）。
PREF_BAND = (1000.0, 4000.0)
PREF_DB = 6.0


# ---------------------------------------------------------------- 色彩（OKLCH）
def _cbrt(x):
    return math.copysign(abs(x) ** (1 / 3), x)


def oklch_to_rgb(L, C, h):
    """OKLCH → sRGB（0–1）。色域外硬夹——这里只用于选线色，损失可接受。"""
    rad = math.radians(h)
    a, b = C * math.cos(rad), C * math.sin(rad)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    f = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055  # noqa: E731
    return tuple(max(0.0, min(1.0, f(max(0.0, min(1.0, c))))) for c in (r, g, bb))


def hue_ramp(n: int, start: float = 30.0) -> list:
    """等色相差的分类色：同一 L、同一 C，只转色相。

    为什么不用 tab10：tab10 的明度参差，深色底上有的线看不清。
    等明度 + 等彩度 → 每条线的"视觉权重"相同，判读才公平。
    """
    return [oklch_to_rgb(0.72, 0.13, (start + i * 360 / n) % 360) for i in range(n)]


def bark(f):
    return 26.81 * f / (1960.0 + f) - 0.53


def bark_to_hz(z):
    return 1960.0 * (z + 0.53) / (26.28 - z)


# ---------------------------------------------------------------- 选型与制图
def pick(models, names):
    """按名单挑；名单里没有的就忽略，不够 2 条就从多源的里补。"""
    by = {}
    for m in models:
        by.setdefault(m["name"], []).append(m)
    got = []
    for want in names:
        if want in by:
            got.append((want, by[want]))
    if len(got) < 2:
        multi = [(n, v) for n, v in by.items() if len(v) >= 2]
        got += multi[:6 - len(got)]
    return got[:6]


def plot(names=None, out=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 中文标签：走 matplotlib 自己的字体表（不碰 Qt，纯脚本场景）
    fams = {f.name for f in fm.fontManager.ttflist}
    for want in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei"):
        if want in fams:
            plt.rcParams["font.sans-serif"] = [want]
            break
    plt.rcParams["axes.unicode_minus"] = False

    data = json.load(open(FEATURES, encoding="utf-8"))
    grid = np.array(data["grid"])
    chosen = pick(data["models"], tuple(names) if names else DEFAULT_MODELS)
    if not chosen:
        raise SystemExit("没挑到可画的型号")

    # 深色底：与工具侧同一套观感（数据是技术的，图不必装素）
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12.0, 7.2), dpi=150, sharex=True,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=.08))
    fig.patch.set_facecolor("#0b0d12")
    for a in (ax, ax2):
        a.set_facecolor("#10131a")
        a.grid(True, which="major", color="#232a38", lw=.6)
        a.grid(True, which="minor", color="#1a2130", lw=.4, alpha=.6)
        for s in a.spines.values():
            s.set_color("#2e3644")
        a.tick_params(colors="#98a2b3", labelsize=8.5)
        a.set_xscale("log")

    # Bark 带底纹：对数轴等距 ≠ 感知等距（spatial.py 同一条理由）
    edges = [20.0] + [e for e in (bark_to_hz(bark(20) + i)
                                  for i in range(1, int(bark(20000) - bark(20)) + 1))
                      if 20 < e < 20000] + [20000.0]
    for i in range(len(edges) - 1):
        if i % 2:
            ax.axvspan(edges[i], edges[i + 1], color="#171b24", lw=0, zorder=0)

    # 参考线：只画有出处的，且只在它覆盖的频段里画
    ax.axhline(0, color="#626c80", lw=.8, ls="-", zorder=2)
    ax.axvspan(PREF_BAND[0], PREF_BAND[1], color="#3d7dff", alpha=.05, lw=0, zorder=1)
    for sgn in (1, -1):
        ax.plot([PREF_BAND[0], PREF_BAND[1]], [sgn * PREF_DB] * 2,
                color="#e0a92e", lw=1.1, ls="--", zorder=2)
    ax.text(math.sqrt(PREF_BAND[0] * PREF_BAND[1]), PREF_DB + 1.2,
            "±6 dB：双盲可区分偏好的量级（Olive & Welti 2013，二手，仅 1–4 kHz）",
            color="#e0a92e", fontsize=7.5, ha="center")

    cols = hue_ramp(len(chosen))
    band_defs = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 500),
                 ("mid", 500, 2000), ("upper_mid", 2000, 6000),
                 ("high", 6000, 16000), ("air", 16000, 20000)]
    centers = [math.sqrt(lo * hi) for _, lo, hi in band_defs]

    for (name, curves), col in zip(chosen, cols):
        dev = np.mean([c["dev"] for c in curves], axis=0)     # 多源取均值
        ax.plot(grid, dev, lw=1.5, color=col, label="%s（%d 源）" % (name, len(curves)),
                zorder=3)
        prof = [float(np.mean([d for f, d in zip(grid, dev) if lo <= f <= hi]))
                for _, lo, hi in band_defs]
        ax2.plot(centers, prof, "o-", lw=1.2, ms=4, color=col, zorder=3)

    ax.set_ylim(-18, 18)
    ax.set_ylabel("相对人群基线 (dB)", color="#98a2b3", fontsize=9)
    ax2.set_ylabel("频段均值 (dB)", color="#98a2b3", fontsize=9)
    ax2.set_xlabel("频率 (Hz，对数轴；底纹为 Bark 临界频带)", color="#98a2b3", fontsize=9)
    ax2.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax2.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
    ax2.set_xlim(20, 20000)
    leg = ax.legend(loc="lower left", fontsize=8, framealpha=.92,
                    facecolor="#171b24", edgecolor="#2e3644", labelcolor="#e8ecf3")
    leg.get_frame().set_linewidth(.8)
    ax.set_title("频响偏差对比（相对全库中位数基线；多来源型号取均值）",
                 color="#e8ecf3", fontsize=11, pad=10)

    os.makedirs(OUT, exist_ok=True)
    out = out or os.path.join(OUT, "fr_compare.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("→ %s（%d 款）" % (out, len(chosen)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="逗号分隔的型号名（须与 features.json 一致）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    names = [s.strip() for s in a.models.split(",") if s.strip()] or None
    plot(names, a.out or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
