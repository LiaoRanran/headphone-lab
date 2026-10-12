#!/usr/bin/env python3
"""空间维度：耳机频响偏差**能改什么、改不了什么**。

## 为什么单独一个模块

README §4.35 第 4 条记着「100–1600 Hz 不规则性影响前后方位判别（ICA 研究）」，
而代码里一行都没用上——只留了个 `audibility.SPATIAL_BAND` 常量当占位。
这是"查了却没用上"的欠账。本模块按**能核到的文献**把这一维落地，
并明确划出"算不了"的部分。

## 地基：一条必须先讲清楚的机制（否则整维都是错的）

耳机重放时，耳机自身的频响偏差 H(f) 是**同时**加在左右两个声道上的
（同一个滤波器分别作用于 L 和 R）。于是

    ILD_out(f) = 20·log10(|X_L·H| / |X_R·H|) = 20·log10(|X_L| / |X_R|) = ILD_in(f)

**ILD 完全不变**；ITD 同理（同一滤波器对两个声道给出同一相移，互相关峰不移动）。
所以"这只耳机频响不平整 → 声场歪了"这句话在物理上是**错的**，除非走下面两条路：

  a. **左右单元不一致**（H_L ≠ H_R）：此时 ILD/ITD 才会真的被改 → 结像偏移、
     声源宽度变化。这需要**逐声道**曲线（本库只有 squig 那批有 L/R 分文件）。
  b. **单耳谱线索**：前后 / 上下（中矢面）判别本来就不靠 ILD/ITD——中矢面上
     ILD 与 ITD 恒为零——而靠耳廓造成的**谱峰谷**。耳机把这个谱形改写了，
     于是记录里携带的"前后"线索被一起改写。

(a) 可以算（有 L/R 数据时）；(b) 可以**报**，但不能当判据——因为测量的不是你的耳廓。

## 文献登记（与 `audibility.py` 同一套规矩：没核到原文的不给数字）

| id | 结论 | 出处 | 状态 |
| --- | --- | --- | --- |
| S1 | 中矢面存在"方向带"：窄带噪声的方向感由**中心频率**决定；宽带信号另有 5 个 "boosted bands" | Blauert, Acustica 22, 205–213 (1969/70) | **经二手核到**：Zieliński et al., EURASIP J. ASMP 2022:3（开放获取，已取全文）。**但具体带边界在该文附录表 4，本机未取到** → 本模块不给数字 |
| S2 | 前后判别最关键的是 **5–6 kHz**（XGBoost 特征重要性：gammatone 第 30/31 通道的 ILD 标准差 ≈5.5/6 kHz；CNN 梯度归因图亦指向 5–6 kHz） | 同上（实验 1 / 5.9 节） | **已核**，数值可用 |
| S3 | 听者**个体化**的谱线索在 **8–16 kHz** | Langendijk & Bronkhorst, JASA 112, 1583 (2002) | 已核（经 S2 同文转述）。**但本库该频段跨源离散度大** → 只标注，不判 |
| S4 | 水平面三类线索里 **ILD 单独约 90%** > IC > ITD | 同上（实验 3） | 已核 |
| S5 | 双工理论边界：>1600 Hz 主要走 ILD，<800 Hz 主要走 ITD | 二手博客转述（vrtonung.de），与经典 duplex theory 一致 | **未取到一手** → 只作方向性说明 |
| S6 | 低频成分参与中矢面定位 | Morimoto et al., Acoust. Sci. & Tech. 24(2), 76–82 (2003) | **文献存在且主题吻合，但结论未取到** → 不用其数字 |
| S6b | 方向带存在**个体差异**；窄带定位与外耳声学 | Itoh/Iida/Morimoto, Appl. Acoust. 68, 909–915 (2007)；Middlebrooks, JASA 92, 2607 (1992) | 经 Iida (2019, Springer) 章节参考文献确认存在；**正文付费墙，结论未取到** → 登记 |
| S7 | 「100–1600 Hz 不规则性影响前后方位判别」 | 项目 README 原记为 ICA 研究 | **未取到原文**（本轮检索未命中）→ **不当阈值**，只登记 |

**因此本模块没有任何"空间可闻阈"**。空间感知的阈值与"音色偏差 6 dB"
不是一回事，文献没给到可核对的数字时，宁可只报数值。
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "data", "features.json")


# ---------------------------------------------------------------- 文献登记

class Band:
    """一个"线索频段"。`usable_as_threshold` 为假时**不得**拿它当判据。"""

    def __init__(self, key, lo, hi, carries, source, verified, threshold, note=""):
        self.key, self.lo, self.hi = key, lo, hi
        self.carries, self.source, self.verified = carries, source, verified
        self.usable_as_threshold, self.note = threshold, note

    def __repr__(self):
        return "<Band %s %.0f–%.0f Hz %s>" % (self.key, self.lo, self.hi,
                                              "可用" if self.verified else "未核")


BANDS: tuple[Band, ...] = (
    Band("frontback", 5000.0, 6000.0, "前后判别最关键（两路方法一致）", "S2",
         True, False, "可报数值；文献给的是「重要性」，不是「偏差多大才听得出」"),
    Band("individual", 8000.0, 16000.0, "听者个体化的谱线索", "S3",
         True, False, "本库该频段跨源离散度大，只标注"),
    Band("itd_dominant", 20.0, 800.0, "ITD 主导（双工理论下段）", "S5",
         False, False, "二手转述，未取到一手"),
    Band("ild_dominant", 1600.0, 20000.0, "ILD 主导（双工理论上段）", "S5",
         False, False, "二手转述，未取到一手"),
    Band("spatial_100_1600", 100.0, 1600.0, "项目原记的「影响前后方位判别」频段", "S7",
         False, False, "**未取到原文**，登记存照，不作任何判据"),
)

BY_KEY = {b.key: b for b in BANDS}

#: 与 `audibility.SPATIAL_BAND` 同源的常量放在这里（那里改为引用，避免两处口径漂移）
SPATIAL_BAND = (BY_KEY["spatial_100_1600"].lo, BY_KEY["spatial_100_1600"].hi)


def usable_bands() -> list[Band]:
    """只有已核到的频段才允许出现在结论里。"""
    return [b for b in BANDS if b.verified]


# ---------------------------------------------------------------- 可计算部分

def decompose(dev_l, dev_r):
    """把左右声道的偏差拆成**共模**（diotic）与**差模**（会改 ILD 的那部分）。

    共模 = (L+R)/2：两耳一起被抬高/压低 → **不改 ILD/ITD**，只改单耳谱形；
    差模 = L−R：这才是真正改写 ILD 的量（ILD_out = ILD_in + 差模）。

    这一拆是"S4：频响不平不等于声场歪"这件事的量化形式。
    """
    n = min(len(dev_l), len(dev_r))
    common = [(dev_l[i] + dev_r[i]) / 2.0 for i in range(n)]
    diff = [dev_l[i] - dev_r[i] for i in range(n)]
    return common, diff


def ild_perturbation(dev_l, dev_r) -> list[float]:
    """耳机引入的 ILD 误差（dB/频点）= L − R。全 0 表示该耳机不改 ILD。"""
    return [l - r for l, r in zip(dev_l, dev_r)]


def band_stats(dev, grid, band: Band) -> dict:
    """某线索频段内的偏差统计。数值都按"已核/未核"标记返回。"""
    sel = [(f, d) for f, d in zip(grid, dev) if band.lo <= f <= band.hi]
    if not sel:
        return {"band": band.key, "n": 0}
    vals = [d for _, d in sel]
    worst = max(sel, key=lambda p: abs(p[1]))
    return {
        "band": band.key,
        "n": len(sel),
        "mean_db": round(sum(vals) / len(vals), 2),
        "worst_db": round(worst[1], 2),
        "worst_hz": round(worst[0], 1),
        "verified": band.verified,
        "usable_as_threshold": band.usable_as_threshold,
    }


def analyse(dev, grid, dev_r=None) -> dict:
    """一条曲线的空间维度体检。

    返回里**刻意分开**"能量化的"与"只能描述的"：
    - `cue_bands`：已核频段上的谱线索偏差（可报数值，无阈值）
    - `ild_error_db`：只有给了逐声道曲线才算得出来；否则为 None（**不是 0**——
      0 会让人以为"这只耳机不改 ILD"，那是没测，不是没差）
    """
    out = {"cue_bands": [band_stats(dev, grid, b) for b in BANDS],
           "ild_error_db": None,
           "ild_error_worst_hz": None,
           "has_channel_data": dev_r is not None}
    if dev_r is not None:
        diff = ild_perturbation(dev, dev_r)
        i = max(range(len(diff)), key=lambda k: abs(diff[k]))
        out["ild_error_db"] = {
            "mean_abs_db": round(sum(abs(d) for d in diff) / len(diff), 2),
            "worst_db": round(diff[i], 2),
        }
        out["ild_error_worst_hz"] = round(grid[i], 1)
    return out


def verdict_text(a: dict) -> str:
    """把结果写成一句不越界的话（措辞与 §4.4 的效力边界一致）。"""
    if not a.get("cue_bands"):
        return "无数据"
    fb = next((b for b in a["cue_bands"] if b["band"] == "frontback"), None)
    parts = []
    if fb and fb.get("n"):
        parts.append("5–6kHz（前后判别最关键区）偏差 %+.1f dB（最差 %.1f @ %.0f Hz）"
                     % (fb["mean_db"], fb["worst_db"], fb["worst_hz"] or 0))
    if a["has_channel_data"]:
        e = a["ild_error_db"]
        parts.append("左右不一致引入的 ILD 误差 平均 %.2f / 最差 %.2f dB"
                     % (e["mean_abs_db"], e["worst_db"]))
    else:
        parts.append("**无逐声道数据 → ILD/ITD 是否被改写，本库不知道**"
                     "（不是「没差」，是没测）")
    parts.append("**注意**：以上都**不是**可闻判据——空间感知的可闻阈本模块没有，"
                 "文献没给到可核对的数字。")
    return " ｜ ".join(parts)


# ---------------------------------------------------------------- 全库普查

def survey(top: int = 12, models=None) -> list[dict]:
    """在特征库上跑一遍，看"前后判别关键区"的偏差分布。

    与 `audibility_survey.py` 同思路：先量分布，再说有没有区分力——
    别拿一个数就当结论。
    """
    if models is None:
        with open(FEATURES, encoding="utf-8") as f:
            data = json.load(f)
        grid, models = data["grid"], data["models"]
    rows = []
    for m in models:
        dev = m.get("dev")
        if not dev:
            continue
        fb = band_stats(dev, grid, BY_KEY["frontback"])
        rows.append({"name": m["name"], "source": m.get("source"),
                     "mean": fb.get("mean_db"), "worst": fb.get("worst_db"),
                     "hz": fb.get("worst_hz")})
    rows.sort(key=lambda r: -abs(r["mean"] or 0))
    return rows[:top]


def source_bias(rows, models) -> list[tuple[str, float, float]]:
    """自查：极端值是否被某个**测量来源**垄断。

    与 `audibility_survey.py` 自查 3 dB 阈值同一条规矩——先怀疑自己的口径。
    若"偏差最大的一批"里某个来源的占比远高于它在全库的占比，那这批排名
    量到的多半是**人工耳/测量架的差异**，不是耳机本身（本库 7 个来源的
    人工耳不同，高频段尤其明显，这是 README §5 第 4 条已有的局限）。
    """
    total = max(1, len(models))
    base = {}
    for m in models:
        base[m.get("source") or "?"] = base.get(m.get("source") or "?", 0) + 1
    hit = {}
    for r in rows:
        hit[r["source"] or "?"] = hit.get(r["source"] or "?", 0) + 1
    out = []
    for s, c in sorted(hit.items(), key=lambda kv: -kv[1]):
        share = 100.0 * c / max(1, len(rows))
        baseline = 100.0 * base.get(s, 0) / total
        if share > baseline * 1.6:          # 超出本底 1.6 倍才提示
            out.append((s, share, baseline))
    return out


def _selfcheck() -> None:
    print("线索频段（只有「已核」的才允许进结论）：")
    for b in BANDS:
        print("  %-16s %8.0f–%-6.0f Hz  %-4s 判据=%s  %s"
              % (b.key, b.lo, b.hi, "已核" if b.verified else "未核",
                 b.usable_as_threshold, b.note))

    print("\n共模/差模拆分（同一滤波器加在两耳上 → ILD 不变）：")
    l = [1.0, 2.0, 3.0]
    r = [1.0, 2.0, 3.0]                 # 两耳完全相同（差模为 0）
    common, diff = decompose(l, r)
    print("  L=R=[1,2,3] → 共模 %s ｜ 差模 %s（期望全 0 = ILD 不变）"
          % ([round(x, 2) for x in common], [round(x, 2) for x in diff]))
    l2 = [1.0, 2.0, 3.0]
    r2 = [0.0, 1.0, 3.0]                # 左右差 1 dB
    print("  L−R=1dB    → ILD 误差 %s（期望 [1,1,0]）"
          % [round(x, 2) for x in ild_perturbation(l2, r2)])

    print("\n全库普查：5–6kHz（前后判别关键区）偏差最大的几款")
    rows = survey(8)
    for r in rows:
        print("  %-42s %-14s 均值 %+6.2f ｜ 最差 %+6.2f @ %5.0f Hz"
              % (r["name"][:42], r["source"] or "", r["mean"], r["worst"], r["hz"] or 0))
    with open(FEATURES, encoding="utf-8") as f:
        models = json.load(f)["models"]
    for s, share, base in source_bias(rows, models):
        print("  ⚠ 来源偏差：%s 占极端值 %.0f%%，而在全库只占 %.0f%%"
              " → 这批排名量的可能是**测量架**，不是耳机" % (s, share, base))
    print("\n提醒：这批数字**没有阈值**可对照——报的是量级，不是「听不听得出」。")


if __name__ == "__main__":
    _selfcheck()
