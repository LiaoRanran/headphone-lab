#!/usr/bin/env python3
"""偏差可闻性：把「差几个 dB」翻译成「听不听得出」。

## 为什么要有这个模块

引擎原来的输出是一个分数（`100 − 7·rms − 6·peak`），系数是手调的。
问题不在系数准不准，而在**这个数不对应任何可验证量**——它没法回答
用户唯一关心的问题：*这只耳机跟目标的差距，我听得出来吗？*

本模块把结论换成可核对的判据：**哪些频段越过了文献里的可闻阈值**。

## 文献依据（附出处与可信度）

1. **Olive & Welti, AES 2013, "Listener Preferences for Different Headphone
   Target Response Curves"**：训练听音者在双盲条件下，需要 **≥6 dB** 的
   频谱偏差、且集中在 **1–4 kHz**，才能对（其余完全相同的）耳机做出
   统计显著的偏好区分。
   ⚠️ **可信度警告**：AES 原文在付费墙后，本机取不到。
   这个数字来自二手转述（thegadgetdigest.com 引述）。
   标为「二手」，可作方向性依据，**不可作为精确阈值引用**。

2. **JAES 2022 耳机研究**：偏好评分 **82%** 的方差可由频响**平滑度**解释，
   其算法是 **1/12 倍频程移动平均后的 RMS 误差**；而「单元类型」
   （平板/动圈/静电）解释的方差 **不足 1%**。
   → 这直接支持了本项目的核心前提（频响是主信号，换能原理不是），
     而且它的度量口径与 `fr_features.py` 的 1/12 倍频程网格**恰好一致**。

3. **Olive 对 283 款耳机的分析**：相对 Harman 目标的 RMS 误差范围
   **2.5–13 dB**。本项目的候选集落在同一量级（见 engine 输出的 RMS 列）。

4. **ICA 研究**：**100–1600 Hz** 频段的不规则性尤其影响耳机听音中的
   **前后方位判别**。→ 频响不只改音色，也改「声音在脑内何处出现」。
   本项目此前完全没建模这一维（只当音色处理）。

5. **Olive 分析**：零售价格与实测频响**无相关性**（仅高价段低频离散度略小）。

## 本模块的立场

阈值全部**按频段分别给**，因为文献支持「偏差落在哪个频段」比「偏差多大」
更要紧（第 1 条）。并且明确区分两类判据：

  * `AUDIBLE`：可闻（有人能听出来）
  * `PREFERENCE`：足以改变偏好（双盲下能做出显著区分）

这两件事差一个量级，混在一起谈是外行话。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 阈值表

#: 训练听音者在双盲下做出统计显著偏好区分所需的频谱偏差（dB）。
#: 来源：Olive & Welti AES 2013（**二手转述，见模块 docstring 警告**）
PREFERENCE_JND_DB = 6.0

#: 该阈值的研究覆盖频段。文献只说 1–4 kHz，其它频段**没有**同等质量的依据，
#: 因此本模块对频段外只做标注、不给同等强度的结论。
PREFERENCE_JND_BAND = (1000.0, 4000.0)

#: 宽带电平变化的可闻阈（dB）。这是「整体响度差」不是「频谱形状差」——
#: 两者机制不同：人耳对**宽带电平**极敏感（约 1 dB 即可察觉），
#: 对**有限频段的形状偏差**则要 6 dB 量级。混为一谈是最常见的错误。
LEVEL_JND_DB = 1.0

#: 低频段：<200 Hz 的搁架偏差。文献（Olive 系列）一致显示适度低频抬升
#: 被偏好（Harman 曲线本身即含 room gain 补偿），但**没有**给出可闻阈的
#: 精确分贝数 → 这里的值标为「工程估计」，用于分级而非判判决。
LOW_SHELF_JND_DB = 3.0

#: 影响前后方位判别的频段。**口径以 `spatial.py` 为准** —— 那一维的文献登记、
#: 机制推导与计算都在那里（`src/spatial.py`），这里只引用，避免两处漂移。
#: 注意它在 `spatial.py` 里是**未取到原文**的条目，因此**不作判据**，
#: 只用于"别把它悄悄算进音色里"这件事。
from spatial import SPATIAL_BAND  # noqa: E402


def classify_band(f: float) -> str:
    """把频率归到文献覆盖的判据类别。"""
    if f < 200:
        return "low_shelf"
    if PREFERENCE_JND_BAND[0] <= f <= PREFERENCE_JND_BAND[1]:
        return "preference_band"
    if 200 <= f < 1000:
        return "mid_low"
    if f > 4000:
        return "high"
    return "other"


def band_threshold(f: float) -> float | None:
    """该频率上的判据阈值（dB）。返回 None 表示**没有可靠依据**。

    **宁缺勿错**：宁可不给阈值，也不把 1–4 kHz 的 6 dB 平移到别的频段——
    文献没有说它对其它频段成立。
    """
    k = classify_band(f)
    if k == "preference_band":
        return PREFERENCE_JND_DB
    if k == "low_shelf":
        return LOW_SHELF_JND_DB
    return None


def analyse(curve, target, grid) -> dict:
    """逐频段评估：哪些段越过了可闻/可改变偏好的阈值。

    返回结构刻意分开「有依据的结论」与「没依据的部分」，
    调用方不能把两者混在一句话里说。
    """
    devs = [(f, c - t) for f, c, t in zip(grid, curve, target)]
    if not devs:
        return {"n": 0}

    pref = [(f, d) for f, d in devs if classify_band(f) == "preference_band"]
    low = [(f, d) for f, d in devs if classify_band(f) == "low_shelf"]
    high = [(f, d) for f, d in devs if classify_band(f) == "high"]
    spatial = [(f, d) for f, d in devs if SPATIAL_BAND[0] <= f <= SPATIAL_BAND[1]]

    def worst(pairs):
        return max(pairs, key=lambda p: abs(p[1])) if pairs else (None, 0.0)

    fw, dw = worst(pref)
    fl, dl = worst(low)
    fh, dh = worst(high)
    fs, _ = worst(spatial)

    return {
        "n": len(devs),
        # ---- 有文献依据的部分 ----
        "pref_band_worst_db": round(dw, 2),
        "pref_band_worst_hz": fw,
        "pref_band_exceeds": abs(dw) >= PREFERENCE_JND_DB,
        "pref_band_pct_over": round(
            100.0 * sum(1 for _, d in pref if abs(d) >= PREFERENCE_JND_DB) / max(1, len(pref)), 1),
        # ---- 工程估计（无同等质量依据，用于分级）----
        "low_shelf_worst_db": round(dl, 2),
        "low_shelf_worst_hz": fl,
        "low_shelf_exceeds": abs(dl) >= LOW_SHELF_JND_DB,
        # ---- 没有依据、只作描述的部分 ----
        "high_worst_db": round(dh, 2),
        "high_worst_hz": fh,
        "spatial_band_worst_hz": fs,
    }


def verdict_text(a: dict) -> str:
    """把分析结果写成一句不越界的话。"""
    if not a.get("n"):
        return "无数据"
    if a["pref_band_exceeds"]:
        return ("1–4kHz 最大偏差 %.1f dB @ %.0f Hz，**越过**双盲可辨阈值（%.0f dB，"
                "Olive & Welti 2013，二手源）→ 这一般的偏差是能被听出来的"
                % (a["pref_band_worst_db"], a["pref_band_worst_hz"] or 0,
                   PREFERENCE_JND_DB))
    return ("1–4kHz 最大偏差 %.1f dB @ %.0f Hz，**未越过**双盲可辨阈值（%.0f dB）"
            "→ 按现有文献，这点偏差不足以让训练听音者在双盲下做出稳定区分。"
            "**注意这说的是「能否区分」，不是「听起来一模一样」。**"
            % (a["pref_band_worst_db"], a["pref_band_worst_hz"] or 0,
               PREFERENCE_JND_DB))


# ---------------------------------------------------------------- 自校验
def _selfcheck():
    print("阈值表：")
    for f in (50.0, 150.0, 500.0, 2000.0, 6000.0, 12000.0):
        t = band_threshold(f)
        print("  %6.0f Hz → %-16s 阈值 %s"
              % (f, classify_band(f), ("%.1f dB" % t) if t else "**无依据**"))
    print()
    print("口径提醒：宽带电平 1 dB 即可察觉；有限频段形状偏差要 6 dB 量级才改偏好。")
    print("这两件事机制不同，不能混为一谈。")

    # 自校验：1-4kHz 给 6 dB，其它频段给 0
    grid = [50.0, 150.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 12000.0]
    target = [0.0] * len(grid)
    curve = [0.0] * len(grid)
    for i, f in enumerate(grid):
        if 1000 <= f <= 4000:
            curve[i] = 6.5
    a = analyse(curve, target, grid)
    print()
    print("自校验（1–4kHz 全给 6.5 dB）：")
    print("  超过阈值 =", a["pref_band_exceeds"], "（期望 True）")
    print("  超阈比例 = %.1f%%（期望 100.0）" % a["pref_band_pct_over"])
    print("  判定：", verdict_text(a))

    curve2 = [0.0] * len(grid)
    for i, f in enumerate(grid):
        if 1000 <= f <= 4000:
            curve2[i] = 5.5
    a2 = analyse(curve2, target, grid)
    print()
    print("自校验（1–4kHz 全给 5.5 dB）：")
    print("  超过阈值 =", a2["pref_band_exceeds"], "（期望 False）")


if __name__ == "__main__":
    _selfcheck()
