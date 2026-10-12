"""可闻性判据：阈值表、分频段归类、以及"宁缺勿错"的边界。

核心断言是 `test_no_threshold_outside_documented_band`——
文献只证明了 1–4kHz 上需要 6 dB，**没有**证明别的频段也一样。
把那个数字平移到 8kHz 是伪造依据，必须被测试挡住。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import audibility as A  # noqa: E402


GRID = [50.0, 100.0, 150.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 12000.0]


def curve_with(bumps):
    """按 {频率: dB} 造一条曲线（未给的频点取 0）。

    收一个 dict 而不是 **kwargs —— 频率是 float，不能当关键字参数名。
    """
    c = [0.0] * len(GRID)
    for f, v in bumps.items():
        if float(f) in GRID:
            c[GRID.index(float(f))] = v
    return c


class TestThresholds(unittest.TestCase):
    def test_preference_band_threshold(self):
        """1–4kHz 必须给 6 dB（Olive & Welti 2013，二手源）。"""
        self.assertEqual(A.PREFERENCE_JND_DB, 6.0)
        for f in (1000.0, 2000.0, 3000.0, 4000.0):
            self.assertEqual(A.band_threshold(f), 6.0)
        for f in (999.0, 4001.0):
            self.assertNotEqual(A.band_threshold(f), 6.0)

    def test_no_threshold_outside_documented_band(self):
        """**核心断言**：文献没覆盖的频段不给阈值，而不是照搬 6 dB。

        把 1–4kHz 的结论平移到 6kHz/12kHz 上，是在伪造文献支持。
        """
        for f in (500.0, 800.0, 5000.0, 6000.0, 12000.0):
            self.assertIsNone(A.band_threshold(f),
                              "%.0fHz 无文献依据，不应给阈值" % f)

    def test_level_jnd_differs_from_spectral_jnd(self):
        """宽带电平 JND（~1dB）与有限频段形状 JND（~6dB）**不是一回事**。

        混为一谈是这个领域最常见的错误：宽带电平人耳极敏感，
        有限频段的形状偏差则要 6 dB 量级才改变偏好。
        """
        self.assertLess(A.LEVEL_JND_DB, A.PREFERENCE_JND_DB)
        self.assertLessEqual(A.LEVEL_JND_DB, 1.0)
        self.assertGreaterEqual(A.PREFERENCE_JND_DB, 5.0)

    def test_spatial_band_is_declared(self):
        """影响前后方位判别的频段必须显式声明（ICA 研究：100–1600Hz）。"""
        self.assertEqual(A.SPATIAL_BAND, (100.0, 1600.0))


class TestAnalyse(unittest.TestCase):
    def test_over_threshold_detected(self):
        c = curve_with({1000.0: 6.5})
        a = A.analyse(c, [0.0] * len(GRID), GRID)
        self.assertTrue(a["pref_band_exceeds"])
        self.assertAlmostEqual(a["pref_band_worst_db"], 6.5, places=6)

    def test_under_threshold_not_flagged(self):
        c = curve_with({2000.0: 5.9})
        a = A.analyse(c, [0.0] * len(GRID), GRID)
        self.assertFalse(a["pref_band_exceeds"])

    def test_threshold_is_inclusive_at_boundary(self):
        """恰好 6.0 dB 应判为越阈（文献说法是「≥6 dB」）。"""
        c = curve_with({3000.0: 6.0})
        a = A.analyse(c, [0.0] * len(GRID), GRID)
        self.assertTrue(a["pref_band_exceeds"])

    def test_negative_deviation_counts_as_much_as_positive(self):
        """凹陷和凸起同样越阈——判据看的是绝对值。"""
        for v in (7.0, -7.0):
            a = A.analyse(curve_with({1000.0: v}), [0.0] * len(GRID), GRID)
            self.assertTrue(a["pref_band_exceeds"], "%+.1f dB 应越阈" % v)

    def test_pct_over_counts_only_preference_band(self):
        """超阈比例的分母只能是 1–4kHz 的点，不能把别的频段算进去。"""
        c = curve_with({1000.0: 8.0, 2000.0: 8.0, 6000.0: 20.0})
        a = A.analyse(c, [0.0] * len(GRID), GRID)
        # GRID 落在 1–4kHz 的只有 4 个点（1000/2000/3000/4000），其中 2 个越阈。
        # 6kHz 那个 20 dB 不计入分母——它不在文献覆盖的频段里。
        self.assertEqual(a["pref_band_pct_over"], 50.0)
        self.assertEqual(A.classify_band(6000.0), "high")

    def test_empty_input_is_safe(self):
        self.assertEqual(A.analyse([], [], []).get("n"), 0)

    def test_verdict_rejects_overclaiming(self):
        """未越阈时不能写成「听起来一样」——只能写「不足以稳定区分」。"""
        a = A.analyse(curve_with({1000.0: 2.0}), [0.0] * len(GRID), GRID)
        txt = A.verdict_text(a)
        self.assertIn("不足以", txt)
        self.assertIn("不是", txt)          # 明确否认「听起来一模一样」
        self.assertNotIn("听起来完全一样", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
