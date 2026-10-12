"""Bark 临界频带：标度自洽性 + 积分域修正的有效性。

核心断言是 `test_octave_grid_overweights_low_frequency`——
它把"1/12 倍频程在低频远窄于 Bark 带"这个数学缺陷钉成可回归的数字。
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import perceptual as pc  # noqa: E402


class TestBarkScale(unittest.TestCase):
    def test_bark_matches_reference_points(self):
        """与文献值对照（Traunmüller 形式）。"""
        for f, exp in ((100.0, 0.8), (1000.0, 8.5), (10000.0, 22.0)):
            self.assertAlmostEqual(pc.bark(f), exp, delta=0.6,
                                   msg="%.0fHz 应约 %.1f Bark" % (f, exp))

    def test_bark_is_monotonic(self):
        prev = -1e9
        for i in range(400):
            f = 20.0 * (20000.0 / 20.0) ** (i / 399.0)
            z = pc.bark(f)
            self.assertGreater(z, prev, "Bark 标度必须单调递增")
            prev = z

    def test_bark_roundtrip(self):
        """bark_to_hz(bark(f)) 必须回到 f（解析逆，不是近似）。"""
        for f in (30.0, 100.0, 440.0, 1000.0, 5000.0, 15000.0):
            self.assertAlmostEqual(pc.bark_to_hz(pc.bark(f)), f, places=6)

    def test_critical_bw_matches_reference(self):
        """1 kHz 处临界带宽 ≈160–170 Hz（文献常引 ~160）。"""
        self.assertTrue(155.0 < pc.critical_bw(1000.0) < 175.0,
                        "实得 %.1f Hz" % pc.critical_bw(1000.0))
        # 带宽必须随频率单调增
        prev = 0.0
        for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
            bw = pc.critical_bw(f)
            self.assertGreater(bw, prev, "临界带宽应随频率增大")
            prev = bw

    def test_critical_bw_consistent_with_bark(self):
        """critical_bw 必须等于 1/(dz/df) —— 两个标度不能互相矛盾。"""
        for f in (100.0, 1000.0, 8000.0):
            h = 1e-3
            dzdf = (pc.bark(f + h) - pc.bark(f - h)) / (2 * h)
            self.assertAlmostEqual(pc.critical_bw(f), 1.0 / dzdf, delta=1.0,
                                   msg="%.0fHz 处两法不一致" % f)


class TestBarkEdges(unittest.TestCase):
    def test_edges_cover_and_monotonic(self):
        e = pc.bark_edges()
        self.assertEqual(e[0], 20.0)
        self.assertEqual(e[-1], 20000.0)
        self.assertTrue(all(e[i] < e[i + 1] for i in range(len(e) - 1)))
        self.assertGreaterEqual(len(e) - 1, 20, "Bark 带数应接近 24")

    def test_edges_equally_spaced_in_bark(self):
        """等宽切在 Bark 域 —— 这是"每个带在人耳看来一样宽"的定义。"""
        e = pc.bark_edges()
        zs = [pc.bark(x) for x in e]
        d = [zs[i + 1] - zs[i] for i in range(len(zs) - 1)]
        self.assertAlmostEqual(max(d) - min(d), 0.0, places=6)

    def test_low_bands_wider_than_high_bands_in_hz_ratio(self):
        """低频 Bark 带宽约 100Hz，高频约 0.2·f —— 形状要对。"""
        self.assertLess(pc.critical_bw(100.0), 120.0)
        self.assertGreater(pc.critical_bw(10000.0), 2000.0)


class TestIntegrationDomain(unittest.TestCase):
    """本模块存在的理由：量化并修正旧积分域的缺陷。"""

    def test_octave_grid_overweights_low_frequency(self):
        """**这是本模块的核心断言**：1/12 倍频程在低频远窄于 Bark 带。

        50 Hz 处窄 26 倍 → 同样的 dB 误差在旧度量里对 RMS 的贡献
        是 1 kHz 处的 26 倍。这个数字必须可回归。
        """
        ratios = {}
        for f in (50.0, 100.0, 1000.0, 8000.0):
            oct12 = f * (2 ** (1 / 12) - 1)
            ratios[f] = pc.critical_bw(f) / oct12
        self.assertGreater(ratios[50.0], 15.0, "50Hz 处倍程带应远窄于 Bark 带")
        self.assertGreater(ratios[100.0], 8.0)
        self.assertLess(ratios[1000.0], 5.0)
        self.assertLess(ratios[8000.0], 6.0)

    def test_bark_rms_downweights_narrowband_low_freq_error(self):
        """只在 40Hz 一个网格点上有 6dB 偏差时，Bark 积分应给出更小的 RMS。

        旧度量：该点权重与其它点相同 → RMS 直接被拉高。
        Bark 度量：40Hz 属于一个很宽的 Bark 带，6dB 被带内其它点稀释。
        """
        import engine
        D = engine.load()
        grid = D["grid"]
        n = len(grid)
        target = [0.0] * n
        curve = [0.0] * n
        # 找最接近 40Hz 的网格点，只在那里加 6dB
        i40 = min(range(n), key=lambda i: abs(grid[i] - 40.0))
        curve[i40] = 6.0

        old = math.sqrt(6.0 ** 2 / n)                      # 等权网格 RMS
        new = pc.bark_rms(curve, target, grid)
        self.assertLess(new, old * 0.6,
                        "Bark 积分应显著稀释窄带低频误差：%.3f vs %.3f" % (new, old))

    def test_bark_rms_equals_grid_rms_when_deviation_is_flat(self):
        """全频段均匀偏差时，两种积分域必须给同一个 RMS（积分域不该引入偏置）。"""
        import engine
        D = engine.load()
        grid = D["grid"]
        n = len(grid)
        target = [0.0] * n
        curve = [3.0] * n                      # 全频 +3dB
        old = 3.0
        new = pc.bark_rms(curve, target, grid)
        self.assertAlmostEqual(new, old, places=6)

    def test_band_means_cover_all_bands(self):
        """带内均值必须覆盖到全部非空 Bark 带（不能 silently 丢带）。"""
        import engine
        D = engine.load()
        grid = D["grid"]
        bm = pc.bark_band_means([0.0] * len(grid), [0.0] * len(grid), grid)
        idx = pc.band_index(grid)
        expected = len({k for k in idx if k >= 0})
        self.assertEqual(len(bm), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
