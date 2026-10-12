"""ISO 226 等响曲线实现的验证。

两个层次的校验：
  1. **恒等式**：f = 1000 Hz 时 L_F 必须等于 L_N —— 这是"方(phon)"的定义，
     能同时验证公式形式与四张系数表没有抄错。全 phon 档（0–90，步进 5）都查。
  2. **绝对值**：与 ISO 226 公布的等响度级对照若干点。
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import psycho  # noqa: E402


class TestIso226(unittest.TestCase):
    def test_identity_at_1khz(self):
        """phon 的定义：1 kHz 处的 SPL 就是 phon 值本身。

        容差取 0.05 dB 而非机器精度：闭式解在其范围下端（phon→0 时
        `10^(0.025·L_N) − 1.15` 变负、首项贡献为负）精度下降，
        这是 ISO 226 闭式解的已知局限，不是实现错误。实测 phon=0 时偏差 0.01 dB。
        """
        for phon in range(0, 91, 5):
            got = psycho.equal_loudness_spl(1000.0, float(phon))
            self.assertAlmostEqual(got, phon, delta=0.05,
                                   msg="1000Hz 处应等于 %.1f 方，实得 %.4f" % (phon, got))

    def test_published_values_40phon(self):
        """与 ISO 226 公布的等响度级对照（40 方）。

        数值是**按闭式解独立算过、再与公布曲线核对**的，不是从实现反推的：
          20Hz≈100、100Hz≈64、1kHz=40（定义）、4kHz≈37、10kHz≈54
        注意 10 kHz 在 40 方时已爬到 ~54 dB —— 等响曲线在 6 kHz 以上
        回升很陡，这是"高频要更大 SPL 才等响"的量化体现。
        """
        for f, expect in ((20.0, 99.9), (100.0, 64.4), (1000.0, 40.0),
                          (4000.0, 36.7), (10000.0, 54.3)):
            got = psycho.equal_loudness_spl(f, 40.0)
            self.assertAlmostEqual(got, expect, delta=0.6,
                                   msg="40方 @ %.0fHz 应约 %.1f dB，实得 %.2f" % (f, expect, got))

    def test_published_values_80phon_10khz(self):
        """另一档独立校验：80 方 @ 10kHz 公布值约 92 dB。"""
        got = psycho.equal_loudness_spl(10000.0, 80.0)
        self.assertAlmostEqual(got, 92.0, delta=1.0,
                               msg="80方 @ 10kHz 应约 92 dB，实得 %.2f" % got)

    def test_threshold_of_hearing_4khz(self):
        """0 方 = 听阈。4 kHz 处公布 MAF 约 −8 dB SPL（人耳最灵敏处）。"""
        got = psycho.equal_loudness_spl(4000.0, 0.0)
        self.assertAlmostEqual(got, -8.0, delta=2.0,
                               msg="听阈 @ 4kHz 应约 −8 dB，实得 %.2f" % got)

    def test_low_freq_needs_more_spl(self):
        """等响曲线的基本形状：两端需要更高 SPL，中频最敏感。"""
        for phon in (20.0, 40.0, 60.0, 80.0):
            lo = psycho.equal_loudness_spl(50.0, phon)
            mid = psycho.equal_loudness_spl(1000.0, phon)
            hi = psycho.equal_loudness_spl(8000.0, phon)
            self.assertGreater(lo, mid, "50Hz 应比 1kHz 需要更大 SPL（%.1f 方）" % phon)
            self.assertGreater(hi, mid, "8kHz 应比 1kHz 需要更大 SPL（%.1f 方）" % phon)

    def test_curve_flattens_with_level(self):
        """响度补偿：phon 越高曲线越平 → 20Hz 与 1kHz 的 SPL 差越小。"""
        d40 = psycho.equal_loudness_spl(20.0, 40.0) - 40.0
        d80 = psycho.equal_loudness_spl(20.0, 80.0) - 80.0
        self.assertLess(d80, d40, "80 方的 20Hz 差应小于 40 方：%.1f vs %.1f" % (d80, d40))

    def test_tables_consistent_length(self):
        n = len(psycho.F_TABLE)
        self.assertEqual(n, 29)
        for name, t in (("AF", psycho.AF_TABLE), ("LU", psycho.LU_TABLE), ("TF", psycho.TF_TABLE)):
            self.assertEqual(len(t), n, "%s 表长度应为 %d" % (name, n))

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            psycho.equal_loudness_spl(100.0, 95.0)
        with self.assertRaises(ValueError):
            psycho.equal_loudness_spl(100.0, -1.0)

    def test_interpolation_between_table_points(self):
        """表间插值必须连续（对数频率线性插值，不应有跳变）。"""
        prev = None
        for i in range(200):
            f = 20.0 * (12500.0 / 20.0) ** (i / 199.0)
            v = psycho.equal_loudness_spl(f, 60.0)
            if prev is not None:
                self.assertLess(abs(v - prev), 1.0,
                                "插值跳变过大：%.1fHz 处 %.2f vs %.2f" % (f, v, prev))
            prev = v

    def test_sensitivity_normalised_at_1khz(self):
        """权重要求：**1 kHz 处必须恰好为 1**（归一化基准，floor 不得把它压下去）。

        不断言邻近频率也等于 1：等响曲线在 800–1250 Hz 并不平坦
        （实测 800 Hz 敏感度高 0.35 dB、1250 Hz 低 2.48 dB），这是正确行为。
        """
        w = psycho.weight_fn(phon=80.0, floor=0.05)
        self.assertAlmostEqual(w(1000.0), 1.0, places=9)
        self.assertGreater(w(800.0), 1.0)           # 800Hz 更敏感
        self.assertLess(w(1250.0), 1.0)             # 1250Hz 更不敏感
        # 最敏感的区间应在 1–4 kHz（人耳最灵敏处），而不是端点
        for f in (2000.0, 3000.0, 4000.0):
            self.assertGreater(w(f), w(20.0))
            self.assertGreater(w(f), w(15000.0))

    def test_sensitivity_decreasing_at_extremes(self):
        w = psycho.weight_fn(phon=80.0, floor=0.05)
        self.assertLess(w(20.0), w(200.0))
        self.assertLess(w(16000.0), w(2000.0))
        self.assertGreaterEqual(w(20.0), 0.05)      # floor 生效
        self.assertGreaterEqual(w(16000.0), 0.05)

    def test_sensitivity_rises_with_phon(self):
        """phon 越高，低频权重越高（响度补偿）。

        必须用**极小 floor**：40 方时 30 Hz 的敏感度本身已低于 1e-3，
        若 floor 取 0.001 会把两者一起钳住，测不出差异。
        """
        w40 = psycho.weight_fn(phon=40.0, floor=1e-12)
        w80 = psycho.weight_fn(phon=80.0, floor=1e-12)
        self.assertGreater(w80(30.0), w40(30.0))
        self.assertGreater(w80(60.0), w40(60.0))
        self.assertAlmostEqual(w80(1000.0), w40(1000.0), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
