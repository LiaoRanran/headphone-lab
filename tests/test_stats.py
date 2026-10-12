"""统计严谨性：CI 计算与分档聚类。

核心断言：**重叠的 CI 必须被聚成同一档** —— 这是"不给伪精度"的机械化保证。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import stats  # noqa: E402


class TestTiers(unittest.TestCase):
    def test_overlapping_ci_merge_into_one_tier(self):
        ann = [
            {"name": "A", "ci_lo": 85.0, "ci_hi": 91.0},
            {"name": "B", "ci_lo": 84.0, "ci_hi": 88.0},   # 与 A 重叠
            {"name": "C", "ci_lo": 86.0, "ci_hi": 89.0},   # 与 A/B 都重叠
        ]
        t = stats.tiers(ann)
        self.assertEqual(len(t), 1, "CI 重叠的三个型号必须同档")
        self.assertEqual(len(t[0]), 3)

    def test_separated_ci_split_into_two_tiers(self):
        ann = [
            {"name": "A", "ci_lo": 90.0, "ci_hi": 95.0},
            {"name": "B", "ci_lo": 91.0, "ci_hi": 94.0},
            {"name": "C", "ci_lo": 80.0, "ci_hi": 84.0},   # 与 A/B 不重叠
        ]
        t = stats.tiers(ann)
        self.assertEqual(len(t), 2)
        self.assertEqual(len(t[0]), 2)
        self.assertEqual(len(t[1]), 1)

    def test_chain_overlap_merges_all(self):
        """A-B 重叠、B-C 重叠，即使 A-C 不重叠也该同档（传递性）。"""
        ann = [
            {"name": "A", "ci_lo": 90.0, "ci_hi": 95.0},
            {"name": "B", "ci_lo": 88.0, "ci_hi": 91.0},   # 与 A 重叠
            {"name": "C", "ci_lo": 85.0, "ci_hi": 89.0},   # 与并集 [88,95] 重叠
        ]
        t = stats.tiers(ann)
        self.assertEqual(len(t), 1, "链式重叠应传递合并")
        self.assertEqual(len(t[0]), 3)

    def test_missing_ci_breaks_tier(self):
        ann = [
            {"name": "A", "ci_lo": 85.0, "ci_hi": 91.0},
            {"name": "B", "ci_lo": None, "ci_hi": None},
            {"name": "C", "ci_lo": 86.0, "ci_hi": 90.0},
        ]
        t = stats.tiers(ann)
        self.assertEqual(len(t), 3, "无 CI 的型号不能并入任何档")


class TestScoreWithCi(unittest.TestCase):
    def _setup(self):
        import engine
        D = engine.load()
        grid = D["grid"]
        W = engine.base_weight()
        target = [0.0] * len(grid)
        return grid, W, target, D

    def test_point_estimate_in_ci(self):
        grid, W, target, D = self._setup()
        G = engine_group()
        name = sorted(G)[0]
        curve = G[name]["curve_t"]
        p, lo, hi, se = stats.score_with_ci(curve, target, grid, W, n_boot=120, seed=1)
        self.assertIsNotNone(p)
        self.assertIsNotNone(lo)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)
        self.assertGreater(se, 0)

    def test_identical_curve_has_tiny_ci(self):
        """与目标完全一致的曲线，得分应接近 100 且 CI 很窄。"""
        grid, W, target, D = self._setup()
        p, lo, hi, se = stats.score_with_ci(list(target), target, grid, W,
                                            n_boot=120, seed=2)
        self.assertAlmostEqual(p, 100.0, places=6)
        self.assertAlmostEqual(lo, 100.0, places=6)
        self.assertAlmostEqual(hi, 100.0, places=6)
        self.assertAlmostEqual(se, 0.0, places=6)

    def test_large_deviation_has_wide_ci(self):
        """**频率相关**的大偏离，bootstrap 的 SE 应显著。

        注意不能用均匀 +6 dB 偏移造这个测试：均匀偏移在任何频点子集下
        得分都完全相同 → SE 恒为 0。那是**正确行为**（偏移与频点选择无关），
        不是缺陷。要测 CI 宽度就必须让偏离随频率变化。
        """
        grid, W, target, D = self._setup()
        n = len(grid)
        # 低频段 +8 dB、高频段 −8 dB：偏离集中在部分频点，
        # 重采样频点会明显改变加权后的 RMS → SE 应大
        bad = [t + (8.0 if grid[i] < 300 else (-8.0 if grid[i] > 4000 else 0.0))
               for i, t in enumerate(target)]
        _, lo, hi, se = stats.score_with_ci(bad, target, grid, W, n_boot=150, seed=3)
        self.assertGreater(se, 0.5, "频率相关大偏离的 SE 应显著，实得 %s" % se)
        self.assertLess(lo, 100.0)

    def test_uniform_offset_has_zero_se(self):
        """均匀偏移 → SE 必须为 0（偏移与频点选择无关）。

        这条钉住一个容易误判为 bug 的正确行为。
        """
        grid, W, target, D = self._setup()
        bad = [t + 6.0 for t in target]
        _, _, _, se = stats.score_with_ci(bad, target, grid, W, n_boot=100, seed=4)
        self.assertAlmostEqual(se, 0.0, places=6)


def engine_group():
    import engine
    D = engine.load()
    return engine.group(D["models"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
