"""测量噪声基线：先量仪器的噪声，再谈被测对象的差异。

## 钉住什么

1. **单源型号不参与基线**——没有"来源之间"就没有离散度可言；
2. **多源完全一致 → 基线为 0**：此时任何真实偏差都算"超出"；
3. **小于基线的偏差不得判为超出**——这条是本模块存在的意义：
   它挡的是"把测量架的差异当成耳机的差异"这种误读；
4. **基线样本不足（<3 个多源型号）时返回 None**：与 `spatial` 同一条规矩，
   没测 ≠ 没差，不知道就说不知道。
"""
from __future__ import annotations

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import repeatability as R  # noqa: E402

GRID = [100.0 * (2 ** (i / 12)) for i in range(80)]      # ~100 Hz–5 kHz


def _dev(offset):
    return [offset] * len(GRID)


class TestSpread(unittest.TestCase):
    def test_single_source_models_are_ignored(self):
        models = [{"name": "A", "dev": _dev(1.0)}]
        self.assertEqual(R.per_model_spread(models, GRID, 100, 5000), [])

    def test_identical_sources_give_zero_floor(self):
        models = [{"name": "A", "dev": _dev(3.0)},
                  {"name": "A", "dev": _dev(3.0)},
                  {"name": "A", "dev": _dev(3.0)}]
        rows = R.per_model_spread(models, GRID, 100, 5000)
        self.assertEqual(rows[0]["range_db"], 0.0)
        nf = R.noise_floor(models, GRID, 100, 5000)
        self.assertEqual(nf["median_range_db"], 0.0)

    def test_floor_tracks_observed_spread(self):
        models = [{"name": "A", "dev": _dev(o)} for o in (1.0, 3.0, 5.0)]
        nf = R.noise_floor(models, GRID, 100, 5000)
        self.assertAlmostEqual(nf["median_range_db"], 4.0, places=2)
        self.assertAlmostEqual(nf["median_std_db"], 1.63, places=2)

    def test_only_counts_within_band(self):
        # 偏差放在带外（<100 Hz）不应影响带内统计
        dev = [5.0 if f < 100 else 0.0 for f in GRID]
        models = [{"name": "A", "dev": dev}, {"name": "A", "dev": _dev(0.0)}]
        rows = R.per_model_spread(models, GRID, 100, 5000)
        self.assertEqual(rows[0]["range_db"], 0.0)


class TestJudge(unittest.TestCase):
    def setUp(self):
        # 三个型号 × 各三个来源（来源间差 1 dB）：基线极差 = 2 dB。
        # **必须 ≥3 个多源型号**——这是本模块自己定的下限，测试也得守。
        self.models = [{"name": name, "dev": _dev(o)}
                       for name in ("A", "B", "C") for o in (-1.0, 0.0, 1.0)]
        self.floor = R.noise_floor(self.models, GRID, 100, 5000)

    def test_below_floor_is_not_exceeding(self):
        r = R.exceeds(_dev(1.5), GRID, 100, 5000, self.floor)
        self.assertFalse(r["exceeds"], "1.5 dB < 基线 2 dB：这是测量架的量级，不该下结论")
        self.assertAlmostEqual(r["ratio"], 0.75, places=2)

    def test_above_floor_exceeds(self):
        r = R.exceeds(_dev(6.0), GRID, 100, 5000, self.floor)
        self.assertTrue(r["exceeds"])
        self.assertAlmostEqual(r["ratio"], 3.0, places=2)

    def test_negative_deviation_counts_by_magnitude(self):
        r = R.exceeds(_dev(-6.0), GRID, 100, 5000, self.floor)
        self.assertTrue(r["exceeds"])

    def test_too_few_models_gives_none(self):
        small = R.noise_floor(self.models[:2], GRID, 100, 5000)   # 只有 1 个多源型号
        self.assertIsNone(R.exceeds(_dev(9.0), GRID, 100, 5000, small),
                          "样本不足必须说不知道，不能给 0")

    def test_no_floor_at_all(self):
        self.assertIsNone(R.exceeds(_dev(1.0), GRID, 100, 5000, {"n_models": 0}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
