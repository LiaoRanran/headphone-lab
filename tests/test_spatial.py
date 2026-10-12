"""空间维度：机制可证伪、口径不越界。

## 钉住什么

1. **共模滤波器不改 ILD**——这是整个模块的物理地基。
   "频响不平 → 声场歪"若成立，这条断言就会失败；它没失败，说明那句话是错的。
2. **没测 ≠ 没差**：无逐声道数据时 `ild_error_db` 必须是 `None`，不能是 0。
   0 会让人读成"这只耳机不改 ILD"，那是把"不知道"说成了"没有"。
3. **没有阈值的条目不许当判据**：与 `tests/test_audibility.py` 那条同源的规矩。
   想给某个频段加可闻阈，先拿得出原文出处，否则 `usable_as_threshold` 必须为假。
4. **普查要自查来源偏差**：极端值被某一个测量来源垄断时，量到的多半是测量架。
"""
from __future__ import annotations

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import spatial as S  # noqa: E402


class TestMechanism(unittest.TestCase):
    """ILD 只由**左右差**决定；两耳共有的滤波器不参与。"""

    def test_common_filter_does_not_change_ild(self):
        # 录音里本来就有 ILD：左耳比右耳高 3 dB
        x_l, x_r = [3.0] * 5, [0.0] * 5
        ild_in = [l - r for l, r in zip(x_l, x_r)]
        # 耳机自身的偏差 H 加在两耳上（共模），哪怕它很不平整
        h = [1.0, -4.0, 6.0, 0.5, -2.0]
        ild_out = [(l + hh) - (r + hh) for l, r, hh in zip(x_l, x_r, h)]
        self.assertEqual(ild_in, ild_out, "共模滤波器改了 ILD → 机制理解有误")

    def test_decompose_splits_common_and_difference(self):
        common, diff = S.decompose([2.0, 4.0], [0.0, 2.0])
        self.assertEqual([round(x, 3) for x in common], [1.0, 3.0])
        self.assertEqual([round(x, 3) for x in diff], [2.0, 2.0])

    def test_identical_channels_mean_zero_ild_error(self):
        self.assertEqual(S.ild_perturbation([1.0, 2.0], [1.0, 2.0]), [0.0, 0.0])


class TestHonestGaps(unittest.TestCase):
    """该说"不知道"的地方不许说成"没有"。"""

    def test_no_channel_data_gives_none_not_zero(self):
        grid = [5000.0, 6000.0]
        a = S.analyse([1.0, 2.0], grid)
        self.assertIsNone(a["ild_error_db"], "没测必须给 None，不能给 0")
        self.assertFalse(a["has_channel_data"])

    def test_with_channel_data_reports_ild_error(self):
        a = S.analyse([1.0, 2.0], [5000.0, 6000.0], dev_r=[0.0, 2.0])
        self.assertEqual(a["ild_error_db"]["worst_db"], 1.0)
        self.assertEqual(a["ild_error_worst_hz"], 5000.0)

    def test_verdict_states_the_gap(self):
        t = S.verdict_text(S.analyse([1.0, 2.0], [5000.0, 6000.0]))
        self.assertIn("本库不知道", t, "措辞必须把「没测」说出来")


class TestNoInventedThresholds(unittest.TestCase):
    """与 audibility 同规矩：没有出处的阈值不许存在。"""

    def test_no_band_is_usable_as_threshold(self):
        for b in S.BANDS:
            self.assertFalse(b.usable_as_threshold,
                             "%s 被标成可作判据了——先拿出原文出处" % b.key)

    def test_unverified_bands_are_flagged(self):
        self.assertFalse(S.BY_KEY["spatial_100_1600"].verified,
                         "100–1600 Hz 那条若已取到原文，请更新登记与 README")
        self.assertFalse(S.BY_KEY["itd_dominant"].verified)
        self.assertFalse(S.BY_KEY["ild_dominant"].verified)

    def test_verified_bands_can_be_reported(self):
        self.assertTrue(S.BY_KEY["frontback"].verified)
        self.assertEqual(S.BY_KEY["frontback"].lo, 5000.0)
        self.assertEqual(S.BY_KEY["frontback"].hi, 6000.0)

    def test_band_stats(self):
        grid = [1000.0, 5500.0, 5800.0, 9000.0]
        st = S.band_stats([0.0, 2.0, -4.0, 1.0], grid, S.BY_KEY["frontback"])
        self.assertEqual(st["n"], 2)
        self.assertAlmostEqual(st["mean_db"], -1.0, places=2)
        self.assertEqual(st["worst_db"], -4.0)
        self.assertEqual(st["worst_hz"], 5800.0)


class TestSurveySelfCheck(unittest.TestCase):
    """普查要先怀疑自己的口径。"""

    def test_source_monopoly_is_detected(self):
        rows = [{"name": "a", "source": "X"}, {"name": "b", "source": "X"},
                {"name": "c", "source": "X"}, {"name": "d", "source": "Y"}]
        models = [{"source": "X"}] * 10 + [{"source": "Y"}] * 90
        bias = S.source_bias(rows, models)
        self.assertTrue(any(s == "X" for s, _, _ in bias),
                        "X 占极端值 75% 而全库只占 10% → 必须被诊断出来")

    def test_no_bias_when_proportional(self):
        rows = [{"name": "a", "source": "X"}, {"name": "b", "source": "Y"}]
        models = [{"source": "X"}, {"source": "Y"}]
        self.assertEqual(S.source_bias(rows, models), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
