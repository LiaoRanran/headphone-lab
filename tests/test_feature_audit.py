"""特征互审：相关性与 VIF 的算法 + 两条纪律。

## 钉住什么

1. **相关/VIF 的实现要对**（同列相关系数为 1、独立列 VIF ≈ 1），
   否则报出来的数字全是错的而没人知道；
2. **特征名写错必须响**——早先把 `sub_extension_db` 写成 `sub_extension`，
   那一对就静默不打印，报告看着一切正常。查不到与没问题长得一样，
   是本项目最忌讳的失败方式；
3. **低频三特征的实测相关性**——这是 README §4.5 那句断言的地基，
   数字变了就说明特征定义被改过，得重新审。
"""
from __future__ import annotations

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import feature_audit as FA  # noqa: E402


class TestMath(unittest.TestCase):
    def test_identical_column_correlates_with_itself(self):
        rows = [{"a": v, "b": v * 2} for v in (1, 2, 3, 4, 5)]
        cor = FA.correlation(["a", "b"], rows)
        self.assertAlmostEqual(cor[("a", "b")], 1.0, places=6)

    def test_anticorrelated_column_is_minus_one(self):
        rows = [{"a": v, "b": -v} for v in (1, 2, 3, 4, 5)]
        self.assertAlmostEqual(FA.correlation(["a", "b"], rows)[("a", "b")], -1.0, places=6)

    def test_independent_columns_give_vif_near_one(self):
        # 交替反相的两列：彼此不相关 → VIF ≈ 1（不是 0，也不是 2）
        rows = [{"a": (1.0 if i % 2 else -1.0), "b": (1.0 if i % 3 == 0 else -1.0)}
                for i in range(12)]
        vs = FA.vif(["a", "b"], rows)
        for k in ("a", "b"):
            self.assertGreater(vs[k], 0.95)
            self.assertLess(vs[k], 1.05 + 1e-9)

    def test_perfectly_collinear_is_not_silently_finite(self):
        rows = [{"a": v, "b": v * 3, "c": (v * 7) % 5} for v in range(1, 20)]
        vs = FA.vif(["a", "b", "c"], rows)
        self.assertTrue(any(v == float("inf") or v > 1e6 for v in vs.values()),
                        "完全共线必须表现成无穷大/极大，不能给个好看的数字")


class TestDiscipline(unittest.TestCase):
    def test_redundant_pairs_are_sorted_by_abs_r(self):
        rows = [{"a": v, "b": v, "c": -v} for v in range(1, 30)]
        pairs = FA.redundant(["a", "b", "c"], rows, threshold=0.8)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([round(p[0], 3) for p in pairs], [1.0, 1.0, 1.0])

    def test_missing_feature_name_is_loud(self):
        names, rows = FA.load()
        keep = FA.LOW_BAND_FEATURES
        try:
            FA.LOW_BAND_FEATURES = ("sub_bass", "no_such_feature", "low_mid")
            text = FA.report()
        finally:
            FA.LOW_BAND_FEATURES = keep
        self.assertIn("no_such_feature", text, "名字写错却什么都没说")

    def test_low_band_correlations_are_what_readme_says(self):
        """低频三特征的实测相关性——README §4.5 的更正就建立在这三个数上。"""
        names, rows = FA.load()
        cor = FA.correlation(names, rows)
        pair = lambda a, b: abs(cor[(a, b)] if (a, b) in cor else cor[(b, a)])  # noqa: E731
        self.assertAlmostEqual(pair("sub_bass", "sub_extension_db"), 0.85, delta=0.03)
        self.assertAlmostEqual(pair("sub_bass", "low_mid"), 0.23, delta=0.03)
        self.assertAlmostEqual(pair("sub_extension_db", "low_mid"), 0.07, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
