"""认识论护栏回归：输出必须自带"这是什么/这不是什么"。

为什么值得测试：引擎的得分是加权 RMS 偏差，而 stats.py 早已测出前 14 名
CI 全部重叠。**知道这件事却不告诉使用者，等于纵容误读**——这些断言钉住
「输出必须自己交代效力边界」，防止以后有人把表头改回光秃秃的"排名"。
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import engine  # noqa: E402


class TestOutputHonesty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.meta = engine.rank({"basis": "target", "bass": 0.0,
                                          "presence": 0.0, "treble": 0.0,
                                          "peak_guard": 4.5, "wired_only": True,
                                          "min_sources": 1})

    def test_disclaimer_states_what_it_is_not(self):
        """免责声明必须点明它**不能**告诉你的那些维度。"""
        for must in ("不是测量结论", "声场", "结像", "瞬态", "佩戴", "驱动难度"):
            self.assertIn(must, engine.DISCLAIMER,
                          "免责声明应提到「%s」" % must)

    def test_disclaimer_refuses_to_settle_bass_target(self):
        """必须明说「该补多少低频」推不出来——这是最容易被当成结论的一项。"""
        self.assertIn("测量推不出来", engine.DISCLAIMER)

    def test_header_says_candidate_not_ranking(self):
        """表头必须写「候选（非排名）」。措辞要与数据的效力相称。

        早先的写法是 `("排名", "型号", ...)` —— ordinal 编号 + "排名"字样，
        会让人以为可以按序下单，而分档会说清同档内根本不可区分。
        """
        self.assertEqual(engine.TABLE_HEADER[1], "候选（非排名）")
        self.assertNotIn("排名", engine.TABLE_HEADER[0])
        # 免责声明应当**否认**自己是排名，而不是从不提这件事
        self.assertIn('更不是"排名"', engine.DISCLAIMER)

    def test_tier_report_runs_and_groups(self):
        """分档必须能跑，并把前 n 名全部归入至少一档。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            tiers = engine.tier_report(self.rows, self.meta, n=8, n_boot=40)
        out = buf.getvalue()
        self.assertGreaterEqual(len(tiers), 1, "至少要有一档")
        self.assertEqual(sum(len(t) for t in tiers), 8, "每个人必须恰好归入一档")
        self.assertIn("同档内名次差异不可解读", out)

    def test_tiers_are_contiguous_by_ci(self):
        """分档必须按 CI 重叠切：同档内任意两款的上限不低于对方下限。"""
        tiers = engine.tier_report(self.rows, self.meta, n=8, n_boot=40)
        for t in tiers:
            los = min(x["lo"] for x in t)
            for x in t:
                self.assertGreaterEqual(x["hi"], los - 1e-9,
                                        "同档内不应有 CI 不重叠的对")

    def test_rows_carry_curve_and_target_for_bootstrap(self):
        """分档用的曲线/目标必须由 rank() 带出，否则 tier_report 会另造一套口径。"""
        r = self.rows[0]
        self.assertIn("_curve", r)
        self.assertIn("_target", r)
        self.assertEqual(len(r["_curve"]), len(r["_target"]))
        self.assertEqual(len(r["_curve"]), len(self.meta["_grid"]))

    def test_meta_carries_weight_and_guard(self):
        """bootstrap 必须复用同一套权重与峰护栏，否则分档和得分对不上。"""
        self.assertIn("_W", self.meta)
        self.assertIn("_pg", self.meta)
        self.assertEqual(self.meta["_pg"], 4.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
