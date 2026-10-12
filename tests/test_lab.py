#!/usr/bin/env python3
"""headphone-lab 回归测试：钉住数据层与引擎的关键行为。

运行:  python -m unittest discover -s tests -v
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ff = load("fr_features")


class TestData(unittest.TestCase):
    def test_manifest_exists(self):
        p = os.path.join(ROOT, "data", "fr_manifest.jsonl")
        self.assertTrue(os.path.exists(p), "缺少 fr_manifest.jsonl（先跑 fetch_fr.py）")
        n = sum(1 for _ in open(p, encoding="utf-8"))
        self.assertGreaterEqual(n, 1000, "实测曲线应 ≥1000 条，实际 %d" % n)

    def test_features_exists(self):
        p = os.path.join(ROOT, "data", "features.json")
        self.assertTrue(os.path.exists(p))
        d = json.load(open(p, encoding="utf-8"))
        self.assertEqual(len(d["grid"]), 121)
        self.assertGreaterEqual(d["n"], 1000)


class TestFeatures(unittest.TestCase):
    def test_grid_is_log_spaced(self):
        r = ff.GRID[-1] / ff.GRID[0]
        self.assertAlmostEqual(r, ff.FHI / ff.FLO, places=3)

    def test_normalize_zeroes_the_band(self):
        c = [10.0] * ff.N
        out = ff.normalize(c)
        self.assertAlmostEqual(sum(out) / len(out), 0.0, places=6)

    def test_flat_curve_has_no_tilt_or_peak(self):
        f = ff.features([0.0] * ff.N)
        self.assertAlmostEqual(f["tilt_1k_10k"], 0.0, places=6)
        self.assertAlmostEqual(f["peak_5_10k_db"], 0.0, places=6)

    def test_peak_is_detected(self):
        c = [0.0] * ff.N
        i = min(range(ff.N), key=lambda k: abs(ff.GRID[k] - 6700))
        c[i] = 9.0
        f = ff.features(c)
        self.assertGreater(f["peak_5_10k_db"], 5)
        self.assertTrue(5000 <= f["peak_5_10k_hz"] <= 10000)


class TestEngineCLI(unittest.TestCase):
    """端到端：引擎必须可跑、确定、且能识别 EDXS 的高频峰。"""

    def run_engine(self, *args):
        p = subprocess.run([sys.executable, os.path.join(SRC, "engine.py"), *args],
                           capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return p.stdout + p.stderr

    def test_explain_edition_xs_flags_peak(self):
        out = self.run_engine("--explain", "Edition XS", "--top", "1")
        self.assertIn("Edition XS", out)
        self.assertIn("峰", out)
        # Edition XS 的 6–7k 尖峰是"难听"的物理来源，必须被检出
        self.assertRegex(out, r"\+[4-9]\.\d+dB@\d+Hz")

    def test_deterministic(self):
        a = self.run_engine("--bass", "3", "--top", "5")
        b = self.run_engine("--bass", "3", "--top", "5")
        self.assertEqual(a, b)

    def test_anchor_changes_ranking(self):
        plain = self.run_engine("--bass", "0", "--top", "3")
        anch = self.run_engine("--bass", "0", "--like", "HD 600", "--like", "HD 650", "--top", "3")
        self.assertNotEqual(plain, anch, "锚点必须改变排序")


if __name__ == "__main__":
    unittest.main(verbosity=2)
