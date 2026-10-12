#!/usr/bin/env python3
"""FR 客观特征层 v1.1：同时产出两种偏差基准，不靠任何主观转述。

1) dev  = 曲线 − **人群中位基线**（数据集自定义，用于描述"这只耳机在人群里长什么样"）
2) devt = 曲线 − **数据源自带目标曲线**（CSV 的 error 列；oratory1990 ≈ Harman 系）
          → 这才是"偏离主流好声标准多少"，推荐打分必须用它，否则开放式会被系统性冤枉。

统一网格 1/12 倍频程 20Hz–20kHz（121 点）；每曲线自身归一化（300–3kHz=0）。
输出 data/features.json。
"""
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(DATA, "fr_manifest.jsonl")

N = 121
FLO, FHI = 20.0, 20000.0
GRID = [FLO * (FHI / FLO) ** (i / (N - 1)) for i in range(N)]
NORM_LO, NORM_HI = 300.0, 3000.0

BANDS = {
    "sub_bass": (20, 60), "bass": (60, 200), "low_mid": (200, 500),
    "mid": (500, 2000), "presence": (2000, 5000), "treble": (5000, 10000),
    "air": (10000, 20000),
}


def load_curve(path):
    """读 AutoEq CSV -> (freqs, raw, error)。error 列为空则返回 None 占位。"""
    fs, raw, err = [], [], []
    idx = None
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        for ln in fh:
            parts = [p.strip() for p in ln.strip().split(",")]
            if len(parts) < 4:
                continue
            if idx is None:
                head = [p.lower() for p in parts]
                if "frequency" in head and "raw" in head:
                    idx = (head.index("frequency"), head.index("raw"),
                           head.index("error") if "error" in head else None)
                    continue
                idx = (0, 1, 2)
            try:
                fs.append(float(parts[idx[0]]))
                raw.append(float(parts[idx[1]]))
                err.append(float(parts[idx[2]]) if idx[2] is not None and parts[idx[2]] else 0.0)
            except ValueError:
                continue
    if len(fs) < 20:
        return None
    return fs, raw, err


def resample(fs, ds):
    out, j = [], 0
    for g in GRID:
        while j < len(fs) - 2 and fs[j + 1] < g:
            j += 1
        while j > 0 and fs[j] > g:
            j -= 1
        f0, f1 = fs[j], fs[j + 1] if j + 1 < len(fs) else fs[j]
        d0, d1 = ds[j], ds[j + 1] if j + 1 < len(ds) else ds[j]
        if f1 <= f0:
            out.append(d0)
            continue
        t = (math.log(g) - math.log(f0)) / (math.log(f1) - math.log(f0))
        out.append(d0 + t * (d1 - d0))
    return out


def normalize(curve):
    sel = [curve[i] for i in range(N) if NORM_LO <= GRID[i] <= NORM_HI]
    m = sum(sel) / len(sel)
    return [v - m for v in curve]


def band_mean(dev, lo, hi):
    sel = [dev[i] for i in range(N) if lo <= GRID[i] <= hi]
    return sum(sel) / len(sel) if sel else 0.0


def features(dev):
    f = {k: round(band_mean(dev, *r), 2) for k, r in BANDS.items()}
    xs = [math.log2(GRID[i]) for i in range(N) if 1000 <= GRID[i] <= 10000]
    ys = [dev[i] for i in range(N) if 1000 <= GRID[i] <= 10000]
    if len(xs) > 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        f["tilt_1k_10k"] = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 3)
    zone = [(dev[i], GRID[i]) for i in range(N) if 5000 <= GRID[i] <= 10000]
    if zone:
        g, fq = max(zone)
        f["peak_5_10k_db"] = round(g, 2)
        f["peak_5_10k_hz"] = int(fq)
    f["sub_extension_db"] = round(band_mean(dev, 20, 40) - band_mean(dev, 60, 200), 2)
    f["roughness"] = round(sum(abs(dev[i + 1] - dev[i]) for i in range(N - 1)) / (N - 1), 3)
    return f


def main():
    rows = []
    with open(MANIFEST, encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            c = load_curve(os.path.join(ROOT, r["csv"]))
            if not c:
                continue
            fs, raw, err = c
            rows.append({"meta": r,
                         "curve": normalize(resample(fs, raw)),   # 绝对形状
                         "devt": resample(fs, err)})              # 相对来源目标曲线
    print("载入曲线 %d 条" % len(rows))
    ref = [st.median([r["curve"][i] for r in rows]) for i in range(N)]

    out = []
    for r in rows:
        dev = [r["curve"][i] - ref[i] for i in range(N)]
        out.append({"name": r["meta"]["name"], "source": r["meta"]["source"],
                    "coupler": r["meta"].get("coupler"), "csv": r["meta"]["csv"],
                    "features": features(dev), "dev": [round(x, 2) for x in dev],
                    "devt": [round(x, 2) for x in r["devt"]],
                    "features_t": features(r["devt"])})
    with open(os.path.join(DATA, "features.json"), "w", encoding="utf-8") as fh:
        json.dump({"grid": GRID, "reference": [round(x, 2) for x in ref],
                   "n": len(out), "models": out}, fh, ensure_ascii=False)
    print("已写 data/features.json：%d 只（含 dev 与 devt 两套）" % len(out))

    # 抽查：EDXS / HD600 相对 Harman 的偏差
    for want in ["Edition XS", "HD 600", "HD 650"]:
        hit = [m for m in out if want.lower() in m["name"].lower()]
        for h in hit[:1]:
            t = h["features_t"]
            print("  %-28s [相对Harman] 低频 %+.1f ｜ presence %+.1f ｜ 高频 %+.1f ｜ 5-10k峰 %+.1f@%sHz ｜ 倾角 %+.2f"
                  % (h["name"][:28], t["bass"], t["presence"], t["treble"],
                     t.get("peak_5_10k_db", 0), t.get("peak_5_10k_hz", "?"), t["tilt_1k_10k"]))


if __name__ == "__main__":
    main()
