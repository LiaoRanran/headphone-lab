#!/usr/bin/env python3
"""squig 曲线（REW 文本）→ 本项目标准 CSV。

输出列：frequency, raw, smoothed, error, target
  raw = 自身归一化曲线（300–3k 均值 = 0）
  error = raw − 数据集人群基线（features.json 的 reference）
  → 因此 dev = devt = 相对人群基线。**这些条目标 basis=population**，与 AutoEq 的
    Harmon 基准（devt = raw − 来源目标曲线）不是同一坐标，混用需先对齐（见 README 下一步）。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr_features as ff  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "squig_raw")
OUT = os.path.join(DATA, "squig_csv")


def parse_rew(path):
    fs, ds = [], []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        ln = ln.strip()
        if not ln or ln[0] in "*#;":
            continue
        parts = [p for p in ln.replace(",", " ").replace("\t", " ").split() if p]
        if len(parts) < 2:
            continue
        try:
            f, d = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if f > 0:
            fs.append(f)
            ds.append(d)
    return (fs, ds) if len(fs) > 20 else None


def main():
    with open(os.path.join(DATA, "features.json"), encoding="utf-8") as f:
        ref = json.load(f)["reference"]
    grid = ff.GRID
    man = [json.loads(l) for l in open(os.path.join(DATA, "squig_manifest.jsonl"), encoding="utf-8")]
    rows, bad = [], 0
    for r in man:
        p = os.path.join(ROOT, r["txt"])
        c = parse_rew(p)
        if not c:
            bad += 1
            continue
        curve = ff.normalize(ff.resample(*c))
        sub = os.path.join(OUT, r["site"])
        os.makedirs(sub, exist_ok=True)
        dst = os.path.join(sub, re.sub(r"[^\w\u4e00-\u9fff.+\-()\[\] ]+", "_", r["stem"]) + ".csv")
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write("frequency,raw,smoothed,error,error_smoothed,target\n")
            for i in range(ff.N):
                err = round(curve[i] - ref[i], 3)
                f.write("%.3f,%.3f,%.3f,%.3f,%.3f,0\n" % (grid[i], curve[i], curve[i], err, err))
        rows.append({"name": r["stem"], "source": "squig:" + r["site"], "coupler": None,
                     "basis": "population", "csv": os.path.relpath(dst, ROOT)})

    # 与 AutoEq 现有型号对比
    have = set()
    for ln in open(os.path.join(DATA, "fr_manifest.jsonl"), encoding="utf-8"):
        have.add(json.loads(ln)["name"].lower())
    new = [r for r in rows if r["name"].lower().replace(" l", "").replace(" r", "") not in have]
    with open(os.path.join(DATA, "squig_csv_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("转换成功 %d ｜ 解析失败 %d ｜ 输出 %s" % (len(rows), bad, OUT))
    print("其中「名字不在 AutoEq 里」的候选新增：%d" % len(new))
    for x in new[:10]:
        print("   ", x["name"], "|", x["source"])


if __name__ == "__main__":
    main()
